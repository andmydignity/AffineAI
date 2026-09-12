import os
import torch
import torch.nn as nn
import pytest
from affine_ai.training.distributed import (
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    get_local_device,
    all_reduce_sum,
    all_reduce_avg,
    broadcast_parameters,
    parse_devices,
    find_free_port,
)
from affine_ai.training.trainer import train, ASDAGTrainer
from affine_ai.models.language_model import ASDAGLanguageModel, ASDAGConfig


def test_parse_devices():
    # Test CPU
    assert parse_devices("cpu") == []

    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    n_gpus = torch.cuda.device_count() if has_cuda else 0

    if not has_cuda:
        assert parse_devices("auto") == []
        assert parse_devices([0, 1]) == []
    else:
        assert len(parse_devices("auto")) == n_gpus
        assert len(parse_devices("all")) == n_gpus
        assert len(parse_devices(None)) == n_gpus
        
        # Test specific strings
        parsed = parse_devices("0")
        assert parsed == [0]
        
        parsed = parse_devices("cuda:0")
        assert parsed == [0]

        # Test list
        parsed = parse_devices([0])
        assert parsed == [0]

        # Test count
        parsed = parse_devices(1)
        assert len(parsed) == 1


def test_find_free_port():
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_single_process_distributed_fallbacks():
    # Outside distributed environment, helpers should return safe single-process defaults
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_process() is True

    t = torch.tensor([1.0, 2.0, 3.0])
    reduced = all_reduce_sum(t)
    assert torch.equal(reduced, torch.tensor([1.0, 2.0, 3.0]))

    t2 = torch.tensor([4.0, 5.0])
    reduced_avg = all_reduce_avg(t2)
    assert torch.equal(reduced_avg, torch.tensor([4.0, 5.0]))

    # Broadcast on unwrapped model does not fail
    m = nn.Linear(4, 4)
    broadcast_parameters(m)


def test_trainer_single_device_equivalence():
    # Test that train() accepts devices parameter without error
    model = ASDAGLanguageModel(vocab_size=256, d_model=64, num_leaves=4)
    data = torch.randint(0, 256, (1000,), dtype=torch.long)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    devices = [0] if torch.cuda.is_available() else "cpu"

    trainer = ASDAGTrainer(
        model=model,
        train_data=data,
        max_steps=2,
        batch_size=2,
        seq_len=64,
        device=device,
        devices=devices,
        use_cuda_graph=False,
    )
    res = trainer.train()
    assert "best_val_loss" in res
    assert res["best_val_loss"] >= 0.0


def _dummy_worker(rank, world_size, port, res_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    import torch.distributed as dist
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
    try:
        t = torch.tensor([float(rank + 1)])
        all_reduce_sum(t)
        all_reduce_avg(t)
        if rank == 0:
            res_queue.put(float(t.item()))
    finally:
        dist.destroy_process_group()


def test_gloo_multiprocess_reduction():
    # Verify multiprocessing reductions with Gloo backend
    import torch.multiprocessing as mp
    port = find_free_port()
    queue = mp.Queue()
    world_size = 2

    mp.spawn(
        _dummy_worker,
        args=(world_size, port, queue),
        nprocs=world_size,
        join=True,
    )

    result = queue.get(timeout=10)
    assert result > 0

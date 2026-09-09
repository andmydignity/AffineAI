import pytest
import torch
import numpy as np
from affine_ai.core.priority_replay import DynamicPriorityReplayBuffer
from affine_ai.models.language_model import ASDAGLanguageModel, ASDAGConfig
from affine_ai.training.trainer import ASDAGTrainer


def test_priority_replay_buffer_basic():
    buf = DynamicPriorityReplayBuffer(max_capacity=10, max_replays=2, min_loss_percentile=50.0)
    assert len(buf) == 0

    # Push some candidates
    indices = [10, 20, 30, 40, 50]
    losses = [1.5, 2.5, 3.2, 2.8, 1.8]
    enqueued = buf.push_candidates(indices, losses, threshold=2.0)
    assert enqueued == 3 # 20 (2.5), 30 (3.2), 40 (2.8)
    assert len(buf) == 3

    # Sample candidates
    rng = np.random.RandomState(42)
    sample = buf.sample(2, rng=rng)
    assert len(sample) == 2
    for s in sample:
        assert buf.buffer[s]["replays"] == 1

    # Sample again: items replayed 2 times should be retired
    sample2 = buf.sample(3, rng=rng)
    assert buf.total_replayed >= 4
    assert buf.total_retired >= 1


def test_priority_replay_eviction_policy():
    buf = DynamicPriorityReplayBuffer(max_capacity=3, max_replays=3)
    buf.push_candidates([1, 2, 3], [2.1, 2.5, 3.0], threshold=2.0)
    assert len(buf) == 3
    assert 1 in buf.buffer

    # Push a harder sample (loss 3.5), it should evict item 1 (lowest loss 2.1)
    buf.push_candidates([4], [3.5], threshold=2.0)
    assert len(buf) == 3
    assert 1 not in buf.buffer
    assert 4 in buf.buffer


def test_priority_replay_trainer_integration():
    model = ASDAGLanguageModel(vocab_size=256, d_model=64, n_layers=2)
    train_data = torch.randint(0, 256, (1000,))
    val_data = torch.randint(0, 256, (200,))

    trainer = ASDAGTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=8,
        seq_len=16,
        max_steps=5,
        use_priority_replay=True,
        replay_ratio=0.25,
        use_lpc=False
    )
    assert trainer.use_priority_replay is True
    assert trainer.replay_buffer is not None

    loss = trainer.train_step(0)
    assert isinstance(loss, (float, torch.Tensor))

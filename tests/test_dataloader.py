import pytest
import numpy as np
import torch
from affine_ai.data.dataloader import (
    PaddedDataLoader,
    FixedShapeDataLoader,
    CUDAGraphDataLoader,
)


def test_padded_dataloader_variable_length_sequences():
    # Variable-length text strings
    texts = [
        "Short",
        "A slightly longer sentence for testing",
        "Hello world!",
        "ASDAG MatMul-Free AI",
        "Tiny",
    ]
    batch_size = 2
    seq_len = 16
    pad_id = 0
    ignore_index = -100

    loader = PaddedDataLoader(
        texts,
        batch_size=batch_size,
        seq_len=seq_len,
        pad_id=pad_id,
        ignore_index=ignore_index,
        pad_remainder=True,
    )

    batches = list(loader)
    # 5 samples with B=2 -> 3 batches (2, 2, 1 padded to 2)
    assert len(batches) == 3

    for bx, by in batches:
        assert bx.shape == (batch_size, seq_len)
        assert by.shape == (batch_size, seq_len)
        assert bx.dtype == torch.long
        assert by.dtype == torch.long

        # Check that padded positions in by have ignore_index
        for i in range(batch_size):
            pad_mask = (bx[i] == pad_id)
            # If the input was padded, targets should also be masked
            if pad_mask.any():
                # Verify that trailing tokens are ignored
                last_non_pad = torch.where(~pad_mask)[0]
                if len(last_non_pad) > 0:
                    last_idx = last_non_pad[-1].item()
                    assert (by[i, last_idx + 1 :] == ignore_index).all()


def test_padded_dataloader_continuous_stream():
    # Continuous 1D byte array
    stream = np.arange(200, dtype=np.uint8)
    batch_size = 4
    seq_len = 16

    loader = FixedShapeDataLoader(
        stream,
        batch_size=batch_size,
        seq_len=seq_len,
        drop_last=False,
        pad_remainder=True,
    )

    for bx, by in loader:
        assert bx.shape == (batch_size, seq_len)
        assert by.shape == (batch_size, seq_len)
        # Next token relationship: y is shifted by 1 from x
        # for non-padded parts
        assert (by[0, :-1] == bx[0, 1:]).all()


def test_padded_dataloader_remainder_handling():
    # 7 items with B=4 -> 2 batches (4 and 3 padded to 4)
    data = [list(range(10)) for _ in range(7)]
    loader = CUDAGraphDataLoader(
        data,
        batch_size=4,
        seq_len=8,
        pad_remainder=True,
        drop_last=False,
    )

    batches = list(loader)
    assert len(batches) == 2
    assert batches[0][0].shape == (4, 8)
    assert batches[1][0].shape == (4, 8)

    # With drop_last=True -> exactly 1 batch
    loader_drop = CUDAGraphDataLoader(
        data,
        batch_size=4,
        seq_len=8,
        drop_last=True,
    )
    batches_drop = list(loader_drop)
    assert len(batches_drop) == 1
    assert batches_drop[0][0].shape == (4, 8)


def test_padded_dataloader_eos_injection():
    # Verify automatic <|endoftext|> injection
    texts = ["Hello world", "Machine learning"]
    loader = PaddedDataLoader(
        texts,
        batch_size=2,
        seq_len=32,
        inject_eos=True,
        eos_token="<|endoftext|>",
    )
    # Check that processed sequences end with <|endoftext|> bytes
    eos_bytes = "<|endoftext|>".encode("utf-8")
    for item in loader.data:
        arr_bytes = bytes(item.astype(np.uint8))
        assert arr_bytes.endswith(eos_bytes)

    # Test with custom int token ID
    int_data = [[1, 2, 3], [4, 5, 6, 7]]
    loader_int = PaddedDataLoader(
        int_data,
        batch_size=2,
        seq_len=10,
        inject_eos=True,
        eos_token=99,
    )
    for item in loader_int.data:
        assert item[-1] == 99


def test_padded_dataloader_as_stream_and_get_batch_by_indices():
    texts = ["Document one content.", "Second document text here."]
    loader = PaddedDataLoader(
        texts,
        batch_size=2,
        seq_len=8,
        as_stream=True,
        inject_eos=True,
    )
    assert loader.is_stream is True
    assert loader.stream_len > 0

    # Test get_batch_by_indices (used by Priority Replay)
    bx, by = loader.get_batch_by_indices([0, 1])
    assert bx.shape == (2, 8)
    assert by.shape == (2, 8)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_padded_dataloader_cuda_graph_integration():
    from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig

    config = TorosHybridConfig(
        context_dim=64,
        byte_dim=32,
        num_layers=2,
        target_patch_size=4,
        vocab_size=256,
        num_experts=1,
    )
    model = TorosHybridLanguageModel(config).cuda().bfloat16()
    opts = model.get_default_lpc_optimizers()

    # Variable length text corpus
    corpus = [
        "Once upon a time in a faraway forest",
        "There lived a curious little fox named Rust",
        "He loved finding shiny things under ancient oak trees",
        "One sunny morning, he found an enchanted compass",
        "The compass pointed towards the highest mountain peak",
        "So began his grand adventure across the valley",
    ]

    loader = PaddedDataLoader(
        corpus,
        batch_size=2,
        seq_len=32,
        device="cuda",
        pad_remainder=True,
    )

    # Verify that forward_lpc_step with default CUDA Graphs executes seamlessly
    step_count = 0
    for bx, by in loader:
        assert bx.is_cuda and by.is_cuda
        assert bx.shape == (2, 32)
        assert by.shape == (2, 32)

        # use_cuda_graph defaults to True!
        res = model.forward_lpc_step(bx, by, opts)
        loss = res["loss"]
        assert torch.isfinite(loss)
        assert loss.item() > 0.0
        step_count += 1

    assert step_count == 3


def test_hf_stream_dict_chunking_and_shift():
    from affine_ai.data.dataloader import HFStreamDataLoader
    docs = [{"text": "Hello world, this is doc %d with enough bytes to chunk. " % i} for i in range(8)]
    loader = HFStreamDataLoader(docs, batch_size=2, seq_len=16, restart=True)
    it = iter(loader)
    bx, by = next(it)
    assert bx.shape == (2, 16) and by.shape == (2, 16)
    assert bx.dtype == torch.long and by.dtype == torch.long
    # next-token shift: y[j] == x[j+1] inside each row (no padding on full rows)
    assert (by[:, :-1] == bx[:, 1:]).all()
    bx2, by2 = next(it)
    assert bx2.shape == (2, 16)
    assert loader.tokens_seen == 2 * 2 * 17


def test_hf_stream_covers_eos_joined_corpus():
    from affine_ai.data.dataloader import HFStreamDataLoader
    corpus = ["alpha beta gamma delta", "one two three four five six", "lorem ipsum dolor sit amet"]
    eos = "<|endoftext|>".encode("utf-8")
    flat = b"".join(s.encode("utf-8") + eos for s in corpus)
    live = HFStreamDataLoader([{"text": s} for s in corpus], batch_size=2, seq_len=16, restart=False)
    live_batches = list(live)
    assert len(live_batches) >= 1
    segs = []
    for lx, ly in live_batches:
        assert lx.shape == (2, 16) and ly.shape == (2, 16)
        for i in range(2):
            valid = (ly[i] != -100)
            if not valid.any():
                continue
            xb = bytes(lx[i][valid].tolist())
            last = int(ly[i][valid][-1].item())
            segs.append(xb + bytes([last]))
    assert b"".join(segs) == flat


def test_hf_stream_one_shot_exhaustion_and_guards():
    from affine_ai.data.dataloader import HFStreamDataLoader
    gen = ({"text": "tiny doc"} for _ in range(2))
    loader = HFStreamDataLoader(gen, batch_size=2, seq_len=16, restart=False)
    batches = list(loader)
    assert len(batches) >= 1
    for bx, by in batches:
        assert bx.shape == (2, 16) and by.shape == (2, 16)
    try:
        len(loader)
        assert False, "expected TypeError"
    except TypeError:
        pass
    try:
        loader.get_batch_by_indices([0, 1])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

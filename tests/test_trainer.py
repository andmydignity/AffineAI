import os
import torch
import numpy as np
from affine_ai.models.language_model import ASDAGLanguageModel
from affine_ai.training.trainer import ASDAGTrainer


def test_asdag_trainer_end_to_end(tmp_path):
    vocab_size = 64
    d_model = 32
    n_layers = 2
    num_leaves = 4

    model = ASDAGLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_leaves=num_leaves
    )

    train_data = np.random.randint(0, vocab_size, size=1000, dtype=np.uint8)
    val_data = np.random.randint(0, vocab_size, size=200, dtype=np.uint8)
    save_path = str(tmp_path / "asdag_test.pt")

    trainer = ASDAGTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=4,
        seq_len=16,
        lr=1e-3,
        max_steps=5,
        eval_interval=2,
        eval_iters=2,
        use_quantized_gates=True,
        use_shift4_act=True
    )

    results = trainer.train(save_path=save_path)

    assert "best_val_loss" in results
    assert "best_val_ppl" in results
    assert os.path.exists(save_path)


def test_asdag_trainer_toros_hybrid(tmp_path):
    from affine_ai.models.hybrid import TorosHybridLanguageModel, TorosHybridConfig
    config = TorosHybridConfig(
        context_dim=64,
        byte_dim=32,
        num_layers=2,
        target_patch_size=4,
        vocab_size=256,
        num_experts=1,
    )
    model = TorosHybridLanguageModel(config)
    train_data = np.random.randint(0, 256, size=500, dtype=np.uint8)
    val_data = np.random.randint(0, 256, size=100, dtype=np.uint8)
    save_path = str(tmp_path / "hybrid_test.pt")

    trainer = ASDAGTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=2,
        seq_len=16,
        max_steps=3,
        eval_interval=2,
        eval_iters=1,
    )

    results = trainer.train(save_path=save_path)
    assert "best_val_loss" in results
    assert "best_val_ppl" in results
    assert os.path.exists(save_path)


def test_affine_ai_train_entrypoint(tmp_path):
    import affine_ai
    from affine_ai import TorosHybridLanguageModel, TorosHybridConfig

    config = TorosHybridConfig(
        context_dim=64,
        byte_dim=32,
        num_layers=2,
        target_patch_size=4,
        vocab_size=256,
        num_experts=1,
    )
    model = TorosHybridLanguageModel(config)
    save_path = str(tmp_path / "top_level_train.pt")

    train_texts = [
        "A quick brown fox jumps over the lazy dog",
        "MatMul-free neural architecture with dynamic sparsity",
        "Zero-overhead CUDA Graphs with PaddedDataLoader",
    ]

    res = affine_ai.train(
        model=model,
        train_data=train_texts,
        val_data=train_texts[:1],
        batch_size=2,
        seq_len=16,
        max_steps=3,
        eval_interval=2,
        eval_iters=1,
        save_path=save_path,
    )

    assert "best_val_loss" in res
    assert "best_val_ppl" in res
    assert os.path.exists(save_path)


def test_affine_ai_train_advanced_options(tmp_path):
    import affine_ai
    from affine_ai import TorosHybridLanguageModel, TorosHybridConfig

    config = TorosHybridConfig(
        dim=64,
        d_byte=32,
        n_encoder_layers=2,
        target_patch_size=4,
    )
    model = TorosHybridLanguageModel(config)
    save_path = str(tmp_path / "advanced_train.pt")

    train_texts = [
        "A quick brown fox jumps over the lazy dog",
        "MatMul-free neural architecture with dynamic sparsity",
        "Zero-overhead CUDA Graphs with PaddedDataLoader",
        "Multi-token prediction foresight loss with auxiliary heads",
    ]

    res = affine_ai.train(
        model=model,
        train_data=train_texts,
        val_data=train_texts[:2],
        batch_size=2,
        context_window=32,
        inject_eos=True,
        eos_token="<|endoftext|>",
        use_mtp=True,
        num_mtp_heads=2,
        mtp_lambda=0.25,
        channel_mixer="asdag_tree",
        time_mixer="gla",
        use_priority_replay=True,
        replay_ratio=0.5,
        max_steps=4,
        eval_interval=2,
        eval_iters=1,
        save_path=save_path,
    )

    assert "best_val_loss" in res
    assert "best_val_ppl" in res
    assert hasattr(model, "mtp")
    assert model.config.use_mtp is True
    assert model.config.num_mtp_heads == 2
    assert model.context_window == 32
    assert os.path.exists(save_path)


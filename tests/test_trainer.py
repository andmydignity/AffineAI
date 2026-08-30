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

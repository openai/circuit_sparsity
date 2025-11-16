
import blobfile as bf
import pytest
import torch

MODEL_SUBDIR = "csp_yolo1"


def test_gpt_forward_pass(monkeypatch):
    pytest.importorskip("mpi4py")
    monkeypatch.setenv("NO_COMMS", "1")

    from circuit_sparsity.inference.gpt import GPT, GPTConfig

    config = GPTConfig(
        block_size=8,
        vocab_size=16,
        n_layer=1,
        n_head=1,
        d_model=8,
        dropout=0.0,
        flash=False,
        grad_checkpointing=False,
    )
    model = GPT(config)

    batch_size = 2
    seq_len = 4
    idx = torch.randint(config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    targets = torch.randint(config.vocab_size, (batch_size, seq_len), dtype=torch.long)

    logits, loss, hidden_states = model(idx, targets=targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss.shape == torch.Size([])
    assert len(hidden_states) == config.n_layer


def test_pretrained_gpt_forward_pass(monkeypatch):
    pytest.importorskip("mpi4py")
    monkeypatch.setenv("NO_COMMS", "1")

    from circuit_sparsity.inference.gpt import load_model
    from circuit_sparsity.registries import MODEL_BASE_DIR

    model_dir = bf.join(MODEL_BASE_DIR, "models", MODEL_SUBDIR)

    model = load_model(model_dir, cuda=False)

    batch_size = 1
    seq_len = 8
    vocab_size = model.config.vocab_size

    idx = torch.randint(vocab_size, (batch_size, seq_len), dtype=torch.long)
    targets = torch.randint(vocab_size, (batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        logits, loss, hidden_states = model(idx, targets=targets)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert len(hidden_states) == model.config.n_layer

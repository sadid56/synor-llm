import torch
import pytest
from synor.config import SynorConfig, get_preset
from synor.model import SynorLM, RMSNorm, FeedForward, precompute_rope_freqs, apply_rotary_emb


def test_rmsnorm():
    norm = RMSNorm(dim=64, eps=1e-5)
    x = torch.randn(2, 10, 64)
    y = norm(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_rope_embedding():
    dim = 32
    seq_len = 16
    cos, sin = precompute_rope_freqs(dim, seq_len)
    x = torch.randn(2, seq_len, 4, dim)
    y = apply_rotary_emb(x, cos, sin)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_swiglu_feedforward():
    cfg = SynorConfig(n_embd=64, hidden_dim=128, dropout=0.0)
    ffn = FeedForward(cfg)
    x = torch.randn(2, 8, 64)
    y = ffn(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_model_forward_and_loss():
    cfg = SynorConfig(
        vocab_size=100,
        block_size=32,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        hidden_dim=128,
        dropout=0.0,
    )
    model = SynorLM(cfg)
    idx = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))

    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 100)
    assert loss is not None
    assert loss.item() > 0.0

    loss.backward()
    assert model.token_embedding.weight.grad is not None

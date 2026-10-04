import torch
import pytest
from synor.config import SynorConfig
from synor.model import SynorLM, KVCache


def test_kv_cache_equivalence():
    torch.manual_seed(42)
    cfg = SynorConfig(
        vocab_size=128,
        block_size=64,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        n_layer=2,
        hidden_dim=128,
        dropout=0.0,
    )
    model = SynorLM(cfg)
    model.eval()

    tokens = torch.randint(0, 128, (1, 8))

    # Full forward pass without cache
    with torch.no_grad():
        full_logits, _ = model(tokens)

    # Step-by-step forward pass with KV cache
    cache = KVCache(len(model.blocks))
    with torch.no_grad():
        # Prefill first 4 tokens
        prefill_logits, _ = model(tokens[:, :4], start_pos=0, kv_cache=cache)
        assert torch.allclose(full_logits[:, :4, :], prefill_logits, atol=1e-4)

        # Step 4th to 7th tokens
        for i in range(4, 8):
            step_logits, _ = model(tokens[:, i : i + 1], start_pos=i, kv_cache=cache)
            assert torch.allclose(full_logits[:, i : i + 1, :], step_logits, atol=1e-4)

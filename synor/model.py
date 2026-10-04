import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from synor.config import SynorConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d_half = x.shape[-1] // 2
    return torch.cat((-x[..., d_half:], x[..., :d_half]), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    t = x.shape[1]
    d_half = x.shape[-1] // 2
    cos_t = cos[start_pos : start_pos + t, :d_half]
    sin_t = sin[start_pos : start_pos + t, :d_half]
    cos_t = torch.cat([cos_t, cos_t], dim=-1).unsqueeze(0).unsqueeze(2)
    sin_t = torch.cat([sin_t, sin_t], dim=-1).unsqueeze(0).unsqueeze(2)
    return (x * cos_t) + (rotate_half(x) * sin_t)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, t, n_kv_head, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(b, t, n_kv_head, n_rep, head_dim)
        .reshape(b, t, n_kv_head * n_rep, head_dim)
    )


class KVCache:
    def __init__(self, n_layers: int):
        self.k: List[Optional[torch.Tensor]] = [None] * n_layers
        self.v: List[Optional[torch.Tensor]] = [None] * n_layers

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k
            self.v[layer_idx] = v
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=1)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=1)
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self) -> None:
        self.k = [None] * len(self.k)
        self.v = [None] * len(self.v)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SynorConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        layer_idx: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        b, t, c = x.size()
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim)

        q = apply_rotary_emb(q, cos, sin, start_pos=start_pos)
        k = apply_rotary_emb(k, cos, sin, start_pos=start_pos)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        is_causal = (t > 1)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(b, t, c)

        return self.resid_dropout(self.out_proj(y))


class SwiGLUExpert(nn.Module):
    def __init__(self, dim_or_config, hidden_dim: Optional[int] = None, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if isinstance(dim_or_config, SynorConfig):
            cfg = dim_or_config
            dim = cfg.n_embd
            hidden_dim = cfg.hidden_dim or int(2 * (4 * dim) / 3)
            bias = cfg.bias
            dropout = cfg.dropout
        else:
            dim = dim_or_config
            if hidden_dim is None:
                hidden_dim = int(2 * (4 * dim) / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoELayer(nn.Module):
    def __init__(self, config: SynorConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        dim = config.n_embd
        hidden_dim = config.hidden_dim or int(2 * (4 * dim) / 3)

        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(dim, hidden_dim, bias=config.bias, dropout=config.dropout)
            for _ in range(self.num_experts)
        ])
        self.shared_expert = SwiGLUExpert(dim, hidden_dim, bias=config.bias, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x_flat = x.view(-1, d)

        # Shared expert processes all tokens
        shared_out = self.shared_expert(x_flat)

        # Router calculates top-k expert weights
        router_logits = self.gate(x_flat)
        weights, indices = torch.topk(router_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # Accumulate routed expert outputs
        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            exp_idx = indices[:, k]
            weight = weights[:, k].unsqueeze(-1)
            for e_id in range(self.num_experts):
                mask = (exp_idx == e_id)
                if mask.any():
                    token_slice = x_flat[mask]
                    routed_out[mask] += weight[mask] * self.experts[e_id](token_slice)

        final_out = (shared_out + routed_out).view(b, t, d)
        return final_out


FeedForward = SwiGLUExpert


class Block(nn.Module):
    def __init__(self, layer_idx: int, config: SynorConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        
        if config.use_moe:
            self.mlp = MoELayer(config)
        else:
            hidden_dim = config.hidden_dim or int(2 * (4 * config.n_embd) / 3)
            self.mlp = SwiGLUExpert(config.n_embd, hidden_dim, bias=config.bias, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        h = x + self.attn(
            self.attn_norm(x),
            cos=cos,
            sin=sin,
            start_pos=start_pos,
            layer_idx=self.layer_idx,
            kv_cache=kv_cache,
        )
        return h + self.mlp(self.ffn_norm(h))


class SynorLM(nn.Module):
    def __init__(self, config: SynorConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(i, config) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

        head_dim = config.n_embd // config.n_head
        max_seq = max(config.block_size * 4, 4096)
        cos, sin = precompute_rope_freqs(head_dim, max_seq, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.token_embedding(idx)

        cos = self.rope_cos
        sin = self.rope_sin

        for block in self.blocks:
            x = block(x, cos, sin, start_pos=start_pos, kv_cache=kv_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )

        return logits, loss

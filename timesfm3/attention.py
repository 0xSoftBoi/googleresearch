"""Multi-head attention used by the alternating TimesFM-3 layers.

Temporal attention uses rotary position embeddings and a strictly causal
mask, so a token only attends to past tokens within its own series.
Cross-variate attention is unordered — series form a set, so no positional
information is injected along the variate axis.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
    """Standard rotary position embedding (RoPE) over sequence position."""

    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings.")
        inv_freq = base ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to (batch, heads, seq, head_dim) queries or keys."""
    return x * cos + _rotate_half(x) * sin


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional causality and rotary embeddings."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        rotary: bool = False,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.causal = causal
        self.dropout = dropout
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.proj = nn.Linear(model_dim, model_dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim) if rotary else None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attends over the middle (sequence) axis of (B', S, D).

        Args:
            x: (B', S, D) token sequence.
            key_padding_mask: optional (B', S) bool, True for VALID keys.
        """
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B', H, S, hd)

        if self.rope is not None:
            cos, sin = self.rope(s, x.device)
            cos = cos.to(x.dtype)[None, None]
            sin = sin.to(x.dtype)[None, None]
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            # (B', 1, 1, S): True = keep.
            attn_mask = key_padding_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)

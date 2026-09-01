"""Alternating temporal / cross-variate transformer layers.

Tokens form a 2D grid of shape (series N, patches P). The stack alternates
between two complementary attention mechanisms:

- Temporal attention: tokens attend horizontally across time. The mask is
  strictly causal — a token can only look at past tokens within its own
  series.
- Cross-variate attention: tokens attend vertically across series. At a
  given time step a token can look at all other time series, which captures
  cross-series correlations.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadAttention
from .configuration import TimesFM3Config


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AlternatingLayer(nn.Module):
    """One pre-norm transformer layer over one axis of the token grid."""

    def __init__(self, config: TimesFM3Config, axis: str):
        super().__init__()
        if axis not in ("time", "variate"):
            raise ValueError(f"Unknown axis: {axis}")
        self.axis = axis
        self.attn_norm = nn.RMSNorm(config.model_dim)
        self.attn = MultiHeadAttention(
            config.model_dim,
            config.num_heads,
            dropout=config.dropout,
            causal=(axis == "time"),
            rotary=(axis == "time"),
        )
        self.ff_norm = nn.RMSNorm(config.model_dim)
        self.ff = FeedForward(config.model_dim, config.ff_dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        variate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies attention along one grid axis.

        Args:
            tokens: (B, N, P, D) token grid.
            variate_mask: optional (B, N) bool, True for real (non-padding)
                series; used as the key mask for cross-variate attention.

        Returns:
            (B, N, P, D) updated token grid.
        """
        b, n, p, d = tokens.shape
        if self.axis == "time":
            # Fold series into the batch; attend causally over patches.
            x = tokens.reshape(b * n, p, d)
            key_mask = None
        else:
            # Fold time into the batch; attend over series at each step.
            x = tokens.transpose(1, 2).reshape(b * p, n, d)
            key_mask = None
            if variate_mask is not None:
                key_mask = (
                    variate_mask[:, None, :]
                    .expand(b, p, n)
                    .reshape(b * p, n)
                )

        x = x + self.dropout(self.attn(self.attn_norm(x), key_padding_mask=key_mask))
        x = x + self.dropout(self.ff(self.ff_norm(x)))

        if self.axis == "time":
            return x.reshape(b, n, p, d)
        return x.reshape(b, p, n, d).transpose(1, 2)


def build_alternating_stack(config: TimesFM3Config) -> nn.ModuleList:
    """Even layers attend over time, odd layers attend across variates."""
    return nn.ModuleList(
        AlternatingLayer(config, axis="time" if i % 2 == 0 else "variate")
        for i in range(config.num_layers)
    )

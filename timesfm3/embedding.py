"""Patching and token construction for TimesFM-3.

Contiguous data points are grouped into patches of ``patch_len`` (32) time
steps. Tokens are then built per series role:

- Targets and past-only covariates: a token is constructed directly from a
  single patch.
- Past-future ("known ahead") covariates: each token concatenates the current
  patch with the next ``lookahead_patches`` future patches, letting the model
  peek at upcoming known signals while temporal attention stays causal.

Masked horizon patches (contiguous patch masking) are marked through an
explicit indicator channel plus a learned mask embedding.
"""

from __future__ import annotations

import torch
from torch import nn

from .configuration import TimesFM3Config

# Series roles.
ROLE_TARGET = 0
ROLE_PAST_COVARIATE = 1
ROLE_FUTURE_COVARIATE = 2
NUM_ROLES = 3


def patchify(values: torch.Tensor, patch_len: int) -> torch.Tensor:
    """Reshapes (B, N, T) into (B, N, P, patch_len); T must divide evenly."""
    b, n, t = values.shape
    if t % patch_len != 0:
        raise ValueError(f"Series length {t} is not a multiple of {patch_len}.")
    return values.reshape(b, n, t // patch_len, patch_len)


class ResidualBlock(nn.Module):
    """The residual MLP used by TimesFM for input embedding and output heads."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.hidden = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.out = nn.Linear(hidden_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.hidden(x))) + self.residual(x)


class TokenEmbedding(nn.Module):
    """Builds the 2D token grid (series x patches) fed to the transformer."""

    def __init__(self, config: TimesFM3Config):
        super().__init__()
        self.config = config
        p = config.patch_len
        d = config.model_dim
        # Standard tokens carry one patch of values plus one patch of
        # observed-flags.
        self.patch_embed = ResidualBlock(2 * p, d, d)
        # Future-covariate tokens carry (1 + lookahead) patches of values and
        # flags each.
        window = 1 + config.lookahead_patches
        self.lookahead_embed = ResidualBlock(2 * p * window, d, d)
        self.role_embed = nn.Embedding(NUM_ROLES, d)
        # Learned embedding added at masked (to-be-forecast) patch positions.
        self.mask_embed = nn.Parameter(torch.zeros(d))

    def forward(
        self,
        values: torch.Tensor,
        observed: torch.Tensor,
        masked: torch.Tensor,
        roles: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds patched series into tokens.

        Args:
            values: (B, N, P, patch_len) normalized values. Masked positions
                must already be zeroed by the caller.
            observed: (B, N, P, patch_len) float in {0, 1}; 1 where the value
                is a real observation visible to the model.
            masked: (B, N, P) float in {0, 1}; 1 where the patch is a masked
                horizon patch to be forecast.
            roles: (B, N) integer role per series.

        Returns:
            (B, N, P, model_dim) token embeddings.
        """
        b, n, p, _ = values.shape
        cfg = self.config

        standard = self.patch_embed(torch.cat([values, observed], dim=-1))

        # Lookahead construction: concatenate the current patch with the next
        # `lookahead_patches` patches (zero-padded past the end of the grid).
        window_vals = [values, ]
        window_obs = [observed, ]
        for step in range(1, cfg.lookahead_patches + 1):
            pad = torch.zeros_like(values[:, :, :step])
            window_vals.append(torch.cat([values[:, :, step:], pad], dim=2))
            window_obs.append(torch.cat([observed[:, :, step:], pad], dim=2))
        lookahead = self.lookahead_embed(
            torch.cat(window_vals + window_obs, dim=-1)
        )

        is_future_cov = (roles == ROLE_FUTURE_COVARIATE)[:, :, None, None]
        tokens = torch.where(is_future_cov, lookahead, standard)

        tokens = tokens + self.role_embed(roles)[:, :, None, :]
        tokens = tokens + masked[..., None] * self.mask_embed
        return tokens

"""The TimesFM-3 model.

Forecasting works via Contiguous Patch Masking: the input grid spans both
the context and the horizon. Over the horizon region, target and past-only
covariate patches are masked (their values hidden and replaced by a learned
mask embedding), while past-future covariate patches remain visible so that
known future signals (holidays, scheduled events, ...) can steer the
forecast. The entire horizon is generated in a single forward pass — every
masked patch position directly emits its own patch of predictions.

For each target series and each horizon time step the model predicts a point
forecast plus 9 quantiles (10th to 90th percentile).
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from .blocks import build_alternating_stack
from .configuration import TimesFM3Config
from .embedding import (
    ROLE_FUTURE_COVARIATE,
    ResidualBlock,
    TokenEmbedding,
    patchify,
)
from .normalization import PerSeriesNormalizer


@dataclasses.dataclass
class TimesFM3Output:
    """Model outputs in original (denormalized) units.

    Attributes:
        point: (B, N, T, 1) point forecast per series and time step.
        quantiles: (B, N, T, Q) quantile forecasts (q10 ... q90).
        masked: (B, N, P) the contiguous patch mask that was applied.
    """

    point: torch.Tensor
    quantiles: torch.Tensor
    masked: torch.Tensor


class TimesFM3Model(nn.Module):
    """Alternating-attention multivariate forecaster."""

    def __init__(self, config: TimesFM3Config):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config)
        self.layers = build_alternating_stack(config)
        self.final_norm = nn.RMSNorm(config.model_dim)
        # Each token decodes its own patch: patch_len steps x (point + Q).
        self.head = ResidualBlock(
            config.model_dim,
            config.model_dim,
            config.patch_len * config.output_dim_per_step,
        )

    def forward(
        self,
        values: torch.Tensor,
        observed: torch.Tensor,
        roles: torch.Tensor,
        num_horizon_patches: int,
        variate_mask: torch.Tensor | None = None,
    ) -> TimesFM3Output:
        """Runs contiguous-patch-masked decoding.

        Args:
            values: (B, N, T) raw series. T covers context AND horizon and
                must be a multiple of patch_len. Horizon values may be
                arbitrary (e.g. zeros) for targets and past-only covariates;
                for past-future covariates they must hold the known future.
            observed: (B, N, T) bool; True where `values` is a real
                observation the model may see. For past-future covariates
                this includes the horizon; for everything else only the
                context. Left-padding is False.
            roles: (B, N) int64 series roles (target / past cov / future cov).
            num_horizon_patches: number of trailing patches to mask and
                forecast.
            variate_mask: optional (B, N) bool, True for real series (allows
                batching examples with different numbers of variates).

        Returns:
            TimesFM3Output with denormalized point and quantile forecasts for
            every series over the full (context + horizon) range. Context
            positions hold the model's reconstruction and are typically
            ignored; slice the trailing horizon steps for the forecast.
        """
        cfg = self.config
        b, n, t = values.shape
        num_patches = t // cfg.patch_len
        if num_horizon_patches <= 0 or num_horizon_patches >= num_patches:
            raise ValueError("num_horizon_patches must be in [1, num_patches).")

        # Contiguous patch mask over the horizon: targets and past-only
        # covariates are hidden; past-future covariates stay visible.
        patch_idx = torch.arange(num_patches, device=values.device)
        in_horizon = patch_idx >= (num_patches - num_horizon_patches)  # (P,)
        hidden_role = roles != ROLE_FUTURE_COVARIATE  # (B, N)
        masked = (hidden_role[:, :, None] & in_horizon[None, None, :]).float()

        # Per-series normalization from visible context statistics only.
        visible_steps = observed & ~masked.bool().repeat_interleave(
            cfg.patch_len, dim=-1
        )
        normalizer = PerSeriesNormalizer(values, visible_steps)
        norm_values = normalizer.normalize(values)

        # Hide masked and unobserved values from the network entirely.
        visible = visible_steps.to(norm_values.dtype)
        norm_values = norm_values * visible

        patches = patchify(norm_values, cfg.patch_len)
        patch_observed = patchify(visible, cfg.patch_len)

        tokens = self.embedding(patches, patch_observed, masked, roles)
        for layer in self.layers:
            tokens = layer(tokens, variate_mask=variate_mask)
        tokens = self.final_norm(tokens)

        out = self.head(tokens)  # (B, N, P, patch_len * (1 + Q))
        out = out.reshape(b, n, num_patches, cfg.patch_len, cfg.output_dim_per_step)
        out = out.reshape(b, n, t, cfg.output_dim_per_step)
        out = normalizer.denormalize(out)

        return TimesFM3Output(
            point=out[..., :1],
            quantiles=out[..., 1:],
            masked=masked,
        )

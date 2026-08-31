"""Training objective: quantile (pinball) loss plus a point loss.

Losses are computed at masked horizon positions, in normalized units, so
the model learns calibrated quantiles under the same contiguous-patch-
masking regime used at inference time. Masked past-only covariate patches
contribute as an auxiliary objective (the model forecasts every masked
series), which densifies the training signal in multivariate examples.
"""

from __future__ import annotations

import torch

from .configuration import TimesFM3Config
from .embedding import ROLE_TARGET
from .model import TimesFM3Output


def quantile_loss(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    quantiles: tuple[float, ...],
) -> torch.Tensor:
    """Pinball loss.

    Args:
        predictions: (..., Q) quantile predictions.
        actuals: (...,) ground-truth values.
        quantiles: quantile levels matching the last prediction axis.

    Returns:
        (..., Q) elementwise pinball losses.
    """
    q = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype)
    error = actuals.unsqueeze(-1) - predictions
    return torch.maximum(q * error, (q - 1.0) * error)


def forecast_loss(
    config: TimesFM3Config,
    output: TimesFM3Output,
    actuals: torch.Tensor,
    roles: torch.Tensor,
    variate_mask: torch.Tensor | None = None,
    covariate_weight: float = 0.5,
) -> torch.Tensor:
    """Combined point (MSE) + quantile loss over masked target patches.

    Both terms are computed in normalized units (errors divided by each
    series' context scale), so series with vastly different magnitudes
    contribute comparably to the objective.

    Args:
        config: model configuration (for quantile levels).
        output: model forward output (denormalized).
        actuals: (B, N, T) ground-truth values over context + horizon.
        roles: (B, N) series roles.
        variate_mask: optional (B, N) bool for real series.
        covariate_weight: relative weight of masked past-only covariate
            patches (targets weigh 1.0).

    Returns:
        Scalar loss.
    """
    # (B, N, T) weight over masked horizon steps: full for targets, reduced
    # for past-only covariates (auxiliary), zero elsewhere.
    step_mask = output.masked.bool().repeat_interleave(
        config.patch_len, dim=-1
    )
    series_weight = torch.where(
        roles == ROLE_TARGET,
        torch.ones_like(roles, dtype=actuals.dtype),
        torch.full_like(roles, covariate_weight, dtype=actuals.dtype),
    )
    weight = step_mask.to(actuals.dtype) * series_weight[:, :, None]
    if variate_mask is not None:
        weight = weight * variate_mask[:, :, None].float()
    denom = weight.sum().clamp(min=1.0)

    std = output.std  # (B, N, 1)
    point_err = ((output.point.squeeze(-1) - actuals) / std) ** 2
    point_loss = (point_err * weight).sum() / denom

    q_loss = quantile_loss(
        output.quantiles / std.unsqueeze(-1), actuals / std, config.quantiles
    )
    q_loss = (q_loss.mean(dim=-1) * weight).sum() / denom

    return point_loss + q_loss

"""Per-time-series reversible normalization.

Input series can have vastly different scales, so TimesFM-3 normalizes each
time series individually before patching and embedding, and maps forecasts
back to the original scale afterwards. Statistics are computed only from the
observed context region (never from masked horizon positions or padding).
"""

from __future__ import annotations

import torch


class PerSeriesNormalizer:
    """Computes and applies per-series normalization statistics.

    Shapes use B = batch, N = number of series (variates), T = time steps.
    """

    def __init__(self, values: torch.Tensor, observed: torch.Tensor):
        """Fits statistics from the observed context.

        Args:
            values: (B, N, T) raw series values (padding may be arbitrary).
            observed: (B, N, T) boolean mask, True where `values` holds a real
                observed context point.
        """
        observed = observed.to(values.dtype)
        count = observed.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (values * observed).sum(dim=-1, keepdim=True) / count
        var = ((values - mean) ** 2 * observed).sum(dim=-1, keepdim=True) / count
        # Floor the scale relative to the series magnitude: a near-constant
        # series (e.g. a flat exchange rate over one window) must not get a
        # microscopic std that amplifies inputs and normalized errors by
        # orders of magnitude.
        self.mean = mean  # (B, N, 1)
        floor = 1e-3 * (1.0 + mean.abs())
        self.std = torch.maximum(torch.sqrt(var), floor)  # (B, N, 1)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalizes (B, N, T) values with the fitted statistics."""
        return (values - self.mean) / self.std

    def denormalize(self, outputs: torch.Tensor) -> torch.Tensor:
        """Maps model outputs back to the original scale.

        Args:
            outputs: (B, N, T, K) tensor whose last dimension holds the point
                forecast and quantiles, all in normalized units.

        Returns:
            (B, N, T, K) tensor in original units.
        """
        return outputs * self.std.unsqueeze(-1) + self.mean.unsqueeze(-1)

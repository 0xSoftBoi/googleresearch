"""High-level numpy-in / numpy-out forecasting API for TimesFM-3.

Handles patch-multiple padding, per-role tensor assembly, and slicing the
horizon out of the model's single-forward-pass decode.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import torch

from .configuration import TimesFM3Config
from .embedding import (
    ROLE_FUTURE_COVARIATE,
    ROLE_PAST_COVARIATE,
    ROLE_TARGET,
)
from .model import TimesFM3Model


@dataclasses.dataclass
class ForecastResult:
    """Zero-shot forecast for the target series.

    Attributes:
        point: (num_targets, horizon) point forecasts.
        quantiles: (num_targets, horizon, Q) quantile forecasts, ordered by
            the configured quantile levels (q10 ... q90 by default).
        quantile_levels: the quantile levels for the last axis.
    """

    point: np.ndarray
    quantiles: np.ndarray
    quantile_levels: tuple[float, ...]


class TimesFM3Forecaster:
    """Wraps :class:`TimesFM3Model` with a convenient forecasting interface."""

    def __init__(
        self,
        config: TimesFM3Config | None = None,
        model: TimesFM3Model | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config or TimesFM3Config.base()
        self.model = model or TimesFM3Model(self.config)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forecast(
        self,
        targets: Sequence[np.ndarray],
        horizon: int,
        past_covariates: Sequence[np.ndarray] = (),
        future_covariates: Sequence[np.ndarray] = (),
    ) -> ForecastResult:
        """Produces point and quantile forecasts for the target series.

        Args:
            targets: one or more 1-D arrays of length `context`; all series
                must share the same context length and be time-aligned.
            horizon: number of future steps to forecast.
            past_covariates: optional 1-D arrays of length `context`.
            future_covariates: optional 1-D arrays of length
                `context + horizon` — their future values are known and stay
                visible to the model over the horizon.

        Returns:
            A :class:`ForecastResult` for the targets, in original units.
        """
        cfg = self.config
        if not targets:
            raise ValueError("At least one target series is required.")
        context = len(targets[0])
        for series in list(targets) + list(past_covariates):
            if len(series) != context:
                raise ValueError(
                    "All targets and past covariates must share one context "
                    f"length, got {len(series)} vs {context}."
                )
        for series in future_covariates:
            if len(series) != context + horizon:
                raise ValueError(
                    "Future covariates must cover context + horizon "
                    f"({context + horizon} steps), got {len(series)}."
                )
        if horizon > cfg.max_horizon_len:
            raise ValueError(
                f"horizon {horizon} exceeds the model's single-pass maximum "
                f"of {cfg.max_horizon_len} steps."
            )

        # Pad context on the left and horizon on the right to patch multiples.
        padded_context = min(
            cfg.max_context_len,
            math.ceil(context / cfg.patch_len) * cfg.patch_len,
        )
        left_pad = padded_context - min(context, padded_context)
        crop = max(0, context - padded_context)  # drop oldest if too long
        horizon_patches = math.ceil(horizon / cfg.patch_len)
        padded_horizon = horizon_patches * cfg.patch_len
        total = padded_context + padded_horizon

        n = len(targets) + len(past_covariates) + len(future_covariates)
        values = np.zeros((1, n, total), dtype=np.float32)
        observed = np.zeros((1, n, total), dtype=bool)
        roles = np.zeros((1, n), dtype=np.int64)

        row = 0
        for series in targets:
            values[0, row, left_pad:padded_context] = series[crop:]
            observed[0, row, left_pad:padded_context] = True
            roles[0, row] = ROLE_TARGET
            row += 1
        for series in past_covariates:
            values[0, row, left_pad:padded_context] = series[crop:]
            observed[0, row, left_pad:padded_context] = True
            roles[0, row] = ROLE_PAST_COVARIATE
            row += 1
        for series in future_covariates:
            values[0, row, left_pad:padded_context] = series[crop : context]
            values[0, row, padded_context : padded_context + horizon] = series[
                context:
            ]
            observed[0, row, left_pad:padded_context] = True
            observed[0, row, padded_context : padded_context + horizon] = True
            roles[0, row] = ROLE_FUTURE_COVARIATE
            row += 1

        output = self.model(
            values=torch.from_numpy(values).to(self.device),
            observed=torch.from_numpy(observed).to(self.device),
            roles=torch.from_numpy(roles).to(self.device),
            num_horizon_patches=horizon_patches,
        )

        start = padded_context
        num_targets = len(targets)
        point = output.point[0, :num_targets, start : start + horizon, 0]
        quantiles = output.quantiles[0, :num_targets, start : start + horizon]
        return ForecastResult(
            point=point.cpu().numpy(),
            quantiles=quantiles.cpu().numpy(),
            quantile_levels=cfg.quantiles,
        )

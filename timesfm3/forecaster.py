"""High-level numpy-in / numpy-out forecasting API for TimesFM-3.

Handles patch-multiple padding, per-role tensor assembly, quantile-crossing
repair, and horizons longer than the model's single-pass maximum by rolling
the contiguous-patch-masked decode forward (the model's own point forecasts
become context for the next chunk).
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
            the configured quantile levels (q10 ... q90 by default; the
            median sits at index 4).
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

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str | torch.device | None = None,
    ) -> "TimesFM3Forecaster":
        """Loads a checkpoint produced by ``timesfm3.train``."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        config: TimesFM3Config = state["config"]
        model = TimesFM3Model(config)
        model.load_state_dict(state["model"])
        return cls(config=config, model=model, device=device)

    @torch.no_grad()
    def forecast(
        self,
        targets: Sequence[np.ndarray],
        horizon: int,
        past_covariates: Sequence[np.ndarray] = (),
        future_covariates: Sequence[np.ndarray] = (),
        fix_quantile_crossing: bool = True,
    ) -> ForecastResult:
        """Produces point and quantile forecasts for the target series.

        Args:
            targets: one or more 1-D arrays of length `context`; all series
                must share the same context length and be time-aligned.
                NaN values (in any input series) are treated as missing
                observations and masked from the model.
            horizon: number of future steps to forecast. Horizons beyond the
                single-pass maximum are decoded in rolling chunks.
            past_covariates: optional 1-D arrays of length `context`.
            future_covariates: optional 1-D arrays of length
                `context + horizon` — their future values are known and stay
                visible to the model over the horizon.
            fix_quantile_crossing: sort each step's quantiles so the levels
                are monotone (crossings can occur since quantiles are
                predicted jointly but independently per level).

        Returns:
            A :class:`ForecastResult` for the targets, in original units.
        """
        cfg = self.config
        if not targets:
            raise ValueError("At least one target series is required.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
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

        # Evolving contexts: the model's own point forecasts are appended so
        # long horizons can roll forward chunk by chunk.
        num_targets = len(targets)
        evolving = [
            np.asarray(s, dtype=np.float32).copy()
            for s in list(targets) + list(past_covariates)
        ]
        known = [np.asarray(s, dtype=np.float32) for s in future_covariates]

        point_chunks: list[np.ndarray] = []
        quantile_chunks: list[np.ndarray] = []
        produced = 0
        while produced < horizon:
            chunk = min(horizon - produced, cfg.max_horizon_len)
            ctx_len = context + produced
            point, quantiles = self._forecast_chunk(
                evolving=evolving,
                num_targets=num_targets,
                known=[s[: ctx_len + chunk] for s in known],
                chunk=chunk,
            )
            point_chunks.append(point[:num_targets])
            quantile_chunks.append(quantiles[:num_targets])
            for i in range(len(evolving)):
                evolving[i] = np.concatenate([evolving[i], point[i]])
            produced += chunk

        point = np.concatenate(point_chunks, axis=1)
        quantiles = np.concatenate(quantile_chunks, axis=1)
        if fix_quantile_crossing:
            quantiles = np.sort(quantiles, axis=-1)
        return ForecastResult(
            point=point,
            quantiles=quantiles,
            quantile_levels=cfg.quantiles,
        )

    def _forecast_chunk(
        self,
        evolving: list[np.ndarray],
        num_targets: int,
        known: list[np.ndarray],
        chunk: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-pass decode of `chunk` steps for every series.

        Args:
            evolving: targets followed by past covariates, each of one shared
                context length (only the context — nothing beyond it).
            num_targets: how many leading rows of `evolving` are targets.
            known: future covariates, each covering context + chunk steps.
            chunk: horizon steps to decode (<= cfg.max_horizon_len).

        Returns:
            (point, quantiles) for the evolving series over the chunk:
            point is (len(evolving), chunk), quantiles is
            (len(evolving), chunk, Q), both denormalized.
        """
        cfg = self.config
        context = len(evolving[0])

        # Pad context on the left and horizon on the right to patch
        # multiples; crop the oldest history beyond the supported context.
        padded_context = min(
            cfg.max_context_len,
            math.ceil(context / cfg.patch_len) * cfg.patch_len,
        )
        left_pad = padded_context - min(context, padded_context)
        crop = max(0, context - padded_context)
        horizon_patches = math.ceil(chunk / cfg.patch_len)
        total = padded_context + horizon_patches * cfg.patch_len

        n = len(evolving) + len(known)
        values = np.zeros((1, n, total), dtype=np.float32)
        observed = np.zeros((1, n, total), dtype=bool)
        roles = np.full((1, n), ROLE_PAST_COVARIATE, dtype=np.int64)
        roles[0, :num_targets] = ROLE_TARGET

        # NaN input values are treated as missing observations.
        for row, series in enumerate(evolving):
            chunk_vals = series[crop:]
            values[0, row, left_pad:padded_context] = np.nan_to_num(chunk_vals)
            observed[0, row, left_pad:padded_context] = np.isfinite(chunk_vals)
        for offset, series in enumerate(known):
            row = len(evolving) + offset
            ctx_vals = series[crop:context]
            hor_vals = series[context:]
            values[0, row, left_pad:padded_context] = np.nan_to_num(ctx_vals)
            values[0, row, padded_context : padded_context + chunk] = (
                np.nan_to_num(hor_vals)
            )
            observed[0, row, left_pad:padded_context] = np.isfinite(ctx_vals)
            observed[0, row, padded_context : padded_context + chunk] = (
                np.isfinite(hor_vals)
            )
            roles[0, row] = ROLE_FUTURE_COVARIATE

        output = self.model(
            values=torch.from_numpy(values).to(self.device),
            observed=torch.from_numpy(observed).to(self.device),
            roles=torch.from_numpy(roles).to(self.device),
            num_horizon_patches=horizon_patches,
        )

        start = padded_context
        rows = len(evolving)
        point = output.point[0, :rows, start : start + chunk, 0]
        quantiles = output.quantiles[0, :rows, start : start + chunk]
        return point.cpu().numpy(), quantiles.cpu().numpy()

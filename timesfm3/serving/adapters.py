"""Adapters that let the registry serve someone else's foundation model.

``ModelRegistry.add_forecaster`` only requires two things of a forecaster:
a ``.model`` (a ``torch.nn.Module``, so the registry can report a parameter
count) and a ``.forecast(targets, horizon, past_covariates, future_covariates)
-> ForecastResult`` method. TimesFM-3 checkpoints meet that contract; nothing
about it is TimesFM-3-specific. ``ChronosAdapter`` below is the reference
proof: a genuine, tested integration against Amazon's own
``chronos-forecasting`` package (an open-weight competitor named in
``docs/BUSINESS.md``), not a stub. Wrapping one more vendor's weights is an
adapter of a few dozen lines, not a rewrite of the serving stack.

Requires the optional ``chronos-forecasting`` package
(``pip install "timesfm3[chronos]"``); nothing else in this package depends
on it, and the registry works exactly as before if it is absent.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..forecaster import ForecastResult
from .registry import DEFAULT_QUANTILES, ModelEntry, ModelRegistry


class ChronosAdapter:
    """Wraps an Amazon Chronos / Chronos-Bolt / Chronos-2 pipeline as a registry forecaster.

    Works with Chronos, Chronos-Bolt and Chronos-2 checkpoints. Each target is
    forecast univariately; the adapter raises on covariates rather than
    silently ignoring inputs a caller expected it to use.
    """

    def __init__(self, model_id: str = "amazon/chronos-bolt-tiny", device: str = "cpu"):
        try:
            from chronos import BaseChronosPipeline
        except ImportError as exc:  # pragma: no cover - exercised via skip
            raise ImportError(
                "ChronosAdapter needs the optional 'chronos-forecasting' package: "
                "pip install chronos-forecasting"
            ) from exc
        self.model_id = model_id
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
        # torch.nn.Module: the registry sums its parameters for GET /v1/models.
        self.model = self.pipeline.model

    def forecast(
        self,
        targets: Sequence[np.ndarray],
        horizon: int,
        past_covariates: Sequence[np.ndarray] = (),
        future_covariates: Sequence[np.ndarray] = (),
    ) -> ForecastResult:
        if len(past_covariates) or len(future_covariates):
            raise ValueError(
                f"{self.model_id}: this adapter forecasts each target univariately "
                "and does not pass covariates (Chronos-Bolt has no covariate input; "
                "Chronos-2 does, but it is not wired up here)."
            )
        import torch

        inputs = [torch.as_tensor(np.asarray(t, dtype=np.float32)) for t in targets]
        levels = list(DEFAULT_QUANTILES)
        quantiles, mean = self.pipeline.predict_quantiles(
            inputs=inputs, prediction_length=horizon, quantile_levels=levels,
        )
        # Chronos/Chronos-Bolt return stacked (C, H, Q) / (C, H) tensors;
        # Chronos-2 returns one (variates, H, Q) / (variates, H) tensor per input.
        if isinstance(quantiles, (list, tuple)):
            quantiles = torch.cat([q.reshape(-1, horizon, len(levels)) for q in quantiles])
            mean = torch.cat([m.reshape(-1, horizon) for m in mean])
        return ForecastResult(
            point=mean.float().numpy().astype(np.float64),
            quantiles=quantiles.float().numpy().astype(np.float64),
            quantile_levels=DEFAULT_QUANTILES,
        )


def add_chronos(
    registry: ModelRegistry,
    model_id: str = "amazon/chronos-bolt-tiny",
    name: str | None = None,
    device: str = "cpu",
    make_default: bool = False,
) -> ModelEntry:
    """Registers a Chronos / Chronos-Bolt checkpoint in ``registry``.

    ``timesfm3 serve --checkpoint ...`` and ``TIMESFM3_CHECKPOINTS`` load this
    repo's own weights; ``TIMESFM3_CHRONOS_MODELS`` (see
    :meth:`ModelRegistry.from_env`) calls this for each listed model ID, so a
    deployment can serve TimesFM-3 and Chronos side by side and let
    ``/v1/backtest`` say which one actually wins on the customer's data.
    """
    adapter = ChronosAdapter(model_id, device=device)
    entry_name = name or model_id.rsplit("/", 1)[-1]
    entry = registry.add_forecaster(
        entry_name, adapter, kind="chronos", supports_covariates=False,
        description=f"Amazon Chronos ({model_id}), served through this registry's "
        "adapter contract — not a TimesFM-3 checkpoint.",
    )
    if make_default:
        registry.default = entry_name
    return entry

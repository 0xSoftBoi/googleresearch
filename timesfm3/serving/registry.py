"""One registry, one interface, every forecaster.

A product needs a stable contract regardless of what is behind it: a
TimesFM-3 checkpoint, the classical baselines, or both side by side so a
customer can *measure* whether the neural model earns its keep on their
data.  Every entry exposes::

    entry.forecast(targets, horizon, past_covariates, future_covariates)
        -> ForecastResult   # point (C, H) + quantiles (C, H, Q), same levels

The classical baselines have no quantile head, so their bands come from a
walk-forward residual estimate *inside the context* (nothing beyond the
forecast origin is used): the baseline is refit at several earlier
origins, its h-step errors collected, and the empirical error quantiles
added to the point forecast.  Where too few origins fit, a Gaussian band
scaled by the one-step residual times sqrt(h) is used instead.
"""

from __future__ import annotations

import dataclasses
import glob
import os
from collections.abc import Sequence

import numpy as np

from ..baselines import AR, Baseline, ContextMean, Drift, EWMA, LastValue
from ..forecaster import ForecastResult

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: Where bundled checkpoints live inside the installed package.
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


def _fill_missing(x: np.ndarray) -> np.ndarray:
    """Forward- then back-fill NaNs (classical baselines cannot skip them)."""
    x = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return np.zeros_like(x)
    idx = np.where(finite, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    first = np.flatnonzero(finite)[0]
    x[:first] = x[first]
    return x


def _normal_ppf(levels: Sequence[float]) -> np.ndarray:
    import torch

    return torch.distributions.Normal(0.0, 1.0).icdf(
        torch.tensor(levels, dtype=torch.float64)
    ).numpy()


def empirical_quantiles(
    baseline: Baseline,
    context: np.ndarray,
    horizon: int,
    point: np.ndarray,
    levels: Sequence[float] = DEFAULT_QUANTILES,
    max_origins: int = 16,
) -> np.ndarray:
    """(horizon, Q) quantile forecasts from in-context walk-forward residuals."""
    n = len(context)
    levels = tuple(levels)
    min_ctx = max(8, n // 4)
    q = np.empty((horizon, len(levels)), dtype=np.float64)
    errors = np.full((0, horizon), np.nan)
    if n - min_ctx >= 2:
        origins = np.unique(
            np.linspace(min_ctx, n - 1, num=min(max_origins, n - min_ctx)).astype(int)
        )
        errors = np.full((len(origins), horizon), np.nan)
        for i, o in enumerate(origins):
            h = min(horizon, n - o)
            fc = baseline.forecast(context[:o], h)
            errors[i, :h] = context[o : o + h] - fc

    one_step = errors[:, 0][np.isfinite(errors[:, 0])] if errors.size else np.empty(0)
    if len(one_step) >= 2:
        sigma1 = float(one_step.std()) if one_step.std() > 0 else float(np.abs(one_step).mean())
    else:
        d = np.diff(context)
        sigma1 = float(d.std()) if len(d) > 1 else float(np.abs(context).mean() * 0.1)
    # Floor relative to the series' scale so a perfectly deterministic context
    # still yields a finite, interpretable band (and anomaly score).
    sigma1 = max(sigma1, 1e-6 * max(1.0, float(np.abs(context).mean())))
    z = _normal_ppf(levels)
    for j in range(horizon):
        gaussian = z * sigma1 * np.sqrt(j + 1)
        e = errors[:, j][np.isfinite(errors[:, j])] if errors.size else np.empty(0)
        if len(e) >= 4:
            # Empirical offsets capture skew and fat tails, but with a handful
            # of origins they are noisy and can collapse; never let the band
            # be narrower than the Gaussian sqrt(h) estimate at any level.
            empirical = np.quantile(e, levels)
            offsets = np.where(np.abs(empirical) >= np.abs(gaussian), empirical, gaussian)
            q[j] = point[j] + offsets
        else:
            q[j] = point[j] + gaussian
    return np.sort(q, axis=-1)


@dataclasses.dataclass
class ModelEntry:
    """A registered forecaster with the metadata the API reports."""

    name: str
    kind: str  # "timesfm3" | "classical"
    description: str
    parameters: int = 0
    meta: dict = dataclasses.field(default_factory=dict)
    supports_covariates: bool = False
    _baseline: Baseline | None = None
    _forecaster: object = None

    def forecast(
        self,
        targets: Sequence[np.ndarray],
        horizon: int,
        past_covariates: Sequence[np.ndarray] = (),
        future_covariates: Sequence[np.ndarray] = (),
    ) -> ForecastResult:
        if self.kind == "timesfm3":
            return self._forecaster.forecast(  # type: ignore[attr-defined]
                targets=[np.asarray(t, dtype=np.float32) for t in targets],
                horizon=horizon,
                past_covariates=[np.asarray(c, dtype=np.float32) for c in past_covariates],
                future_covariates=[np.asarray(c, dtype=np.float32) for c in future_covariates],
            )
        assert self._baseline is not None
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        points, quants = [], []
        for series in targets:
            ctx = _fill_missing(series)
            if len(ctx) < 2:
                raise ValueError("Each target needs at least 2 observations.")
            p = np.asarray(self._baseline.forecast(ctx, horizon), dtype=np.float64)
            points.append(p)
            quants.append(empirical_quantiles(self._baseline, ctx, horizon, p))
        return ForecastResult(
            point=np.stack(points),
            quantiles=np.stack(quants),
            quantile_levels=DEFAULT_QUANTILES,
        )

    def info(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "parameters": self.parameters,
            "supports_covariates": self.supports_covariates,
            "meta": self.meta,
        }


CLASSICAL: dict[str, tuple[Baseline, str]] = {
    "last-value": (LastValue(), "Random walk: holds the last observation flat."),
    "ctx-mean": (ContextMean(), "Holds the context mean flat."),
    "drift": (Drift(), "Random walk with drift: extrapolates the context slope."),
    "ewma": (EWMA(), "Exponentially weighted level, smoothing fit in-context."),
    "ar1": (AR(1), "AR(1) fit by OLS on the context, iterated forward."),
    "ar4": (AR(4), "AR(4) fit by OLS on the context, iterated forward."),
}


class ModelRegistry:
    """Holds every servable forecaster and picks the default."""

    def __init__(self, include_classical: bool = True):
        self._entries: dict[str, ModelEntry] = {}
        self.default: str | None = None
        if include_classical:
            for name, (baseline, desc) in CLASSICAL.items():
                self._entries[name] = ModelEntry(
                    name=name, kind="classical", description=desc, _baseline=baseline
                )
            self.default = "ewma"

    # -- population -------------------------------------------------------

    def add_checkpoint(
        self, path: str, name: str | None = None, device: str | None = None,
        make_default: bool = True,
    ) -> ModelEntry:
        from ..forecaster import TimesFM3Forecaster

        forecaster = TimesFM3Forecaster.from_checkpoint(path, device=device)
        meta = dict(forecaster.meta)
        name = name or meta.get("name") or os.path.splitext(os.path.basename(path))[0]
        params = sum(p.numel() for p in forecaster.model.parameters())
        cfg = forecaster.config
        entry = ModelEntry(
            name=name,
            kind="timesfm3",
            description=meta.get(
                "description",
                f"TimesFM-3 checkpoint ({cfg.num_layers} layers, dim {cfg.model_dim}).",
            ),
            parameters=int(params),
            meta={
                **meta,
                "path": path,
                "max_context": cfg.max_context_len,
                "max_horizon_single_pass": cfg.max_horizon_len,
                "device": str(forecaster.device),
            },
            supports_covariates=True,
            _forecaster=forecaster,
        )
        self._entries[name] = entry
        if make_default or self.default is None or self._entries[self.default].kind != "timesfm3":
            self.default = name
        return entry

    def add_forecaster(self, name: str, forecaster, description: str = "") -> ModelEntry:
        """Registers an in-memory :class:`TimesFM3Forecaster` (tests, notebooks)."""
        params = sum(p.numel() for p in forecaster.model.parameters())
        entry = ModelEntry(
            name=name, kind="timesfm3",
            description=description or "In-memory TimesFM-3 forecaster.",
            parameters=int(params), supports_covariates=True,
            _forecaster=forecaster,
        )
        self._entries[name] = entry
        if self.default is None or self._entries[self.default].kind != "timesfm3":
            self.default = name
        return entry

    @classmethod
    def from_env(
        cls,
        checkpoints: Sequence[str] = (),
        include_bundled: bool = True,
        device: str | None = None,
        default: str | None = None,
    ) -> "ModelRegistry":
        """Registry from explicit paths, ``TIMESFM3_CHECKPOINTS``,
        ``TIMESFM3_MODEL_DIR`` and the bundled starter checkpoint.

        ``checkpoints`` entries may be ``path`` or ``name=path``.  The last
        checkpoint added becomes the default unless ``TIMESFM3_DEFAULT_MODEL``
        or ``default`` names one.
        """
        reg = cls()
        specs: list[str] = list(checkpoints)
        env = os.environ.get("TIMESFM3_CHECKPOINTS", "")
        specs += [s for s in env.split(",") if s.strip()]
        model_dir = os.environ.get("TIMESFM3_MODEL_DIR")
        if model_dir:
            specs += sorted(glob.glob(os.path.join(model_dir, "*.pt")))
        if include_bundled and os.environ.get("TIMESFM3_NO_BUNDLED", "") != "1":
            specs = sorted(glob.glob(os.path.join(ASSET_DIR, "*.pt"))) + specs
        for spec in specs:
            name, _, path = spec.partition("=") if "=" in spec else (None, None, spec)
            reg.add_checkpoint(path.strip(), name=(name or "").strip() or None, device=device)
        chosen = default or os.environ.get("TIMESFM3_DEFAULT_MODEL")
        if chosen:
            if chosen not in reg._entries:
                raise KeyError(f"default model {chosen!r} not among {reg.names()}")
            reg.default = chosen
        return reg

    # -- lookup -----------------------------------------------------------

    def names(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, name: str | None = None) -> ModelEntry:
        key = name or self.default
        if key is None or key not in self._entries:
            raise KeyError(
                f"Unknown model {key!r}; available: {', '.join(self.names())}"
            )
        return self._entries[key]

    def entries(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def describe(self) -> list[dict]:
        return [
            {**e.info(), "default": e.name == self.default} for e in self._entries.values()
        ]

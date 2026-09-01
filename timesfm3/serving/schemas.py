"""Request / response contracts for the forecasting API (Pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Series(BaseModel):
    """One time series. ``null`` values are missing observations."""

    name: str | None = Field(default=None, max_length=128)
    values: list[float | None] = Field(min_length=2)

    @field_validator("values")
    @classmethod
    def _at_least_one_observation(cls, v):
        if not any(x is not None for x in v):
            raise ValueError("a series needs at least one non-null value")
        return v


class ForecastRequest(BaseModel):
    targets: list[Series] = Field(min_length=1, description="Series to forecast.")
    past_covariates: list[Series] = Field(
        default_factory=list, description="History-only features (same length as targets)."
    )
    future_covariates: list[Series] = Field(
        default_factory=list,
        description="Known-ahead features covering context + horizon steps.",
    )
    horizon: int = Field(ge=1, le=4096, description="Steps to forecast.")
    model: str | None = Field(default=None, description="Registry name; default if omitted.")
    quantiles: bool = Field(default=True, description="Include the 9 quantile paths.")
    timestamps: list[str] | None = Field(
        default=None, description="Context timestamps (ISO 8601), one per step."
    )
    freq: str | None = Field(
        default=None, description="Step size for output timestamps, e.g. '15min', '1h', 'D'."
    )


class TargetForecast(BaseModel):
    name: str
    point: list[float]
    quantiles: dict[str, list[float]] | None = None


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    quantile_levels: list[float]
    timestamps: list[str] | None = None
    forecasts: list[TargetForecast]
    latency_ms: float


class BacktestRequest(BaseModel):
    """Walk-forward comparison of models on the caller's own series."""

    series: list[Series] = Field(min_length=1)
    context: int = Field(ge=8, le=16384, description="Context steps per window.")
    horizon: int = Field(ge=1, le=1024)
    models: list[str] | None = Field(default=None, description="Default: every model.")
    reference: str = Field(default="last-value", description="Model the others are tested against.")
    windows: int = Field(default=20, ge=3, le=500, description="Forecast origins per series.")
    metric: Literal["mae", "mse"] = "mae"
    overlap: bool = Field(
        default=False,
        description="Allow overlapping windows (more origins, less independence).",
    )


class ModelScore(BaseModel):
    model: str
    mean_loss: float
    ratio: float = Field(description="mean loss / reference mean loss; < 1 is better")
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    p_adjusted: float | None
    win_rate: float
    n: int
    n_effective: float
    verdict: str


class BacktestResponse(BaseModel):
    reference: str
    reference_loss: float
    metric: str
    context: int
    horizon: int
    windows_per_series: int
    scores: list[ModelScore]
    latency_ms: float


class VolatilityRequest(BaseModel):
    """Variance forecast + Moreira-Muir sizing from daily returns or prices."""

    returns: list[float] | None = Field(default=None, description="Daily log returns.")
    prices: list[float] | None = Field(default=None, description="Alternatively, daily prices.")
    horizon: int = Field(default=5, ge=1, le=252)
    vol_target: float = Field(default=0.10, gt=0, le=2.0, description="Annualized target vol.")
    max_leverage: float = Field(default=3.0, gt=0, le=20.0)

    @field_validator("prices")
    @classmethod
    def _positive_prices(cls, v):
        if v is not None and any(p is None or p <= 0 for p in v):
            raise ValueError("prices must be positive")
        return v


class VolatilityModelForecast(BaseModel):
    model: str
    variance_path: list[float] = Field(description="Daily variance forecast per horizon step.")
    mean_variance: float
    annualized_vol: float
    weight: float = Field(description="vol_target / forecast vol, capped at max_leverage.")


class VolatilityResponse(BaseModel):
    horizon: int
    n_returns: int
    realized_vol_annualized: float = Field(description="Trailing 21-day realized vol.")
    forecasts: list[VolatilityModelForecast]
    latency_ms: float


class ModelInfo(BaseModel):
    name: str
    kind: str
    description: str
    parameters: int
    supports_covariates: bool
    default: bool
    meta: dict


class Health(BaseModel):
    status: str
    version: str
    models: int
    default_model: str | None
    device: str

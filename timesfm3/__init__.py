"""TimesFM-3: a zero-shot foundation model for multivariate forecasting.

Independent PyTorch implementation of the architecture described in
https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
"""

from .configuration import TimesFM3Config
from .forecaster import ForecastResult, TimesFM3Forecaster
from .loss import forecast_loss, quantile_loss
from .model import TimesFM3Model, TimesFM3Output

__all__ = [
    "TimesFM3Config",
    "TimesFM3Forecaster",
    "ForecastResult",
    "TimesFM3Model",
    "TimesFM3Output",
    "forecast_loss",
    "quantile_loss",
]

__version__ = "0.4.0"

"""Zero-shot multivariate forecasting with TimesFM-3.

Builds two correlated target series, one past-only covariate and one
known-future covariate, then decodes a 128-step horizon in a single forward
pass and prints the point forecast with its q10/q90 band.

Note: without pre-trained weights the numbers come from a randomly
initialized network — this example demonstrates the API and shapes.
"""

import numpy as np

from timesfm3 import TimesFM3Config, TimesFM3Forecaster

CONTEXT = 512
HORIZON = 128


def main() -> None:
    rng = np.random.default_rng(0)
    time = np.arange(CONTEXT + HORIZON)

    # A known-future covariate, e.g. a scheduled promotion calendar.
    promo = (np.sin(2 * np.pi * time / 64) > 0.8).astype(np.float32)

    # Two correlated targets driven by shared seasonality plus the promo.
    season = np.sin(2 * np.pi * time / 32)
    target_a = 10 * season + 5 * promo + rng.normal(0, 0.3, time.shape)
    target_b = -6 * season + 3 * promo + rng.normal(0, 0.3, time.shape)

    # A past-only covariate (its future is unknown at forecast time).
    temperature = 20 + 4 * np.sin(2 * np.pi * time / 96) + rng.normal(0, 0.5, time.shape)

    forecaster = TimesFM3Forecaster(TimesFM3Config.small())
    result = forecaster.forecast(
        targets=[target_a[:CONTEXT], target_b[:CONTEXT]],
        past_covariates=[temperature[:CONTEXT]],
        future_covariates=[promo],
        horizon=HORIZON,
    )

    print(f"point forecast shape:    {result.point.shape}")
    print(f"quantile forecast shape: {result.quantiles.shape}")
    print(f"quantile levels:         {result.quantile_levels}")
    for i in range(result.point.shape[0]):
        q10 = result.quantiles[i, :5, 0]
        q90 = result.quantiles[i, :5, -1]
        print(f"target {i} first 5 steps: point={np.round(result.point[i, :5], 2)}")
        print(f"          q10={np.round(q10, 2)}  q90={np.round(q90, 2)}")


if __name__ == "__main__":
    main()

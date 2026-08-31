"""Plots a TimesFM-3 forecast (point + q10-q90 band) against ground truth.

Loads a checkpoint produced by `python -m timesfm3.train` and forecasts a
held-out multivariate example: two correlated seasonal targets plus a
known-future covariate that spikes the targets on scheduled events.

Usage:
    python examples/plot_forecast.py --checkpoint timesfm3_checkpoint.pt \
        [--output forecast.png]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from timesfm3 import TimesFM3Forecaster


def make_series(context: int, horizon: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    time = np.arange(context + horizon)
    promo = (np.sin(2 * np.pi * time / 48 + 1.0) > 0.9).astype(np.float32)
    season = np.sin(2 * np.pi * time / 32)
    trend = 0.004 * time
    target_a = 3.0 * season + 2.0 * promo + trend + rng.normal(0, 0.15, time.shape)
    target_b = -2.0 * season + 1.5 * promo - trend + rng.normal(0, 0.15, time.shape)
    return time, promo, target_a, target_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="forecast.png")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=64)
    args = parser.parse_args()

    forecaster = TimesFM3Forecaster.from_checkpoint(args.checkpoint)
    context, horizon = args.context, args.horizon
    time, promo, target_a, target_b = make_series(context, horizon)

    result = forecaster.forecast(
        targets=[target_a[:context], target_b[:context]],
        future_covariates=[promo],
        horizon=horizon,
    )

    truths = [target_a, target_b]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    show_from = max(0, context - 160)
    for i, ax in enumerate(axes):
        ax.plot(time[show_from:context + 1], truths[i][show_from:context + 1],
                color="#444", lw=1.2, label="context")
        ax.plot(time[context:], truths[i][context:],
                color="#444", lw=1.2, ls=":", label="ground truth")
        ax.plot(time[context:], result.point[i],
                color="#d62728", lw=1.6, label="point forecast")
        ax.fill_between(time[context:], result.quantiles[i, :, 0],
                        result.quantiles[i, :, -1], color="#d62728",
                        alpha=0.18, label="q10–q90")
        ax.axvline(context, color="#999", lw=0.8)
        ax.set_ylabel(f"target {chr(ord('A') + i)}")
        if i == 0:
            ax.legend(loc="upper left", ncols=4, fontsize=9)
    axes[-1].set_xlabel("time step")
    fig.suptitle("TimesFM-3 zero-shot forecast (single-pass decode, known-future covariate)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

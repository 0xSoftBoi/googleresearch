"""Zero-shot evaluation on the ETT (Electricity Transformer Temperature) data.

Slides forecasting windows over the tail of ETTh1 (hourly, 7 variables,
~17k steps), forecasts all 7 series jointly as multivariate targets, and
reports scaled MAE against last-value and seasonal-naive (period 24)
baselines. The model never sees ETT during training, so this measures
zero-shot transfer from the (synthetic) pre-training corpus.

Get the data:
    curl -sLO https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

Usage:
    python examples/evaluate_ett.py --checkpoint ckpt.pt --data ETTh1.csv
"""

import argparse
import csv

import numpy as np

from timesfm3 import TimesFM3Forecaster


def load_ett(path: str) -> np.ndarray:
    """Returns (num_series, num_steps) with the date column dropped."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = [[float(x) for x in row[1:]] for row in reader]
    return np.asarray(rows, dtype=np.float32).T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--season", type=int, default=24, help="daily period")
    args = parser.parse_args()

    series = load_ett(args.data)
    n, t = series.shape
    forecaster = TimesFM3Forecaster.from_checkpoint(args.checkpoint)

    # Windows spaced evenly over the second half of the data (the model is
    # zero-shot either way; the tail avoids the well-behaved early region).
    span = args.context + args.horizon
    starts = np.linspace(t // 2, t - span - 1, args.windows).astype(int)

    model_err, last_err, seasonal_err = [], [], []
    for start in starts:
        ctx = series[:, start : start + args.context]
        truth = series[:, start + args.context : start + span]
        result = forecaster.forecast(
            targets=[ctx[i] for i in range(n)], horizon=args.horizon
        )
        reps = -(-args.horizon // args.season)  # ceil division
        for i in range(n):
            scale = max(ctx[i].std(), 1e-6)
            seasonal = np.tile(ctx[i, -args.season :], reps)[: args.horizon]
            model_err.append(np.abs(result.point[i] - truth[i]).mean() / scale)
            last_err.append(np.abs(ctx[i, -1] - truth[i]).mean() / scale)
            seasonal_err.append(np.abs(seasonal - truth[i]).mean() / scale)

    print(
        f"ETTh1 zero-shot: {args.windows} windows x {n} series, "
        f"context {args.context}, horizon {args.horizon}"
    )
    print(f"scaled MAE  model:          {np.mean(model_err):.4f}")
    print(f"scaled MAE  last-value:     {np.mean(last_err):.4f}")
    print(f"scaled MAE  seasonal-naive: {np.mean(seasonal_err):.4f}")


if __name__ == "__main__":
    main()

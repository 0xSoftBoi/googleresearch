"""Forecast Polymarket prediction-market prices with TimesFM-3.

Pulls hourly order-book data from the public Pendulum Flow archive
(https://archive.pendulumflow.com/), reconstructs each active outcome token's
mid-price series on a regular time grid, and forecasts the tail horizon of
every series jointly as multivariate targets. Reports scaled MAE against a
last-value (random-walk) baseline -- a strong baseline for near-efficient
prediction-market prices -- and optionally plots one asset.

Outcome-token prices are probabilities in [0, 1]; the mid price is
(best_bid + best_ask) / 2 from the archive's best_bid_ask / price_change events.

Usage:
    # Plumbing demo with an untrained model (shows the API end to end):
    python examples/forecast_polymarket.py --start 2026-08-28T00 --end 2026-08-28T02

    # With a checkpoint from `python -m timesfm3.train`, and a plot:
    python examples/forecast_polymarket.py --start 2026-08-28T00 --end 2026-08-28T05 \
        --checkpoint timesfm3_checkpoint.pt --output polymarket_forecast.png

Requires pyarrow (`pip install pyarrow`).
"""

import argparse
import datetime as dt

import numpy as np

from timesfm3 import TimesFM3Config, TimesFM3Forecaster
from timesfm3.data.polymarket import (
    PolymarketArchive,
    build_mid_grid,
    top_assets,
)


def _parse_hour(text: str) -> dt.datetime:
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"unrecognised hour: {text!r} (use YYYY-MM-DDTHH)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=_parse_hour, required=True,
                        help="first hour, e.g. 2026-08-28T00 (UTC)")
    parser.add_argument("--end", type=_parse_hour, required=True,
                        help="last hour, inclusive")
    parser.add_argument("--cache-dir", default="data/polymarket")
    parser.add_argument("--version", default="v3")
    parser.add_argument("--num-assets", type=int, default=16,
                        help="most active outcome tokens to forecast jointly")
    parser.add_argument("--freq-seconds", type=float, default=5.0)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--checkpoint", default=None,
                        help="TimesFM-3 checkpoint; omit for an untrained demo model")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip SHA-256 checksum verification of downloads")
    parser.add_argument("--output", default=None, help="optional PNG plot path")
    args = parser.parse_args()

    archive = PolymarketArchive(
        cache_dir=args.cache_dir, version=args.version, verify=not args.no_verify
    )
    print(f"Downloading {args.start:%Y-%m-%dT%H} .. {args.end:%Y-%m-%dT%H} "
          f"({args.version}) into {args.cache_dir} ...")
    paths = archive.download_range(args.start, args.end)

    assets = top_assets(paths, args.num_assets)
    if not assets:
        raise SystemExit("No active assets found in that range.")
    values, assets, grid_us = build_mid_grid(
        paths, assets, freq_seconds=args.freq_seconds
    )
    # Keep only assets fully observed over the grid, so eval windows are clean.
    coverage = np.isfinite(values).mean(axis=1)
    full = coverage >= 0.999
    values = values[full]
    kept = [a for a, ok in zip(assets, full) if ok]
    n, t = values.shape
    print(f"Grid: {n} assets x {t} steps @ {args.freq_seconds:g}s "
          f"({n} of {len(assets)} assets fully observed)")

    if args.checkpoint:
        forecaster = TimesFM3Forecaster.from_checkpoint(args.checkpoint)
    else:
        print("No --checkpoint given: using an untrained small model "
              "(pipeline demo; forecasts are not meaningful).")
        forecaster = TimesFM3Forecaster(TimesFM3Config.small())

    span = args.context + args.horizon
    if t < span:
        raise SystemExit(
            f"Need context+horizon={span} steps but grid has only {t}; "
            "widen the date range or lower --context/--horizon/--freq-seconds."
        )
    starts = np.linspace(0, t - span, min(args.windows, t - span + 1)).astype(int)

    model_err, last_err = [], []
    last_result = None
    for start in starts:
        ctx = values[:, start : start + args.context]
        truth = values[:, start + args.context : start + span]
        result = forecaster.forecast(
            targets=[ctx[i] for i in range(n)], horizon=args.horizon
        )
        last_result = (start, ctx, truth, result)
        for i in range(n):
            scale = max(float(ctx[i].std()), 1e-6)
            model_err.append(np.abs(result.point[i] - truth[i]).mean() / scale)
            last_err.append(np.abs(ctx[i, -1] - truth[i]).mean() / scale)

    print(
        f"\nPolymarket {args.version}: {len(starts)} windows x {n} series, "
        f"context {args.context}, horizon {args.horizon}"
    )
    print(f"scaled MAE  model:      {np.mean(model_err):.4f}")
    print(f"scaled MAE  last-value: {np.mean(last_err):.4f}")

    if args.output and last_result is not None:
        _plot(args, kept, *last_result)


def _plot(args, assets, start, ctx, truth, result) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Most-variable asset in this window makes the clearest picture.
    i = int(np.argmax(ctx.std(axis=1)))
    c, h = args.context, args.horizon
    x_ctx = np.arange(c)
    x_hor = np.arange(c, c + h)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x_ctx, ctx[i], color="#444", lw=1.2, label="context")
    ax.plot(x_hor, truth[i], color="#444", lw=1.2, ls=":", label="ground truth")
    ax.plot(x_hor, result.point[i], color="#d62728", lw=1.6, label="point forecast")
    ax.fill_between(x_hor, result.quantiles[i, :, 0], result.quantiles[i, :, -1],
                    color="#d62728", alpha=0.18, label="q10–q90")
    ax.axvline(c, color="#999", lw=0.8)
    ax.set_ylabel("mid price (probability)")
    ax.set_xlabel(f"time step ({args.freq_seconds:g}s)")
    ax.set_title(f"TimesFM-3 forecast — Polymarket asset {assets[i].hex()[:16]}…")
    ax.legend(loc="upper left", ncols=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

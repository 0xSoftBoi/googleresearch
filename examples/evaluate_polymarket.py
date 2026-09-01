"""Benchmark forecasters on Polymarket prediction-market microstructure.

Downloads hours from the public Pendulum Flow archive, rebuilds each liquid
market's outcome-token series on a regular grid, and scores a forecast horizon
per feature channel against two trivial predictors: last-value (random walk)
and context-mean. Baselines are always reported; a TimesFM-3 checkpoint is
scored alongside them when ``--checkpoint`` is given.

Why not scaled MAE. The repo's other benchmarks divide MAE by the context
standard deviation, but prediction-market contexts are frequently *perfectly
flat* -- a market can sit at one price for an hour -- which collapses that
denominator and produces meaningless six-figure "errors". This script reports
MAE in native units (probability for ``mid``, counts for activity) plus a
scale-free skill ratio against the better baseline.

Usage:
    # Baselines only -- no model required:
    python examples/evaluate_polymarket.py --start 2026-08-29T03 --end 2026-08-29T15

    # With a trained checkpoint, and a plot:
    python examples/evaluate_polymarket.py --start 2026-08-29T03 --end 2026-08-29T15 \
        --checkpoint timesfm3_polymarket.pt --plot polymarket.png

Requires pyarrow (`pip install -e .[polymarket]`).
"""

import argparse
import datetime as dt

import numpy as np

from timesfm3.data.polymarket import (
    COUNT_CHANNELS,
    PolymarketArchive,
    build_market_panels,
    select_covered,
    top_markets,
)


def _hour(text: str) -> dt.datetime:
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"expected YYYY-MM-DDTHH, got {text!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=_hour, required=True)
    p.add_argument("--end", type=_hour, required=True)
    p.add_argument("--cache-dir", default="data/polymarket")
    p.add_argument("--channels", default="mid,spread,abs_ret,quotes")
    p.add_argument("--candidates", type=int, default=300,
                   help="markets to rank by trade count before coverage filtering")
    p.add_argument("--freq-seconds", type=float, default=15.0)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--windows", type=int, default=12)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--plot", default=None)
    p.add_argument("--accept-unverified", action="store_true",
                   help="accept hours whose bytes disagree with SHA256SUMS.txt")
    args = p.parse_args()
    channels = tuple(args.channels.split(","))

    archive = PolymarketArchive(cache_dir=args.cache_dir)
    paths = archive.download_range(
        args.start, args.end,
        on_mismatch="warn" if args.accept_unverified else "raise",
    )
    print(f"{len(paths)} verified hours; ranking markets by trade count ...")
    markets = top_markets(paths, args.candidates)
    panels = build_market_panels(paths, markets, freq_seconds=args.freq_seconds)
    panels = select_covered(panels, channel="mid", min_coverage=0.98)
    if not panels:
        raise SystemExit("No continuously quoted markets in that range.")
    print(f"{len(panels)} continuously quoted markets x {panels[0].num_steps} "
          f"steps @ {args.freq_seconds:g}s")

    usable = _usable_markets(panels, channels)
    if not usable:
        raise SystemExit(f"No market has all of {channels} fully observed.")
    print(f"{len(usable)} of {len(panels)} markets fully observed across {channels}")
    series = {c: _channel_matrix(panels, c, usable) for c in channels}
    forecaster = None
    if args.checkpoint:
        from timesfm3 import TimesFM3Forecaster
        forecaster = TimesFM3Forecaster.from_checkpoint(args.checkpoint)

    results = _score(series, channels, args, forecaster)
    _report(results, channels, args)
    if args.plot:
        _plot(panels, args)


def _channel_matrix(panels, channel, keep):
    """One outcome per market: p_yes + p_no == 1 exactly in this archive, so
    the second token is an arithmetic mirror carrying no extra information.
    Step 0 is dropped because ``ret``/``abs_ret`` are undefined there."""
    return np.asarray([panels[i].features[channel][0, 1:].astype(np.float64)
                       for i in keep])


def _usable_markets(panels, channels):
    """Markets whose every requested channel is fully observed after step 0."""
    keep = []
    for i, panel in enumerate(panels):
        if all(np.isfinite(panel.features[c][0, 1:]).all() for c in channels):
            keep.append(i)
    return keep


def _score(series, channels, args, forecaster):
    """Per-window errors, tagging windows whose target never moves.

    Most windows in this archive are frozen -- the price does not change at all
    over the horizon -- and there last-value is exactly right by construction,
    so a model can only lose. Those windows are counted separately rather than
    quietly dominating the average.
    """
    ctx_len, hor = args.context, args.horizon
    out = {c: {"last": [], "cmean": [], "model": [], "frozen": []} for c in channels}
    ref = series[channels[0]]
    starts = np.unique(
        np.linspace(1, ref.shape[1] - ctx_len - hor - 1, args.windows).astype(int)
    )
    n_markets = min(series[c].shape[0] for c in channels)
    for m in range(n_markets):
        for s in starts:
            ctxs = {c: series[c][m, s:s + ctx_len] for c in channels}
            trus = {c: series[c][m, s + ctx_len:s + ctx_len + hor] for c in channels}
            preds = {}
            if forecaster is not None:
                # Channels are forecast jointly, so cross-variate attention can
                # let activity inform price and vice versa.
                targets = [_precondition(ctxs[c], c) for c in channels]
                res = forecaster.forecast(targets=targets, horizon=hor)
                preds = {c: _postcondition(res.point[i], c)
                         for i, c in enumerate(channels)}
            for c in channels:
                out[c]["frozen"].append(bool(trus[c].std() == 0))
                out[c]["last"].append(np.abs(ctxs[c][-1] - trus[c]).mean())
                out[c]["cmean"].append(np.abs(ctxs[c].mean() - trus[c]).mean())
                if preds:
                    out[c]["model"].append(np.abs(preds[c] - trus[c]).mean())
    return out


def _precondition(x, channel):
    """log1p-compress heavy-tailed count channels before the model sees them."""
    return np.log1p(np.abs(x)) * np.sign(x) if channel in COUNT_CHANNELS else x


def _postcondition(y, channel):
    return (np.expm1(np.abs(y)) * np.sign(y)) if channel in COUNT_CHANNELS else y


def _report(results, channels, args):
    have_model = bool(results[channels[0]]["model"])
    minutes = args.horizon * args.freq_seconds / 60.0
    for active_only in (False, True):
        scope = "active windows only" if active_only else "all windows"
        print(f"\nMAE {minutes:.0f} min ahead, context {args.context} — {scope}")
        head = f"{'channel':<10}{'last-value':>12}{'ctx-mean':>11}"
        if have_model:
            head += f"{'model':>11}{'model/best':>12}{'win rate':>10}"
        head += f"{'frozen':>9}{'n':>7}"
        print(head)
        for c in channels:
            r = results[c]
            keep = [i for i, f in enumerate(r["frozen"]) if not (active_only and f)]
            if not keep:
                print(f"{c:<10}{'(no windows in scope)':>32}")
                continue
            last = float(np.mean([r["last"][i] for i in keep]))
            cmean = float(np.mean([r["cmean"][i] for i in keep]))
            frozen = float(np.mean([r["frozen"][i] for i in keep]))
            line = f"{c:<10}{last:>12.5f}{cmean:>11.5f}"
            if have_model:
                model = float(np.mean([r["model"][i] for i in keep]))
                wins = np.mean([r["model"][i] < min(r["last"][i], r["cmean"][i])
                                for i in keep])
                line += f"{model:>11.5f}{model / min(last, cmean):>12.3f}{wins:>10.2f}"
            line += f"{frozen:>9.2f}{len(keep):>7}"
            print(line)
    if have_model:
        print("\n'model/best' below 1.0 means the model beats the better baseline.")
        print("'frozen' is the fraction of windows whose target never moves; there")
        print("last-value is exact by construction and no forecaster can win.")


def _plot(panels, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel = max(panels, key=lambda p: np.nanstd(p.features["mid"][0]))
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    t = np.arange(panel.num_steps) * panel.freq_seconds / 3600.0
    for ax, ch, lab in zip(axes, ("mid", "spread", "quotes"),
                           ("mid price", "spread", "quote updates")):
        ax.plot(t, panel.features[ch][0], lw=1.0, color="#d62728")
        ax.set_ylabel(lab)
    axes[0].set_title(f"Polymarket microstructure — {panel.info.label[:70]}")
    axes[-1].set_xlabel("hours into window")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()

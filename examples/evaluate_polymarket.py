"""Benchmark forecasters on Polymarket prediction-market microstructure.

Rebuilds each liquid market's outcome-token series from the public Pendulum
Flow archive and scores a forecast horizon per feature channel against a panel
of classical baselines, with the statistics such a comparison actually needs.

Three properties of this data drive the evaluation design:

1. **Most horizons are frozen.** A prediction market can sit at one price for
   an hour, so a large fraction of windows have a target that never moves.
   There last-value is exactly right by construction and no forecaster can beat
   it. Frozen and active windows are reported separately rather than letting
   the frozen majority set the average.
2. **Scaled MAE is unusable.** Dividing by the context standard deviation --
   what this repo's other benchmarks do -- collapses on those flat contexts and
   produces meaningless six-figure "errors". Losses are reported in native
   units, and comparisons are made as ratios.
3. **Sliding windows are not independent.** Overlapping windows share most of
   their context, and windows from one market share its regime. Windows are
   therefore taken **non-overlapping**, significance uses a HAC-corrected
   Diebold-Mariano test, confidence intervals come from a bootstrap that
   resamples whole markets, and p-values are Holm-corrected across channels.

Usage:
    # Baselines only -- no model needed:
    python examples/evaluate_polymarket.py --start 2026-08-29T03 --end 2026-08-29T15

    # Add a TimesFM-3 checkpoint to the comparison:
    python examples/evaluate_polymarket.py --start 2026-08-29T03 --end 2026-08-29T15 \
        --checkpoint timesfm3_polymarket.pt

Requires pyarrow (`pip install -e .[polymarket]`).
"""

import argparse
import datetime as dt
import json
import pickle

import numpy as np

from timesfm3.baselines import DEFAULT_BASELINES
from timesfm3.data.polymarket import (
    COUNT_CHANNELS,
    PolymarketArchive,
    build_market_panels,
    select_covered,
    top_markets,
)
from timesfm3.evaluation import compare

MODEL = "timesfm3"


def _hour(text: str) -> dt.datetime:
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"expected YYYY-MM-DDTHH, got {text!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=_hour)
    p.add_argument("--end", type=_hour)
    p.add_argument("--panels", default=None,
                   help="pickled panels from a previous run (skips download)")
    p.add_argument("--cache-dir", default="data/polymarket")
    p.add_argument("--save-panels", default=None)
    p.add_argument("--channels", default="mid,spread,abs_ret,quotes")
    p.add_argument("--candidates", type=int, default=300)
    p.add_argument("--freq-seconds", type=float, default=15.0)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--reference", default="last-value",
                   help="baseline every forecaster is compared against")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--compare-space", default="native",
                   choices=["native", "log"],
                   help="'native' inverts the model's log1p compression before "
                        "scoring counts; 'log' scores every forecaster in log1p "
                        "space instead. Reported results should not depend on "
                        "which is chosen.")
    p.add_argument("--accept-unverified", action="store_true")
    p.add_argument("--split", default=None,
                   help="split JSON written by train_polymarket.py")
    p.add_argument("--eval-markets", default="heldout",
                   choices=["heldout", "train", "all"],
                   help="which markets to score when --split is given")
    p.add_argument("--time-tail", type=float, default=None,
                   help="score only the final fraction of the time axis "
                        "(defaults to what --split says was held out)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    channels = tuple(args.channels.split(","))
    panels = _load_panels(args)

    tail = args.time_tail
    if args.split:
        with open(args.split) as f:
            split = json.load(f)
        if tuple(split["channels"]) != channels:
            raise SystemExit(
                f"split was written for channels {split['channels']}, "
                f"but --channels is {list(channels)}"
            )
        key = {"heldout": "heldout_markets", "train": "train_markets"}.get(
            args.eval_markets
        )
        chosen = (
            list(range(len(panels))) if key is None
            else [i for i in split[key] if i < len(panels)]
        )
        if tail is None:
            tail = 1.0 - float(split["train_fraction"])
        print(f"scoring {args.eval_markets} markets from {args.split}: "
              f"{len(chosen)} of {len(panels)} panels, "
              f"final {tail:.0%} of the time axis (never trained on)")
        panels = [panels[i] for i in chosen]
    elif tail is None:
        tail = 1.0

    usable = [
        p for p in panels
        if all(np.isfinite(p.features[c][0, 1:]).all() for c in channels)
    ]
    if not usable:
        raise SystemExit(f"No market has all of {channels} fully observed.")
    print(f"{len(usable)} of {len(panels)} markets fully observed across {channels}")
    args._tail = tail

    forecaster = None
    if args.checkpoint:
        from timesfm3 import TimesFM3Forecaster
        forecaster = TimesFM3Forecaster.from_checkpoint(args.checkpoint)

    losses, groups, frozen = _collect_losses(usable, channels, args, forecaster)
    _report(losses, groups, frozen, channels, args)


def _load_panels(args):
    if args.panels:
        with open(args.panels, "rb") as f:
            return pickle.load(f)
    if not (args.start and args.end):
        raise SystemExit("give --start/--end, or --panels from an earlier run")
    archive = PolymarketArchive(cache_dir=args.cache_dir)
    paths = archive.download_range(
        args.start, args.end,
        on_mismatch="warn" if args.accept_unverified else "raise",
    )
    print(f"{len(paths)} verified hours; ranking markets by trade count ...")
    markets = top_markets(paths, args.candidates)
    panels = select_covered(
        build_market_panels(paths, markets, freq_seconds=args.freq_seconds),
        channel="mid", min_coverage=0.98,
    )
    if args.save_panels:
        with open(args.save_panels, "wb") as f:
            pickle.dump(panels, f)
        print(f"saved panels to {args.save_panels}")
    return panels


def _collect_losses(panels, channels, args, forecaster):
    """Per-window MAE for every forecaster, on non-overlapping windows."""
    ctx_len, hor = args.context, args.horizon
    step = ctx_len + hor                    # non-overlapping: no shared context
    names = [b.name for b in DEFAULT_BASELINES] + ([MODEL] if forecaster else [])
    losses = {c: {n: [] for n in names} for c in channels}
    groups = {c: [] for c in channels}
    frozen = {c: [] for c in channels}

    tail = getattr(args, "_tail", 1.0)
    log_space = args.compare_space == "log"
    for market_id, panel in enumerate(panels):
        series = {c: panel.features[c][0, 1:].astype(np.float64) for c in channels}
        full = min(len(v) for v in series.values())
        # Score only the tail of the time axis, which training never saw.
        start = int(full * (1.0 - tail))
        series = {c: v[start:] for c, v in series.items()}
        length = full - start
        for s in range(0, length - step + 1, step):
            ctxs = {c: series[c][s:s + ctx_len] for c in channels}
            trus = {c: series[c][s + ctx_len:s + step] for c in channels}
            if log_space:
                # Score everyone in the space the model works in, so the
                # comparison cannot turn on the expm1 back-transform (which is
                # a biased estimator of a mean).
                ctxs = {c: _pre(v, c) for c, v in ctxs.items()}
                trus = {c: _pre(v, c) for c, v in trus.items()}
            preds = {c: {} for c in channels}
            for c in channels:
                for b in DEFAULT_BASELINES:
                    preds[c][b.name] = b.forecast(ctxs[c], hor)
            if forecaster is not None:
                # Channels are forecast jointly so cross-variate attention can
                # let activity inform price and vice versa.
                targets = [ctxs[c] if log_space else _pre(ctxs[c], c)
                           for c in channels]
                res = forecaster.forecast(targets=targets, horizon=hor)
                for i, c in enumerate(channels):
                    preds[c][MODEL] = (
                        res.point[i] if log_space else _post(res.point[i], c)
                    )
            for c in channels:
                groups[c].append(market_id)
                frozen[c].append(bool(trus[c].std() == 0))
                for n in names:
                    losses[c][n].append(float(np.abs(preds[c][n] - trus[c]).mean()))
    return losses, groups, frozen


def _pre(x, channel):
    """log1p-compress heavy-tailed count channels before the model sees them."""
    return np.log1p(np.abs(x)) * np.sign(x) if channel in COUNT_CHANNELS else x


def _post(y, channel):
    return (np.expm1(np.abs(y)) * np.sign(y)) if channel in COUNT_CHANNELS else y


def _report(losses, groups, frozen, channels, args):
    minutes = args.horizon * args.freq_seconds / 60.0
    for active_only in (False, True):
        scope = "ACTIVE windows" if active_only else "all windows"
        print(f"\n{'='*82}\n{scope}: MAE {minutes:.0f} min ahead, "
              f"context {args.context}, non-overlapping\n{'='*82}")
        for c in channels:
            mask = np.array(
                [not (active_only and f) for f in frozen[c]], dtype=bool
            )
            if mask.sum() < 4:
                print(f"\n{c}: too few windows ({int(mask.sum())})")
                continue
            sub = {n: np.asarray(v)[mask] for n, v in losses[c].items()}
            grp = np.asarray(groups[c])[mask]
            if args.reference not in sub:
                raise SystemExit(f"unknown --reference {args.reference!r}")
            results = compare(sub, args.reference, grp, resamples=args.resamples)
            n_win = int(mask.sum())
            frozen_frac = float(np.mean(np.asarray(frozen[c])[mask]))
            print(f"\n{c}  —  {n_win} windows, {len(np.unique(grp))} markets, "
                  f"{frozen_frac:.0%} frozen, reference = {args.reference} "
                  f"(MAE {sub[args.reference].mean():.5g})")
            print(f"  {'forecaster':<12}{'MAE':>11}{'ratio':>8}"
                  f"{'95% CI':>18}{'p(Holm)':>10}  verdict")
            for name, r in sorted(results.items(), key=lambda kv: kv[1].ratio):
                ci = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"
                p = r.p_adjusted if r.p_adjusted is not None else r.p_value
                mark = "*" if r.significant else " "
                print(f"  {name:<12}{r.mean_loss:>11.5g}{r.ratio:>8.3f}"
                      f"{ci:>18}{p:>10.3g}{mark} {r.verdict}")
            eff = np.mean([r.n_effective for r in results.values()])
            print(f"  (effective sample size ≈ {eff:.0f} of {n_win} windows)")
    print("\n* = significant at 5% after Holm correction across forecasters.")
    print("ratio < 1 means lower error than the reference baseline.")


if __name__ == "__main__":
    main()

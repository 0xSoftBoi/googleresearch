"""Volatility TimesFM-3, attempt 2: more assets, implied vol, an asset holdout.

Protocol: docs/VERIFICATION.md section 14. Attempt 1 (sections 11-13) tied
RiskMetrics on 2020-2026 and was not released; that test period is spent
for its 16 assets, so attempt 2 is judged only on assets never used for
training, validation or selection.

    python examples/hedge_fund/train_vol_model_v2.py data                # list assets, splits
    python examples/hedge_fund/train_vol_model_v2.py finetune --out data/vol/v2_a.pt [...]
    python examples/hedge_fund/train_vol_model_v2.py test --candidate data/vol/v2_a.pt   # ONCE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_vol_model import CHRONOS, STARTER, load_model, table  # noqa: E402

from timesfm3.checkpoint import load_checkpoint, package_checkpoint  # noqa: E402
from timesfm3.data.markets import (  # noqa: E402
    CRYPTO_UNIVERSE, DEFAULT_UNIVERSE, EXTRA_UNIVERSE, IMPLIED_VOL, TRADING_DAYS,
    fetch_fred_csv, load_universe,
)
from timesfm3.quant import volmodel as vm  # noqa: E402

V1_CANDIDATE = "data/vol/ft6.pt"
TRAIN_EXTRA = ("MXNUSD", "INRUSD", "CNYUSD", "TWDUSD", "THBUSD", "MYRUSD", "UST1Y", "UST30Y")
HOLDOUT = ("NASDAQ100", "DJIA", "KRWUSD", "BRLUSD", "ZARUSD", "SGDUSD", "DKKUSD", "UST5Y",
           "BTCUSD", "ETHUSD", "LTCUSD")
ALPHA = 0.05


def load_assets(cache_dir: str = "data/fred") -> tuple[list[vm.AssetRV], list[vm.AssetRV]]:
    """(training assets, holdout assets), each with implied variance attached where mapped."""
    assets: list[vm.AssetRV] = []
    for universe in (DEFAULT_UNIVERSE, EXTRA_UNIVERSE, CRYPTO_UNIVERSE):  # separate calendars
        assets += vm.rv_panel(load_universe(universe, cache_dir=cache_dir, verbose=False))
    iv_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for a in assets:
        sid = IMPLIED_VOL.get(a.name)
        if sid is None:
            continue
        if sid not in iv_cache:
            iv_cache[sid] = fetch_fred_csv(sid, cache_dir=cache_dir)
        d, v = iv_cache[sid]
        idx = np.searchsorted(d, a.dates)
        idx = np.clip(idx, 0, len(d) - 1)
        same_day = d[idx] == a.dates          # only the same day's close; never a later one
        iv = np.full(len(a.dates), np.nan)
        iv[same_day] = (v[idx[same_day]] / 100.0) ** 2 / TRADING_DAYS
        a.set_implied_var(iv)
    train_names = {i.name for i in DEFAULT_UNIVERSE} | set(TRAIN_EXTRA)
    train = [a for a in assets if a.name in train_names]
    holdout = [a for a in assets if a.name in HOLDOUT]
    missing = (train_names | set(HOLDOUT)) - {a.name for a in assets} - {"GOLD"}
    if missing:
        raise RuntimeError(f"assets failed to load: {sorted(missing)}")
    return train, holdout


def competitors() -> dict[str, vm.Forecaster]:
    from chronos import BaseChronosPipeline

    models = {
        "riskmetrics": vm.riskmetrics(),
        "har": vm.har(),
        "har-log": vm.har(log_space=True),
        "har-iv": vm.har_iv(),
        "iv-or-riskmetrics": vm.iv_or_riskmetrics(),
        "starter-small": vm.timesfm(load_model(STARTER)),
        "timesfm3-vol-v1": vm.timesfm(load_model(V1_CANDIDATE), channels=vm.CHANNELS[:3]),
    }
    for name, mid in CHRONOS.items():
        models[name] = vm.chronos(BaseChronosPipeline.from_pretrained(mid, device_map="cpu"))
    return models


def cmd_data(args) -> None:
    train, holdout = load_assets()
    for label, group in (("TRAIN/VAL", train), ("HOLDOUT (test only)", holdout)):
        print(f"== {label}")
        for a in group:
            ivn = int(np.isfinite(a.implied_var).sum())
            print(f"  {a.name:10s} {a.dates[0]} .. {a.dates[-1]}  train {vm.split_origins(a, 'train').size:6d}"
                  f"  val {vm.split_origins(a, 'val').size:5d}  test {vm.split_origins(a, 'test').size:5d}"
                  f"  implied-vol days {ivn}")
    print("totals: train", sum(vm.split_origins(a, "train").size for a in train),
          "val", sum(vm.split_origins(a, "val").size for a in train),
          "holdout test", sum(vm.split_origins(a, "test").size for a in holdout))


def cmd_finetune(args) -> None:
    torch.set_num_threads(args.threads)
    train, _ = load_assets()
    model = load_model(args.base)
    channels = tuple(args.channels.split(","))
    t0 = time.time()
    val = vm.collect(train, "val", {"_": vm.timesfm(model, channels=channels)})
    best_state, history = vm.finetune_qlike(
        model, train, val=val, val_assets=train, steps=args.steps, batch_size=args.batch_size,
        lr=args.lr, pinball_weight=args.pinball_weight, val_every=args.val_every, seed=args.seed,
        channels=channels, step_horizon=args.step_horizon, agg_weight=args.agg_weight,
        iv_fraction=args.iv_fraction,
    )
    scratch = args.out + ".train.pt"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"config": model.config, "model": best_state}, scratch)
    best_val = min(h["val"] for h in history if "val" in h)
    package_checkpoint(scratch, args.out, meta={
        "name": args.name,
        "description": "TimesFM-3 small fine-tuned for daily variance forecasting. Inputs over a "
                       "512-day context: log squared daily returns (target), causal EWMA(0.94) and "
                       "22-day-mean variance, and CBOE implied variance where the asset has an "
                       "index (else unobserved). Output: log-variance forecasts; exp(point) is "
                       "the variance forecast. QLIKE-trained.",
        "base_checkpoint": load_checkpoint(args.base)["meta"].get("name", args.base),
        "context": vm.CONTEXT, "input_channels": list(channels),
        "train_assets": sorted(a.name for a in train),
        "corpus": f"FRED, training assets only, windows with targets before {vm.TRAIN_END}",
        "objective": f"QLIKE per step (first {args.step_horizon or 32}) + {args.agg_weight} x "
                     f"5-day QLIKE + {args.pinball_weight} x pinball on log variance",
        "iv_fraction": args.iv_fraction,
        "finetune_steps": args.steps, "finetune_lr": args.lr, "batch_size": args.batch_size,
        "seed": args.seed, "best_val_qlike_5d_calibrated": best_val,
        "train_minutes": round((time.time() - t0) / 60, 1),
    })
    os.remove(scratch)
    with open(args.out + ".history.json", "w") as fh:
        json.dump(history, fh)
    print(f"saved {args.out}  best validation QLIKE(5d, calibrated) {best_val:.4f}")


def cmd_test(args) -> None:
    if os.path.exists(args.json) and not args.force:
        raise SystemExit(f"{args.json} exists: that test has been run. A pre-registered test runs "
                         "once; pass --json for a new file (or --force to overwrite).")
    train, holdout = load_assets()
    models = competitors()
    channels = tuple(load_checkpoint(args.candidate)["meta"]["input_channels"])
    print(f"candidate {args.candidate}: input channels {channels}")
    models[args.name] = vm.timesfm(load_model(args.candidate), channels=channels)
    calibration = vm.fit_calibration(vm.collect(train, "val", models, verbose=True))
    test = vm.collect(holdout, "test", models, verbose=True)
    dates, losses = vm.daily_losses(test, calibration)
    result = {
        "candidate": args.candidate, "candidate_name": args.name,
        "holdout_assets": sorted(set(test.assets.tolist())),
        "test_dates": [str(dates[0]), str(dates[-1]), int(len(dates))],
        "test_forecasts": int(len(test.dates)),
        "calibration_fit_on_validation": {m: c.tolist() for m, c in calibration.items()},
        "vs_candidate": table(losses, args.name, "HOLDOUT ASSETS, 2020-onward (gate view)"),
        "vs_riskmetrics": table(losses, "riskmetrics", "HOLDOUT, reference RiskMetrics (context)"),
    }
    with_iv = np.isin(test.assets, [a.name for a in holdout if np.isfinite(a.implied_var).any()])
    if with_iv.any():
        sub = vm.Forecasts(test.dates[with_iv], test.assets[with_iv], test.target[with_iv],
                           {m: p[with_iv] for m, p in test.preds.items()})
        _, iv_losses = vm.daily_losses(sub, calibration)
        result["holdout_with_implied_vol_vs_candidate"] = table(
            iv_losses, args.name, "HOLDOUT assets with an implied-vol index only (secondary)")
    gate = {k: v for k, v in result["vs_candidate"][f"{vm.EVAL_HORIZON}d"].items()
            if not k.startswith("_")}
    passed = all(v["ratio_vs_reference"] > 1 and v["p_holm"] < ALPHA for v in gate.values())
    result["gate"] = {"horizon_days": vm.EVAL_HORIZON, "alpha_holm": ALPHA, "passed": passed,
                      "failing": [k for k, v in gate.items()
                                  if not (v["ratio_vs_reference"] > 1 and v["p_holm"] < ALPHA)]}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nGATE ({vm.EVAL_HORIZON}-day QLIKE on holdout assets, candidate significantly better "
          f"than every competitor, Holm p < {ALPHA}): {'PASSED' if passed else 'FAILED'}")
    if not passed:
        print("  not beaten:", ", ".join(result["gate"]["failing"]))
    print(f"wrote {args.json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data").set_defaults(func=cmd_data)
    f = sub.add_parser("finetune")
    f.add_argument("--base", default=STARTER)
    f.add_argument("--out", required=True)
    f.add_argument("--name", default="timesfm3-vol-small")
    f.add_argument("--channels", default=",".join(vm.CHANNELS))
    f.add_argument("--steps", type=int, default=6000)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--pinball-weight", type=float, default=0.1)
    f.add_argument("--step-horizon", type=int, default=5)
    f.add_argument("--agg-weight", type=float, default=2.0)
    f.add_argument("--val-every", type=int, default=500)
    f.add_argument("--iv-fraction", type=float, default=0.0,
                   help="share of each batch drawn from windows with implied vol")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--threads", type=int, default=4)
    f.set_defaults(func=cmd_finetune)
    t = sub.add_parser("test")
    t.add_argument("--candidate", required=True)
    t.add_argument("--name", default="timesfm3-vol-small-v2")
    t.add_argument("--json", default="results/vol_test_v2.json")
    t.add_argument("--force", action="store_true", help="overwrite an existing result file")
    t.set_defaults(func=cmd_test)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

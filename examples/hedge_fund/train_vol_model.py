"""Build, select and test a volatility-specialized TimesFM-3 (protocol: docs/VERIFICATION.md §11).

    python examples/hedge_fund/train_vol_model.py zeroshot            # validation only
    python examples/hedge_fund/train_vol_model.py finetune --steps 1500 --lr 5e-5 --out data/vol/ft.pt
    python examples/hedge_fund/train_vol_model.py test --candidate data/vol/ft.pt   # run ONCE

``zeroshot`` and ``finetune`` touch only the 2015-2019 validation period.
``test`` fits every model's calibration scalar on validation, scores
2020-onward once, writes the full result to ``--json``, and prints the
pre-registered gate verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from timesfm3.checkpoint import load_checkpoint, package_checkpoint
from timesfm3.data.markets import load_universe
from timesfm3.model import TimesFM3Model
from timesfm3.quant import volmodel as vm

STARTER = os.path.join(os.path.dirname(__file__), "..", "..", "timesfm3", "assets", "starter-small.pt")
CHRONOS = {
    "chronos-bolt-tiny": "amazon/chronos-bolt-tiny",
    "chronos-bolt-small": "amazon/chronos-bolt-small",
    "chronos-2": "amazon/chronos-2",
}
ALPHA = 0.05


def load_model(path: str) -> TimesFM3Model:
    state = load_checkpoint(path)
    model = TimesFM3Model(state["config"])
    model.load_state_dict(state["model"])
    model.eval()
    return model


def competitors(with_chronos: bool = True) -> dict[str, vm.Forecaster]:
    models = {
        "riskmetrics": vm.riskmetrics(),
        "har": vm.har(),
        "har-log": vm.har(log_space=True),
        "starter-small": vm.timesfm(load_model(STARTER)),
    }
    if with_chronos:
        from chronos import BaseChronosPipeline

        for name, mid in CHRONOS.items():
            models[name] = vm.chronos(BaseChronosPipeline.from_pretrained(mid, device_map="cpu"))
    return models


def table(losses: dict[str, np.ndarray], reference: str, title: str) -> dict:
    out = {}
    print(f"\n{title}  (reference: {reference}; ratio > 1 means the reference is better)")
    for col, h in enumerate(vm.HORIZONS):
        res = vm.compare_models(losses, reference, col)
        print(f"  horizon {h}d: reference mean QLIKE {losses[reference][:, col].mean():.4f}")
        rows = sorted(res.values(), key=lambda r: r.ratio)
        for r in rows:
            print(f"    {r.name:20s} QLIKE {r.mean_loss:.4f}  ratio {r.ratio:6.3f} "
                  f"[{r.ci_low:.3f}, {r.ci_high:.3f}]  p_holm {r.p_adjusted:.2e}  "
                  f"ref wins {100 * (1 - r.win_rate):4.1f}%  [{r.verdict}]")
        out[f"{h}d"] = {r.name: {"mean_qlike": r.mean_loss, "ratio_vs_reference": r.ratio,
                                 "ci": [r.ci_low, r.ci_high], "p_holm": r.p_adjusted,
                                 "p_raw": r.p_value, "n_dates": r.n,
                                 "n_effective": r.n_effective, "verdict": r.verdict}
                        for r in rows}
        out[f"{h}d"]["_reference_mean_qlike"] = float(losses[reference][:, col].mean())
    return out


def cmd_zeroshot(args) -> None:
    assets = vm.rv_panel(load_universe(cache_dir="data/fred", verbose=False))
    f = vm.collect(assets, "val", competitors(not args.no_chronos), verbose=True)
    _, losses = vm.daily_losses(f, vm.fit_calibration(f))
    table(losses, "riskmetrics", "VALIDATION 2015-2019, calibrated in-sample (selection view only)")


def cmd_finetune(args) -> None:
    torch.set_num_threads(args.threads)
    assets = vm.rv_panel(load_universe(cache_dir="data/fred", verbose=False))
    model = load_model(args.base)
    channels = tuple(args.channels.split(","))
    t0 = time.time()
    val = vm.collect(assets, "val", {"_": vm.timesfm(model, channels=channels)})
    best_state, history = vm.finetune_qlike(
        model, assets, val=val, val_assets=assets, steps=args.steps, batch_size=args.batch_size,
        lr=args.lr, pinball_weight=args.pinball_weight, val_every=args.val_every, seed=args.seed,
        channels=channels, step_horizon=args.step_horizon, agg_weight=args.agg_weight,
    )
    scratch = args.out + ".train.pt"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"config": model.config, "model": best_state}, scratch)
    best_val = min(h["val"] for h in history if "val" in h)
    base_meta = load_checkpoint(args.base)["meta"]
    package_checkpoint(scratch, args.out, meta={
        "name": args.name,
        "description": "TimesFM-3 small fine-tuned for daily variance forecasting: input log "
                       "variance over a 512-day context, output log-variance forecasts; "
                       "exp(point) is the variance forecast. QLIKE-trained.",
        "base_checkpoint": base_meta.get("name", args.base),
        "task": "variance forecasting (input: log squared daily returns, 512-day context)",
        "context": vm.CONTEXT,
        "input_channels": list(channels),
        "corpus": "FRED multi-asset panel (timesfm3.data.markets.DEFAULT_UNIVERSE), "
                  f"windows with targets before {vm.TRAIN_END}",
        "objective": f"QLIKE on exp(point) per step (first {args.step_horizon or 32} steps) + "
                     f"{args.agg_weight} x QLIKE of the 5-day mean + "
                     f"{args.pinball_weight} x pinball on log variance",
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
    assets = vm.rv_panel(load_universe(cache_dir="data/fred", verbose=False))
    models = competitors()
    channels = tuple(load_checkpoint(args.candidate)["meta"].get("input_channels", ["log_rv"]))
    print(f"candidate {args.candidate}: input channels {channels}")
    models[args.name] = vm.timesfm(load_model(args.candidate), channels=channels)
    val = vm.collect(assets, "val", models, verbose=True)
    calibration = vm.fit_calibration(val)
    test = vm.collect(assets, "test", models, verbose=True)
    dates, losses = vm.daily_losses(test, calibration)
    result = {
        "candidate": args.candidate, "candidate_name": args.name,
        "test_dates": [str(dates[0]), str(dates[-1]), int(len(dates))],
        "test_forecasts": int(len(test.dates)),
        "assets": sorted(set(test.assets.tolist())),
        "calibration_fit_on_validation": {m: c.tolist() for m, c in calibration.items()},
        "vs_candidate": table(losses, args.name, "TEST 2020-onward, validation-fitted calibration"),
        "vs_riskmetrics": table(losses, "riskmetrics", "TEST, reference RiskMetrics (context)"),
    }
    # Secondary, not gating: SP500 never appears in training (FRED starts it in 2016).
    sp = test.assets == "SP500"
    if sp.any():
        sub = vm.Forecasts(test.dates[sp], test.assets[sp], test.target[sp],
                           {m: p[sp] for m, p in test.preds.items()})
        _, sp_losses = vm.daily_losses(sub, calibration)
        result["sp500_only_vs_candidate"] = table(sp_losses, args.name, "TEST, SP500 only (secondary)")
    gate = result["vs_candidate"][f"{vm.EVAL_HORIZON}d"]
    others = {k: v for k, v in gate.items() if not k.startswith("_")}
    passed = all(v["ratio_vs_reference"] > 1 and v["p_holm"] < ALPHA for v in others.values())
    result["gate"] = {"horizon_days": vm.EVAL_HORIZON, "alpha_holm": ALPHA, "passed": passed,
                      "failing": [k for k, v in others.items()
                                  if not (v["ratio_vs_reference"] > 1 and v["p_holm"] < ALPHA)]}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nGATE ({vm.EVAL_HORIZON}-day QLIKE, candidate significantly better than every "
          f"competitor, Holm p < {ALPHA}): {'PASSED' if passed else 'FAILED'}")
    if not passed:
        print("  not beaten:", ", ".join(result["gate"]["failing"]))
    print(f"wrote {args.json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zeroshot")
    z.add_argument("--no-chronos", action="store_true")
    z.set_defaults(func=cmd_zeroshot)
    f = sub.add_parser("finetune")
    f.add_argument("--base", default=STARTER)
    f.add_argument("--out", required=True)
    f.add_argument("--name", default="timesfm3-vol-small")
    f.add_argument("--steps", type=int, default=1500)
    f.add_argument("--lr", type=float, default=5e-5)
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--pinball-weight", type=float, default=0.1)
    f.add_argument("--val-every", type=int, default=100)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--threads", type=int, default=4)
    f.add_argument("--step-horizon", type=int, default=None,
                   help="per-step QLIKE over only the first k forecast steps")
    f.add_argument("--agg-weight", type=float, default=1.0)
    f.add_argument("--channels", default="log_rv",
                   help=f"comma-separated, log_rv first; available: {','.join(vm.CHANNELS)}")
    f.set_defaults(func=cmd_finetune)
    t = sub.add_parser("test")
    t.add_argument("--candidate", required=True)
    t.add_argument("--name", default="timesfm3-vol-small")
    t.add_argument("--json", default="results/vol_test.json")
    t.add_argument("--force", action="store_true", help="overwrite an existing result file")
    t.set_defaults(func=cmd_test)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

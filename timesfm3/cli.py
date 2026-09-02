"""``timesfm3`` command-line interface.

    timesfm3 serve [--port 8000] [--checkpoint PATH ...] [--api-key KEY]
    timesfm3 forecast data.csv --horizon 24 [--model NAME] [--output out.csv]
    timesfm3 backtest data.csv --context 256 --horizon 24 [--models a,b]
    timesfm3 models [--checkpoint PATH ...]
    timesfm3 anomalies data.csv [--context 96] [--threshold 2] [--model NAME]
    timesfm3 finetune data.csv --out my-model.pt [--from CKPT] [--steps 300]
    timesfm3 credits buy --api URL --count 25 [--api-key K | --private-key 0x..] [--wallet credits.json]
    timesfm3 credits status [--wallet credits.json]
    timesfm3 pack train_ckpt.pt packaged.pt --name my-model
    timesfm3 train --config small --steps 6000 ...     (see timesfm3.train)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from . import __version__


def _registry(args, include_bundled: bool = True):
    from .serving.registry import ModelRegistry

    return ModelRegistry.from_env(
        checkpoints=args.checkpoint or (),
        include_bundled=include_bundled and not getattr(args, "no_bundled", False),
        device=getattr(args, "device", None),
        default=getattr(args, "default_model", None),
    )


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", "-c", action="append", metavar="[NAME=]PATH",
                   help="TimesFM-3 checkpoint to register (repeatable).")
    p.add_argument("--no-bundled", action="store_true", help="Skip the bundled starter model.")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.add_argument("--default-model", default=None, help="Registry name to serve by default.")


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The server needs the 'serve' extra:  pip install 'timesfm3[serve]'", file=sys.stderr)
        return 2
    from .serving.app import create_app

    if args.api_key:
        os.environ["TIMESFM3_API_KEY"] = args.api_key
    registry = _registry(args)
    app = create_app(registry=registry)
    names = ", ".join(f"{e.name}{'*' if e.name == registry.default else ''}" for e in registry.entries())
    print(f"TimesFM-3 Forecast Service v{__version__}: {len(registry)} models ({names})")
    x402 = getattr(app.state, "x402", None)
    if x402 is not None:
        print(f"x402 pay-per-call: {x402.network} -> {x402.pay_to} via {x402.facilitator}"
              + ("" if x402.mainnet else "  (testnet: fund wallets with Base Sepolia USDC)"))
    print(f"dashboard http://{args.host}:{args.port}/   docs http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_models(args) -> int:
    registry = _registry(args)
    for info in registry.describe():
        star = "*" if info["default"] else " "
        params = f"{info['parameters'] / 1e6:.1f}M" if info["parameters"] else "-"
        print(f"{star} {info['name']:16s} {info['kind']:10s} {params:>7s}  {info['description']}")
    return 0


def cmd_forecast(args) -> int:
    from .tabular import future_timestamps, infer_step, parse_freq, read_series_csv, write_forecast_csv

    table = read_series_csv(args.input)
    if args.columns:
        wanted = [c.strip() for c in args.columns.split(",")]
        idx = [table.names.index(c) for c in wanted]
        names, values = wanted, table.values[idx]
    else:
        names, values = table.names, table.values
    registry = _registry(args)
    entry = registry.get(args.model)
    result = entry.forecast([v for v in values], args.horizon)
    stamps = None
    if table.timestamps:
        step = parse_freq(args.freq) if args.freq else infer_step(table.timestamps)
        if step is not None:
            stamps = future_timestamps(table.timestamps[-1], args.horizon, step)
    if args.format == "json" or (args.output and args.output.endswith(".json")):
        keys = [f"q{int(round(q * 100))}" for q in result.quantile_levels]
        payload = {
            "model": entry.name, "horizon": args.horizon, "timestamps": stamps,
            "forecasts": [
                {"name": n, "point": result.point[i].tolist(),
                 "quantiles": {k: result.quantiles[i, :, j].tolist() for j, k in enumerate(keys)}}
                for i, n in enumerate(names)
            ],
        }
        text = json.dumps(payload, indent=None if args.output else 2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(text)
        else:
            print(text)
    else:
        write_forecast_csv(args.output or sys.stdout, names, result.point, result.quantiles,
                           result.quantile_levels, stamps)
    if args.output:
        print(f"{entry.name}: {len(names)} series x {args.horizon} steps -> {args.output}",
              file=sys.stderr)
    return 0


def cmd_backtest(args) -> int:
    from .serving.app import run_backtest
    from .tabular import read_series_csv

    table = read_series_csv(args.input)
    registry = _registry(args)
    models = [m.strip() for m in args.models.split(",")] if args.models else registry.names()
    report = run_backtest(
        registry, [v for v in table.values], args.context, args.horizon, models,
        args.reference, args.windows, args.metric, args.overlap,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"{args.input}: {table.num_series} series, {report['windows_per_series']} "
          f"non-overlapping windows/series, context {args.context}, horizon {args.horizon}, "
          f"{args.metric.upper()} ratio vs {args.reference} (< 1 is better)")
    print(f"{'model':16s} {'mean ' + args.metric:>11s} {'ratio':>7s} {'95% CI':>17s} "
          f"{'p (Holm)':>9s} {'win':>5s} {'n (eff)':>10s}  verdict")
    for s in report["scores"]:
        ci = f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]" if s["ci_low"] is not None else "-"
        p = f"{s['p_adjusted']:.3f}" if s["p_adjusted"] is not None else "-"
        win = f"{100 * s['win_rate']:.0f}%" if s["verdict"] != "reference" else "-"
        print(f"{s['model']:16s} {s['mean_loss']:11.4g} {s['ratio']:7.3f} {ci:>17s} {p:>9s} "
              f"{win:>5s} {s['n']:4d} ({s['n_effective']:4.0f})  {s['verdict']}")
    return 0


def _print_backtest(report: dict, title: str) -> None:
    print(title)
    print(f"{'model':16s} {'mean mae':>11s} {'ratio':>7s} {'95% CI':>17s} "
          f"{'p (Holm)':>9s} {'win':>5s} {'n (eff)':>10s}  verdict")
    for s in report["scores"]:
        ci = f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]" if s["ci_low"] is not None else "-"
        p = f"{s['p_adjusted']:.3f}" if s["p_adjusted"] is not None else "-"
        win = f"{100 * s['win_rate']:.0f}%" if s["verdict"] != "reference" else "-"
        print(f"{s['model']:16s} {s['mean_loss']:11.4g} {s['ratio']:7.3f} {ci:>17s} {p:>9s} "
              f"{win:>5s} {s['n']:4d} ({s['n_effective']:4.0f})  {s['verdict']}")


def cmd_anomalies(args) -> int:
    from .anomaly import detect_anomalies
    from .tabular import read_series_csv

    table = read_series_csv(args.input)
    registry = _registry(args)
    entry = registry.get(args.model)
    results = []
    for name, x in zip(table.names, table.values):
        rep = detect_anomalies(entry, x, args.context, args.block, args.threshold)
        results.append((name, rep, rep.anomalies(x, table.timestamps)))
    if args.json:
        print(json.dumps({
            "model": entry.name, "context": args.context, "block": args.block,
            "threshold": args.threshold,
            "series": [{"name": n, "n_scored": int(np.isfinite(r.scores).sum()),
                        "n_flagged": int(r.flagged.sum()), "anomalies": a}
                       for n, r, a in results],
        }, indent=2))
        return 0
    print(f"{args.input}: model {entry.name}, context {args.context}, block {args.block}, "
          f"threshold {args.threshold} (1.0 = 80% band edge)")
    for name, rep, anomalies in results:
        print(f"\n{name}: {len(anomalies)} anomalies in {int(np.isfinite(rep.scores).sum())} scored steps")
        for a in anomalies:
            when = a.get("timestamp", f"#{a['index']}")
            print(f"  {when:>20s}  value {a['value']:>10.4g}  expected {a['expected']:>10.4g} "
                  f"[{a['lower']:.4g}, {a['upper']:.4g}]  score {a['score']:.2f} {a['direction']}")
    return 0


def cmd_finetune(args) -> int:
    from .finetune import finetune
    from .serving.registry import ASSET_DIR
    from .tabular import read_series_csv

    base = args.base
    if base is None:
        bundled = sorted(glob.glob(os.path.join(ASSET_DIR, "*.pt")))
        if not bundled:
            print("error: no bundled checkpoint; pass --from PATH", file=sys.stderr)
            return 1
        base = bundled[0]
    table = read_series_csv(args.input)
    periods = tuple(int(p) for p in args.periods.split(",")) if args.periods else ()
    report = finetune(
        table.values, base, args.out, name=args.name, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, context_patches=args.context_patches,
        horizon_patches=args.horizon_patches, train_fraction=args.train_fraction,
        periods=periods, synthetic_fraction=args.synthetic, device=args.device,
        evaluate=not args.no_eval, eval_windows=args.windows,
    )
    print(f"\nfine-tuned {os.path.basename(base)} -> {report.output} "
          f"({report.steps} steps, {report.minutes:.1f} min, best val loss {report.best_val_loss:.4f})")
    if report.evaluation:
        if "error" in report.evaluation:
            print(f"held-out evaluation skipped: {report.evaluation['error']}")
        else:
            ev = report.evaluation
            _print_backtest(
                ev, f"held-out tail ({(1 - args.train_fraction):.0%} of the data, "
                    f"{ev['windows_per_series']} windows/series, context {ev['context']}, "
                    f"horizon {ev['horizon']}): MAE ratio vs last-value")
            mine = next(s for s in ev["scores"] if s["model"] == args.name)
            base_row = next(s for s in ev["scores"] if s["model"] == "base")
            print(f"\nfine-tuned vs base: MAE {mine['mean_loss']:.4g} vs {base_row['mean_loss']:.4g} "
                  f"({(mine['mean_loss'] / base_row['mean_loss'] - 1) * 100:+.1f}%); "
                  f"verdict vs last-value: {mine['verdict']}")
            print(f"serve it:  timesfm3 serve --checkpoint {args.name}={report.output}")
    return 0


def cmd_credits(args) -> int:
    from .client import ForecastClient
    from .credits import CreditWallet

    wallet = CreditWallet(args.wallet)
    if args.action == "status":
        print(f"{args.wallet}: {len(wallet)} unspent credit(s)"
              + (f", pool key {wallet.pool['kid']}" if wallet.pool else ""))
        return 0
    if args.private_key:
        # Pay for the batch with x402 from the given EVM key (needs `pip install "x402[evm]"`).
        try:
            from eth_account import Account
            from x402 import x402ClientSync
            from x402.http.clients import x402_requests
            from x402.mechanisms.evm import EthAccountSigner
            from x402.mechanisms.evm.exact import ExactEvmClientScheme
        except ImportError:
            print("x402 payment needs:  pip install 'timesfm3[x402]'", file=sys.stderr)
            return 2
        client = x402ClientSync().register("eip155:*", ExactEvmClientScheme(EthAccountSigner(Account.from_key(args.private_key))))
        session = x402_requests(client)
        pool = session.get(f"{args.api}/v1/credits/pool", timeout=60).json()
        pending = wallet.prepare(pool, args.count)
        r = session.post(f"{args.api}/v1/credits/buy/{args.count}",
                         json={"blinded": [format(p.blinded, "x") for p in pending]}, timeout=120)
        if r.status_code != 200:
            print(f"error: HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
            return 1
        added = wallet.finish(pending, r.json()["blind_signatures"])
        paid = r.headers.get("PAYMENT-RESPONSE", "")
        print(f"bought {added} credits with x402 ({pool['price_per_credit_usd'] * added:.4f} USD)"
              + ("; settlement receipt in PAYMENT-RESPONSE" if paid else ""))
    else:
        fc = ForecastClient(args.api, api_key=args.api_key, credits=wallet)
        added = fc.buy_credits(args.count, wallet)
        print(f"bought {added} credits" + (" on your plan" if args.api_key else ""))
    print(f"{args.wallet}: {len(wallet)} unspent credit(s); use ForecastClient(credits=CreditWallet('{args.wallet}'))")
    return 0


def cmd_pack(args) -> int:
    from .checkpoint import package_checkpoint

    meta = {"name": args.name} if args.name else {}
    if args.description:
        meta["description"] = args.description
    out = package_checkpoint(args.src, args.dst, meta=meta, half=not args.fp32)
    size = os.path.getsize(args.dst) / 1e6
    print(f"packaged {args.src} -> {args.dst} ({size:.1f} MB, {out['dtype']})")
    return 0


def cmd_train(args) -> int:
    from .train import main as train_main

    sys.argv = ["timesfm3 train"] + args.rest
    train_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="timesfm3", description="TimesFM-3 forecasting toolkit.")
    p.add_argument("--version", action="version", version=f"timesfm3 {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="Run the forecast service (REST API + dashboard).")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    s.add_argument("--api-key", default=None, help="Require this key (or set TIMESFM3_API_KEY).")
    s.add_argument("--log-level", default="info")
    _add_model_args(s)
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("models", help="List servable models.")
    _add_model_args(m)
    m.set_defaults(func=cmd_models)

    f = sub.add_parser("forecast", help="Forecast every column of a CSV.")
    f.add_argument("input")
    f.add_argument("--horizon", "-H", type=int, required=True)
    f.add_argument("--model", "-m", default=None)
    f.add_argument("--columns", default=None, help="Comma-separated subset of columns.")
    f.add_argument("--freq", default=None, help="Output timestamp step (e.g. 1h, D).")
    f.add_argument("--output", "-o", default=None, help="Write here (.csv or .json); default stdout.")
    f.add_argument("--format", choices=["csv", "json"], default="csv")
    _add_model_args(f)
    f.set_defaults(func=cmd_forecast)

    b = sub.add_parser("backtest", help="Compare models walk-forward on a CSV.")
    b.add_argument("input")
    b.add_argument("--context", type=int, required=True)
    b.add_argument("--horizon", "-H", type=int, required=True)
    b.add_argument("--models", default=None, help="Comma-separated; default all.")
    b.add_argument("--reference", default="last-value")
    b.add_argument("--windows", type=int, default=20)
    b.add_argument("--metric", choices=["mae", "mse"], default="mae")
    b.add_argument("--overlap", action="store_true")
    b.add_argument("--json", action="store_true")
    _add_model_args(b)
    b.set_defaults(func=cmd_backtest)

    a = sub.add_parser("anomalies", help="Flag anomalous points in every column of a CSV.")
    a.add_argument("input")
    a.add_argument("--model", "-m", default=None)
    a.add_argument("--context", type=int, default=96)
    a.add_argument("--block", type=int, default=24)
    a.add_argument("--threshold", type=float, default=2.0)
    a.add_argument("--json", action="store_true")
    _add_model_args(a)
    a.set_defaults(func=cmd_anomalies)

    ft = sub.add_parser("finetune", help="Fine-tune a checkpoint on a CSV and evaluate on its tail.")
    ft.add_argument("input")
    ft.add_argument("--out", "-o", required=True, help="Packaged checkpoint to write.")
    ft.add_argument("--from", dest="base", default=None, help="Base checkpoint (default: bundled).")
    ft.add_argument("--name", default="finetuned")
    ft.add_argument("--steps", type=int, default=300)
    ft.add_argument("--batch-size", type=int, default=16)
    ft.add_argument("--lr", type=float, default=1e-4)
    ft.add_argument("--context-patches", type=int, default=8)
    ft.add_argument("--horizon-patches", type=int, default=2)
    ft.add_argument("--train-fraction", type=float, default=0.8)
    ft.add_argument("--periods", default=None, help="Calendar periods in steps, e.g. 24,168.")
    ft.add_argument("--synthetic", type=float, default=0.2, help="Fraction of synthetic windows mixed in.")
    ft.add_argument("--windows", type=int, default=20, help="Held-out evaluation windows per series.")
    ft.add_argument("--no-eval", action="store_true")
    ft.add_argument("--device", default=None)
    ft.set_defaults(func=cmd_finetune)

    cr = sub.add_parser("credits", help="Buy / inspect unlinkable prepaid credits.")
    cr.add_argument("action", choices=["buy", "status"])
    cr.add_argument("--api", default="http://localhost:8000")
    cr.add_argument("--count", type=int, default=25, help="10, 25 or 100")
    cr.add_argument("--api-key", default=None, help="Pay with a plan (points).")
    cr.add_argument("--private-key", default=None, help="Pay with x402 from this EVM private key.")
    cr.add_argument("--wallet", default="credits.json", help="Wallet file for the tokens.")
    cr.set_defaults(func=cmd_credits)

    k = sub.add_parser("pack", help="Package a training checkpoint (fp16 + metadata).")
    k.add_argument("src")
    k.add_argument("dst")
    k.add_argument("--name", default=None)
    k.add_argument("--description", default=None)
    k.add_argument("--fp32", action="store_true")
    k.set_defaults(func=cmd_pack)

    t = sub.add_parser("train", help="Pre-train a model (arguments pass through to timesfm3.train).")
    t.add_argument("rest", nargs=argparse.REMAINDER)
    t.set_defaults(func=cmd_train)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

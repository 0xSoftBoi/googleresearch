"""``timesfm3`` command-line interface.

    timesfm3 serve [--port 8000] [--checkpoint PATH ...] [--api-key KEY]
    timesfm3 forecast data.csv --horizon 24 [--model NAME] [--output out.csv]
    timesfm3 backtest data.csv --context 256 --horizon 24 [--models a,b]
    timesfm3 models [--checkpoint PATH ...]
    timesfm3 pack train_ckpt.pt packaged.pt --name my-model
    timesfm3 train --config small --steps 6000 ...     (see timesfm3.train)
"""

from __future__ import annotations

import argparse
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

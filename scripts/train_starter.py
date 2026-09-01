"""Train the bundled *starter* checkpoint that ``timesfm3 serve`` ships with.

The recipe is the one from ``notebooks/timesfm3_real_data.ipynb``: the
``small`` config (~5M parameters) pre-trained on a real corpus (ETTh1,
ETTm1, ETTm2, daily exchange rates -- ``bash data/download.sh`` fetches
them) mixed 70/30 with the synthetic generator, with calendar covariates
and role randomization.  ETTh2 is never seen, so ``timesfm3 backtest`` on
it is a genuine zero-shot number.

The result is packaged as a half-precision checkpoint with provenance
metadata so it is small enough to ship inside the repository / image.

Run:  python scripts/train_starter.py [--steps 6000] [--out models/starter-small.pt]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from timesfm3 import TimesFM3Config
from timesfm3.checkpoint import package_checkpoint
from timesfm3.data import (
    MixedCorpus,
    RealWindowDataset,
    SyntheticMultivariateCorpus,
    load_csv_dataset,
)
from timesfm3.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--config", choices=["tiny", "small"], default="small")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="models/starter-small.pt")
    parser.add_argument("--scratch", default="starter_train.pt",
                        help="where the full-precision best checkpoint is written")
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    context_patches, horizon_patches = 8, 2

    sources = [
        load_csv_dataset(f"{args.data}/ETTh1.csv", "ETTh1", periods=(24, 168)),
        load_csv_dataset(f"{args.data}/ETTm1.csv", "ETTm1", periods=(96, 672)),
        load_csv_dataset(f"{args.data}/ETTm2.csv", "ETTm2", periods=(96, 672)),
        load_csv_dataset(
            f"{args.data}/exchange_rate.txt", "exchange", periods=(7,),
            skip_first_col=False,
        ),
    ]
    total = sum(s.values.size for s in sources)
    print(f"training corpus: {total / 1e6:.2f}M real points across {len(sources)} datasets")

    cfg = getattr(TimesFM3Config, args.config)()
    real = RealWindowDataset(
        cfg, sources, context_patches=context_patches,
        horizon_patches=horizon_patches, seed=11,
    )
    synth = SyntheticMultivariateCorpus(
        cfg, context_patches=context_patches, horizon_patches=horizon_patches, seed=12
    )
    val_real = RealWindowDataset(
        cfg, sources, context_patches=context_patches,
        horizon_patches=horizon_patches, seed=987_654,
    )
    history: list = []
    t0 = time.time()
    train(
        cfg, steps=args.steps, batch_size=args.batch_size,
        context_patches=context_patches, horizon_patches=horizon_patches,
        dataset=MixedCorpus(real, synth, primary_prob=0.7, seed=13),
        val_dataset=val_real, checkpoint_path=args.scratch, history=history,
        log_every=200, val_every=500,
    )
    minutes = (time.time() - t0) / 60
    best_val = min(h["val_loss"] for h in history if "val_loss" in h)
    print(f"wall time: {minutes:.1f} min, best val loss {best_val:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    package_checkpoint(
        args.scratch, args.out,
        meta={
            "name": f"starter-{args.config}",
            "description": (
                "TimesFM-3 %s config pre-trained on ETTh1/ETTm1/ETTm2/exchange-rate "
                "(70%%) + synthetic corpus (30%%) with calendar covariates." % args.config
            ),
            "training_steps": args.steps,
            "batch_size": args.batch_size,
            "context_patches": context_patches,
            "horizon_patches": horizon_patches,
            "best_val_loss": float(best_val),
            "train_minutes": float(minutes),
            "corpus": [s.name for s in sources],
            "holdout": ["ETTh2"],
        },
    )
    print(f"packaged starter checkpoint -> {args.out}")


if __name__ == "__main__":
    main()

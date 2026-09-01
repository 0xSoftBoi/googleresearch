"""Train a TimesFM-3 model on Polymarket microstructure.

Writes the market split alongside the checkpoint so evaluation can be held out
honestly: `examples/evaluate_polymarket.py --split <path>` then scores only
markets this run never saw, over the tail of the time axis it never trained on.

Usage:
    python examples/train_polymarket.py --panels panels.pkl --steps 4000 \
        --checkpoint timesfm3_polymarket.pt
"""

import argparse
import json
import pickle
import time

import numpy as np

from timesfm3 import TimesFM3Config
from timesfm3.data.polymarket import COUNT_CHANNELS
from timesfm3.data.real import RealSource, RealWindowDataset
from timesfm3.train import train


def channel_values(panel, channels):
    """One outcome per market, log1p-compressed counts, first cell dropped.

    Only one outcome is used because this archive's two outcome tokens are
    exactly complementary (p_yes + p_no == 1, zero variance), so the second is
    an arithmetic mirror. The first cell is dropped because `ret`/`abs_ret` are
    undefined there.
    """
    out = []
    for c in channels:
        v = panel.features[c][0:1, 1:].astype(np.float32).copy()
        if c in COUNT_CHANNELS:
            v = np.sign(v) * np.log1p(np.abs(v))
        out.append(v)
    return np.concatenate(out, axis=0)


def split_markets(panels, channels, holdout: float, seed: int):
    """Deterministic train/held-out split over markets usable for `channels`."""
    usable = [
        i for i, p in enumerate(panels)
        if np.isfinite(channel_values(p, channels)).all()
        and channel_values(p, channels).std() > 0
    ]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(usable))
    n_test = max(1, int(round(len(usable) * holdout)))
    heldout = sorted(usable[i] for i in perm[:n_test])
    train_ids = sorted(usable[i] for i in perm[n_test:])
    return train_ids, heldout, usable


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panels", required=True)
    p.add_argument("--channels", default="mid,spread,abs_ret,quotes")
    p.add_argument("--config", default="tiny", choices=["tiny", "small", "base"])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--context-patches", type=int, default=8)
    p.add_argument("--horizon-patches", type=int, default=2)
    p.add_argument("--train-fraction", type=float, default=0.8,
                   help="fraction of each market's time axis available to train")
    p.add_argument("--holdout", type=float, default=0.25,
                   help="fraction of markets excluded from training entirely")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0,
                   help="seeds model init and the training window stream")
    p.add_argument("--checkpoint", default="timesfm3_polymarket.pt")
    args = p.parse_args()
    channels = tuple(args.channels.split(","))

    with open(args.panels, "rb") as f:
        panels = pickle.load(f)
    train_ids, heldout, usable = split_markets(
        panels, channels, args.holdout, args.split_seed
    )
    print(f"{len(panels)} panels; {len(usable)} usable for {channels}; "
          f"{len(train_ids)} train / {len(heldout)} held-out markets")

    split_path = args.checkpoint + ".split.json"
    with open(split_path, "w") as f:
        json.dump(
            {
                "channels": list(channels),
                "train_markets": train_ids,
                "heldout_markets": heldout,
                "train_fraction": args.train_fraction,
                "holdout": args.holdout,
                "split_seed": args.split_seed,
                "panels": args.panels,
            },
            f,
            indent=2,
        )
    print(f"wrote split to {split_path}")

    import torch
    torch.manual_seed(args.seed)
    cfg = getattr(TimesFM3Config, args.config)()
    sources = [
        RealSource(f"pm{i}", channel_values(panels[i], channels), ())
        for i in train_ids
    ]
    common = dict(
        context_patches=args.context_patches,
        horizon_patches=args.horizon_patches,
        max_variates=len(channels),
        train_fraction=args.train_fraction,
        calendar=False,
        demote_prob=0.0,
    )
    ds = RealWindowDataset(cfg, sources, seed=11 + args.seed, **common)
    val = RealWindowDataset(cfg, sources, seed=987_654 + args.seed, **common)

    t0 = time.time()
    train(
        cfg,
        steps=args.steps,
        batch_size=args.batch_size,
        context_patches=args.context_patches,
        horizon_patches=args.horizon_patches,
        dataset=ds,
        val_dataset=val,
        checkpoint_path=args.checkpoint,
        log_every=500,
        val_every=1000,
    )
    print(f"wall time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

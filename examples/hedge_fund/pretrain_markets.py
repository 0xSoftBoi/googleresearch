"""Pre-train a small TimesFM-3 on real market data, for the signal harness.

Rahimikia et al. ("Re(Visiting) Time Series Foundation Models in Finance",
2025, Man Group-affiliated) report that generic-pretrained time-series
foundation models transfer poorly to financial returns, while the *same
architectures pre-trained on financial data* (plus synthetic augmentation)
produce real forecasting and portfolio gains.  This script is that recipe
on this repo's stack:

1. load the FRED multi-asset panel (``timesfm3.data.markets``),
2. expose it as a ``RealSource`` of log-price levels,
3. mix real windows (70%) with the synthetic corpus (30%) exactly as the
   existing training pipeline does for ETT,
4. pre-train the ``tiny`` config on CPU and checkpoint the best-validation
   weights.

``RealWindowDataset`` samples training windows only from the first 80% of
each series (``train_fraction``), so roughly 2013-onward data is never
seen in training: evaluate the checkpoint on recent years and the result
is out-of-sample in time.

Run:   python examples/hedge_fund/pretrain_markets.py [steps]
Then:  python examples/hedge_fund/model_signal.py data/markets_tiny.pt
"""

from __future__ import annotations

import sys

from timesfm3.configuration import TimesFM3Config
from timesfm3.data.markets import load_universe, to_real_source
from timesfm3.data.real import MixedCorpus, RealWindowDataset
from timesfm3.data.synthetic import SyntheticMultivariateCorpus
from timesfm3.train import train

CHECKPOINT = "data/markets_tiny.pt"


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    config = TimesFM3Config.tiny()
    panel = load_universe(cache_dir="data/fred", verbose=False)
    source = to_real_source(panel)
    print(f"real source: {source.values.shape[0]} channels x {source.values.shape[1]} steps")

    context_patches, horizon_patches = 12, 2  # 384-day context, 64-day horizon
    real = RealWindowDataset(
        config, [source],
        context_patches=context_patches, horizon_patches=horizon_patches,
        seed=7,
    )
    synthetic = SyntheticMultivariateCorpus(
        config, context_patches=context_patches, horizon_patches=horizon_patches,
        seed=11,
    )
    val = RealWindowDataset(
        config, [source],
        context_patches=context_patches, horizon_patches=horizon_patches,
        seed=987_654_321,
    )
    train(
        config,
        steps=steps,
        batch_size=16,
        context_patches=context_patches,
        horizon_patches=horizon_patches,
        dataset=MixedCorpus(real, synthetic, primary_prob=0.7, seed=13),
        val_dataset=val,
        checkpoint_path=CHECKPOINT,
    )
    print(f"checkpoint written to {CHECKPOINT}")


if __name__ == "__main__":
    main()

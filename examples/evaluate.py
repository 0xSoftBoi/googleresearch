"""Evaluates a trained TimesFM-3 checkpoint against naive baselines.

Draws held-out examples from the synthetic corpus (a seed never used in
training), forecasts the trailing horizon, and reports scaled MAE — mean
absolute error divided by each series' context standard deviation — for:

- the model's point forecast,
- a last-value baseline (repeat the final context value),
- a context-mean baseline.

Usage:
    python examples/evaluate.py --checkpoint timesfm3_checkpoint.pt
"""

import argparse

import numpy as np
import torch

from timesfm3 import TimesFM3Forecaster
from timesfm3.data.synthetic import SyntheticMultivariateCorpus
from timesfm3.embedding import ROLE_FUTURE_COVARIATE, ROLE_TARGET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--horizon-patches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=987_654)
    args = parser.parse_args()

    forecaster = TimesFM3Forecaster.from_checkpoint(args.checkpoint)
    cfg = forecaster.config
    horizon = args.horizon_patches * cfg.patch_len
    corpus = SyntheticMultivariateCorpus(
        cfg,
        context_patches=8,
        horizon_patches=args.horizon_patches,
        seed=args.seed,
    )

    model_err, last_err, mean_err = [], [], []
    stream = iter(corpus)
    with torch.no_grad():
        for _ in range(args.examples):
            ex = next(stream)
            values = ex["values"].numpy()
            observed = ex["observed"].numpy()
            roles = ex["roles"].numpy()
            # The corpus left-truncates with unobserved zeros; drop that
            # prefix so the forecaster only sees real observations.
            first = int(np.argmax(observed[0]))
            values = values[:, first:]
            observed = observed[:, first:]
            context = values.shape[1] - horizon

            target_rows = np.where(roles == ROLE_TARGET)[0]
            future_rows = np.where(roles == ROLE_FUTURE_COVARIATE)[0]
            past_rows = [
                r for r in range(len(roles))
                if r not in target_rows and r not in future_rows
            ]
            result = forecaster.forecast(
                targets=[values[r, :context] for r in target_rows],
                past_covariates=[values[r, :context] for r in past_rows],
                future_covariates=[values[r] for r in future_rows],
                horizon=horizon,
            )

            for k, r in enumerate(target_rows):
                ctx = values[r, :context][observed[r, :context]]
                scale = max(ctx.std(), 1e-6)
                truth = values[r, context:]
                model_err.append(np.abs(result.point[k] - truth).mean() / scale)
                last_err.append(np.abs(ctx[-1] - truth).mean() / scale)
                mean_err.append(np.abs(ctx.mean() - truth).mean() / scale)

    print(f"held-out examples: {args.examples}, target series: {len(model_err)}")
    print(f"scaled MAE  model:      {np.mean(model_err):.4f}")
    print(f"scaled MAE  last-value: {np.mean(last_err):.4f}")
    print(f"scaled MAE  ctx-mean:   {np.mean(mean_err):.4f}")


if __name__ == "__main__":
    main()

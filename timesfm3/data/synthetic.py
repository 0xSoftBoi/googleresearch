"""Synthetic multivariate pre-training corpus.

TimesFM-3 was pre-trained on a mixture of real-world and synthetic series
totalling over a trillion time points. This module implements the synthetic
side: randomly composed trends, seasonalities and ARMA-style noise, with a
random linear mixing step so that variates within one example are genuinely
correlated — the signal the cross-variate attention layers must learn.

Each generated example is a dict of tensors ready for the model:
    values:   (N, T) float32, context + horizon
    observed: (N, T) bool
    roles:    (N,)   int64
"""

from __future__ import annotations

import numpy as np
import torch

from ..configuration import TimesFM3Config
from ..embedding import ROLE_FUTURE_COVARIATE, ROLE_PAST_COVARIATE, ROLE_TARGET


class SyntheticMultivariateCorpus(torch.utils.data.IterableDataset):
    """Infinite stream of synthetic multivariate forecasting examples."""

    def __init__(
        self,
        config: TimesFM3Config,
        max_variates: int = 8,
        context_patches: int = 16,
        horizon_patches: int = 4,
        seed: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.max_variates = max_variates
        self.context_patches = context_patches
        self.horizon_patches = horizon_patches
        self.seed = seed

    @property
    def total_len(self) -> int:
        return (self.context_patches + self.horizon_patches) * self.config.patch_len

    def _base_signals(self, rng: np.random.Generator, k: int, t: int) -> np.ndarray:
        """Draws k independent latent signals of length t."""
        time = np.arange(t, dtype=np.float64)
        signals = np.zeros((k, t))
        for i in range(k):
            # Trend: random low-order polynomial.
            trend = np.zeros(t)
            for power in range(rng.integers(0, 3)):
                trend += rng.normal(0, 0.5) * (time / t) ** (power + 1)
            # Seasonality: a few random sinusoids. Kept dominant relative to
            # noise so the corpus rewards learning predictable structure.
            season = np.zeros(t)
            for _ in range(rng.integers(1, 4)):
                period = rng.uniform(8, t / 2)
                phase = rng.uniform(0, 2 * np.pi)
                season += rng.exponential(1.0) * np.sin(
                    2 * np.pi * time / period + phase
                )
            # AR(1)-style noise; a small fraction of series is noise-heavy.
            noise = np.zeros(t)
            phi = rng.uniform(-0.9, 0.95)
            noise_scale = rng.exponential(0.5 if rng.uniform() < 0.15 else 0.1)
            eps = rng.normal(0, noise_scale, size=t)
            for step in range(1, t):
                noise[step] = phi * noise[step - 1] + eps[step]
            signals[i] = trend + season + noise
        return signals

    def _sample(self, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        t = self.total_len
        n = int(rng.integers(1, self.max_variates + 1))
        # Mix latent signals so the variates are cross-correlated.
        k = max(1, n // 2 + int(rng.integers(0, 2)))
        latents = self._base_signals(rng, k, t)
        mixing = rng.normal(0, 1, size=(n, k))
        values = mixing @ latents + rng.normal(0, 0.05, size=(n, t))
        # Random affine scale per series (models see vastly different scales).
        scale = rng.lognormal(0, 2, size=(n, 1))
        offset = rng.normal(0, 10, size=(n, 1))
        values = values * scale + offset

        roles = np.full(n, ROLE_TARGET, dtype=np.int64)
        if n > 1:
            # Randomly demote some series to covariates; keep >= 1 target.
            for i in range(1, n):
                draw = rng.uniform()
                if draw < 0.25:
                    roles[i] = ROLE_PAST_COVARIATE
                elif draw < 0.5:
                    roles[i] = ROLE_FUTURE_COVARIATE

        observed = np.ones((n, t), dtype=bool)
        # Random left truncation to vary effective context length.
        cut = int(rng.integers(0, self.context_patches)) * self.config.patch_len
        observed[:, :cut] = False
        values[:, :cut] = 0.0

        return {
            "values": torch.from_numpy(values.astype(np.float32)),
            "observed": torch.from_numpy(observed),
            "roles": torch.from_numpy(roles),
        }

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng(
            None if self.seed is None else self.seed + worker_id
        )
        while True:
            yield self._sample(rng)


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pads examples with different variate counts into one batch."""
    max_n = max(item["values"].shape[0] for item in batch)
    t = batch[0]["values"].shape[1]
    b = len(batch)
    values = torch.zeros(b, max_n, t)
    observed = torch.zeros(b, max_n, t, dtype=torch.bool)
    roles = torch.zeros(b, max_n, dtype=torch.int64)
    variate_mask = torch.zeros(b, max_n, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["values"].shape[0]
        values[i, :n] = item["values"]
        observed[i, :n] = item["observed"]
        roles[i, :n] = item["roles"]
        variate_mask[i, :n] = True
    return {
        "values": values,
        "observed": observed,
        "roles": roles,
        "variate_mask": variate_mask,
    }

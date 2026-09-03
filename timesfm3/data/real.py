"""Real-data training corpus: windows sampled from public benchmark datasets.

Builds TimesFM-3 training examples from real multivariate series (e.g. the
ETT electricity-transformer datasets, exchange rates) with three tricks
that make a small pre-training budget productive:

- **Multi-frequency augmentation**: windows are subsampled with a random
  stride, so one hourly dataset also teaches 2-hourly and 4-hourly
  dynamics and the model generalizes across sampling rates.
- **Calendar covariates**: sin/cos features of the dataset's natural
  periods (day, week) are appended as past-future covariates — they are
  known arbitrarily far into the future, exactly the covariate type
  TimesFM-3's lookahead tokens are built for, and they let the model
  anchor phase instead of inferring it purely from context.
- **Role randomization**: real channels are randomly demoted to past-only
  covariates so the model trains on the same target/covariate mixtures it
  will see at inference time.

Examples match the synthetic corpus schema, so the two mix freely.
"""

from __future__ import annotations

import csv
import dataclasses

import numpy as np
import torch

from ..configuration import TimesFM3Config
from ..embedding import ROLE_FUTURE_COVARIATE, ROLE_PAST_COVARIATE, ROLE_TARGET


@dataclasses.dataclass
class RealSource:
    """One dataset: (channels, time) values plus its natural periods."""

    name: str
    values: np.ndarray  # (C, T) float32
    periods: tuple[int, ...]  # e.g. (24, 168) for hourly data

    @property
    def num_steps(self) -> int:
        return self.values.shape[1]


def load_csv_dataset(
    path: str, name: str, periods: tuple[int, ...], skip_first_col: bool = True
) -> RealSource:
    """Loads a CSV/TSV with time as rows (optionally a leading date column)."""
    rows = []
    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "," if sample.count(",") >= sample.count(" ") else " "
        reader = csv.reader(f, delimiter=delimiter, skipinitialspace=True)
        first = next(reader)
        has_header = not _is_float(first[-1])
        if not has_header:
            rows.append(first)
        for row in reader:
            rows.append(row)
    start = 1 if skip_first_col else 0
    data = np.asarray(
        [[float(x) for x in row[start:]] for row in rows], dtype=np.float32
    ).T
    return RealSource(name=name, values=data, periods=periods)


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def calendar_covariates(
    start: int, length: int, stride: int, periods: tuple[int, ...]
) -> np.ndarray:
    """Returns (2 * len(periods), length) sin/cos phase features.

    Phases are computed from absolute positions (start + stride * i), so a
    subsampled window still carries the true calendar phase.
    """
    idx = start + stride * np.arange(length, dtype=np.float64)
    feats = []
    for period in periods:
        angle = 2 * np.pi * idx / period
        feats.append(np.sin(angle))
        feats.append(np.cos(angle))
    return np.asarray(feats, dtype=np.float32)


class RealWindowDataset(torch.utils.data.IterableDataset):
    """Infinite stream of real-data windows in the synthetic-corpus schema."""

    def __init__(
        self,
        config: TimesFM3Config,
        sources: list[RealSource],
        context_patches: int = 8,
        horizon_patches: int = 2,
        max_variates: int = 6,
        strides: tuple[int, ...] = (1, 2, 4),
        train_fraction: float = 0.8,
        calendar: bool = True,
        demote_prob: float = 0.2,
        seed: int | None = None,
        tail: bool = False,
    ):
        """``tail=True`` samples windows from the last ``1 - train_fraction``
        of each series instead of the first ``train_fraction`` -- a held-out
        validation stream that never overlaps the training windows."""
        super().__init__()
        if not sources:
            raise ValueError("At least one real source is required.")
        self.tail = tail
        self.config = config
        self.sources = sources
        self.window = (context_patches + horizon_patches) * config.patch_len
        self.max_variates = max_variates
        self.strides = strides
        self.train_fraction = train_fraction
        self.calendar = calendar
        self.demote_prob = demote_prob
        self.seed = seed
        # Sample sources proportionally to their usable length.
        weights = np.asarray([s.num_steps * len(s.values) for s in sources], float)
        self.source_probs = weights / weights.sum()

    def _sample(self, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        source = self.sources[rng.choice(len(self.sources), p=self.source_probs)]
        train_end = int(source.num_steps * self.train_fraction)
        lo, hi = (train_end, source.num_steps) if self.tail else (0, train_end)

        stride = int(rng.choice(self.strides))
        span = self.window * stride
        if span > hi - lo:
            stride = 1
            span = self.window
        if span > hi - lo:
            raise ValueError(
                f"Source {source.name!r}: region of {hi - lo} steps is shorter than one "
                f"window of {span} steps."
            )
        start = int(rng.integers(lo, hi - span + 1))
        window = source.values[:, start : start + span : stride]

        channels = list(rng.permutation(len(window))[: self.max_variates])
        values = window[channels].astype(np.float32)
        roles = np.full(len(channels), ROLE_TARGET, dtype=np.int64)
        # Keep >= 1 target; demote later channels to past-only covariates.
        for i in range(1, len(channels)):
            if rng.uniform() < self.demote_prob:
                roles[i] = ROLE_PAST_COVARIATE

        if self.calendar and source.periods:
            cal = calendar_covariates(
                start, self.window, stride, source.periods
            )
            values = np.concatenate([values, cal], axis=0)
            roles = np.concatenate(
                [roles, np.full(len(cal), ROLE_FUTURE_COVARIATE, dtype=np.int64)]
            )

        observed = np.isfinite(values)
        return {
            "values": torch.from_numpy(np.nan_to_num(values)),
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


class MixedCorpus(torch.utils.data.IterableDataset):
    """Mixes two example streams (e.g. real and synthetic) by probability."""

    def __init__(
        self,
        primary: torch.utils.data.IterableDataset,
        secondary: torch.utils.data.IterableDataset,
        primary_prob: float = 0.7,
        seed: int | None = None,
    ):
        super().__init__()
        self.primary = primary
        self.secondary = secondary
        self.primary_prob = primary_prob
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng(
            None if self.seed is None else self.seed + worker_id
        )
        primary, secondary = iter(self.primary), iter(self.secondary)
        while True:
            yield next(primary if rng.uniform() < self.primary_prob else secondary)

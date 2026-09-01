"""Prediction-market training/eval corpus from the Pendulum Flow archive.

Turns the public Polymarket order-book archive at
``https://archive.pendulumflow.com/`` into TimesFM-3 examples. Each Polymarket
*outcome token* (``asset_id``) is a slowly-moving probability series in
``[0, 1]``; the archive records every top-of-book and level update with
microsecond arrival timestamps. This module:

1. downloads the hourly parquet files (with optional SHA-256 verification
   against the archive's ``SHA256SUMS.txt``), caching them on disk;
2. streams them row-group by row-group (a single hour is ~1 GB / ~86 M rows,
   so nothing is ever fully materialised), keeping only the columns needed to
   reconstruct the mid price;
3. resamples the most active assets onto a regular time grid by
   forward-filling the mid price ``(best_bid + best_ask) / 2``; and
4. wraps the result as a :class:`~timesfm3.data.real.RealSource`, so it drops
   straight into :class:`~timesfm3.data.real.RealWindowDataset` and
   :class:`~timesfm3.data.real.MixedCorpus` alongside the ETT / synthetic
   corpora.

The archive schema (V3) carries one row per event, discriminated by
``event_type`` (``best_bid_ask``, ``price_change``, ``last_trade_price``,
``book``, ``new_market``, ``market_resolved``, ``tick_size_change``). We use
the two high-frequency quote events -- ``best_bid_ask`` and ``price_change``,
both of which carry ``best_bid`` / ``best_ask`` -- to build the mid-price grid.

Requires ``pyarrow`` (an optional dependency): ``pip install pyarrow``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import os
import urllib.request
from collections import Counter

import numpy as np

from .real import RealSource

try:  # pyarrow is an optional dependency (only this data path needs it).
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised only without pyarrow
    raise ImportError(
        "timesfm3.data.polymarket requires pyarrow. Install it with "
        "`pip install pyarrow` (or `pip install -e .[polymarket]`)."
    ) from exc


DEFAULT_BASE_URL = "https://archive.pendulumflow.com"
DEFAULT_VERSION = "v3"

# Event types that carry a usable top-of-book (best_bid / best_ask).
QUOTE_EVENTS = ("best_bid_ask", "price_change")


@dataclasses.dataclass
class PolymarketArchive:
    """Client for the Pendulum Flow Polymarket archive.

    Parameters
    ----------
    cache_dir:
        Local directory where downloaded parquet files are stored (mirrors the
        archive's ``YYYY-MM-DD/HH/`` layout).
    base_url, version:
        Archive location. ``version`` selects the era: ``"v3"`` (default,
        "military grade"), ``"pmxt/v2"``, ``"pmxt/v1"`` or ``"third-party/ag6"``.
    verify:
        When ``True`` (default) each download is checked against the archive's
        ``SHA256SUMS.txt``. Hours newer than the last-published checksum are
        downloaded with a warning rather than rejected.
    """

    cache_dir: str = "data/polymarket"
    base_url: str = DEFAULT_BASE_URL
    version: str = DEFAULT_VERSION
    verify: bool = True

    def __post_init__(self) -> None:
        self._version_url = f"{self.base_url.rstrip('/')}/{self.version.strip('/')}"
        self._checksums: dict[str, str] | None = None

    # -- URLs and paths ---------------------------------------------------

    @staticmethod
    def _relpath(hour: dt.datetime) -> str:
        """Archive-relative path of an hour's parquet, e.g.
        ``2026-09-01/04/2026-09-01T04.parquet``."""
        return (
            f"{hour:%Y-%m-%d}/{hour:%H}/{hour:%Y-%m-%d}T{hour:%H}.parquet"
        )

    def hour_url(self, hour: dt.datetime) -> str:
        return f"{self._version_url}/{self._relpath(hour)}"

    def local_path(self, hour: dt.datetime) -> str:
        return os.path.join(self.cache_dir, self.version, self._relpath(hour))

    # -- checksums --------------------------------------------------------

    def _load_checksums(self) -> dict[str, str]:
        if self._checksums is None:
            url = f"{self._version_url}/SHA256SUMS.txt"
            checksums: dict[str, str] = {}
            with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted host)
                for line in resp.read().decode("utf-8", "replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    digest, _, path = line.partition("  ")
                    if path:
                        checksums[path.strip()] = digest.strip()
            self._checksums = checksums
        return self._checksums

    def expected_sha256(self, hour: dt.datetime) -> str | None:
        """Published SHA-256 for an hour, or ``None`` if not yet listed."""
        try:
            return self._load_checksums().get(self._relpath(hour))
        except OSError:
            return None

    # -- download ---------------------------------------------------------

    def download_hour(
        self, hour: dt.datetime, *, force: bool = False
    ) -> str:
        """Download one hour's parquet into the cache and return its path.

        Cached files are reused unless ``force`` is set. When
        :attr:`verify` is on, the file is checked against the published
        checksum (hours newer than the checksum manifest pass with a warning).
        """
        path = self.local_path(hour)
        if os.path.exists(path) and not force:
            if self.verify:
                self._verify(path, hour)
            return path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        url = self.hour_url(hour)
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:  # noqa: S310
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        os.replace(tmp, path)
        if self.verify:
            self._verify(path, hour)
        return path

    def _verify(self, path: str, hour: dt.datetime) -> None:
        expected = self.expected_sha256(hour)
        if expected is None:
            print(
                f"[polymarket] no published checksum for {self._relpath(hour)} "
                "(hour newer than SHA256SUMS.txt?); skipping verification"
            )
            return
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {path}: expected {expected}, got {actual}"
            )

    def download_range(
        self, start: dt.datetime, end: dt.datetime, *, force: bool = False
    ) -> list[str]:
        """Download every hour in ``[start, end]`` (inclusive) and return paths."""
        return [
            self.download_hour(hour, force=force)
            for hour in hours_between(start, end)
        ]


def hours_between(start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
    """Inclusive list of hour-aligned datetimes from ``start`` to ``end``."""
    start = start.replace(minute=0, second=0, microsecond=0)
    end = end.replace(minute=0, second=0, microsecond=0)
    out, cur = [], start
    while cur <= end:
        out.append(cur)
        cur += dt.timedelta(hours=1)
    return out


# -- parquet -> regular grid ---------------------------------------------


def count_asset_activity(
    paths: list[str], event_types: tuple[str, ...] = QUOTE_EVENTS
) -> Counter:
    """Count quote events per ``asset_id`` across files (streaming, vectorised)."""
    counts: Counter = Counter()
    et_set = list(event_types)
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns=["event_type", "asset_id"])
            keep = pc.is_in(
                table["event_type"], value_set=_arrow_str_set(et_set)
            )
            asset = table["asset_id"].filter(keep)
            vc = pc.value_counts(asset)
            for entry in vc.to_pylist():
                key = entry["values"]
                if key is not None:
                    counts[key] += entry["counts"]
    return counts


def top_assets(
    paths: list[str], n: int, event_types: tuple[str, ...] = QUOTE_EVENTS
) -> list[bytes]:
    """Return the ``n`` most active ``asset_id`` values (as raw bytes)."""
    return [asset for asset, _ in count_asset_activity(paths, event_types).most_common(n)]


def _arrow_str_set(values):
    import pyarrow as pa

    return pa.array(values, type=pa.string())


def _arrow_bin_set(values):
    import pyarrow as pa

    return pa.array(values, type=pa.binary())


def build_mid_grid(
    paths: list[str],
    assets: list[bytes],
    *,
    freq_seconds: float,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> tuple[np.ndarray, list[bytes], np.ndarray]:
    """Forward-fill each asset's mid price onto a regular time grid.

    Returns ``(values, assets, grid_us)`` where ``values`` is ``(len(assets),
    T)`` float32 with NaN before an asset's first quote, ``grid_us`` is the
    grid's microsecond timestamps, and ``T`` spans ``[start, end)`` (bounds
    default to the data's own min/max arrival time).

    Mid price is ``(best_bid + best_ask) / 2`` from ``best_bid_ask`` and
    ``price_change`` events; within a grid cell the last-arriving quote wins.
    """
    if not assets:
        raise ValueError("`assets` must be non-empty.")
    asset_index = {a: i for i, a in enumerate(assets)}
    value_set = _arrow_bin_set(assets)
    et_set = _arrow_str_set(list(QUOTE_EVENTS))

    ts_all: list[np.ndarray] = []
    code_all: list[np.ndarray] = []
    mid_all: list[np.ndarray] = []

    cols = ["event_type", "timestamp_received", "asset_id", "best_bid", "best_ask"]
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns=cols)
            keep = pc.and_(
                pc.is_in(table["event_type"], value_set=et_set),
                pc.is_in(table["asset_id"], value_set=value_set),
            )
            keep = pc.and_(
                keep,
                pc.and_(
                    pc.is_valid(table["best_bid"]), pc.is_valid(table["best_ask"])
                ),
            )
            table = table.filter(keep)
            if table.num_rows == 0:
                continue
            ts = pc.cast(table["timestamp_received"], "int64").to_numpy(
                zero_copy_only=False
            )
            bb = pc.cast(table["best_bid"], "float64").to_numpy(zero_copy_only=False)
            ba = pc.cast(table["best_ask"], "float64").to_numpy(zero_copy_only=False)
            mid = ((bb + ba) * 0.5).astype(np.float32)
            asset_py = table["asset_id"].to_pylist()
            codes = np.fromiter(
                (asset_index[a] for a in asset_py), dtype=np.int64, count=len(asset_py)
            )
            ts_all.append(ts)
            code_all.append(codes)
            mid_all.append(mid)

    if not ts_all:
        raise ValueError("No quote events found for the requested assets.")
    ts = np.concatenate(ts_all)
    codes = np.concatenate(code_all)
    mid = np.concatenate(mid_all)

    start_us = int(start.timestamp() * 1e6) if start else int(ts.min())
    end_us = int(end.timestamp() * 1e6) if end else int(ts.max()) + 1
    step_us = int(round(freq_seconds * 1e6))
    if step_us <= 0:
        raise ValueError("freq_seconds must be positive.")
    n_steps = max(1, (end_us - start_us + step_us - 1) // step_us)
    grid_us = start_us + step_us * np.arange(n_steps, dtype=np.int64)

    in_range = (ts >= start_us) & (ts < start_us + step_us * n_steps)
    ts, codes, mid = ts[in_range], codes[in_range], mid[in_range]
    cell = ((ts - start_us) // step_us).astype(np.int64)

    values = np.full((len(assets), n_steps), np.nan, dtype=np.float32)
    # Stable sort by (asset, cell, arrival) so the last arrival in a cell wins.
    order = np.lexsort((ts, cell, codes))
    values[codes[order], cell[order]] = mid[order]

    _forward_fill_rows(values)
    return values, assets, grid_us


def _forward_fill_rows(values: np.ndarray) -> None:
    """In-place left-to-right forward fill per row; leading NaNs are preserved."""
    n_rows, n_cols = values.shape
    col_idx = np.arange(n_cols)
    for r in range(n_rows):
        row = values[r]
        observed = ~np.isnan(row)
        if not observed.any():
            continue
        last = np.where(observed, col_idx, -1)
        np.maximum.accumulate(last, out=last)
        first = int(observed.argmax())
        filled = np.where(last >= 0, row[last.clip(min=0)], np.nan)
        filled[:first] = np.nan
        values[r] = filled


def load_polymarket_source(
    start: dt.datetime,
    end: dt.datetime,
    *,
    archive: PolymarketArchive | None = None,
    num_assets: int = 32,
    freq_seconds: float = 5.0,
    name: str = "polymarket",
    periods: tuple[int, ...] | None = None,
) -> RealSource:
    """Download ``[start, end]`` and return one :class:`RealSource`.

    Channels are the ``num_assets`` most active outcome tokens over the range,
    each a mid-price series resampled to ``freq_seconds``. Prices live in
    ``[0, 1]`` (they are outcome probabilities). ``periods`` defaults to a daily
    seasonal period (in grid steps) when the range spans at least two days, else
    an empty tuple (intraday microstructure has no fixed calendar period).
    """
    archive = archive or PolymarketArchive()
    paths = archive.download_range(start, end)
    assets = top_assets(paths, num_assets)
    if not assets:
        raise ValueError("No active assets found in the requested range.")
    values, _, _ = build_mid_grid(paths, assets, freq_seconds=freq_seconds)

    if periods is None:
        span_days = (end - start).total_seconds() / 86400.0
        steps_per_day = int(round(86400.0 / freq_seconds))
        periods = (steps_per_day,) if span_days >= 2 else ()

    return RealSource(name=name, values=values, periods=periods)

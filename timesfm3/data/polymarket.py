"""Prediction-market corpus from the Pendulum Flow Polymarket archive.

Turns the public Polymarket order-book archive at
``https://archive.pendulumflow.com/`` into TimesFM-3 examples.

**Shape of the data.** Polymarket markets are binary: every market
(``condition_id``) has exactly two outcome tokens. In this archive their mid
prices are *exactly* complementary -- ``p_yes + p_no == 1`` to the last tick,
with zero variance -- so the second token is an arithmetic mirror of the first
and carries no independent information. Stacking both outcomes as separate
variates therefore buys nothing; take one outcome per market and build the
multivariate structure from the market's own microstructure instead, which is
what the channels below provide.

The archive records every quote and fill with microsecond arrival timestamps,
so each market carries spread, order-flow imbalance and book-update intensity
alongside its price.

**What this module produces.** A :class:`MarketPanel` per market -- a stack of
per-outcome feature series on a regular time grid:

===============  ==========================================================
``mid``          ``(best_bid + best_ask) / 2``, last quote in cell, ffilled
``spread``       ``best_ask - best_bid``, last quote in cell, ffilled
``ret``          first difference of ``mid`` (probability change)
``abs_ret``      ``|ret|`` -- a realized-volatility proxy
``volume``       summed trade size in the cell
``signed_flow``  signed fill size: BUY positive, SELL negative; a fill
                 with no recorded side counts positive
``trades``       trade count in the cell
``quotes``       quote-update count in the cell (book activity)
===============  ==========================================================

Panels convert to :class:`~timesfm3.data.real.RealSource` objects, so they mix
freely with the ETT and synthetic corpora in
:class:`~timesfm3.data.real.RealWindowDataset` / ``MixedCorpus``.

The archive is large (~1 GB and ~10^8 rows per hour), so every read streams
row-group by row-group and only the needed columns are touched.

Requires ``pyarrow``: ``pip install -e .[polymarket]``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import os
import time
import urllib.request
from collections import Counter, defaultdict

import numpy as np

from .real import RealSource

try:  # pyarrow is an optional dependency (only this data path needs it).
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised only without pyarrow
    raise ImportError(
        "timesfm3.data.polymarket requires pyarrow. Install it with "
        "`pip install pyarrow` (or `pip install -e .[polymarket]`)."
    ) from exc


DEFAULT_BASE_URL = "https://archive.pendulumflow.com"
DEFAULT_VERSION = "v3"

# Events carrying a top-of-book quote, and the event carrying fills.
QUOTE_EVENTS = ("best_bid_ask", "price_change")
TRADE_EVENT = "last_trade_price"

#: Per-outcome feature channels produced by :func:`build_market_panels`.
FEATURES = (
    "mid",
    "spread",
    "ret",
    "abs_ret",
    "volume",
    "signed_flow",
    "trades",
    "quotes",
)

#: Channels that are counts/sizes -- heavy-tailed and mostly zero, so
#: :func:`panels_to_sources` log1p-compresses them before training.
COUNT_CHANNELS = frozenset({"volume", "trades", "quotes"})


# --------------------------------------------------------------------------
# Archive client
# --------------------------------------------------------------------------


@dataclasses.dataclass
class PolymarketArchive:
    """Client for the Pendulum Flow Polymarket archive.

    Parameters
    ----------
    cache_dir:
        Local directory for downloads (mirrors the archive's ``YYYY-MM-DD/HH/``
        layout).
    base_url, version:
        Archive location. ``version`` selects the capture era: ``"v3"``
        (default, the highest-fidelity feed), ``"pmxt/v2"``, ``"pmxt/v1"`` or
        ``"third-party/ag6"``.
    verify:
        When ``True`` (default) each file is checked against the archive's
        ``SHA256SUMS.txt``. Hours newer than the published manifest are
        accepted with a warning rather than rejected.
    """

    cache_dir: str = "data/polymarket"
    base_url: str = DEFAULT_BASE_URL
    version: str = DEFAULT_VERSION
    verify: bool = True

    def __post_init__(self) -> None:
        self._version_url = f"{self.base_url.rstrip('/')}/{self.version.strip('/')}"
        self._checksums: dict[str, str] | None = None

    @staticmethod
    def _relpath(hour: dt.datetime) -> str:
        """``2026-08-29/15/2026-08-29T15.parquet``."""
        return f"{hour:%Y-%m-%d}/{hour:%H}/{hour:%Y-%m-%d}T{hour:%H}.parquet"

    def hour_url(self, hour: dt.datetime) -> str:
        return f"{self._version_url}/{self._relpath(hour)}"

    def local_path(self, hour: dt.datetime) -> str:
        return os.path.join(self.cache_dir, self.version, self._relpath(hour))

    def _load_checksums(self) -> dict[str, str]:
        if self._checksums is None:
            url = f"{self._version_url}/SHA256SUMS.txt"
            sums: dict[str, str] = {}
            with urllib.request.urlopen(url) as resp:  # noqa: S310 (fixed host)
                for line in resp.read().decode("utf-8", "replace").splitlines():
                    digest, _, path = line.strip().partition("  ")
                    if path:
                        sums[path.strip()] = digest.strip()
            self._checksums = sums
        return self._checksums

    def expected_sha256(self, hour: dt.datetime) -> str | None:
        """Published SHA-256 for an hour, or ``None`` if not yet listed."""
        try:
            return self._load_checksums().get(self._relpath(hour))
        except OSError:
            return None

    def download_hour(
        self,
        hour: dt.datetime,
        *,
        force: bool = False,
        attempts: int = 3,
        on_mismatch: str = "raise",
    ) -> str:
        """Download one hour's parquet into the cache and return its path.

        Hourly files are ~1 GB and large transfers do get truncated in flight,
        so a download is verified *before* it is promoted into the cache, and a
        cached file that fails its checksum is re-fetched rather than raising
        immediately -- otherwise one bad byte poisons the cache for every later
        run.

        Two different failures are distinguished, because they need different
        responses. If repeated downloads disagree with each other, the transfer
        is at fault and retrying helps. If they agree with each other but
        disagree with ``SHA256SUMS.txt``, the archive is serving a file its own
        manifest does not describe; retrying cannot fix that. ``on_mismatch``
        chooses what to do in the second case -- ``"raise"`` (default) or
        ``"warn"`` to accept the bytes the archive actually serves.
        """
        if on_mismatch not in ("raise", "warn"):
            raise ValueError("on_mismatch must be 'raise' or 'warn'")
        path = self.local_path(hour)
        rel = self._relpath(hour)
        expected = self.expected_sha256(hour) if self.verify else None
        if self.verify and expected is None:
            print(f"[polymarket] {rel}: not in SHA256SUMS.txt "
                  "(newer than the manifest?); integrity cannot be checked")

        if os.path.exists(path) and not force:
            if expected is None or _sha256(path) == expected:
                return path
            print(f"[polymarket] {rel}: cached copy fails its checksum; re-downloading")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        seen: set[str] = set()
        problem = ""
        for attempt in range(1, attempts + 1):
            try:
                _fetch(self.hour_url(hour), tmp)
            except OSError as exc:
                problem = f"transfer failed ({exc})"
            else:
                if expected is None:
                    os.replace(tmp, path)
                    return path
                actual = _sha256(tmp)
                if actual == expected:
                    os.replace(tmp, path)
                    return path
                seen.add(actual)
                problem = f"checksum mismatch (sha256 {actual[:16]}…)"
            print(f"[polymarket] {rel}: {problem} [attempt {attempt}/{attempts}]")
            if attempt < attempts:
                time.sleep(2 ** attempt)

        # Every attempt produced the same bytes: the archive and its manifest
        # disagree, and no number of retries will reconcile them.
        stable = len(seen) == 1
        detail = (
            f"the archive consistently serves sha256 {seen.pop()[:16]}… but "
            f"SHA256SUMS.txt lists {expected[:16]}…" if stable
            else f"downloads disagreed with each other ({len(seen)} distinct "
                 "hashes); the transfer is unreliable"
        )
        if stable and on_mismatch == "warn":
            os.replace(tmp, path)
            print(f"[polymarket] {rel}: {detail}; accepting as requested")
            return path
        if os.path.exists(tmp):
            os.remove(tmp)
        raise ValueError(
            f"could not fetch a verified copy of {rel} after {attempts} "
            f"attempts: {detail}. Pass on_mismatch='warn' to accept the "
            "archive's bytes anyway."
        )

    def download_range(
        self,
        start: dt.datetime,
        end: dt.datetime,
        *,
        force: bool = False,
        on_mismatch: str = "raise",
    ) -> list[str]:
        """Download every hour in ``[start, end]`` inclusive; return paths."""
        return [
            self.download_hour(h, force=force, on_mismatch=on_mismatch)
            for h in hours_between(start, end)
        ]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: str) -> int:
    """Stream ``url`` to ``dest``; raise if the transfer came up short."""
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (fixed host)
        declared = resp.headers.get("Content-Length")
        written = 0
        with open(dest, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                written += len(chunk)
    if declared is not None and written != int(declared):
        raise OSError(f"short read: {written} of {declared} bytes")
    return written


def hours_between(start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
    """Inclusive list of hour-aligned datetimes from ``start`` to ``end``."""
    cur = start.replace(minute=0, second=0, microsecond=0)
    end = end.replace(minute=0, second=0, microsecond=0)
    out = []
    while cur <= end:
        out.append(cur)
        cur += dt.timedelta(hours=1)
    return out


# --------------------------------------------------------------------------
# Market selection
# --------------------------------------------------------------------------


@dataclasses.dataclass
class MarketInfo:
    """A market's identity and its outcome tokens."""

    market: bytes
    assets: list[bytes]
    trades: int = 0
    quotes: int = 0
    question: str | None = None
    slug: str | None = None
    outcomes: list[str] | None = None

    @property
    def label(self) -> str:
        """Human-readable name (question, slug, or truncated condition id)."""
        return self.question or self.slug or self.market.hex()[:16]


def scan_markets(paths: list[str]) -> dict[bytes, MarketInfo]:
    """Map every market to its outcome tokens and activity counts.

    One streaming pass per file. Markets are ranked downstream by ``trades``
    rather than quote traffic: quote churn is dominated by market-maker
    requoting, while fills indicate real liquidity. ``new_market`` rows, when
    present in the window, supply the human-readable question and slug.
    """
    infos: dict[bytes, MarketInfo] = {}
    assets_by_market: dict[bytes, set] = defaultdict(set)
    trades: Counter = Counter()
    quotes: Counter = Counter()
    quote_set = pa.array(list(QUOTE_EVENTS), type=pa.string())
    cols = ["event_type", "market", "asset_id", "assets_ids", "outcomes",
            "question", "slug"]

    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=cols)
            et = t["event_type"]

            for mask, counter in (
                (pc.equal(et, TRADE_EVENT), trades),
                (pc.is_in(et, value_set=quote_set), quotes),
            ):
                sub = t.filter(mask)
                if sub.num_rows == 0:
                    continue
                for e in pc.value_counts(sub["market"]).to_pylist():
                    if e["values"] is not None:
                        counter[e["values"]] += e["counts"]

            # asset -> market membership (present on quote and trade rows).
            # group_by collapses ~10^5 rows to a handful of distinct pairs in
            # C++; a Python zip over every row dominates runtime otherwise.
            sub = t.filter(
                pc.and_(pc.is_valid(t["market"]), pc.is_valid(t["asset_id"]))
            )
            if sub.num_rows:
                pairs = (
                    sub.select(["market", "asset_id"])
                    .group_by(["market", "asset_id"])
                    .aggregate([])
                )
                for m, a in zip(pairs["market"].to_pylist(),
                                pairs["asset_id"].to_pylist()):
                    assets_by_market[m].add(a)

            # new_market rows carry question / slug / outcomes / assets_ids.
            sub = t.filter(pc.equal(et, "new_market"))
            for i in range(sub.num_rows):
                m = sub["market"][i].as_py()
                if m is None:
                    continue
                info = infos.setdefault(m, MarketInfo(market=m, assets=[]))
                info.question = sub["question"][i].as_py() or info.question
                info.slug = sub["slug"][i].as_py() or info.slug
                info.outcomes = sub["outcomes"][i].as_py() or info.outcomes
                for a in sub["assets_ids"][i].as_py() or []:
                    assets_by_market[m].add(a)

    for m, assets in assets_by_market.items():
        info = infos.setdefault(m, MarketInfo(market=m, assets=[]))
        info.assets = sorted(assets)
        info.trades = trades.get(m, 0)
        info.quotes = quotes.get(m, 0)
    return infos


def top_markets(
    paths: list[str], n: int, *, min_outcomes: int = 2, min_trades: int = 1
) -> list[MarketInfo]:
    """The ``n`` most-traded markets that have at least ``min_outcomes`` tokens."""
    infos = scan_markets(paths)
    eligible = [
        i for i in infos.values()
        if len(i.assets) >= min_outcomes and i.trades >= min_trades
    ]
    eligible.sort(key=lambda i: (i.trades, i.quotes), reverse=True)
    return eligible[:n]


# --------------------------------------------------------------------------
# Feature grid
# --------------------------------------------------------------------------


@dataclasses.dataclass
class MarketPanel:
    """One market's outcome tokens as aligned feature series.

    ``features[name]`` is ``(num_assets, num_steps)`` float32; ``assets`` gives
    the outcome-token order and ``info.outcomes`` their labels when known.
    """

    info: MarketInfo
    assets: list[bytes]
    features: dict[str, np.ndarray]
    grid_us: np.ndarray
    freq_seconds: float

    @property
    def num_steps(self) -> int:
        return len(self.grid_us)

    def stack(self, channels: tuple[str, ...]) -> np.ndarray:
        """Stack ``channels`` for every asset into ``(assets*channels, steps)``."""
        return np.concatenate([self.features[c] for c in channels], axis=0)


def build_market_panels(
    paths: list[str],
    markets: list[MarketInfo],
    *,
    freq_seconds: float = 15.0,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> list[MarketPanel]:
    """Build per-market feature panels in a single streaming pass.

    Quote features (``mid``, ``spread``) take the last quote in each grid cell
    and forward-fill; activity features (``volume``, ``signed_flow``,
    ``trades``, ``quotes``) are summed within the cell and are zero where
    nothing happened. Cells before an asset's first quote stay NaN, which the
    model reads through its observed-mask.
    """
    if not markets:
        raise ValueError("`markets` must be non-empty.")

    assets: list[bytes] = []
    asset_market: list[int] = []
    for mi, info in enumerate(markets):
        for a in info.assets:
            assets.append(a)
            asset_market.append(mi)
    index = {a: i for i, a in enumerate(assets)}
    asset_set = pa.array(assets, type=pa.binary())
    quote_set = pa.array(list(QUOTE_EVENTS), type=pa.string())

    start_us, end_us, step_us = _grid_bounds(paths, asset_set, start, end, freq_seconds)
    n_steps = max(1, int((end_us - start_us + step_us - 1) // step_us))
    grid_us = start_us + step_us * np.arange(n_steps, dtype=np.int64)
    n_assets = len(assets)

    mid = np.full((n_assets, n_steps), np.nan, dtype=np.float32)
    spread = np.full((n_assets, n_steps), np.nan, dtype=np.float32)
    volume = np.zeros((n_assets, n_steps), dtype=np.float32)
    signed = np.zeros((n_assets, n_steps), dtype=np.float32)
    trades = np.zeros((n_assets, n_steps), dtype=np.float32)
    quotes = np.zeros((n_assets, n_steps), dtype=np.float32)

    cols = ["event_type", "timestamp_received", "asset_id", "best_bid", "best_ask",
            "price", "size", "side"]
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=cols)
            t = t.filter(pc.is_in(t["asset_id"], value_set=asset_set))
            if t.num_rows == 0:
                continue
            ts = pc.cast(t["timestamp_received"], "int64").to_numpy(zero_copy_only=False)
            keep = (ts >= start_us) & (ts < start_us + step_us * n_steps)
            if not keep.any():
                continue
            cell = ((ts - start_us) // step_us).astype(np.int64)
            code = np.fromiter(
                (index[a] for a in t["asset_id"].to_pylist()),
                dtype=np.int64, count=t.num_rows,
            )
            et = np.asarray(t["event_type"].to_pylist(), dtype=object)
            bb = pc.cast(t["best_bid"], "float64").to_numpy(zero_copy_only=False)
            ba = pc.cast(t["best_ask"], "float64").to_numpy(zero_copy_only=False)

            # -- quotes: last in cell wins, plus an update count.
            q = keep & np.isin(et, QUOTE_EVENTS) & np.isfinite(bb) & np.isfinite(ba)
            if q.any():
                qc, qcell, qts = code[q], cell[q], ts[q]
                order = np.lexsort((qts, qcell, qc))
                mid[qc[order], qcell[order]] = ((bb[q] + ba[q]) * 0.5)[order]
                spread[qc[order], qcell[order]] = (ba[q] - bb[q])[order]
                np.add.at(quotes, (qc, qcell), 1.0)

            # -- trades: summed size, signed flow, count.
            tr = keep & (et == TRADE_EVENT)
            if tr.any():
                sz = pc.cast(t["size"], "float64").to_numpy(zero_copy_only=False)[tr]
                sz = np.nan_to_num(sz)
                side = np.asarray(t["side"].to_pylist(), dtype=object)[tr]
                sign = np.where(side == "SELL", -1.0, 1.0)
                tc, tcell = code[tr], cell[tr]
                np.add.at(volume, (tc, tcell), sz)
                np.add.at(signed, (tc, tcell), sz * sign)
                np.add.at(trades, (tc, tcell), 1.0)

    _forward_fill_rows(mid)
    _forward_fill_rows(spread)
    ret = np.diff(mid, axis=1, prepend=np.nan)
    feats_all = {
        "mid": mid, "spread": spread, "ret": ret, "abs_ret": np.abs(ret),
        "volume": volume, "signed_flow": signed, "trades": trades, "quotes": quotes,
    }

    panels = []
    am = np.asarray(asset_market)
    for mi, info in enumerate(markets):
        rows = np.flatnonzero(am == mi)
        if rows.size == 0:
            continue
        panels.append(
            MarketPanel(
                info=info,
                assets=[assets[r] for r in rows],
                features={k: v[rows] for k, v in feats_all.items()},
                grid_us=grid_us,
                freq_seconds=freq_seconds,
            )
        )
    return panels


def _grid_bounds(paths, asset_set, start, end, freq_seconds):
    """Resolve grid bounds, scanning parquet statistics when not supplied."""
    step_us = int(round(freq_seconds * 1e6))
    if step_us <= 0:
        raise ValueError("freq_seconds must be positive.")
    if start is not None and end is not None:
        return int(start.timestamp() * 1e6), int(end.timestamp() * 1e6), step_us
    lo, hi = None, None
    for path in paths:
        pf = pq.ParquetFile(path)
        # Locate timestamp_received by name; column order is not guaranteed.
        try:
            col = pf.schema_arrow.names.index("timestamp_received")
        except ValueError:
            raise ValueError(f"{path} has no timestamp_received column") from None
        for rg in range(pf.metadata.num_row_groups):
            st = pf.metadata.row_group(rg).column(col).statistics
            if st is None or st.min is None:
                continue
            a, b = _to_us(st.min), _to_us(st.max)
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
    if lo is None:
        raise ValueError("Could not determine a time range from parquet statistics.")
    if start is not None:
        lo = int(start.timestamp() * 1e6)
    if end is not None:
        hi = int(end.timestamp() * 1e6)
    return lo, hi + 1, step_us


def _to_us(value) -> int:
    """Parquet stats come back as datetime or int microseconds."""
    if isinstance(value, dt.datetime):
        return int(value.timestamp() * 1e6)
    return int(value)


def _forward_fill_rows(values: np.ndarray) -> None:
    """In-place left-to-right forward fill per row; leading NaNs are kept."""
    n_cols = values.shape[1]
    col_idx = np.arange(n_cols)
    for r in range(values.shape[0]):
        row = values[r]
        observed = ~np.isnan(row)
        if not observed.any():
            continue
        last = np.where(observed, col_idx, -1)
        np.maximum.accumulate(last, out=last)
        filled = row[np.maximum(last, 0)]
        filled[: int(observed.argmax())] = np.nan
        values[r] = filled


# --------------------------------------------------------------------------
# TimesFM-3 corpus adapters
# --------------------------------------------------------------------------


def select_covered(
    panels: list[MarketPanel], *, channel: str = "mid", min_coverage: float = 0.98
) -> list[MarketPanel]:
    """Keep panels quoted for (nearly) the whole window.

    The archive is full of very short-lived markets -- five-minute "Up or Down"
    contracts are created continuously -- which are mostly NaN across a
    multi-hour grid. Ranking by trade count alone surfaces plenty of them, so
    select on coverage before building a corpus.
    """
    out = []
    for p in panels:
        v = p.features[channel]
        if np.isfinite(v).mean() >= min_coverage:
            out.append(p)
    return out


def panels_to_sources(
    panels: list[MarketPanel],
    channels: tuple[str, ...] = ("mid", "spread", "abs_ret", "volume"),
    *,
    log1p_counts: bool = True,
    min_coverage: float = 0.98,
) -> list[RealSource]:
    """Convert panels to :class:`RealSource` objects for training.

    Each panel becomes one source whose channels are ``assets x channels``, so
    a training window sees both outcomes of a market together and the
    cross-variate layers can exploit their coupling. Count-like channels are
    ``log1p``-compressed (they are heavy-tailed and mostly zero).
    ``periods`` is left empty: intraday microstructure has no fixed calendar
    period at these frequencies.
    """
    sources = []
    for panel in panels:
        stacked = []
        for c in channels:
            # Step 0 is dropped: `ret` and `abs_ret` are undefined at the first
            # cell, and a leading NaN would otherwise ride into every window.
            v = panel.features[c][:, 1:].copy()
            if log1p_counts and c in COUNT_CHANNELS:
                v = np.sign(v) * np.log1p(np.abs(v))
            stacked.append(v)
        values = np.concatenate(stacked, axis=0).astype(np.float32)
        if np.isfinite(values).mean() < min_coverage:
            continue
        sources.append(
            RealSource(name=f"pm:{panel.info.label[:40]}", values=values, periods=())
        )
    return sources


def load_polymarket_sources(
    start: dt.datetime,
    end: dt.datetime,
    *,
    archive: PolymarketArchive | None = None,
    num_markets: int = 64,
    freq_seconds: float = 15.0,
    channels: tuple[str, ...] = ("mid", "spread", "abs_ret", "volume"),
) -> list[RealSource]:
    """Download ``[start, end]`` and return one :class:`RealSource` per market."""
    archive = archive or PolymarketArchive()
    paths = archive.download_range(start, end)
    markets = top_markets(paths, num_markets)
    if not markets:
        raise ValueError("No traded markets found in the requested range.")
    panels = build_market_panels(paths, markets, freq_seconds=freq_seconds)
    return panels_to_sources(panels, channels)

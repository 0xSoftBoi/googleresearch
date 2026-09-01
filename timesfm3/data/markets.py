"""Real daily market data from FRED, shaped for forecasting and backtesting.

Systematic funds trade a diversified universe of liquid futures and FX
forwards.  This module builds a faithful stand-in for that universe from the
Federal Reserve's public FRED archive (no API key, stable CSV endpoint,
histories back to 1971 for the majors):

- **FX majors** -- the nine most liquid USD crosses.  FRED quotes some pairs
  as USD-per-foreign and some as foreign-per-USD; the loader inverts the
  latter so every series is "value of one unit of foreign currency in USD"
  and a positive return always means the foreign currency appreciated.
- **Equity indices** -- NASDAQ Composite (from 1971) and S&P 500 (FRED only
  redistributes the last ~10 years).
- **Commodities** -- WTI and Brent spot, Henry Hub natural gas, LBMA gold.
- **Rates** -- 2y and 10y constant-maturity Treasury *yields*, converted to
  approximate total returns of a constant-maturity par bond (duration +
  carry + convexity), the standard trick for simulating bond futures
  histories when only yields are archived (Swinkels, "Simulating historical
  inflation-linked bond returns", and the managed-futures literature).

Everything is plain numpy: dates as ``datetime64[D]``, one aligned
``(assets, time)`` price panel with NaN before an asset is born, and
NaN-aware log returns.  ``to_real_source`` bridges the panel into the
existing pre-training corpus (`timesfm3.data.real.RealSource`), so the same
data that backtests strategies can also pre-train the model.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import os
import urllib.request

import numpy as np

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


@dataclasses.dataclass(frozen=True)
class Instrument:
    """One tradable series in the universe.

    ``kind`` is ``"price"`` for series whose log returns are directly the
    asset's return, or ``"yield"`` for constant-maturity Treasury yields that
    must first be converted into bond returns.  ``invert`` flips a
    foreign-per-USD FX quote into USD-per-foreign.
    """

    name: str
    series_id: str
    asset_class: str  # "fx" | "equity" | "commodity" | "bond"
    invert: bool = False
    kind: str = "price"
    maturity_years: float = 0.0  # only for kind == "yield"


#: The default universe: liquid, public, and long-history.  Mirrors the
#: asset-class mix of a managed-futures program (Moskowitz-Ooi-Pedersen 2012
#: trade 58 futures/forwards across the same four classes).
DEFAULT_UNIVERSE: tuple[Instrument, ...] = (
    Instrument("EURUSD", "DEXUSEU", "fx"),
    Instrument("GBPUSD", "DEXUSUK", "fx"),
    Instrument("AUDUSD", "DEXUSAL", "fx"),
    Instrument("NZDUSD", "DEXUSNZ", "fx"),
    Instrument("JPYUSD", "DEXJPUS", "fx", invert=True),
    Instrument("CADUSD", "DEXCAUS", "fx", invert=True),
    Instrument("CHFUSD", "DEXSZUS", "fx", invert=True),
    Instrument("SEKUSD", "DEXSDUS", "fx", invert=True),
    Instrument("NOKUSD", "DEXNOUS", "fx", invert=True),
    Instrument("NASDAQ", "NASDAQCOM", "equity"),
    Instrument("SP500", "SP500", "equity"),
    Instrument("WTI", "DCOILWTICO", "commodity"),
    Instrument("BRENT", "DCOILBRENTEU", "commodity"),
    Instrument("NATGAS", "DHHNGSP", "commodity"),
    Instrument("GOLD", "GOLDPMGBD228NLBM", "commodity"),
    Instrument("UST10Y", "DGS10", "bond", kind="yield", maturity_years=10.0),
    Instrument("UST2Y", "DGS2", "bond", kind="yield", maturity_years=2.0),
)

TRADING_DAYS = 252


def fetch_fred_csv(
    series_id: str, cache_dir: str = "data/fred", refresh: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Download (or read from cache) one FRED series.

    Returns ``(dates, values)`` with dates as ``datetime64[D]`` and missing
    observations (FRED prints ``"."``) already dropped.  The raw CSV is
    cached verbatim so experiments are reproducible offline; a cached file
    that no longer parses is re-fetched rather than poisoning every later
    run.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{series_id}.csv")
    if refresh or not os.path.exists(path):
        url = FRED_CSV_URL.format(series_id=series_id)
        with urllib.request.urlopen(url) as resp:  # noqa: S310 (fixed host)
            raw = resp.read()
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return parse_fred_csv(fh.read())
    except ValueError:
        if refresh:
            raise
        os.remove(path)
        return fetch_fred_csv(series_id, cache_dir=cache_dir, refresh=True)


def parse_fred_csv(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse FRED's two-column CSV (``observation_date,<ID>``)."""
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header is None or len(header) != 2 or header[0] != "observation_date":
        raise ValueError("not a FRED observation CSV")
    dates, values = [], []
    for row in reader:
        if len(row) != 2 or row[1] in (".", ""):
            continue
        dates.append(np.datetime64(row[0], "D"))
        values.append(float(row[1]))
    if not dates:
        raise ValueError("FRED CSV contained no observations")
    return np.array(dates, dtype="datetime64[D]"), np.array(values, dtype=np.float64)


def bond_returns_from_yields(yields_pct: np.ndarray, maturity_years: float) -> np.ndarray:
    """Approximate daily total returns of a constant-maturity par bond.

    From day t-1 to t with annualized par yields ``y`` (decimal):

        return = carry + duration effect + convexity effect
               = y[t-1]/252  -  D * (y[t] - y[t-1])  +  0.5 * C * (y[t]-y[t-1])^2

    with the modified duration and convexity of a par bond evaluated at
    y[t-1].  This is the standard reconstruction used when only yield
    histories exist; against actual Treasury futures it captures the level
    dynamics that trend strategies trade (it omits roll-down and the
    futures/spot financing spread).  Returns length ``len(yields) - 1``.
    """
    y = np.asarray(yields_pct, dtype=np.float64) / 100.0
    y_prev = np.clip(y[:-1], 1e-4, None)  # zero yields would blow up duration
    dy = y[1:] - y[:-1]
    n = maturity_years
    # Par bond with annual coupons at rate y: closed-form modified duration
    # [1 - (1+y)^-n] / y and the matching convexity.
    duration = (1.0 - (1.0 + y_prev) ** (-n)) / y_prev
    convexity = 2.0 / (y_prev**2) * (1.0 - (1.0 + y_prev) ** (-n)) - (
        2.0 * n / (y_prev * (1.0 + y_prev) ** (n + 1))
    )
    return y_prev / TRADING_DAYS - duration * dy + 0.5 * convexity * dy**2


@dataclasses.dataclass
class MarketPanel:
    """An aligned multi-asset daily panel.

    ``prices[i, t]`` is NaN before asset ``i`` starts (or on a day it did not
    print after forward-fill), and ``returns[i, t]`` is the log return from
    ``t-1`` to ``t`` (``returns[:, 0]`` is NaN).  For ``kind == "yield"``
    instruments, ``prices`` holds a cumulated total-return index built from
    :func:`bond_returns_from_yields`, so downstream code treats every asset
    identically.
    """

    dates: np.ndarray  # (T,) datetime64[D]
    names: list[str]
    asset_classes: list[str]
    prices: np.ndarray  # (N, T) float64, NaN-padded
    returns: np.ndarray  # (N, T) float64, NaN-padded log returns

    @property
    def num_assets(self) -> int:
        return len(self.names)

    def __len__(self) -> int:
        return len(self.dates)

    def slice(self, start: np.datetime64 | str | None, end: np.datetime64 | str | None):
        """Restrict the panel to ``start <= date <= end``."""
        mask = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            mask &= self.dates >= np.datetime64(start, "D")
        if end is not None:
            mask &= self.dates <= np.datetime64(end, "D")
        return MarketPanel(
            self.dates[mask],
            list(self.names),
            list(self.asset_classes),
            self.prices[:, mask],
            self.returns[:, mask],
        )


def _forward_fill(values: np.ndarray, limit: int) -> np.ndarray:
    """Forward-fill NaNs, but only across gaps of at most ``limit`` days."""
    out = values.copy()
    last, age = np.nan, limit + 1
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last, age = out[i], 0
        else:
            age += 1
            if age <= limit and np.isfinite(last):
                out[i] = last
    return out


def load_universe(
    instruments: tuple[Instrument, ...] = DEFAULT_UNIVERSE,
    cache_dir: str = "data/fred",
    start: np.datetime64 | str | None = None,
    end: np.datetime64 | str | None = None,
    fill_limit: int = 5,
    verbose: bool = True,
) -> MarketPanel:
    """Download the universe and align it on the union of observation dates.

    An instrument whose download or parse fails is skipped with a warning
    rather than failing the whole universe -- FRED occasionally retires a
    series (gold's LBMA license ended in 2024) and a backtest on sixteen
    assets should not die because one archive moved.
    """
    loaded: list[tuple[Instrument, np.ndarray, np.ndarray]] = []
    for inst in instruments:
        try:
            dates, values = fetch_fred_csv(inst.series_id, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - degrade per-instrument
            if verbose:
                print(f"[markets] {inst.name} ({inst.series_id}): skipped ({exc})")
            continue
        if inst.invert:
            values = 1.0 / values
        if inst.kind == "yield":
            rets = bond_returns_from_yields(values, inst.maturity_years)
            values = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(np.log1p(rets)))])
        loaded.append((inst, dates, values))
        if verbose:
            print(
                f"[markets] {inst.name:8s} {inst.series_id:18s} "
                f"{len(values):6d} obs  {dates[0]} .. {dates[-1]}"
            )
    if not loaded:
        raise RuntimeError("no instruments could be loaded")

    all_dates = np.unique(np.concatenate([d for _, d, _ in loaded]))
    prices = np.full((len(loaded), len(all_dates)), np.nan)
    for i, (_, dates, values) in enumerate(loaded):
        idx = np.searchsorted(all_dates, dates)
        prices[i, idx] = values
        first = idx[0]
        prices[i, first:] = _forward_fill(prices[i, first:], fill_limit)

    with np.errstate(invalid="ignore", divide="ignore"):
        log_prices = np.log(prices)
    returns = np.full_like(prices, np.nan)
    returns[:, 1:] = log_prices[:, 1:] - log_prices[:, :-1]

    panel = MarketPanel(
        dates=all_dates,
        names=[inst.name for inst, _, _ in loaded],
        asset_classes=[inst.asset_class for inst, _, _ in loaded],
        prices=prices,
        returns=returns,
    )
    return panel.slice(start, end)


def to_real_source(panel: MarketPanel, name: str = "fred-markets"):
    """Bridge the panel into the pre-training corpus as a ``RealSource``.

    Channels are cumulative log-price levels (the model forecasts levels and
    normalizes per-window), NaNs filled by holding the last level so early
    history with fewer live assets is still usable.  Weekly seasonality (5
    trading days) is the only calendar period that plausibly exists in daily
    financial data.
    """
    from .real import RealSource  # local import: keeps markets torch-free

    values = np.empty_like(panel.prices, dtype=np.float32)
    for i in range(panel.num_assets):
        row = panel.prices[i]
        filled = _forward_fill(row, limit=len(row))
        first = np.flatnonzero(np.isfinite(filled))
        base = filled[first[0]] if len(first) else 1.0
        filled = np.where(np.isfinite(filled), filled, base)
        # WTI printed negative in April 2020; clamp to a small positive
        # floor so log-levels stay finite (the episode becomes a spike to
        # the floor rather than a NaN that poisons training windows).
        floor = max(1e-6, 0.005 * float(np.nanmedian(np.abs(filled))))
        values[i] = np.log(np.maximum(filled, floor)).astype(np.float32)
    return RealSource(name=name, values=values, periods=(5,))

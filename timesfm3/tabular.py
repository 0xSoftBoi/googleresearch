"""Reading and writing time series as tables (CSV), shared by CLI and server.

Layout: one row per time step, one column per series.  A leading column
that does not parse as a number is treated as the timestamp column; a
first row that does not parse as numbers is treated as the header.  Empty
cells and ``nan``/``NA`` become NaN, which the forecasters treat as missing.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import re

import numpy as np

_MISSING = {"", "nan", "na", "n/a", "null", "none"}


def _to_float(token: str) -> float | None:
    """Float for a numeric token, NaN for a missing marker, None otherwise."""
    t = token.strip()
    if t.lower() in _MISSING:
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return None


@dataclasses.dataclass
class SeriesTable:
    """A (num_series, num_steps) panel with names and optional timestamps."""

    names: list[str]
    values: np.ndarray  # (C, T) float64, NaN for missing
    timestamps: list[str] | None = None

    @property
    def num_series(self) -> int:
        return self.values.shape[0]

    @property
    def num_steps(self) -> int:
        return self.values.shape[1]


def parse_series_csv(text: str) -> SeriesTable:
    """Parses CSV text (see module docstring for the layout rules)."""
    sample = text[:4096]
    delimiter = ","
    if sample.count("\t") > sample.count(","):
        delimiter = "\t"
    elif sample.count(";") > sample.count(","):
        delimiter = ";"
    rows = [
        r for r in csv.reader(io.StringIO(text), delimiter=delimiter, skipinitialspace=True)
        if any(cell.strip() for cell in r)
    ]
    if not rows:
        raise ValueError("The table is empty.")

    header: list[str] | None = None
    first_parsed = [_to_float(c) for c in rows[0]]
    if any(v is None for v in first_parsed[1:]) or (
        len(rows[0]) == 1 and first_parsed[0] is None
    ):
        header = [c.strip() for c in rows[0]]
        rows = rows[1:]
    if not rows:
        raise ValueError("The table has a header but no data rows.")

    width = len(rows[0])
    has_time = _to_float(rows[0][0]) is None
    start = 1 if has_time else 0
    timestamps: list[str] | None = [] if has_time else None
    data: list[list[float]] = []
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"Row {i + 1} has {len(row)} cells, expected {width}.")
        if has_time:
            timestamps.append(row[0].strip())  # type: ignore[union-attr]
        parsed = []
        for cell in row[start:]:
            v = _to_float(cell)
            if v is None:
                raise ValueError(f"Row {i + 1}: {cell!r} is not a number.")
            parsed.append(v)
        data.append(parsed)
    values = np.asarray(data, dtype=np.float64).T
    if values.shape[0] == 0:
        raise ValueError("The table has no value columns.")
    if header is not None and len(header) == width:
        names = [h or f"series_{j}" for j, h in enumerate(header[start:])]
    else:
        names = [f"series_{j}" for j in range(values.shape[0])]
    return SeriesTable(names=names, values=values, timestamps=timestamps)


def read_series_csv(path: str) -> SeriesTable:
    with open(path, newline="") as f:
        return parse_series_csv(f.read())


_FREQ = re.compile(r"^\s*(\d+)?\s*([a-zA-Z]+)\s*$")
_UNITS = {
    "s": "s", "sec": "s", "second": "s", "seconds": "s",
    "m": "m", "min": "m", "minute": "m", "minutes": "m", "t": "m",
    "h": "h", "hr": "h", "hour": "h", "hours": "h",
    "d": "D", "day": "D", "days": "D",
    "w": "W", "wk": "W", "week": "W", "weeks": "W",
}


def parse_freq(freq: str) -> np.timedelta64:
    """'15min', '1h', 'D', '2W' -> numpy timedelta."""
    m = _FREQ.match(freq)
    if not m or m.group(2).lower() not in _UNITS:
        raise ValueError(
            f"Unsupported frequency {freq!r}; use e.g. '15min', '1h', 'D', 'W'."
        )
    n = int(m.group(1) or 1)
    return np.timedelta64(n, _UNITS[m.group(2).lower()])


def infer_step(timestamps: list[str]) -> np.timedelta64 | None:
    """The most common spacing of the timestamps, or None if unparseable."""
    try:
        ts = np.asarray(timestamps[-64:], dtype="datetime64[s]")
    except ValueError:
        return None
    if len(ts) < 2:
        return None
    diffs = np.diff(ts)
    vals, counts = np.unique(diffs, return_counts=True)
    step = vals[np.argmax(counts)]
    return step if step > np.timedelta64(0, "s") else None


def future_timestamps(
    last: str, horizon: int, step: np.timedelta64
) -> list[str]:
    """``horizon`` timestamps after ``last`` at the given spacing."""
    base = np.datetime64(last, "s")
    return [str(base + (k + 1) * step.astype("timedelta64[s]")) for k in range(horizon)]


def write_forecast_csv(
    path_or_buffer,
    names: list[str],
    point: np.ndarray,
    quantiles: np.ndarray | None,
    levels: tuple[float, ...],
    timestamps: list[str] | None = None,
) -> None:
    """Long-format CSV: step, [timestamp], series, point, q10 ... q90."""
    close = False
    if isinstance(path_or_buffer, str):
        f = open(path_or_buffer, "w", newline="")
        close = True
    else:
        f = path_or_buffer
    try:
        w = csv.writer(f)
        head = ["step"] + (["timestamp"] if timestamps else []) + ["series", "point"]
        if quantiles is not None:
            head += [f"q{int(round(q * 100))}" for q in levels]
        w.writerow(head)
        for i, name in enumerate(names):
            for j in range(point.shape[1]):
                row = [j + 1] + ([timestamps[j]] if timestamps else []) + [name, f"{point[i, j]:.6g}"]
                if quantiles is not None:
                    row += [f"{v:.6g}" for v in quantiles[i, j]]
                w.writerow(row)
    finally:
        if close:
            f.close()

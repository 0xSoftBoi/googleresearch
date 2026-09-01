import io

import numpy as np
import pytest

from timesfm3.tabular import (
    future_timestamps,
    infer_step,
    parse_freq,
    parse_series_csv,
    write_forecast_csv,
)


def test_header_and_timestamps():
    t = parse_series_csv("date,a,b\n2024-01-01,1,2\n2024-01-02,3,\n2024-01-03,5,6\n")
    assert t.names == ["a", "b"]
    assert t.timestamps == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert t.values.shape == (2, 3)
    assert np.isnan(t.values[1, 1])


def test_bare_numbers_and_delimiters():
    assert parse_series_csv("1,2\n3,4\n").values.tolist() == [[1, 3], [2, 4]]
    assert parse_series_csv("1\t2\n3\t4\n").values.tolist() == [[1, 3], [2, 4]]
    t = parse_series_csv("x\n1\n2\n3\n")
    assert t.names == ["x"] and t.values.shape == (1, 3)


def test_missing_markers_and_errors():
    t = parse_series_csv("a\n1\nNA\nnan\n4\n")
    assert np.isnan(t.values[0, 1]) and np.isnan(t.values[0, 2])
    with pytest.raises(ValueError):
        parse_series_csv("a,b\n1,2\n3\n")
    with pytest.raises(ValueError):
        parse_series_csv("")
    with pytest.raises(ValueError):
        parse_series_csv("a,b\n1,x\n")


def test_freq_and_timestamps():
    assert parse_freq("15min") == np.timedelta64(15, "m")
    assert parse_freq("D") == np.timedelta64(1, "D")
    assert parse_freq("2 h") == np.timedelta64(2, "h")
    with pytest.raises(ValueError):
        parse_freq("fortnight")
    step = infer_step(["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"])
    assert step == np.timedelta64(3600, "s")
    assert infer_step(["a", "b"]) is None
    assert future_timestamps("2024-01-01T02:00", 2, step) == [
        "2024-01-01T03:00:00", "2024-01-01T04:00:00"
    ]


def test_write_forecast_csv_roundtrip():
    buf = io.StringIO()
    point = np.array([[1.0, 2.0]])
    q = np.arange(2 * 9, dtype=float).reshape(1, 2, 9)
    write_forecast_csv(buf, ["s"], point, q, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                       ["t1", "t2"])
    lines = buf.getvalue().strip().splitlines()
    assert lines[0] == "step,timestamp,series,point,q10,q20,q30,q40,q50,q60,q70,q80,q90"
    assert lines[1].startswith("1,t1,s,1,0,1,2")
    assert len(lines) == 3

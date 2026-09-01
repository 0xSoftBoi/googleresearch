"""Unit tests for the FRED market-data loader (no network needed)."""

import numpy as np
import pytest

from timesfm3.data.markets import (
    _forward_fill,
    bond_returns_from_yields,
    parse_fred_csv,
)

TRADING_DAYS = 252


class TestParseFredCsv:
    def test_parses_dates_and_values(self):
        text = "observation_date,DEXUSEU\n1999-01-04,1.1812\n1999-01-05,1.1760\n"
        dates, values = parse_fred_csv(text)
        assert dates[0] == np.datetime64("1999-01-04")
        assert values == pytest.approx([1.1812, 1.1760])

    def test_drops_missing_observations(self):
        text = "observation_date,X\n2020-01-01,1.0\n2020-01-02,.\n2020-01-03,2.0\n"
        _, values = parse_fred_csv(text)
        assert values == pytest.approx([1.0, 2.0])

    def test_rejects_non_fred_header(self):
        with pytest.raises(ValueError):
            parse_fred_csv("date,close\n2020-01-01,1.0\n")

    def test_rejects_empty_series(self):
        with pytest.raises(ValueError):
            parse_fred_csv("observation_date,X\n")


class TestBondReturns:
    def test_flat_yields_earn_carry_only(self):
        y = np.full(11, 5.0)  # 5% forever
        r = bond_returns_from_yields(y, maturity_years=10.0)
        assert r == pytest.approx(np.full(10, 0.05 / TRADING_DAYS))

    def test_yield_rise_loses_roughly_duration_times_move(self):
        y = np.array([5.0, 5.10])  # +10bp in one day
        (r,) = bond_returns_from_yields(y, maturity_years=10.0)
        duration = (1 - 1.05**-10) / 0.05  # ~7.72 for a 10y par bond
        expected = 0.05 / TRADING_DAYS - duration * 0.001
        assert r == pytest.approx(expected, rel=0.05)  # convexity helps a little
        assert r < 0

    def test_convexity_makes_big_moves_asymmetric(self):
        up = bond_returns_from_yields(np.array([5.0, 6.0]), 10.0)[0]
        down = bond_returns_from_yields(np.array([5.0, 4.0]), 10.0)[0]
        # Positive convexity: the gain from -100bp exceeds the loss from +100bp.
        assert down > -up


class TestForwardFill:
    def test_fills_short_gaps_only(self):
        x = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        filled = _forward_fill(x, limit=2)
        assert filled[1] == 1.0 and filled[2] == 1.0
        assert np.isnan(filled[3])  # third consecutive gap exceeds the limit
        assert filled[4] == 5.0

    def test_leading_nans_stay(self):
        x = np.array([np.nan, np.nan, 3.0])
        filled = _forward_fill(x, limit=5)
        assert np.isnan(filled[0]) and np.isnan(filled[1]) and filled[2] == 3.0

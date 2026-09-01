"""Tests for the Polymarket archive loader.

Builds a small parquet file in the archive's V3 schema with hand-chosen events,
so every grid cell has an expected value that can be checked exactly.
"""

import datetime as dt
import os
import decimal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timesfm3.data.polymarket import (
    MarketInfo,
    PolymarketArchive,
    _forward_fill_rows,
    build_market_panels,
    hours_between,
    scan_markets,
    select_covered,
)

MARKET = b"\x01" * 32
YES, NO = b"\xaa" * 32, b"\xbb" * 32
T0 = dt.datetime(2026, 8, 29, 12, 0, 0, tzinfo=dt.timezone.utc)

SCHEMA = pa.schema([
    ("event_type", pa.string()),
    ("timestamp_received", pa.timestamp("us", tz="UTC")),
    ("sequence", pa.uint64()),
    ("timestamp", pa.timestamp("ms", tz="UTC")),
    ("market", pa.binary()),
    ("asset_id", pa.binary()),
    ("best_bid", pa.decimal128(9, 4)),
    ("best_ask", pa.decimal128(9, 4)),
    ("spread", pa.decimal128(9, 4)),
    ("price", pa.decimal128(9, 4)),
    ("size", pa.decimal128(18, 6)),
    ("side", pa.string()),
    ("assets_ids", pa.list_(pa.binary())),
    ("outcomes", pa.list_(pa.string())),
    ("question", pa.string()),
    ("slug", pa.string()),
])


def _dec(x, places):
    return None if x is None else decimal.Decimal(str(x)).quantize(
        decimal.Decimal(1).scaleb(-places)
    )


def _row(event_type, offset_s, asset=None, bid=None, ask=None, size=None,
         side=None, market=MARKET, **extra):
    row = {
        "event_type": event_type,
        "timestamp_received": T0 + dt.timedelta(seconds=offset_s),
        "sequence": 0,
        "timestamp": T0 + dt.timedelta(seconds=offset_s),
        "market": market,
        "asset_id": asset,
        "best_bid": _dec(bid, 4),
        "best_ask": _dec(ask, 4),
        "spread": None,
        "price": None,
        "size": _dec(size, 6),
        "side": side,
        "assets_ids": None,
        "outcomes": None,
        "question": None,
        "slug": None,
    }
    row.update(extra)
    return row


@pytest.fixture
def archive_file(tmp_path):
    """One 60 s window, 10 s cells. Cell k covers [10k, 10k+10) seconds."""
    rows = [
        # market metadata
        _row("new_market", 0, assets_ids=[YES, NO], outcomes=["Yes", "No"],
             question="Will it rain?", slug="will-it-rain"),
        # cell 0: two quotes for YES -- the later one must win.
        _row("best_bid_ask", 1, YES, bid=0.40, ask=0.44),
        _row("best_bid_ask", 8, YES, bid=0.50, ask=0.52),
        # cell 0: one quote for NO.
        _row("best_bid_ask", 2, NO, bid=0.48, ask=0.50),
        # cell 1: no quotes at all -> both forward-filled.
        # cell 2: YES quote via price_change (also carries top of book).
        _row("price_change", 22, YES, bid=0.60, ask=0.62),
        # cell 2: two trades for YES.
        _row("last_trade_price", 23, YES, size=10.0),
        _row("last_trade_price", 24, YES, size=5.0),
        # cell 4: one trade for NO.
        _row("last_trade_price", 45, NO, size=7.5, side="BUY"),
        # cell 3: mixed taker sides for YES -- signed flow must net out.
        _row("last_trade_price", 31, YES, size=9.0, side="BUY"),
        _row("last_trade_price", 32, YES, size=4.0, side="SELL"),
    ]
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    path = tmp_path / "2026-08-29T12.parquet"
    pq.write_table(table, path)
    return str(path)


def test_scan_markets_finds_outcomes_and_metadata(archive_file):
    infos = scan_markets([archive_file])
    assert set(infos) == {MARKET}
    info = infos[MARKET]
    assert set(info.assets) == {YES, NO}
    assert info.question == "Will it rain?"
    assert info.outcomes == ["Yes", "No"]
    assert info.trades == 5          # five last_trade_price rows
    assert info.quotes == 4          # three best_bid_ask + one price_change
    assert info.label == "Will it rain?"


def test_panel_grid_semantics(archive_file):
    info = scan_markets([archive_file])[MARKET]
    info.assets = [YES, NO]          # pin order for assertions
    panel, = build_market_panels(
        [archive_file], [info], freq_seconds=10.0,
        start=T0, end=T0 + dt.timedelta(seconds=60),
    )
    assert panel.num_steps == 6
    yes, no = 0, 1
    mid, spread = panel.features["mid"], panel.features["spread"]

    # cell 0: last quote in the cell wins (0.50/0.52, not 0.40/0.44).
    assert mid[yes, 0] == pytest.approx(0.51)
    assert spread[yes, 0] == pytest.approx(0.02)
    # cell 1: no quote -> forward-filled from cell 0.
    assert mid[yes, 1] == pytest.approx(0.51)
    # cell 2: price_change updates the top of book.
    assert mid[yes, 2] == pytest.approx(0.61)
    # ... and holds for the rest of the window.
    assert mid[yes, 5] == pytest.approx(0.61)

    # ret is the first difference of mid; step 0 is undefined.
    ret = panel.features["ret"]
    assert np.isnan(ret[yes, 0])
    assert ret[yes, 2] == pytest.approx(0.10, abs=1e-6)
    assert panel.features["abs_ret"][yes, 2] == pytest.approx(0.10, abs=1e-6)

    # activity channels sum within the cell and are zero elsewhere.
    assert panel.features["volume"][yes, 2] == pytest.approx(15.0)
    assert panel.features["trades"][yes, 2] == pytest.approx(2.0)
    assert panel.features["volume"][yes, 0] == 0.0
    assert panel.features["quotes"][yes, 0] == pytest.approx(2.0)
    assert panel.features["quotes"][yes, 2] == pytest.approx(1.0)
    assert panel.features["volume"][no, 4] == pytest.approx(7.5)


def test_signed_flow_nets_taker_sides(archive_file):
    """BUY adds, SELL subtracts; volume stays the gross total."""
    info = MarketInfo(market=MARKET, assets=[YES, NO])
    panel, = build_market_panels(
        [archive_file], [info], freq_seconds=10.0,
        start=T0, end=T0 + dt.timedelta(seconds=60),
    )
    # cell 3 holds a 9.0 BUY and a 4.0 SELL.
    assert panel.features["volume"][0, 3] == pytest.approx(13.0)
    assert panel.features["signed_flow"][0, 3] == pytest.approx(5.0)
    assert panel.features["trades"][0, 3] == pytest.approx(2.0)
    # cell 2's trades carry no side; they must not be counted as sells.
    assert panel.features["signed_flow"][0, 2] == pytest.approx(15.0)


def test_leading_cells_stay_nan_until_first_quote(archive_file):
    """A token with no quote in early cells must stay NaN, not back-fill."""
    info = MarketInfo(market=MARKET, assets=[NO])
    panel, = build_market_panels(
        [archive_file], [info], freq_seconds=1.0,
        start=T0, end=T0 + dt.timedelta(seconds=5),
    )
    mid = panel.features["mid"][0]
    assert np.isnan(mid[0]), "no quote before t=2s, must not look ahead"
    assert np.isnan(mid[1])
    assert mid[2] == pytest.approx(0.49)
    assert mid[3] == pytest.approx(0.49)  # forward-filled


def test_forward_fill_preserves_leading_nan():
    a = np.array([[np.nan, 1.0, np.nan, np.nan, 2.0, np.nan]], dtype=np.float32)
    _forward_fill_rows(a)
    assert np.isnan(a[0, 0])
    np.testing.assert_allclose(a[0, 1:], [1.0, 1.0, 1.0, 2.0, 2.0])


def test_forward_fill_all_nan_row_is_untouched():
    a = np.full((1, 4), np.nan, dtype=np.float32)
    _forward_fill_rows(a)
    assert np.isnan(a).all()


def test_select_covered_drops_sparse_panels(archive_file):
    info = MarketInfo(market=MARKET, assets=[NO])
    panel, = build_market_panels(
        [archive_file], [info], freq_seconds=1.0,
        start=T0, end=T0 + dt.timedelta(seconds=5),
    )
    assert select_covered([panel], min_coverage=0.5) == [panel]
    assert select_covered([panel], min_coverage=0.99) == []


def test_hours_between_is_inclusive():
    hrs = hours_between(dt.datetime(2026, 8, 29, 22), dt.datetime(2026, 8, 30, 1))
    assert [h.hour for h in hrs] == [22, 23, 0, 1]


def test_archive_paths():
    a = PolymarketArchive(cache_dir="/tmp/x")
    hour = dt.datetime(2026, 8, 29, 15)
    assert a.hour_url(hour).endswith("/v3/2026-08-29/15/2026-08-29T15.parquet")
    assert a.local_path(hour).endswith("/v3/2026-08-29/15/2026-08-29T15.parquet")


def test_stack_orders_channels(archive_file):
    info = scan_markets([archive_file])[MARKET]
    info.assets = [YES, NO]
    panel, = build_market_panels(
        [archive_file], [info], freq_seconds=10.0,
        start=T0, end=T0 + dt.timedelta(seconds=60),
    )
    stacked = panel.stack(("mid", "spread"))
    assert stacked.shape == (4, 6)
    np.testing.assert_allclose(stacked[:2], panel.features["mid"])
    np.testing.assert_allclose(stacked[2:], panel.features["spread"])


# --------------------------------------------------------------------------
# Download integrity policy
# --------------------------------------------------------------------------


class _FakeTransfer:
    """Stands in for the network: writes chosen bytes on each attempt."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def __call__(self, url, dest):
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        if payload is None:
            raise OSError("short read: 1 of 2 bytes")
        with open(dest, "wb") as f:
            f.write(payload)
        return len(payload)


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """An archive whose fetches are faked and whose checksum is pinned."""
    import timesfm3.data.polymarket as pm

    def make(payloads, expected):
        transfer = _FakeTransfer(payloads)
        monkeypatch.setattr(pm, "_fetch", transfer)
        monkeypatch.setattr(pm, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))
        archive = pm.PolymarketArchive(cache_dir=str(tmp_path))
        monkeypatch.setattr(archive, "expected_sha256", lambda hour: expected)
        return archive, transfer

    return make


def _sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


HOUR = dt.datetime(2026, 8, 29, 2, tzinfo=dt.timezone.utc)


def test_good_download_is_promoted(patched):
    archive, transfer = patched([b"good"], _sha(b"good"))
    path = archive.download_hour(HOUR)
    assert open(path, "rb").read() == b"good"
    assert transfer.calls == 1


def test_cached_good_file_is_not_refetched(patched):
    archive, transfer = patched([b"good"], _sha(b"good"))
    archive.download_hour(HOUR)
    archive.download_hour(HOUR)
    assert transfer.calls == 1, "a verified cached file must not be re-downloaded"


def test_flaky_transfer_is_retried_until_it_verifies(patched):
    archive, transfer = patched([b"junk1", b"junk2", b"good"], _sha(b"good"))
    path = archive.download_hour(HOUR, attempts=3)
    assert open(path, "rb").read() == b"good"
    assert transfer.calls == 3


def test_unstable_corruption_reports_the_transfer(patched):
    archive, _ = patched([b"junk1", b"junk2", b"junk3"], _sha(b"good"))
    with pytest.raises(ValueError, match="disagreed with each other"):
        archive.download_hour(HOUR, attempts=3)


def test_stable_mismatch_is_reported_as_an_archive_disagreement(patched):
    """Every attempt agrees but the manifest doesn't: retrying cannot help."""
    archive, transfer = patched([b"served"], _sha(b"manifest"))
    with pytest.raises(ValueError, match="consistently serves"):
        archive.download_hour(HOUR, attempts=3)
    assert transfer.calls == 3


def test_stable_mismatch_can_be_accepted_explicitly(patched):
    archive, _ = patched([b"served"], _sha(b"manifest"))
    path = archive.download_hour(HOUR, attempts=2, on_mismatch="warn")
    assert open(path, "rb").read() == b"served"


def test_corrupt_cached_file_is_repaired(patched, tmp_path):
    archive, transfer = patched([b"good"], _sha(b"good"))
    path = archive.local_path(HOUR)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"corrupt")
    assert open(archive.download_hour(HOUR), "rb").read() == b"good"


def test_unverifiable_hour_is_still_downloaded(patched):
    """Hours newer than SHA256SUMS.txt have no checksum; accept with a warning."""
    archive, _ = patched([b"whatever"], None)
    assert open(archive.download_hour(HOUR), "rb").read() == b"whatever"


def test_bad_policy_rejected(patched):
    archive, _ = patched([b"x"], _sha(b"x"))
    with pytest.raises(ValueError, match="on_mismatch"):
        archive.download_hour(HOUR, on_mismatch="ignore")

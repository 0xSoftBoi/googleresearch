#!/usr/bin/env bash
# Downloads hourly Polymarket order-book parquet files from the public
# Pendulum Flow archive (https://archive.pendulumflow.com/) into
# data/polymarket/<version>/, verifying each file against SHA256SUMS.txt.
#
# Usage:
#   data/download_polymarket.sh START_HOUR END_HOUR [VERSION]
# Example (three hours of the current "military grade" V3 feed):
#   data/download_polymarket.sh 2026-08-28T00 2026-08-28T02
#
# Hours are UTC, inclusive, formatted YYYY-MM-DDTHH. VERSION defaults to v3
# (other eras: pmxt/v2, pmxt/v1, third-party/ag6).
#
# NOTE: one hour of V3 is roughly 1 GB, and large transfers do get truncated in
# flight, so every file is checked against SHA256SUMS.txt. For programmatic use
# prefer timesfm3.data.polymarket.PolymarketArchive, which additionally verifies
# before caching, retries flaky transfers, and repairs a bad cached file.
#
# A mismatch is reported, not silently ignored: the archive has been observed to
# serve at least one hour whose bytes its own manifest does not describe.
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -lt 2 ]; then
  echo "usage: $0 START_HOUR END_HOUR [VERSION]   (hours as YYYY-MM-DDTHH, UTC)" >&2
  exit 2
fi

START="$1"
END="$2"
VERSION="${3:-v3}"
BASE="https://archive.pendulumflow.com/${VERSION}"
OUT="polymarket/${VERSION}"
mkdir -p "$OUT"

# to_epoch YYYY-MM-DDTHH -> unix seconds (GNU date, or BSD/macOS date fallback).
to_epoch() {
  date -u -d "${1/T/ }:00:00" +%s 2>/dev/null || \
    date -u -j -f "%Y-%m-%dT%H" "$1" +%s
}
start_s=$(to_epoch "$START")
end_s=$(to_epoch "$END")
if [ "$start_s" -gt "$end_s" ]; then
  echo "START_HOUR is after END_HOUR" >&2
  exit 2
fi

echo "Fetching checksum manifest ..."
sums="$OUT/SHA256SUMS.txt"
curl -sSL -o "$sums" "$BASE/SHA256SUMS.txt"

verify() {  # verify RELPATH LOCALFILE
  local rel="$1" file="$2" expected
  expected=$(grep -F "  $rel" "$sums" | head -n1 | awk '{print $1}')
  if [ -z "$expected" ]; then
    echo "  ! no published checksum for $rel (hour newer than manifest?); skipping"
    return 0
  fi
  local actual
  actual=$(sha256sum "$file" | awk '{print $1}')
  if [ "$actual" != "$expected" ]; then
    echo "  ! CHECKSUM MISMATCH for $rel" >&2
    echo "      expected $expected" >&2
    echo "      actual   $actual" >&2
    echo "      re-run to retry; if it repeats, the archive and its manifest disagree" >&2
    bad=$((bad + 1))
    return 0
  fi
  echo "  ok $rel"
}

bad=0
s="$start_s"
while [ "$s" -le "$end_s" ]; do
  day=$(date -u -d "@$s" +%Y-%m-%d 2>/dev/null || date -u -r "$s" +%Y-%m-%d)
  hh=$(date -u -d "@$s" +%H 2>/dev/null || date -u -r "$s" +%H)
  rel="${day}/${hh}/${day}T${hh}.parquet"
  dest="$OUT/$rel"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    echo "have  $rel"
  else
    echo "get   $rel"
    curl -sSL -o "$dest.part" "$BASE/$rel"
    mv "$dest.part" "$dest"
  fi
  verify "$rel" "$dest"
  s=$((s + 3600))
done

if [ "$bad" -gt 0 ]; then
  echo "Done, but $bad file(s) failed verification. Files in $OUT/" >&2
  exit 1
fi
echo "Done, all files verified. Files in $OUT/"

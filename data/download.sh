#!/usr/bin/env bash
# Downloads the public benchmark datasets used by the real-data notebook
# into data/raw/ (ETT electricity-transformer data + daily exchange rates).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw
for f in ETTh1 ETTh2 ETTm1 ETTm2; do
  [ -f "raw/$f.csv" ] || curl -sL -o "raw/$f.csv" \
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/$f.csv" &
done
if [ ! -f raw/exchange_rate.txt ]; then
  curl -sL -o raw/exchange_rate.txt.gz \
    "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz" &
fi
wait
[ -f raw/exchange_rate.txt.gz ] && gunzip -f raw/exchange_rate.txt.gz
ls -la raw

for f in electricity/electricity solar-energy/solar_AL traffic/traffic; do
  n=$(basename "$f")
  [ -f "raw/$n.txt" ] || { curl -sL -o "raw/$n.txt.gz" \
    "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/$f.txt.gz" \
    && gunzip -f "raw/$n.txt.gz"; } &
done
wait
ls -la raw

#!/usr/bin/env bash
set -euo pipefail

# Base URL for the pbp release assets
BASE_URL="https://github.com/nflverse/nflverse-data/releases/download/pbp"

# Directory to store downloaded parquet files
OUT_DIR="./data/raw"
mkdir -p "$OUT_DIR"

# Change these if you only want a subset of years
START_YEAR=1999
END_YEAR=2025

for year in $(seq "$START_YEAR" "$END_YEAR"); do
  file="play_by_play_${year}.parquet"
  url="${BASE_URL}/${file}"

  echo "Downloading ${file} ..."
  # -L follows redirects, -f fails on HTTP errors, -sS is quiet but shows errors
  if curl -fLsS "$url" -o "${OUT_DIR}/${file}"; then
    echo "  -> saved to ${OUT_DIR}/${file}"
  else
    echo "  -> ${file} not found, skipping."
  fi
done

echo "Done."

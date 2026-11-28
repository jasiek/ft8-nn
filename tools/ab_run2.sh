#!/bin/bash
# Quick A/B for a decoder change: 100 noise-only slots (false decodes) + validation block set
# (default SET=small: 16 blocks / 128 scored slots across all 10 bands and 6 continents; set SET=third
# for the 265-block / 2,120-slot set before committing), scored against PSK Reporter + cross-receiver xconf.
#
#   tools/ab_run2.sh nn_v4_xx [--erase-osd-dmax 3 ...]
#
# Keep a change only if, vs the current default tag: decodes and confirmed (slot + xconf) rise, unconf does
# not rise, and noise stays at 0 false. Then run tools/validate_full.sh before committing.
# Env: V (validation set dir), SET (default small), WORKERS (default 6), DECODER (default ft8nn_py; set
# DECODER=ft8nn to run the C CLI). Logs: data/logs/<tag>/.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$1; shift
V=${V:-$PWD/datasets/ft8-validation-set}
SET=${SET:-small}
DECODER=${DECODER:-ft8nn_py}
W=${WORKERS:-6}
L=data/logs/$TAG; mkdir -p "$L"

uv run python bench/run_ours.py data/bench/noise --model models/v4/model.pt --also-classical \
    --tag "$TAG" --workers "$W" "$@" > "$L/noise.log" 2>&1
V="$V" uv run python tools/run_set.py --set "$SET" --decoder "$DECODER" --tag "$TAG" \
    --workers "$W" --verify --list -- "$@" > "$L/set_${SET}.txt" 2>&1

echo "== $TAG noise (want 0 false)"; tail -1 "$L/noise.log"
echo "== $TAG set $SET"; cat "$L/set_${SET}.txt"

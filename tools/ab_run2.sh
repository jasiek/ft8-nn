#!/bin/bash
# Quick A/B for a decoder change: 100 noise-only slots (false decodes) + validation run 2 (unseen receivers),
# scored against PSK Reporter spots. Extra args go to both bench scripts, e.g. --erase-osd-dmax 0.
#
#   tools/ab_run2.sh nn_v2_xx [--erase-osd-dmax 3 ...]
#
# Keep a change only if, vs the current default tag: run 2 decodes and slot-confirmed rise, unconf does not
# rise, noise stays at 0 false. Then run tools/validate_full.sh before committing.
# Env: V (validation set dir), WORKERS (default 6). Logs: data/logs/<tag>/.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$1; shift
V=${V:-$(cd .. && pwd)/ft8-validation-set}
W=${WORKERS:-6}
L=data/logs/$TAG; mkdir -p "$L"
M="--model models/v2/model.pt --also-classical --tag $TAG --workers $W"

uv run python bench/run_ours.py data/bench/noise $M "$@" > "$L/noise.log" 2>&1
uv run python bench/run_real.py "$V/run2/recordings" $M "$@" --out data/val/run2 > "$L/run2.log" 2>&1
V="$V" python3 collect/verify.py --recordings "$V/run2/recordings" --spots "$V/run2/pskr_spots.jsonl" \
    --decodes data/val/run2 --tag "$TAG" --list > "$L/verify_run2.txt" 2>&1

echo "== $TAG noise (want 0 false)"; tail -1 "$L/noise.log"
echo "== $TAG run 2"; grep -E "^station|^TOTAL" "$L/verify_run2.txt"
sed -n '/^unconfirmed:/,$p' "$L/verify_run2.txt"

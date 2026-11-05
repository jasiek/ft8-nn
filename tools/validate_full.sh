#!/bin/bash
# Full validation of a decoder change before committing (after tools/ab_run2.sh passed). Takes ~1.5 h on an
# idle M1 Pro with 6 workers. Steps: validation run 1 + spots, training recordings vs jt9, AWGN sensitivity,
# C build, C == Python golden tests (2 training stations, 1 run-2 station with early-DT signals, 60 AWGN files).
#
#   tools/validate_full.sh nn_v2_xx [decoder flags for the bench scripts]
#
# Pass if: run 1 decodes/slot-confirmed rise and unconf does not; training decodes rise; AWGN 50% not worse
# (−22.37 dB as of nn_v2_f31); every golden line says identical. Then set tools/golden_c.py's default --tag,
# update STATUS.md, commit.
# Env: V, T (validation set dir, training recordings dir), WORKERS (default 6). Logs: data/logs/<tag>/.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$1; shift
V=${V:-$(cd .. && pwd)/ft8-validation-set}
T=${T:-$(cd .. && pwd)/ft8-training-data/recordings}
W=${WORKERS:-6}
L=data/logs/$TAG; mkdir -p "$L"
M="--model runs/v2/model.pt --also-classical --tag $TAG --workers $W"
step() { echo "[$(date +%H:%M)] $*"; }

step "validation run 1";  uv run python bench/run_real.py "$V/recordings" $M "$@" --out data/val/run1 > "$L/run1.log" 2>&1
python3 "$V/tools/verify.py" --decodes data/val/run1 --tag "$TAG" --list > "$L/verify_run1.txt" 2>&1
step "training recordings"; uv run python bench/run_real.py "$T" $M "$@" > "$L/train.log" 2>&1
uv run python bench/run_real.py "$T" --compare-only --tag "$TAG" > "$L/train_cmp.txt" 2>&1
step "AWGN";              uv run python bench/run_ours.py data/bench/awgn $M "$@" > "$L/awgn.log" 2>&1
uv run python bench/report.py data/bench/awgn --dt-max 0.5 > "$L/awgn_report.txt" 2>&1
step "C build + golden";  make -C c > "$L/make.log" 2>&1
# macOS malloc fills new and freed blocks with garbage: a read of uninitialised or freed memory then changes the
# output reliably instead of now and then (found a use-after-free of the candidate list this way, 2026-11-05)
export MallocScribble=1 MallocPreScribble=1
uv run python tools/golden_c.py real "$T/00_40m_kiwisdr-dynpc-net-8073" --n 20 --tag "$TAG" > "$L/golden_00_40m.txt" 2>&1
uv run python tools/golden_c.py real "$T/08_80m_178-115-248-93-8073" --n 20 --tag "$TAG" > "$L/golden_08_80m.txt" 2>&1
uv run python tools/golden_c.py real "$V/run2/recordings/02_80m_82-65-236-171-8073" --n 20 --tag "$TAG" \
    --out data/val/run2 > "$L/golden_run2_02_80m.txt" 2>&1
uv run python tools/golden_c.py awgn data/bench/awgn --n 60 --workers "$W" --tag "$TAG" > "$L/golden_awgn.txt" 2>&1

step "done; summaries:"
grep -E "^TOTAL" "$L/verify_run1.txt"; sed -n '/^unconfirmed:/,$p' "$L/verify_run1.txt"
head -20 "$L/train_cmp.txt"
grep -E "SNR|50%" "$L/awgn_report.txt"
tail -n 3 "$L"/golden_*.txt

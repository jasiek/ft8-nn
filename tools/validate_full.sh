#!/bin/bash
# Full validation of a decoder change before committing (after tools/ab_run2.sh passed).
# Steps: validation set `sets/third.txt` (265 blocks, 2,120 scored slots across runs/r02, r05-r14) + PSK Reporter
# & cross-receiver xconf, training recordings vs jt9, AWGN sensitivity, C build, and C == Python golden tests
# (2 training stations, 1 unseen validation station from runs/r07, 60 AWGN files).
#
#   tools/validate_full.sh nn_v4_xx [decoder flags for the Python bench scripts]
#
# Pass if: third-set decodes and confirmed (slot + xconf) rise and unconf does not; training decodes do not drop;
# AWGN 50% not worse (−22.55 dB as of nn_v4); every golden line says identical. Then set tools/golden_c.py's
# default --tag, update STATUS.md, commit.
# Env: V, T (validation set dir, training recordings dir), SET (default third), WORKERS (default 6),
# ONNX (C-side model of the golden tests, default models/v4/ft8nn.onnx). Logs: data/logs/<tag>/.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$1; shift
V=${V:-$PWD/datasets/ft8-validation-set}
T=${T:-$PWD/datasets/ft8-training-data/recordings}
SET=${SET:-third}
W=${WORKERS:-6}
G=${ONNX:+--onnx $ONNX}
L=data/logs/$TAG; mkdir -p "$L"
M="--model models/v4/model.pt --also-classical --tag $TAG --workers $W"
step() { echo "[$(date +%H:%M)] $*"; }

step "validation set ($SET)"; V="$V" uv run python tools/run_set.py --set "$SET" --decoder ft8nn_py \
    --tag "$TAG" --workers "$W" --verify --list -- "$@" > "$L/verify_${SET}.txt" 2>&1
step "training recordings"; uv run python bench/run_real.py "$T" $M "$@" > "$L/train.log" 2>&1
uv run python bench/run_real.py "$T" --compare-only --tag "$TAG" > "$L/train_cmp.txt" 2>&1
step "AWGN";              uv run python bench/run_ours.py data/bench/awgn $M "$@" > "$L/awgn.log" 2>&1
uv run python bench/report.py data/bench/awgn --dt-max 0.5 > "$L/awgn_report.txt" 2>&1
step "C build + golden";  make -C c > "$L/make.log" 2>&1
# macOS malloc fills new and freed blocks with garbage: a read of uninitialised or freed memory then changes the
# output reliably instead of now and then (found a use-after-free of the candidate list this way, 2026-11-05)
export MallocScribble=1 MallocPreScribble=1
uv run python tools/golden_c.py real "$T/00_40m_kiwisdr-dynpc-net-8073" --n 20 --tag "$TAG" $G > "$L/golden_00_40m.txt" 2>&1
uv run python tools/golden_c.py real "$T/08_80m_178-115-248-93-8073" --n 20 --tag "$TAG" $G > "$L/golden_08_80m.txt" 2>&1
uv run python bench/run_real.py "$V/runs/r07/recordings" --stations 00_40m $M "$@" --out data/val/r07 > "$L/r07_00_40m.log" 2>&1
uv run python tools/golden_c.py real "$V/runs/r07/recordings/00_40m_fenuradio-kiwi2-ddns-net-8074" --n 20 --tag "$TAG" $G \
    --out data/val/r07 > "$L/golden_r07_00_40m.txt" 2>&1
uv run python tools/golden_c.py awgn data/bench/awgn --n 60 --workers "$W" --tag "$TAG" $G > "$L/golden_awgn.txt" 2>&1

step "done; summaries:"
cat "$L/verify_${SET}.txt"
head -20 "$L/train_cmp.txt"
grep -E "SNR|50%" "$L/awgn_report.txt"
tail -n 3 "$L"/golden_*.txt

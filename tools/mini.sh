#!/bin/bash
# Mirror this checkout + data + Claude Code memory to a second Mac (ssh alias "mini") and run experiments there.
# One-time setup: docs/mac-mini.md.
#
#   tools/mini.sh push                         code (incl. uncommitted changes), model, data, memory; uv sync; make
#   tools/mini.sh run tools/ab_run2.sh TAG ... run a command in the mini's checkout (in the foreground)
#   tools/mini.sh pull                         copy decodes/verified/results files and logs back here
#
# rsync -u never overwrites a newer file on the receiving side, so results produced on either machine survive.
# The data submodules (datasets/, unpacked) go with the checkout, so the mini needs no LFS download.
set -euo pipefail
H=${MINI:-mini}
REPO=$(cd "$(dirname "$0")/.." && pwd)
key() { echo "$1" | tr / -; }   # Claude Code's per-project dir name: absolute path with / -> -
RPATH='export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH'
RESULTS=(--include '*/' --include 'decodes_*' --include 'verified_*' --include 'results_*' --exclude '*' -m)

case ${1:-} in
push)
  rsync -a -u --exclude .venv --exclude c/build --exclude __pycache__ "$REPO/" "$H:tmp/ft8-nn/"
  RH=$(ssh "$H" 'echo $HOME')
  RMEM=".claude/projects/$(key "$RH/tmp/ft8-nn")/memory"
  ssh "$H" "mkdir -p '$RMEM'"
  rsync -a -u "$HOME/.claude/projects/$(key "$REPO")/memory/" "$H:$RMEM/"
  ssh "$H" "$RPATH; cd tmp/ft8-nn && uv sync -q && tools/get_onnxruntime.sh >/dev/null && make -s -C c"
  echo "pushed to $H:~/tmp/ft8-nn"
  ;;
run)
  shift
  ssh "$H" "$RPATH; cd tmp/ft8-nn && $*"
  ;;
pull)
  rsync -a -u "${RESULTS[@]}" "$H:tmp/ft8-nn/data/" "$REPO/data/"
  rsync -a -u "$H:tmp/ft8-nn/data/logs/" "$REPO/data/logs/" 2>/dev/null || true
  echo "pulled results from $H"
  ;;
*)
  sed -n '2,10p' "$0"; exit 1
  ;;
esac

# Working on ft8-nn

Start by reading `STATUS.md`: "In flight" and the numbered plan say where things stand; results, reproduce commands
and past decisions are there too. `README.md` covers setup; `docs/data-collection.md` how to get the recordings.

## Working agreements with the user

- **Ask before every commit and every push.** An autonomous loop (commits without asking) needs the user's explicit
  OK with stop rules; record it in STATUS.md and go back to asking when it ends.
- **After every commit, update the STATUS.md plan** (done / results / next), preferably in the same commit.
- **Explain in detail:** lead with the current state or next action, then what the problem was, why it happens, what
  the change does, how it was tested and what the numbers mean. Define terms on first use (candidate, DT, LLR, BP,
  OSD, a-priori/AP, slot-confirmed / unconfirmed). Tags like `nn_v2_rs` are labels, never the explanation.
- **Whenever a run starts, say how long it will take and the clock time it should finish**, from the last comparable
  run (file times in `data/logs/<tag>/`); say when the estimate is shaky.
- Don't build or compile third-party C without asking. On macOS `c/src/nn_accel.c` runs `SoftDemod` via Apple
  `-framework Accelerate` (`cblas_sgemm` AMX) with ONNX Runtime fallback; elsewhere (Linux/Windows, or
  `FT8NN_BACKEND=ort` on macOS) keep ONNX Runtime (the official build from `tools/get_onnxruntime.sh`; Homebrew's is
  2.3x slower) paired with `models/v4/screener.onnx`. The FFT and everything else in `c/` is our own code.

## Data and evaluation rules

- The recordings are git submodules (Git LFS, packed): `datasets/ft8-training-data`, `datasets/ft8-validation-set`.
  Fetch and unpack them as in `docs/data-collection.md`. They are read-only; outputs go under `data/` (gitignored).
- The validation set (`datasets/ft8-validation-set/runs/r02`, `runs/r05`–`r14`, block lists `sets/{small,third,full}.txt`,
  all unseen receiver sites) is the honest test: never train or tune thresholds on it (nor on archived v1 runs 2–4).
- Decoder changes: Python first (`ft8nn/`), then the C port (`c/`); C must equal Python (`tools/golden_c.py`).
  Accept a change with `tools/ab_run2.sh TAG` (`SET=small` quick check, `SET=third` before committing: decodes and
  confirmed `slot + xconf` up, `unconf` not up, noise 0 false), then `tools/validate_full.sh TAG` (`sets/third.txt`,
  training recordings, AWGN, all golden tests).
- `models/v4` is the shipped model (`models/v3` is kept as its starting point). New trainings write to `runs/`
  (gitignored). Retraining moves unconfirmed decodes by several (1 vs 5 for the same recipe), so a new network
  needs the AP/OSD gates re-checked; judge false decodes on `sets/third.txt` (or `full.txt`) and `data/bench/noise10k`.
- The data repos (Git LFS) share 10 GiB/month of download bandwidth: avoid needless full clones.

## Machine notes (two Mac minis: an 8-core M1 and an M4, ssh alias `mini`, driven by `tools/mini.sh`)

- `sysctl -n machdep.cpu.brand_string` says which one you are on; timings differ ~1.8x, so always name the machine.

- Long runs: start only when the Mac is fully awake and wrap with `caffeinate -dims`; afterwards check
  `pmset -g log` for Sleep/DarkWake (dark-wake runs crawl and give garbage timings).
- Timing only on an idle machine, one job at a time; never run training and benchmarks together.
- Local CI: `act` + Colima (`colima start` after a reboot); the command is in STATUS.md.

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
- Don't build or compile third-party C without asking. Keep ONNX Runtime for the NN (the official build from
  `tools/get_onnxruntime.sh`; Homebrew's is 2.3x slower). The FFT and everything else in `c/` is our own code.

## Data and evaluation rules

- The recordings are git submodules (Git LFS, packed): `datasets/ft8-training-data`, `datasets/ft8-validation-set`.
  Fetch and unpack them as in `docs/data-collection.md`. They are read-only; outputs go under `data/` (gitignored).
- Validation runs 2 and 3 (unseen receivers) are the honest tests: never train or tune thresholds on them.
- Decoder changes: Python first (`ft8nn/`), then the C port (`c/`); C must equal Python (`tools/golden_c.py`).
  Accept a change with `tools/ab_run2.sh TAG` (run 2 decodes and slot-confirmed up, unconfirmed not up, noise 0
  false), then `tools/validate_full.sh TAG` (run 1, training recordings, AWGN, all golden tests).
- `models/v2` is the shipped model. New trainings write to `runs/` (gitignored). Retraining moves run-2 unconfirmed
  decodes by several (1 vs 5 for the same recipe), so a new network needs the AP/OSD gates re-checked.
- The data repos (Git LFS) share 10 GiB/month of download bandwidth: avoid needless full clones.

## Machine notes (Mac mini M4)

- Long runs: start only when the Mac is fully awake and wrap with `caffeinate -dims`; afterwards check
  `pmset -g log` for Sleep/DarkWake (dark-wake runs crawl and give garbage timings).
- Timing only on an idle machine, one job at a time; never run training and benchmarks together.
- Local CI: `act` + Colima (`colima start` after a reboot); the command is in STATUS.md.

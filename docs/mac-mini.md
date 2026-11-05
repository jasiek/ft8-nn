# Running experiments on the Mac mini

The decoder benchmarks are CPU-bound (about 80% of a slot is NumPy/BP/OSD, only the NN could use a GPU), so a second
idle Mac roughly doubles experiment throughput. The main machine is an M1 Pro (8P+2E, 16 GB); the mini is a
Mac mini (2024), M4 (4P+6E) or M4 Pro (8–10P+4E). Folder layout is the same on both machines:

```
~/tmp/ft8-nn                          this repo
~/tmp/ft8-validation-set              KiwiSDR recordings + PSK Reporter spots (read-only)
~/tmp/ft8-training-data/recordings    another agent's folder (read-only; the mini holds a copy)
```

## One-time setup

### On the mini (at its keyboard)

1. System Settings → General → Sharing → **Remote Login** on.
2. `scutil --get LocalHostName; whoami; sysctl -n machdep.cpu.brand_string` (note the host name and user).
3. Homebrew (also installs the compiler, git, make), then uv and ONNX Runtime:
   ```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"
   brew install uv onnxruntime
   ```

### On the main machine

4. A plain SSH key for the mini (the default key is a hardware security key that needs a touch per connection,
   which blocks unattended runs). Replace HOST and USER:
   ```sh
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_mini -N "" -C "ft8nn mini"
   cat >> ~/.ssh/config <<'EOF'

   Host mini
     HostName HOST.local
     User USER
     IdentityFile ~/.ssh/id_ed25519_mini
     IdentitiesOnly yes
   EOF
   ssh-copy-id -i ~/.ssh/id_ed25519_mini.pub mini
   ssh mini 'uname -m; sysctl -n machdep.cpu.brand_string hw.ncpu; /opt/homebrew/bin/uv --version'
   ```
5. Copy everything and build (about 3 GB the first time; later pushes only send changes):
   ```sh
   tools/mini.sh push
   ```
   This copies the working tree **including uncommitted changes** (no git push needed), the model in `runs/`,
   `data/`, the validation set, the training recordings, and the Claude Code memory for this project
   (`~/.claude/projects/<path>/memory/`). Then it runs `uv sync` and `make -C c` on the mini.
6. Smoke test (C library + ONNX model):
   ```sh
   tools/mini.sh run './c/build/ft8nn_cli runs/v2/ft8nn.onnx $(ls ~/tmp/ft8-validation-set/recordings/02_40m_kiwisdr-k1ra-us-8073/slots/*.wav | sed -n 6p) | head -5'
   ```
7. Check that both machines decode identically before trusting any A/B from the mini: decode one receiver there
   with the current default tag and diff against this machine's file:
   ```sh
   tools/mini.sh run 'uv run python bench/run_real.py ~/tmp/ft8-validation-set/run2/recordings --stations 07 --model runs/v2/model.pt --also-classical --tag mini_check --workers 6 --out data/val/run2'
   tools/mini.sh pull
   diff <(cut -f1-5 data/val/run2/07_12m_203-129-16-92-8073/decodes_mini_check.txt) \
        <(cut -f1-5 data/val/run2/07_12m_203-129-16-92-8073/decodes_nn_v2_eg.txt) && echo IDENTICAL
   ```
   (Use the current default tag from STATUS.md in place of `nn_v2_eg`.)

## Working on the mini directly

Everything also runs in the mini's own checkout (no SSH layer). Two extra setup steps there: `.python-version`
pins Python 3.12 (uv otherwise picks 3.15, which has no onnxruntime wheel), and `brew install onnxruntime` for
the C library. WSJT-X is absent, so the 16 tests that call `ft8code` fail; the rest pass. Speed: about 2 s per
slot per core; `WORKERS=10` uses all 10 cores (4P+6E). Quick A/B about 1 h, full check about 1.5 h. Decodes were
verified identical to the main machine (run 2 receiver 07, 382/382 lines). macOS background work (Siri, Spotlight
indexing after a big copy) can slow a run several-fold: check `uptime` before trusting a timing.

## Day to day

```sh
tools/mini.sh push                                         # after every code change
tools/mini.sh run 'tools/ab_run2.sh nn_v2_xx'              # quick A/B: noise + validation run 2
tools/mini.sh run 'tools/validate_full.sh nn_v2_xx'        # full check before a commit
tools/mini.sh pull                                         # bring decodes + logs back; score/compare here
```

Long runs: start them under `nohup` so a closed SSH session does not kill them, e.g.
`tools/mini.sh run 'nohup tools/validate_full.sh nn_v2_xx > data/logs/nn_v2_xx.out 2>&1 &'`.
Set `WORKERS=` for the mini's core count (M4: 8, M4 Pro: 10–12).

Commits are made on the main machine only, so there is a single git history; the mini is a compute box.

## Resuming the Claude Code work on the mini itself

To work there directly rather than drive it from the main machine:

1. Install Claude Code on the mini, `cd ~/tmp/ft8-nn`, run `claude`.
2. Say: "carry on with the plan in STATUS.md". The memory copied by `tools/mini.sh push` holds the working
   preferences (explain in detail, update STATUS after each commit, read-only folders, compare with jt9 by exact text).
3. Not available on the mini: WSJT-X (`jt9`, `ft8sim`). Only needed to make new synthetic benchmark sets or new jt9
   baselines; the existing ones are in `data/bench` and the recordings folders.

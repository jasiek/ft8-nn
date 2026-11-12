# Status and plan (resume here)

Last updated 2026-11-12 (**repeat stage scores only competitors a station could send, tag `nn_v3_ccrpc`** (plan
12): a CQ's competitors no longer include `CQ K1ABC R FN42` or `CQ K1ABC RR73/-12`, which had beaten the margin on
226 of the 320 repeats lost that way; a CQ-only sender then needs ρ ≥ 0.50 (`REPEAT_SOLO` 0.35), set on the run-1
wrong-receiver null (0 of 223,643, as before); runs 2–4 pooled **107,108 / 78,782 / 4** (+557 decodes / +458
slot-confirmed over `nn_v3_cc`, the same four unconfirmed, 0 stale of 319 checkable; +24.2% / +23.2% over jt9,
+14.7% over JTDX's defaults, +9.6% over its ceiling); run 1 34,727 / 26,615 / 1 (+142 / +110); C golden PASS;
`validate_full.sh nn_v3_ccrpc` PASS. Also the **runs 2–4 miss profile** (`docs/miss-profile-dtcrp.md`): the
2,495 slot-confirmed pairs jt9, JTDX or deep JTDX get and `nn_v3_dtcrp` missed, by mechanism and ranked lever; and
the 12 doubtful `rp` decodes checked in the audio: none false. Before that **co-channel retry shipped, tag `nn_v3_cc`** (plan 7.1 item 1c): after the last pass, every
decode's own cell is searched in the residual with a data-symbol sync metric for a second station whose sync the
subtraction took away; runs 2–4 pooled **106,551 / 78,324 / 4** (+477 decodes / +302 slot-confirmed over
`nn_v3_dtcrp`, unconfirmed the same four; +23.5% / +22.5% over jt9, +14.1% over JTDX's defaults, +9.0% over its
ceiling); the plausibility rules now also check non-standard-call, DXpedition, RTTY Roundup and EU VHF messages
(plan 7.4); C port golden PASS (1 thread and 4 threads streaming); `validate_full.sh nn_v3_cc` PASS: run 1 34,585 / 26,505 / 1 (+290 / +206), training recordings 18,489 -> 18,563 (jt9-only 209 -> 197), AWGN identical. Before that **README gains a JTDX section**: per-run tables of jt9 / JTDX defaults / JTDX deep / `nn_v3` /
`nn_v3_dtcrp` and ft8nn's margin over JTDX. Before that **rebased onto `main` and measured together with DT re-centring**: tag `nn_v3_dtcrp`, runs
2–4 pooled **106,074 / 78,022 / 4** (+23.0% / +22.0% over jt9; +13.6% over JTDX's defaults, +8.6% over its ceiling),
run 1 / training / AWGN identical to `nn_v3_rpf`, golden PASS on the threaded C (1 thread and 4 threads streaming).
Before that **repeat stage shipped** (plan 12, tag `nn_v3_rpf`, Python + C, golden PASS): a context
message from 30/60 s earlier is tested as a whole known codeword on the residual; runs 2–4 pooled 100,630 → **105,154
decodes (+4.5%)**, slot-confirmed 73,876 → **77,369 (+4.7%)**, unconf 3 → 4 (the new one an ordinary `osd0`, not
the stage); now +21.9% over jt9, +12.6% over JTDX's defaults and +7.6% over JTDX's ceiling. Before that **where the
JTDX-only confirmed decodes come from** (plan 12): 55% are exact repeats of our own decodes 30/60 s earlier that
context AP misses. Before that **compared against JTDX and FT8TW on the unseen validation runs** (plan 11,
`docs/decoder-comparison.md`): pooled over runs 2+3+4, `nn_v3` is +7.8% decodes / +6.8% slot-confirmed over JTDX
2.2.x at its defaults, which is itself +8.3% over stock jt9, and **+3.0% / +2.4% over JTDX at its ceiling**
(`-w -c 3 -s 2`, now measured on all three runs: 44 unconfirmed against our 3); FT8TW publishes no decoder source, so
ft8_lib stands in for it and finds half as much as we do. Before that **decoding runs on every performance core and streams within a pass**, plan 6 step (3):
same decodes in every field, 0.3% more instructions; 40 busy run-1 `01_40m` slots through the sidecar on the M4 mini
with other jobs running, median time to first decodes 5.20 -> 0.33 s, whole slot 13.0 -> 4.6 s, every decode within
WSJT-X's 15 s. Before that **Windows CI added**: x64 MinGW-w64 build, C == Python 57/57 on
synthetic slots, plan 3c. Before that **decodes now stream to WSJT-X pass by pass**, plan 6 step (1): on 40 busy slots
median time to first decodes 14.05 -> 5.26 s, and decodes shown within WSJT-X's 15 s timeout 79.5% -> 98.0% on
this M1. Before that **remaining misses re-profiled, plan 7.1 item 1c**: 96-98% of jt9-only decodes are spot-confirmed real; the biggest fixable class is co-channel collisions, ~400 pooled, diagnosed but parked because the whole jt9-only pool is now ~1.3% of our decodes - **next is low-latency decoding, plan 6**. Before that **DT re-centring shipped**, `dt_recenter=0.8`, tag `nn_v3_dtc`, plan 7.1 item 1b: the
candidate window follows the median DT of the decodes already in hand, so receivers whose clock or audio stream is
off are decoded instead of half-missed - +504 decodes on run 4's `08_12m` and +305 on run 3's `03_20m`, pooled runs
2+3+4 now +17.6% over jt9 with unconf still 3, and run 1 / training / AWGN bit-identical because the gate cannot arm
there. The user authorised this one commit and the fast-forward merge into `main` conditional on `validate_full`
passing, which it did; asking resumes for every commit after it. Before that **the two fading sets re-measured with `models/v3`**: moderate −20.12, disturbed −20.16,
so 0.72 and 0.97 dB over jt9, and the README table is now all current measurements. Before that **validation run 4
scored with `nn_v3`**: 45,744 decodes vs jt9's 38,432, +19.0%, 2
unconfirmed; the results vs stock jt9 are now summarised in `README.md`. Before that **`wsjtx/ft8nn_sidecar`**:
the decoder as WSJT-X's external FT8 decoder,
`WSJTX_FT8_DECODER`, plan 9; before that **`models/v3` shipped**: v2 fine-tuned with real receiver noise, tag `nn_v3`, plan 7.5,
validation run 3 and a 10,000-slot noise-only set, data repos as submodules in `datasets/`).

## Improve loop (2026-11-05 on the Mac mini, user-approved) — FINISHED 18:40, stop condition (1) met

Each iteration: one idea in Python + C → `tools/ab_run2.sh TAG` → if it passes, `tools/validate_full.sh TAG` → commit
+ update this file (commits without asking are approved for this loop only). Stop at the first of:
(1) `nn_v2_eh` committed **and** run 2 slot-confirmed ≥ 19,500 with unconf ≤ 1 and noise 0 false;
(2) 3 consecutive ideas fail the A/B; (3) about 8 h of loop time. Log every tried idea and its numbers here.
Running directly on the mini (M4 4P+6E, `WORKERS=10`): quick A/B ~50 min, full check another ~1.5 h
(background Spotlight/Siri load can slow it a lot).
Local CI: `act push -j build-test --matrix os:ubuntu-24.04-arm -P ubuntu-24.04-arm=catthehacker/ubuntu:act-24.04
--container-architecture linux/arm64 --container-daemon-socket -` (Colima running; DOCKER_HOST=unix://$HOME/.colima/default/docker.sock).
Mini needs `.python-version` 3.12 and `tools/get_onnxruntime.sh` (official ONNX Runtime); WSJT-X 3.0.2 (Improved) installed, all 26 tests pass;
decodes verified identical to the main machine.

## In flight

- **Validation set expanded 2026-11-09.** Run 3 (`$V/run3`, data repo `794a63d`): 12 fresh Kiwis, none at a training,
  run-1 or run-2 site, Mon 05:32–06:02 UTC (run 2 was Wed 22:30): Oceania/Asia/Qatar daytime 20/15/10 m, Europe dawn
  40 m and 60 m (new band), western USA 80 m. jt9 23,524 decodes / 17,888 slot-confirmed / 17 unconf, the size of run 2
  (24,291 / 17,993 / 16). `nn_v2_pl`: 26,535 / 19,982 / 1 (`CQ K39HUN FG27`), `nn_v3` 26,688 / 20,105 / 0; noise10k 0 false for both. Treat it like run 2: never train or tune on it. Judge
  false-decode changes on runs 2–4 pooled (jt9 62 unconf of 84,618 checkable decodes, `nn_v3` 3). Noise-only: `data/bench/noise10k` (10,000 slots, seed 19, README); the
  `data/bench/noise` set is 100 slots (its "998" counts the invisible −80 dB signals, 10 per slot), so 0 false there
  only bounds the rate below ~3 per 100 slots; 0 on noise10k bounds it below ~3 per 10,000. Run 4 (`$V/run4`, data
  repo `0635402`, unpacked here 2026-11-09):
  15 more fresh Kiwis (no training, run-2 or run-3 site), Mon 15:00–15:30 UTC: European afternoon / US morning on
  20/17/15/12/10 m, Alberta 40 m, Okayama 80 m. jt9 38,432 / 28,053 / 29 unconf (all garbage calls), the biggest run;
  runs 2–4 together are about 3.5× run 2's checkable decodes. **Scored with `nn_v3` 2026-11-09:** 45,744 decodes /
  33,070 slot-confirmed / 2 unconf (`CQ U87GYB/R LC30` at 07_20m, `TNX 73 GL` at 09_15m), i.e. +19.0% decodes and
  +17.9% slot-confirmed over jt9, the largest gain of the four runs (its afternoon high bands are also the busiest
  audio we have, up to 45 decodes per slot); jt9-only 948, 59 of them at ≥ 0 dB; ours-only 8,259, 99% naming a
  callsign jt9 heard at that receiver. Decoding it took 1 h 40 min for 1,790 slots with `--workers 6` on an
  8-core M1 (not the mini; 35 min of that went on the three busiest stations alone, at ~45 decodes per slot), so
  estimate a run from its decodes per slot, not its slot count.
- **Compared against JTDX and FT8TW 2026-11-10 (plan 11).** Report: `docs/decoder-comparison.md`. Runs 2+3+4
  pooled (4,302 slots, 36 unseen receivers), the same slots and the same scorer as the `nn_v3` and jt9 numbers
  above, so the rows are directly comparable: ft8_lib 49,835 decodes / 37,113 slot-confirmed / 4 unconf; jt9
  86,247 / 63,934 / 62; **JTDX at its own defaults 93,392 / 69,159 / 11; `nn_v3` 100,630 / 73,876 / 3**, i.e. ours
  is +7.8% decodes and +6.8% slot-confirmed over JTDX while JTDX is +8.3% over jt9 — about half of our margin over
  WSJT-X is margin JTDX also finds. Per run, ours vs JTDX: run 2 +7.2%/+6.5%, run 3 +5.6%/+5.1%, run 4
  +9.4%/+8.1%. **JTDX at its ceiling** (`-w -c 3 -s 2`: SWL mode, 9 subtraction passes, `lft8subpass`), completed
  on all three runs 2026-11-10 on the Mac mini (whole comparison re-run there: jt9, JTDX defaults, `nn_v3` and runs
  2–3 deep all reproduced the first machine's numbers exactly), buys **+4.6% decodes / +4.3% slot-confirmed**
  over its own defaults for 5× the CPU (20.5 vs 4.0 s/slot/core on the M4; 71.8 + 74.2 + 89.7 min), and pays for
  it in false decodes: unconf 1→16 (run 2), 3→9 (run 3), 7→19 (run 4). Pooled: deep JTDX 97,711 / 72,160 / **44**,
  defaults 93,392 / 69,159 / 11, jt9 86,247 / 63,934 / 62, `nn_v3` 100,630 / 73,876 / **3** — ours is **+3.0%
  decodes / +2.4% slot-confirmed** ahead of JTDX's ceiling (run 2 +2.8%, run 3 +1.4%, run 4 +4.0%) with a
  fifteenth of its false decodes. **Overlap** (slot, sending callsign), `tools/compare_decoders.py`: vs default
  JTDX, both 87,232, JTDX-only 4,955 of which **3,069 slot-confirmed**, ours-only 11,413 (7,840 slot-confirmed);
  vs deep JTDX, deep-only 6,035 (3,780 slot-confirmed, 38 unconf), ours-only 8,270 (5,577, 2 unconf). The
  JTDX-only slot-confirmed decodes are real signals a classical decoder gets from the same audio and we miss —
  the best list of recoverable misses we have (plan 7.1 territory).
  Two findings worth keeping: (a) over half of JTDX's output (`^`, 53.8%) comes from its `ft8sd1`/`ft8mf1`/
  `ft8mfcq` tone-matching stages, which bypass the LDPC CRC entirely, and PSK Reporter contradicted **none** of
  50,350 checkable ones — every false decode JTDX makes comes from its ordinary BP/OSD path; (b) JTDX misses 3,449
  of jt9's callsign-slots, **228 of them at ≥ 0 dB**, the same class of strong miss as plan 7.1 — strong signals
  two independent decoders disagree on are worth a look.
- **Fading sets re-measured 2026-11-10.** `models/v3` on regenerated moderate and disturbed sets (seeds 2, 3):
  50% −20.12 and −20.16, 90% −17.48 and −18.09, 0 false decodes; the AWGN control reproduced −22.49 / −21.24
  exactly, so the regenerated sets equal the originals and the recorded jt9 baselines (−19.40, −19.19) still apply.
  Generating all three sets took 78 s and decoding them 11 min with `--workers 6` on the 8-core M1 (synthetic slots
  carry 10 signals, so they cost about a third of a busy real slot). `data/bench/{awgn,moderate,disturbed}` now
  exist on this machine again; `data/bench/noise10k` does not.
- **WSJT-X integration added 2026-11-09 (plan 9).** `wsjtx/ft8nn_sidecar` is `libft8nn` as
  WSJT-X's *external FT8 decoder*: with `WSJTX_FT8_DECODER="<path>/ft8nn_sidecar --model <path>/ft8nn.onnx"` set,
  WSJT-X starts it once and per decode pass sends one compact JSON line (`utc`, `pass` = `nzhsym`, `final`, `disk`,
  `period_start`, and every `dec_data.params` field) followed by `samples * 2` bytes of mono s16le 12 kHz audio;
  we answer one JSON line per decode (`snr`, `dt`, `freq`, `msg`, `ap`) and one `{"seq":N,"done":true}`. WSJT-X
  formats those into `jt9`'s FT8 lines itself, so filtering, ALL.TXT, UDP, PSK Reporter and auto-sequencing are
  unchanged. Only FT8 arrives here (every other mode, Fox and SuperFox stay with WSJT-X's own `jt9`, which also
  takes FT8 back if we exit), so there is nothing to forward, and no shared memory and no Qt in this build - the
  protocol is JSON lines and raw audio on a pipe. Protocol: `doc/ft8-sidecar-protocol.md` in the wsjtx fork
  (commit `9f6f04258`, `Decoder/Ft8Sidecar.cpp`). `wsjtx/README.md` has the install steps and the limits;
  `wsjtx/test_sidecar.sh` (`make -C wsjtx test`, also in CI, 116 s for 2 slots on the M1) drives the whole protocol
  with `wsjtx/fake_wsjtx` and checks 8 things: the protocol path against a plain decode of the same audio, the
  `jt9` column layout of every line WSJT-X would build, `[nfa, nfb]`, the early pass, a re-decode (`samples: 0`),
  a non-JSON line, exit on `quit`, and a missing model. The first attempt, a `jt9` shared-memory shim (`jt9nn`,
  Qt 5 + the 46 MiB `dec_data` segment, so stock WSJT-X would work too), was dropped: the fork has the sidecar
  interface and stock WSJT-X is not a target.
- Last commit: the repeat stage with only competitors a station could send, tag `nn_v3_ccrpc` (plan 12), on top of
  the co-channel retry (`nn_v3_cc`, plan 7.1 item 1c); Python + C, golden PASS, `validate_full.sh nn_v3_ccrpc` PASS;
  plus the runs 2-4 miss profile that ranked it (`docs/miss-profile-dtcrp.md`).
  Next: idle C timing (nothing else running; the user schedules it), then the miss profile's other levers - more
  lag tracks per frequency in the candidate search (396 misses decode when forced at the true lag, which the
  per-frequency tracks lose to alias lags), and a look-back repeat stage (offline only: 564 misses are a station's
  first overs that we decode 30/60 s later) - and the early pass (plan 6a).
- Architecture study finished 2026-11-07 (plan 7.3d): keep the transformer. Its code (`train.py --kind conv|gru`,
  `--teacher`, `model.build`) is merged; models in `runs/{tf,conv,gru,tf_l2,tf_l2kd}_v{1,2}`, scripts `data/tmp/` (gitignored).
- Loop log (finished): `nn_v2_eh` pass, `nn_v2_ob` fail (1 ghost), `nn_v2_og` pass, `nn_v2_p4` pass → run 2
  slot-confirmed 19,550 ≥ 19,500 with unconf 1, noise 0 false: loop stopped. Commits need the user's OK again.
- Portability done 2026-11-08: model in `models/v2`, data in the GitHub LFS repos, benchmark-set commands recovered
  (seeds 1/2/3/9, README; regenerated byte-identical), `CLAUDE.md` holds the working agreements.

## Where we are

Best model: `models/v3/model.pt` (+ `ft8nn.onnx`, training `log.json`), shipped 2026-11-09; `models/v2` stays in git as
its starting point. `runs/` is the gitignored output folder for new training runs (the study models are there,
`runs/v2` = `models/v2`, `runs/rn_real_seed1` = `models/v3`). Recipe: v1 (6000 steps, `--cfg v1`), then v2 =
`train.py --cfg v2 --init runs/v1/model.pt --lr 5e-4 --steps 8000` (about 4 h on the M1 Pro, 74 min on the M4 mini, MPS),
then v3 = 4000 steps from v2 with real noise in half the rows (21 min; README has the commands, plan 7.5 the results). `--kind conv|gru` trains the study's alternatives, `--teacher CKPT` distils from a checkpoint.

Decoder defaults: `decoder.decode_slot`. These are 4 passes with subtraction, 2 fine-sync hypotheses,
NN metric with classical nsym 1/2/3 fallback (`--also-classical`), OSD order 2 gated on expected errors < 40,
CQ a-priori decoding (expected errors < 44), burst-erasure retry (expected errors < 15) with a wide-window OSD stage
(`osd_er`), OSD on summed BP posteriors after iterations 1–2 (`osd_bp`), ghost filter, candidate search 150–3100 Hz plus an early-DT track down to −2.5 s, candidate reuse in passes 2–4
(`reuse_df` 62.5 Hz, plan 7.3b), message plausibility rules on every stage's result (plan 7.4),
context a-priori decoding from our own decodes 30/60 s earlier (`bp_cx`/`osd_cx`, expected errors < 48; needs
`context=` in Python, `ft8nn_decode_at` in C; the bench passes it, see plan 7.2c), DT re-centring (`dt_recenter`,
plan 7.1 item 1b), and after the last pass, on the residual, the co-channel retry (`bp_cc`/`osd_cc`/`osd_bp_cc`,
plan 7.1 item 1c) and, with context, the repeat stage (`rp`, plan 12; competitors only messages a station could
send, a CQ-only sender needs ρ ≥ 0.50).

### Results vs WSJT-X jt9 (`-d 3`)

50% decode threshold, DT ≤ 0.5 s, 10 signals per slot:

| set | jt9 | v1 | v2 | **v3 (shipped)** |
|---|---|---|---|---|
| AWGN | −21.31 | −22.20 | −22.37 | **−22.49** |
| moderate fading (0.5 Hz / 1 ms) | −19.40 | −20.12 | −20.12 | **−20.12** |
| disturbed fading (1 Hz / 2 ms) | −19.19 | −19.85 | −20.09 | **−20.16** |
| noise only, 100 slots | – | 0 false | 0 false | **0 false** |

The v3 column is `models/v3` with today's decoder defaults, re-measured 2026-11-10 on the regenerated sets
(seeds 1/2/3; 90% thresholds −21.24 / −17.48 / −18.09, 0 false decodes in all three). The AWGN 50% and 90% numbers
matched the mini's run to the 0.01 dB printed, which confirms the regenerated sets match the ones behind the
older columns. jt9 was not re-run: no WSJT-X on the machine used (8-core M1), so its column is the recorded baseline for
the same sets. Gains vs jt9: AWGN 1.18 dB, moderate 0.72 dB, disturbed 0.97 dB.

Real KiwiSDR recordings (10 stations, 1197 slots, `datasets/ft8-training-data/recordings`, read-only submodule):
**Current default `nn_v3` decodes 17706, jt9 decodes 15729 (+12.6%)** (biased upward: `models/v3` was trained with
these recordings' residual noise; `nn_v2_pl` 17583, +11.8%, is the clean number). 2282 decodes are ours only (2204 name
a callsign jt9 heard at that station); 305 are jt9 only.
History: `nn_v2` 16055 (jt9-only 663) → baseline fix `nn_v2_bm` 16336 (jt9-only 512) → erasure `nn_v2_er` 16443 (jt9-only 466) → fmax 3100 `nn_v2_f31` 16452 (jt9-only 463) → early-DT + osd_er `nn_v2_eh` 16460 (463) → osd_bp `nn_v2_og` 16621 (405) → 4 passes `nn_v2_p4` 16638 (400) → context AP `nn_v2_cy` 17586 (309) → plausibility `nn_v2_pl` 17583 (309) → model v3 `nn_v3` 17706 (305).
08_80m: 3477 (`nn_v3` 3506) vs jt9 3258. AWGN 50% with `nn_v2_og`/`nn_v2_p4`/`nn_v2_cy`/`nn_v2_rs`/`nn_v2_pl`: −22.52 (90%: −21.28; `nn_v2_pl` −21.24); `nn_v3` −22.49 / −21.24.

### Validation set with ground truth (`datasets/ft8-validation-set`, added by the user 2026-11-04)

KiwiSDR recordings plus every PSK Reporter FT8 spot logged at the same time; its `README.md` explains it all.
Run 1 (`recordings/`) = 9 training-set Kiwis on other bands/times; **run 2 (`run2/recordings`) = 9 Kiwis never
used for training** (the honest generalisation test); **run 3 (`run3/recordings`, 2026-11-09) = 12 more unseen Kiwis
at another day and time** (no site shared with training, run 1 or run 2); **run 4 (`run4/recordings`, 2026-11-09
15:00 UTC) = 15 more unseen Kiwis, European afternoon / US morning on the high bands**. 120 slots per station, jt9 baseline in each `decodes.txt`.
Its `tools/verify.py` puts each decode's sender into a tier: `slot` (spotted by others in that slot) > `active`
> `repeat` > `unconf`. **`unconf` ≈ false decodes** (jt9: 13 of 27,462 checkable in run 1, 16 of 23,958 in run 2, 17 of 23,139 in run 3, 29 of 37,521 in run 4).
Treat the folder as read-only; our decodes go to `data/val/run{1,2,3,4}/<station>/decodes_<tag>.txt`. Both data
folders are git submodules under `datasets/` (GitHub LFS repos `jasiek/ft8-validation-set`, `jasiek/ft8-training-data`;
`git submodule update --init`, then `tools/datarepo.py unpack`; see `docs/data-collection.md`):

```sh
V=datasets/ft8-validation-set
uv run python bench/run_real.py $V/recordings --model models/v3/model.pt --also-classical --tag nn_v3 --workers 6 --out data/val/run1
uv run python bench/run_real.py $V/run2/recordings --model models/v3/model.pt --also-classical --tag nn_v3 --workers 6 --out data/val/run2
python3 collect/verify.py --decodes data/val/run1 --tag nn_v3 --list
python3 collect/verify.py --recordings $V/run2/recordings --spots $V/run2/pskr_spots.jsonl --decodes data/val/run2 --tag nn_v3 --list
uv run python bench/run_real.py $V/run3/recordings --model models/v3/model.pt --also-classical --tag nn_v3 --workers 6 --out data/val/run3
python3 collect/verify.py --recordings $V/run3/recordings --spots $V/run3/pskr_spots.jsonl --decodes data/val/run3 --tag nn_v3 --list
uv run python bench/run_real.py $V/run4/recordings --model models/v3/model.pt --also-classical --tag nn_v3 --workers 6 --out data/val/run4
python3 collect/verify.py --recordings $V/run4/recordings --spots $V/run4/pskr_spots.jsonl --decodes data/val/run4 --tag nn_v3 --list
```

Scores (vs jt9 baseline, checkable = sender call is spotted-checkable):

| run | decoder | decodes | slot-confirmed | unconf (≈ false) | jt9-only | ours-only (99% callsign-corroborated) |
|---|---|---|---|---|---|---|
| 1 (training sites) | jt9 | 27,730 | 21,492 | 13 | – | – |
| 1 | nn_v2_er | 29,372 (+5.9%) | 22,669 (+5.5%) | 3 | 1543 | 3185 |
| 1 | nn_v2_f31 | 29,497 (+6.4%) | 22,792 (+6.0%) | 3 | | |
| 1 | nn_v2_eh | 29,584 (+6.7%) | 22,852 (+6.3%) | 3 | | |
| 1 | nn_v2_og | 29,929 (+7.9%) | 23,090 (+7.4%) | 3 | | |
| 1 | nn_v2_p4 | 30,116 (+8.6%) | 23,230 (+8.1%) | 3 | | |
| 1 | nn_v2_cy | 31,957 (+15.2%) | 24,449 (+13.8%) | 3 | | |
| 1 | nn_v2_rs | 31,952 (+15.2%) | 24,444 (+13.7%) | 3 | | |
| 1 | nn_v2_pl | 31,951 (+15.2%) | 24,444 (+13.7%) | 2 | | |
| 1 | **nn_v3** | **32,244 (+16.3%)** | **24,660 (+14.7%)** | **1** | | |
| 1 | nn_v3_dtcrp | 34,295 (+23.7%) | 26,299 (+22.4%) | 1 | | |
| 1 | nn_v3_cc | 34,585 (+24.7%) | 26,505 (+23.3%) | 1 | | |
| 1 | **nn_v3_ccrpc** | **34,727 (+25.2%)** | **26,615 (+23.8%)** | **1** | | |
| 2 (unseen sites) | jt9 | 24,291 | 17,993 | 16 | – | – |
| 2 | nn_v2_er | 25,994 (+7.0%) | 19,244 (+7.0%) | 1 | 886 | 2590 |
| 2 | nn_v2_f31 | 26,032 (+7.2%) | 19,281 (+7.2%) | 1 | 850 | |
| 2 | nn_v2_eh | 26,091 (+7.4%) | 19,320 (+7.4%) | 1 | | |
| 2 | nn_v2_og | 26,340 (+8.4%) | 19,480 (+8.3%) | 1 | | |
| 2 | nn_v2_p4 | 26,439 (+8.8%) | 19,550 (+8.7%) | 1 | | |
| 2 | nn_v2_cy | 27,930 (+15.0%) | 20,505 (+14.0%) | 1 | | |
| 2 | nn_v2_rs | 27,929 (+15.0%) | 20,504 (+14.0%) | 1 | | |
| 2 | nn_v2_pl | 27,928 (+15.0%) | 20,504 (+14.0%) | 0 | | |
| 2 | **nn_v3** | **28,198 (+16.1%)** | **20,701 (+15.1%)** | **1** | | |
| 2 | **nn_v3_dtc** | **28,198 (+16.1%)** | **20,701 (+15.1%)** | **1** | | identical to `nn_v3`: the gate never arms here |
| 2 | nn_v3_dtcrp | 29,435 (+21.2%) | 21,646 (+20.3%) | 1 | | |
| 2 | nn_v3_cc | 29,588 (+21.8%) | 21,751 (+20.9%) | 1 | | |
| 2 | **nn_v3_ccrpc** | **29,691 (+22.2%)** | **21,820 (+21.3%)** | **1** | | |
| 3 (unseen sites, other day) | jt9 | 23,524 | 17,888 | 17 | – | – |
| 3 | nn_v2_pl | 26,535 (+12.8%) | 19,982 (+11.7%) | 1 | | |
| 3 | **nn_v3** | **26,688 (+13.5%)** | **20,105 (+12.4%)** | **0** | | |
| 3 | **nn_v3_dtc** † | **26,993 (+14.7%)** | **20,342 (+13.7%)** | **0** | | |
| 3 | nn_v3_dtcrp | 28,206 (+19.9%) | 21,266 (+18.9%) | 1 | | |
| 3 | nn_v3_cc | 28,295 (+20.3%) | 21,324 (+19.2%) | 1 | | |
| 3 | **nn_v3_ccrpc** | **28,418 (+20.8%)** | **21,421 (+19.8%)** | **1** | | |
| 4 (unseen sites, high bands, afternoon) | jt9 | 38,432 | 28,053 | 29 | – | – |
| 4 | **nn_v3** | **45,744 (+19.0%)** | **33,070 (+17.9%)** | **2** | 948 | 8,259 |
| 4 | **nn_v3_dtc** † | **46,248 (+20.3%)** | **33,397 (+19.0%)** | **2** | 727 | 8,542 |
| 4 | nn_v3_dtcrp | 48,433 (+26.0%) | 35,110 (+25.2%) | 2 | | |
| 4 | nn_v3_cc | 48,668 (+26.6%) | 35,249 (+25.7%) | 2 | | |
| 4 | **nn_v3_ccrpc** | **48,999 (+27.5%)** | **35,541 (+26.7%)** | **2** | | |
| **2 + 3 + 4 pooled (unseen)** | jt9 | 86,247 | 63,934 | 62 | – | – |
| **2 + 3 + 4 pooled (unseen)** | **nn_v3** | **100,630 (+16.7%)** | **73,876 (+15.6%)** | **3** | | |
| **2 + 3 + 4 pooled (unseen)** | **nn_v3_dtc** † | **101,439 (+17.6%)** | **74,440 (+16.4%)** | **3** | | |
| **2 + 3 + 4 pooled (unseen)** | nn_v3_dtcrp | 106,074 (+23.0%) | 78,022 (+22.0%) | 4 | | |
| **2 + 3 + 4 pooled (unseen)** | nn_v3_cc | 106,551 (+23.5%) | 78,324 (+22.5%) | 4 | | |
| **2 + 3 + 4 pooled (unseen)** | **nn_v3_ccrpc** | **107,108 (+24.2%)** | **78,782 (+23.2%)** | **4** | | |

† `nn_v3_dtc` on runs 3 and 4 = the re-measured drifting receiver (`03_20m`, `08_12m`) plus the other
stations' unchanged `nn_v3` decodes; the gate provably cannot fire on those (replaying it over their
decoded DTs never comes within 0.2 s of the 0.8 s threshold). Run 2 was re-run in full.

The gain is larger on unseen receivers than on training sites, and it holds up against outside spots:
false decodes are below jt9's. **Caveat for `nn_v2_cy`:** a context decode's sender is by construction a station
we decoded 30/60 s earlier, so the spot tiers can't catch a false context decode well; its false-alarm evidence is
the shuffle test and the free-bits plausibility check in plan 7.2c. Odd: jt9-only includes many strong signals (run 1: 289 at ≥ 0 dB, run 2: 114),
e.g. run 2 00_40m `E70A KU2X FN13` at +4…+9 dB missed in 3 slots. Many were above our old 3000 Hz search limit
(fixed by `nn_v2_f31`: run 2 strong jt9-only 114 → 82); the rest are plan step 7.1.

### Reproduce the numbers

```sh
uv run pytest -q                                                    # 26 tests
uv run python bench/run_ours.py data/bench/awgn --model models/v3/model.pt --also-classical --tag nn_v3 --workers 6
uv run python bench/report.py data/bench/awgn --dt-max 0.5
uv run python bench/run_real.py datasets/ft8-training-data/recordings --model models/v3/model.pt --also-classical --tag nn_v3 --workers 6
uv run python bench/run_real.py datasets/ft8-training-data/recordings --compare-only --tag nn_v3
```
`run_real.py` decodes each (station, slot parity) chain in time order so context AP sees the decoder's own earlier
slots; `--no-context` turns that off (per-slot decodes, the `nn_v2_p4` behaviour). `run_ours.py` has no context.

Real decode run is about 20 min with 6 workers (about 4 s/slot/core). The synthetic sets take about 5 min each.
Don't run training and benchmarks at the same time, because memory pressure stalls both.

### Candidate-score baseline fix (2026-11-04)

`find_candidates(base="median")` (default): score baseline = 1.77 × median over all (lag, freq) cells.
The old baseline (`base="max40"`, 40th percentile of per-frequency max) is computed from real signals on crowded bands:
4.29 on 00_40m vs 2.30 on noise, which nearly doubled the effective threshold. 1.77 = old/new ratio on noise and synthetic slots,
so synthetic results are unchanged (AWGN 50% −22.37, 0 false on 100 noise slots).
Example miss it fixes: 00_40m 261104_033130 2795 Hz (raw ratio 4.3, normalised 0.93 < 1.3; decodes by BP when forced).

### What we know about the remaining real-data misses (analysis predates the baseline fix)

These come from `tools/diag_strong_miss.py`. It looked at 45 strong jt9-only decodes (jt9 SNR ≥ −9) in 40 slots:
- 28 had no candidate.
- 16 had a candidate, but the decode failed.
- 1 decodes in the residual.

- The misses cluster on station `00_40m` (Hamburg). There the 2.7–3 kHz region is about 7 dB below the band median.
  The noise level also jumps within a frame, probably from receiver AGC.
  Signals there are real: the matched-tone test passes and the per-symbol tone SNR is 10–50. But our Costas score
  is only about 1.0 at that spot.
- Our noise normalisation is global: one median PSD over 200–3000 Hz (`Baseband.noise_psd`). In quiet
  regions this makes `estimate_snr` give nonsense (−56 dB), and it probably makes the NN input scale wrong too.
- Tried and rejected: per-symbol Costas ratio (`find_candidates(per_symbol=True)`). Real decodes fell to 15808,
  strong-signal misses went from 35 to 254, and AWGN was 0.2 dB worse.
- Not tried: frequency-dependent noise whitening. That means estimating the noise floor per ~50 Hz from the spectrogram
  and normalising both the candidate search and `Baseband.downsample` by the local floor rather than the global one.
- jt9 decodes above 3000 Hz (our old search limit): fixed, default is now 3100 Hz (plan step 7.1).

## Plan

1. **ONNX export of the NN — DONE.**
   `uv run python tools/export_onnx.py models/v2/model.pt models/v2/ft8nn.onnx` writes a 3.5 MB graph
   (input `features` (B,79,48) float32 → output `llr` (B,174), dynamic batch, opset 17) and checks it with onnxruntime:
   max |diff| 1.9e-5 on 512 v2-mix samples, 100% sign agreement. `tests/test_onnx.py` covers it.
   `_conv_bl` switches to plain `Conv1d` under `torch.onnx.is_in_onnx_export()` (unfold doesn't export).
   Speed: ORT CPU 0.75 ms/candidate vs torch 0.35 ms (batch 128). Fine for now; try ORT transformer fusion later if needed.
2. **C library `libft8nn` — first version DONE, matches Python exactly.**
   - Code in `c/` (`make -C c` → `c/build/libft8nn.{a,dylib}`, `c/build/ft8nn_cli`). Own C for everything,
     including a mixed-radix 2/3/5 FFT; the only dependency is ONNX Runtime (official release via
     `tools/get_onnxruntime.sh`, plan 7.3c; Homebrew's build is the fallback and 2.3x slower).
     API in `c/include/ft8nn.h`: `ft8nn_open(onnx, n_threads)`, `ft8nn_decode(ctx, audio, n, out, max)`,
     `ft8nn_decode_at(ctx, audio, n, slot_time, out, max)` (same, plus context AP from the context's own decodes
     at slot_time − 30/60 s; call it every slot in time order), `ft8nn_reset_hashes`, `ft8nn_close`.
     `ft8nn_cli` uses `ft8nn_decode_at` when the file name is a `yymmdd_HHMMSS` slot time. One context per thread; the context's hash table learns callsigns.
     `ft8nn_decode` normalises audio to unit std internally (as `bench/run_real.py` does).
   - LDPC tables: `c/src/ldpc_tables.h`, generated by `tools/gen_c_tables.py`.
   - Golden test `tools/golden_c.py` (uses `models/v3/ft8nn.onnx`; compares against tag `nn_v3_rpf` by default;
     `nn_v3_rpf` 2026-11-11: 785/785 + 627/627 real, run 2 02_80m 749/749, AWGN 429/429; `nn_v3`
     2026-11-09: 764/764 + 607/607 real, run 2 02_80m 733/733, AWGN 429/429; before that
     verified with `nn_v2_pl`: 756/756 + 599/599 real (08_80m all 120 slots too), run 2 02_80m 733/733, 430/430 AWGN; the real-data golden
     decodes the slots in order through `ft8nn_decode_at`, so context AP is covered):
     real 00_40m + 08_80m, first 20 slots each: 1231/1231 decodes identical in every output field;
     AWGN 60 files: 418/418 identical payloads.
     `uv run python tools/golden_c.py real <station_dir> --n 20`, `... awgn data/bench/awgn --n 60`.
   - Speed (single core, `FT8NN_PROFILE=1` prints per-pass stage times): about 6.4 s/slot vs Python 4 s.
     Per slot: NN 4.3 s, BP 1.1 s, fine sync 0.5 s, candidates 0.3 s, classical metrics 0.2 s.
     Done so far: `-fcx-limited-range`, BP via expm1/log, split re/im sync grid, per-tone maxima in nsym 2/3
     metrics. All kept output byte-identical (golden test re-run).
     The NN cost is ONNX Runtime's CPU GEMM: 2.4 ms/hypothesis single-thread vs torch 0.82 ms (torch uses
     Accelerate/AMX). Tried, no gain: ORT transformer fusion, static-batch export. CoreML EP: 1.6 ms, partial
     graph support. **User decision 2026-11-04: keep ONNX Runtime.**
   - Possible later speedups (would change results, so need Python + C change and an A/B): BP early stop when
     the unsatisfied-check count stalls (most of the 6000 BP rows per slot are noise running all 30 iterations).
3. **Local noise whitening — replaced by the baseline fix above.** Whitening would not have helped: the Costas
   score is already a local ratio (tone vs median of the other 7 tones), and NN features are normalised by each
   candidate's own median. Remaining 00_40m misses that have a candidate fail at decode with expected errors 28–35
   (fading within the frame, e.g. 261104_031245 1858 Hz: Costas array 0 absent, arrays 1–2 strong).
   Committed as `0a795d0`.
4. **Miss diagnosis after the baseline fix — DONE (2026-11-04).**
   `tools/diag_strong_miss.py models/v2/model.pt nn_v2_bm -9 2000` over all 98 slots with strong jt9-only misses:
   114 misses = 64 FAIL (candidate found, decode fails), 47 NOCAND, 3 OK.
   - 22 of the 64 FAILs are *confident* (NN expected errors 0–11, signals up to +21 dB). Cause: the sender changed
     the message mid-transmission (WSJT-X switches tones at once). Example 08_80m 261104_031300 1331 Hz: OM2WQ sent
     `KC4ES OM2WQ -21` the two slots before; data symbols 7–15 still carry KC4ES, the rest is `RA6HN OM2WQ +00`.
     Result: 15–26 bit errors with |LLR| ≈ 10, the same confidence as the correct bits. BP and OSD can't fix it, and
     rescaling LLRs doesn't help (3/64). The wrong symbols form contiguous bursts (start 7–16, end 53–71, or mid 29–35).
   - The other 42 FAILs have expected errors 13–73 (weak or faded); 47 NOCAND are mostly −9…−4 dB.
5. **Burst-erasure retry — DONE (2026-11-04), default on.**
   `decoder._try_erase`: if the NN metric fails with expected errors < `erase_gate` (15), zero the LLRs of each of
   96 windows of data symbols (widths 6/10/14/18, stride 2) and run BP; the first CRC pass wins. Method `bp_er`
   (C: `FT8NN_BP_ERASE`). Prototype on the 21 confident FAILs: 16 correct, 0 wrong, 58 ms/candidate (Python).
   C: `try_erase` in `c/src/decode.c` (`ERASE_GATE`). Bench flag `--erase-gate` (0 = off).
   A/B vs `nn_v2_bm`: real 16336 → 16443 (+107), jt9-only 512 → 466, AWGN 50% −22.37 unchanged,
   0 false on 998 noise slots. The 111 `bp_er` decodes: 53 also decoded by jt9, 58 ours-only with a jt9-heard
   callsign, 0 uncorroborated.
   Golden C vs Python (tag `nn_v2_er`, first 20 slots): 00_40m 683/683, 08_80m 582/582 identical in every field;
   AWGN 60 files 419/419 identical payloads.
6. **Score `nn_v2_er` on the validation set — DONE (2026-11-04).** Results in the table above: +7.0% decodes and
   +7.0% slot-confirmed on unseen receivers, 1 unconfirmed vs jt9's 16. From now on every decoder change is A/B'd here as well (run 2 is the deciding set),
   because `ours-only corroborated` on the training recordings only checks the callsign, not that the station sent it.
7. **Next, ranked:**
   1. **Strong jt9-only misses on the validation set.**
      - **fmax 3100 — DONE (2026-11-04).** Run 1 had 289 strong (≥ 0 dB) jt9-only misses, 119 above 2990 Hz; run 2 had
        114, 33 above. jt9 decodes up to f0 = 3052 Hz; our search stopped at 3000. Now `frontend.find_candidates(fmax=3100)`,
        bench `--fmax 3100`, C `FMAX` in `c/src/decode.c` (the noise-estimate band 200–3000 Hz is unchanged).
        A/B `nn_v2_er` → `nn_v2_f31`: run 2 25,994 → 26,032 decodes, slot-confirmed 19,244 → 19,281, unconf 1 → 1,
        jt9-only 886 → 850 (strong 114 → 82); run 1 29,372 → 29,497, slot-confirmed 22,669 → 22,792, unconf 3 → 3;
        training recordings 16,443 → 16,452 (jt9-only 466 → 463); AWGN 50% −22.37 unchanged; noise 0 false / 998.
        Golden C vs Python (`nn_v2_f31`): 00_40m 684/684, 08_80m 582/582 all fields; AWGN 419/419.
      - **Next: diagnose the remaining strong misses on run 2** (82 at ≥ 0 dB):
        `uv run python tools/diag_strong_miss.py models/v2/model.pt nn_v2_f31 0 2000 --root $V/run2/recordings --out data/val/run2`
        (script takes `--root`, `--out`, `--max-snr`). Sort into NOCAND / FAIL (by expected errors) / OK, as in step 4.
      - **Diagnosis run 2 (2026-11-04):** 80 strong misses in 68 slots = 47 NOCAND, 32 FAIL, 1 OK.
        **19 of the 47 NOCAND have DT −2.2…−2.3 s** (station SO3O and partners, clock ≈ 2.3 s early, seen on 3 Kiwis).
        Our search starts at `dtmin` −1.5; jt9 searches to −2.5. Spot check with `dtmin=-2.5`: 4/4 test slots decode
        with plain BP. **Plain `dtmin=-2.5` (tag `nn_v2_dt25`) fails the A/B:** run 2 26,032 → 25,988 decodes,
        slot-confirmed 19,281 → 19,249, unconf 1 → 2 (+56 decodes at DT < −1.5, ~100 lost elsewhere: edge lags,
        where only 2 Costas arrays are in the slot, spike and steal the main lag tracks; noise 0 false).
        **Replacement: separate early-DT track** (`find_candidates(early_dt=-2.5, early_min=3.0)`, uncommitted):
        main search unchanged (verified: identical candidate list), plus per frequency the best lag in [−2.5, −1.5),
        kept if it is a frequency peak, ≥ 3.0, and beats every main lag at that frequency (strong on-time signals
        also score high at early lags). Appended after the 600 budget. ~35 extra candidates per real slot, 0 on noise.
        C ported (`EARLY_DT`, `EARLY_MIN` in `c/src/decode.c`; candidate buffer now MAX_CAND + nf).
        **A/B `nn_v2_et` (early track only): passes.** Run 2 26,032 → 26,061 decodes, slot-confirmed 19,281 → 19,304,
        unconf 1 → 1, 46 decodes at DT < −1.5; noise 0 false.
      - **A/B `nn_v2_eo` (early track + osd_er): fails.** 26,095 decodes, slot 19,320, but unconf 1 → 2 and 31 osd_er
        decodes of which 4 are garbage (`<5:…>`, `<6:…>`, `U+LVARAJ CD?C`, `4Q9OIV LQ7Q H R LL46`); 14 slot-confirmed,
        13 active. Distance/expected errors don't separate them. **All 4 garbage sit within 1 Hz of a real decode in
        the same slot at a DT 1.2–2.6 s away; none of the 27 real ones does** → time-shifted ghosts of strong signals
        (wrong lag, NN confidently reads the real tones in the wrong places, OSD fits a CRC-valid random message).
      - **Ghost filter** (`decoder.GHOST_DF/GHOST_DDT` = 3 Hz / 0.5 s, C same names): at the end of each pass, drop an
        osd_er decode if another decode (earlier pass, or this pass and not osd_er) is that close in frequency and
        that far in DT. bp_er left alone (its neighbour cases are real, e.g. `AP5ARS EA3FCQ 73` also decoded by jt9).
        **A/B `nn_v2_eg` (early + osd_er + ghost filter): passes.** Run 2 26,061 (`nn_v2_et`) → 26,091 decodes,
        slot-confirmed 19,304 → 19,320, unconf 1 → 1 (the same old `CQ 0V1YXG GG88`), noise 0 false / 998.
        osd_er decodes 31 → 27: exactly the 4 ghosts removed, all 27 left are plausible messages
        (e.g. `CQ F4CTJ JN09` on two Kiwis, `SP3JHZ CT3HQ RR73` on two Kiwis).
        The 32 FAILs: expected errors mostly 20–45 (weak/faded, plan item 7.2); 2 at E=2 are mid-transmission
        message changes with bursts wider than the 18-symbol BP erasure windows: 02_80m 223430 `CQ F4CTJ JN09` (data
        symbols 35–57 wrong) and 08_80m 223530 `K0VM CT3HQ -15` (3–28, plus stray errors). BP fails at every width
        20–30 (too many erasures). **OSD-0 on the erased window works** for F4CTJ: window (34,58) → correct, distance 0.0.
      - **Wide-window OSD erasure stage (`osd_er`)** (A/B'd as `nn_v2_eo` above, kept with the ghost filter as `nn_v2_eg`).
        `decoder._try_erase(llr, osd_dmax)`: after the BP windows fail, OSD-0 on 57 windows of width 20/22/24
        (`ERASE_OSD_WINDOWS`); smallest-distance CRC pass wins if distance < `erase_osd_dmax` (3.0). ≤ 24 symbols
        erased keeps ≥ 102 bits, so ≥ 11 parity bits + CRC-14 still check the result. 0.8 ms per window (Python).
        Bench flag `--erase-osd-dmax` (0 = off). End-to-end: F4CTJ decodes, nothing else in the 2 slots changes.
        C ported: `try_erase` in `c/src/decode.c` (`ERASE_OSD_DMAX`), method enum `FT8NN_OSD_ERASE = 5`, CLI name `osd_er`.
      - **`nn_v2_eh` — DONE (2026-11-05, Mac mini).** Early-DT track + `osd_er` + two fixes after `nn_v2_eg` produced 3
        garbage decodes on run 1: (1) early-DT candidates get only BP, OSD and `bp_ap` (no OSD under a-priori bits, no
        erasure stages; `Candidate.early`, C `candidate.early`); (2) `decoder._is_ghost` (C `is_ghost`) replaces the
        3 Hz / 0.5 s ghost rule: drop an `osd_er` decode whose data tones equal a neighbour's (within 50 Hz) tones
        shifted by whole symbols and whole tones on ≥ 40% of ≥ 20 comparable symbols (offline: flags all 6 known ghosts,
        none of 82 real `osd_er` decodes). Results vs `nn_v2_f31`: run 2 26,032 → 26,091 decodes, slot-confirmed
        19,281 → 19,320, unconf 1 → 1; run 1 29,497 → 29,584, slot 22,792 → 22,852, unconf 3 → 3; training recordings
        16,452 → 16,460 (jt9-only 463); AWGN 50% −22.37; noise 0 false / 998. Golden C vs Python: 00_40m 685/685,
        08_80m 583/583, run 2 02_80m 720/720 all fields; AWGN 419/419.
        **C bug found on the way:** `decode_pass` freed the candidate list before the decode loop reads `cands[ci].early`
        (use after free). On the M1 the stale memory happened to hold the right values; on the mini under load the AWGN
        golden test lost 2–6 `osd_ap0` decodes (non-deterministically). Fixed; `validate_full.sh` now runs the golden
        tests with `MallocScribble=1 MallocPreScribble=1`, which makes such reads fail reliably.
        Some jt9-only "strong" decodes look like jt9 false decodes (`HYZDB82E2JV <...> RR73`, `GE2EDD 5Y4KKW GP53`).
   1b. **DT re-centring for receivers whose clock or stream is off (`dt_recenter`) - DONE 2026-11-10 as `nn_v3_dtc`.**
      Miss profile of validation run 4 with `nn_v3` (948 jt9-only decodes, scratch `miss_profile.py`): **214 sit at
      DT < -1.5 s and 203 of those come from one receiver**, `08_12m` (n9azz), whose KiwiSDR stream slips about
      2.3 s over the 30 min recording - its per-slot median DT walks from +0.2 (first slot) to -2.1 (last), while the
      spread *within* a slot stays ~0.3 s. 37% of that station's jt9 decodes are below -1.5 s, where our main search
      does not go: the window is DT [-1.5, 2.5] plus the restricted early-DT track (plain BP/OSD only, score >= 3.0,
      must beat every main lag). Run 3's `03_20m` is the same pathology (42% of its decodes below -1.5 s, slot median
      down to -2.6); no other receiver in runs 1-4 or the training set drifts (per-slot median |DT| <= 0.5 s).
      This is a candidate-*search* problem, not a metric one: `fine_sync2` cuts the NN's input window relative to each
      candidate's own (f0, dt), so a signal at DT -2.1 s looks identical to one at +0.2 s once the search finds its lag.
      Change: `decoder.decode_slot(dt_recenter=0.8)` takes the median DT of the decodes already in hand - this slot's
      plus the context's from 30/60 s earlier, >= `DT_RECENTER_MIN` 6 of them, clamped to `DT_RECENTER_MAX` 2.5 s and
      snapped to the 40 ms lag grid - and, when it is 0.8 s or more from 0, shifts the whole candidate window by it
      (`frontend.find_candidates(dt_shift=)`: main range, typical-DT track and early band move together, in whole
      spectrogram frames, so with `dt_shift=0` the code is bit-identical to before). The estimate comes from
      CRC-checked decodes, not from the sync score, because senders' own clocks scatter only a few tenths of a second
      while the receiver's offset is common to every signal in the slot. (Score-based estimators were tried first and
      abandoned: a max over ~180 lags inflates noise, and row-normalised lag profiles pick edge lags, where fewer
      Costas terms are in the slot - they put the run-4 station at +2.16 s when the truth was -2.10 s.)
      C: `DT_RECENTER*` and `dt_recenter_shift` in `c/src/decode.c`, `dt_shift` in `find_candidates`
      (`c/src/frontend.c`); the tried table grew to lags -160..159 frames so candidate reuse still applies when the
      window is shifted. Bench flag `--dt-recenter` (0 = off). Tests: `tests/test_recenter.py` (gate/clamp/snap, early
      candidate becomes an ordinary one, and a slot at DT -2.9 s that the window does not reach at all: 0 of 4 decoded
      before, 4 with).
      **Inert where it is not needed:** replaying the gate over every station of runs 1-3 (jt9's own DTs) and over
      run 4's saved `nn_v3` decodes, it fires only on those two receivers (23 of 36 scorable run-3 slots, 81 of 119
      run-4 slots) and on no other station - run 1 and run 2 never come within 0.2 s of it. Synthetic sets: 1 slot of
      300 reaches |median| 0.82 s, and every synthetic signal (DT in [-0.4, 1.0]) stays inside the shifted window.
      Noise-only slots have no decodes, so the gate cannot fire there.
      **Results (2026-11-10), the two drifting receivers, 119 slots each, scored against PSK Reporter spots:**
      run 4 `08_12m` 3,643 -> **4,147** decodes (jt9 3,359; ours/jt9 1.085 -> **1.235**), slot-confirmed 2,428 -> 2,755,
      active-only 1,138 -> 1,315, unconf 0 -> 0, jt9-only 283 -> **62**; run 3 `03_20m` 339 -> **644** (jt9 318;
      1.066 -> **2.025**), slot-confirmed 235 -> 472, active 101 -> 169, unconf 0 -> 0, jt9-only 38 -> **18**, and the
      slot-confirmed share of checkable decodes rises 69.9% -> 73.6%. Run totals: run 3 26,688 -> 26,993 decodes
      (+13.5% -> **+14.7%** vs jt9), slot-confirmed 20,105 -> 20,342; run 4 45,744 -> 46,248 (+19.0% -> **+20.3%**),
      slot-confirmed 33,070 -> 33,397, jt9-only 948 -> **727**. Note `run_real.py`'s `corrob.` column *falls* on run 3
      (97% -> 89%): it only asks whether jt9 heard the same callsign on the same receiver, so it understates a decoder
      that finds stations jt9 misses entirely - the spot tiers (100.0% confirmed, 0 unconf) are the real check.
      **`tools/ab_run2.sh nn_v3_dtc`:** run 2 28,198 decodes / 20,701 slot-confirmed / 1 unconf - *identical* to
      `nn_v3`, because none of run 2's nine receivers has a clock far enough off to arm the gate; noise 0 false / 998.
      The written A/B rule (run-2 decodes and slot-confirmed must rise) cannot be met by a change that is inert on
      run 2, so it stands here as a no-regression check and the gain is carried by runs 3-4.
      **`tools/validate_full.sh nn_v3_dtc` (23:15-01:34, 2 h 19 min): PASS, and bit-identical to `nn_v3`
      everywhere the gate cannot arm** - which is every set in this suite, so it is a pure no-regression result.
      Validation run 1: 32,244 decodes / 24,660 slot-confirmed / 1 unconf (`CQ KR9ZUC/R AO63`, the same one as
      `nn_v3`), 100.0% confirmed on all nine receivers, every figure unchanged. Training recordings: 17,706
      decodes, jt9-only 305 - unchanged. AWGN: identical row by row, 50% -22.49 dB, 90% -21.24 dB, 0 false
      decodes at every SNR. Golden C vs Python under `MallocScribble=1 MallocPreScribble=1`: 00_40m 764/764
      all fields equal, 08_80m 607/607, run2 02_80m 733/733, awgn 429/429 - four PASS (the one place the port
      had to be checked hard, since it threads `dt_shift` through `find_candidates` and doubles the `tried`
      table). `pmset -g log`: no Sleep, no DarkWake; `caffeinate` held the machine awake for the full run.
      Tooling, found on the way: `ab_run2.sh` and `validate_full.sh` call bare `uv`, which the asdf shim could not
      resolve in a worktree ("No version is set for command uv"), and `set -euo pipefail` made that a silent abort.
      `.tool-versions` now pins `uv 0.13.7` alongside `python 3.12.14` (`uv lock --check` clean and
      `uv sync --dry-run` "would make no changes", so the pinned 0.13.7 leaves the locked environment untouched).
   1c. **Co-channel collisions: the strong signal's subtraction destroys the weak one's sync - DONE 2026-11-12 as
      the co-channel retry, tag `nn_v3_cc` (the shipped default; Python + C, golden PASS).** Diagnosis first
      (2026-11-11), then the stage (below the diagnosis). With 1b in, the remaining jt9-only decodes were re-profiled from scratch.
      **They are real, not jt9 false decodes.** Re-emitting each station's jt9-only decodes as their own decode
      file and running `collect/verify.py` over it: run 1 1,027 misses = 788 slot-confirmed / 199 active / 3
      repeat / 14 unconf, run 2 583 = 419 / 126 / 0 / 16, run 4 727 = 493 / 174 / 0 / 29. So 96-98% are
      genuine signals we miss. The suspect remainder (59 pooled) is jt9's own false-decode rate concentrated in
      this subset - 1.4-4.2% here against ~0.05% over all jt9 decodes, a 30-90x enrichment, with the expected
      look (`GE2EDD 5Y4KKW GP53` at +9, `SK5OCT R4MRJ R KI00` at +20, `CQ XO7FYG EA52`). **This retires the old
      note that many strong jt9-only decodes are jt9 false decodes:** run 1's 128 strong (>= 0 dB) misses are
      83 slot-confirmed + 43 active + 2 n/a and **0 unconfirmed**. They are ours to find.
      **Stage breakdown** (`tools/diag_strong_miss.py models/v3/model.pt nn_v3_dtc 0 2000 --root $V/recordings
      --out data/val/run1`, 99 slots, 90 s): 120 strong misses = **74 NOCAND, 44 FAIL, 2 OK**. Not a budget
      problem: re-searching each NOCAND cell with `min_score=0.6, lag_excl=0, max_cand=4000` recovers only 5 of
      74; for 66 the Costas score at the target cell is at noise level. The 44 FAILs are the known weak/faded
      population (E 13-40: 22, E > 40: 16, plan 7.2).
      **The mechanism, for the 18 NOCAND that have a decode of ours within 10 Hz / 0.4 s:** two stations in the
      same slot within a few Hz at the same DT. Every FT8 signal carries the *same* three Costas arrays, so
      theirs overlap coherently and the pair occupies one candidate cell (spectrogram bins are 3.125 Hz). We
      take the cell, decode the stronger, subtract it - and because the amplitude was fitted to the *sum*, the
      subtraction takes the weaker signal's sync with it. Its data symbols encode a different message, so those
      survive. Worked example, `02_40m` 1700-1703 Hz, W1TC at +24/+25/+19 over AD8RW at +13/+14/+12, all at
      DT +0.1: we get W1TC, jt9 gets both; in the residual no candidate exists within 6 Hz / 0.3 s at any
      threshold. **Forcing a hypothesis at the cell we just subtracted from, with no sync score consulted** (a
      df grid +-7 Hz / 0.5 Hz on the subtracted decode's own f0, dt +- 6 baseband samples, straight into the NN
      + LDPC) recovers `EA3CCI AD8RW EN80` in all three slots at E = 18.6 / 3.4 / 9.5.
      **Class size** (jt9-only miss with a decode of ours within 10 Hz / 0.4 s, all SNRs): run 1 179 of 1,027
      (17.4%), run 2 112 of 583 (19.2%), run 4 111 of 727 (15.3%) - about 400 pooled, 90-96% spot-confirmed.
      **The stage (`decoder._cochannel_stage`, C `cochannel_stage`; methods `bp_cc` / `osd_cc` / `osd_bp_cc`,
      `FT8NN_BP_CC` = 10 / `FT8NN_OSD_CC` = 11 / `FT8NN_OSD_BP_CC` = 12).** After the last pass, on the residual
      (every decode subtracted), at every decode's own cell: a **data-symbol sync** search
      (`frontend.data_sync`, C `data_sync`) over df +-8 Hz x dt +-50 baseband samples (+-0.25 s) on a 0.5 Hz x
      2-sample grid - per data symbol the strongest of the 8 tone powers minus the stronger of the two tones
      just outside the band (-1 and 8), summed over the 58 data symbols, over the summed mean in-band power. The
      out-of-band term is what makes it work: without it every signal scores almost the same at +-6.25 Hz (7 of
      its 8 tones), and those aliases filled the top peaks (run-1 class recoveries 41 -> 42 with half the
      hypotheses once removed). Then the 2 best local maxima within +-24 samples and the best one beyond, each
      scoring >= 3.0, go to the NN as 3 x 3 hypotheses (+-0.4 Hz, +-1 sample); NN metric -> BP -> OSD-2
      (expected errors < 40, hard errors <= 40) -> OSD on BP sums; plausibility rules; in order, one decode
      per peak, and a payload already decoded or a ghost (`_is_ghost`) of any decode in the slot is skipped.
      Stage decodes are subtracted, then the repeat stage runs. Run before the repeat stage, so a station both
      would find is decoded with a CRC check rather than tested (151 of the 552 on runs 2-4 replaced `rp`
      decodes). C: the grid search per cell and the per-hypothesis decoding run on the thread pool, then a
      serial merge in Python's order; `data_sync` is a sliding 32-sample DFT over the ~1,080 window starts
      (restarted from the direct sum every 64 samples), ~2 M operations per cell instead of 36 M direct; the
      last pass is now always subtracted (it was only with context); stage decodes go out in the stream's
      `pass = -1` batch. Bench flag `--cochannel NEAR,FAR,DF,NEAR_DT,FAR_DT,MIN` (default `2,1,8,24,50,3`,
      `off` disables it).
      **Choosing it (validation run 1 and the training recordings only; scratch scripts `cc_*.py`).** Offline
      emulation: subtract a slot's saved `nn_v3_dtcrp` decodes, run the stage at every cell. On the run-1 class
      (146 jt9-only decodes within 10 Hz / 0.4 s of ours, 179 before the repeat stage took some): the earlier
      hand-test grid (+-7 Hz / +-30 ms, 377 hypotheses per cell) 42, a brute grid +-8 Hz / +-80 ms (561) 69,
      Costas fine sync on the residual 31, data sync without the out-of-band term 41-43, with it and a 3 x 3
      refinement (18 per cell) 67, and with the far band (+-0.25 s) **72**. The far band came from the runs 2-4
      geometry of the other fork's miss profile (ΔDT p75 0.19 s, p90 0.28 s; a +-0.125 s window reached only
      53% of that class) and was chosen on run 1: one +-0.25 s window with 2 peaks lost 6 near recoveries to far
      peaks while gaining 7 far ones, two zones kept both. ±10 Hz gained nothing.
      Run 1, all 34,295 cells: **+253 decodes = 174 slot-confirmed, 74 active, 0 unconfirmed, 5 unscorable**
      (DXpedition/hashed formats), 14.7 NN hypotheses per cell. Training recordings: +50 decodes, 49 with a
      callsign jt9 heard at that receiver (the other, `N9AOT KN4WJY -18`, plausible but unconfirmed), 1 ghost
      filtered. **Noise:** the cells of 1,000 busy real slots forced onto noise-only audio (43,474 cells): best
      data-sync score <= 2.77, so no hypothesis reaches the NN; with no score limit at all 1,173,798 NN
      hypotheses and **0 false decodes**. AWGN: identical (no co-channel pairs in synthetic slots).
      **A/B `nn_v3_cc` (runs 2-4, user-approved run 2026-11-12, scored against `nn_v3_dtcrp`):**

      | run | `nn_v3_dtcrp` | `nn_v3_cc` | change |
      |---|---|---|---|
      | noise (998 signals) | 0 false | 0 false | - |
      | 2 | 29,435 / 21,646 / 1 | 29,588 / 21,751 / 1 | +153 / +105 |
      | 3 | 28,206 / 21,266 / 1 | 28,295 / 21,324 / 1 | +89 / +58 |
      | 4 | 48,433 / 35,110 / 2 | 48,668 / 35,249 / 2 | +235 / +139 |
      | **pooled** | 106,074 / 78,022 / 4 | **106,551 / 78,324 / 4** | **+477 / +302** |

      Unconfirmed: the same four as before. The stage made 552 decodes = 356 slot-confirmed, 181 active, 0
      unconfirmed, 15 unscorable (mostly the special-event LZ80RKVT's hashed messages). 552 -> +477: 151 replace
      `rp` decodes of the same message, +79 more come from other stages in later slots (stage decodes become
      context), 3 `rp` decodes were lost (`JA9BFN AC2Y FN03` slot-confirmed, two active). Against the
      references: 249 agree with jt9/JTDX/deep JTDX decoding the same sender in that slot, 18 differ only in
      hash formatting or are LZ80RKVT's several streams, 2 conflict with JTDX's CRC-less `^` stage only
      (`9Y4C KX4TE R+23` vs `R-01`, `5B4AMX W2BDM -14` vs `PE2K W2BDM -14`; jt9 decoded neither), 283 have
      no reference decode. Doubtful: `FN3EPRKOCCN <...> 73` (run 2, an `i3=4` non-standard call no rule can
      tell from a real one; plan 7.4). Of the stage's new (slot, sender) pairs 101 are on the other fork's
      list of reference-only slot-confirmed misses (68 inside its 130-decode "forced at the true spot" bound),
      282 are signals no reference decoded as a slot-confirmed miss. Wall times (shaky: the other fork ran
      Python jobs, and run 2 met a burst of macOS background work after the wake): run 2 49:48, run 3 44:58,
      run 4 53:44 vs `nn_v3_rpf` 38.7 / 38.7 / 52.9 min; Python CPU on 36 sample slots +3.3%, NN hypotheses
      +18.8% (436 per slot at 29 decodes).
      **C port:** golden C == Python under MallocScribble, run 2 02_80m 753/753 all fields at 1 thread and at 4
      threads streaming (3 stage decodes among them, `FN3EPRKOCCN` included); `tools/ci_smoke.py` 51/51 (51, not
      57: plan 7.4); `wsjtx/test_sidecar.sh` all 8 checks.
      **`tools/validate_full.sh nn_v3_cc` (2026-11-12 11:09:38-13:29:16, 2 h 20 min with 8 workers while the other
      fork ran 8 Python workers too; no Sleep/DarkWake after the full wake at 07:48:18): PASS.** Run 1 34,295 /
      26,299 / 1 -> **34,585 / 26,505 / 1** (+290 / +206; the same unconfirmed `CQ KR9ZUC/R AO63`); the stage made
      322 decodes there = 227 slot-confirmed, 88 active, 1 repeat, 6 unscorable, 0 unconfirmed; at payload level
      294 gained, 4 lost (3 slot-confirmed: `CQ SV3AUW KM17` and `KG4OLE IW9FRA JM68` from `rp`, `D44PX SV1AZL
      KM17` from `osd`). Training recordings 18,489 -> 18,563 decodes, jt9-only 209 -> 197, corroborated 97%. AWGN
      identical row by row (50% -22.49, 90% -21.24, 0 false). Golden 00_40m 807/807, 08_80m 627/627, run 2 02_80m
      753/753 all fields, AWGN 429/429. Step times: run 1 75 min, training 58, AWGN 2, build + golden 5.

   2. **Deeper OSD for expected-error 13–40 FAILs: OSD on summed BP posteriors (`osd_bp`) — DONE as `nn_v2_og`.**
      WSJT-X's decode174_91 keeps the running sum of BP posteriors after the first iterations (zsave) and runs OSD on
      those too; we ran OSD on the NN LLRs only. Prototype (scratch script, 2026-11-05): on run 2, for every hypothesis
      the `nn_v2_f31` decoder left undecoded (expected errors < 40, BP and OSD failed), OSD-2 on the sums after
      iterations 1, 2, 3: 285 new decodes, 57 of them exactly jt9-only messages, none contradicting jt9.
      Scored against spots: 281 checkable = 178 slot, 101 active, 1 unconf (`IT4AGC F40UUB KI04`, snapshot 3, 38 hard
      errors). With the usual hard-error limit (≤ 40 vs the NN LLRs) and snapshots 1–2 only: 243 = 154 slot, 88 active,
      0 unconf. 0 hypotheses reach the stage on noise-only slots (all have expected errors ≥ 40).
      Implementation: `ldpc.bp_posterior_sums`, `decoder._try_osd_bp` (after plain OSD fails, NN metric only, not for
      early-DT candidates), bench flag `--osd-snaps` (0 = off); C `bp_posterior_sums`, `try_osd_bp`, `OSD_SNAPS`,
      method `FT8NN_OSD_BP = 6` / `osd_bp`.
      **A/B `nn_v2_ob`: fails by one decode.** Run 2 26,091 → 26,341 decodes, slot-confirmed 19,320 → 19,480,
      noise 0 false, but unconf 1 → 2: `0J4SMG/R TS6KAQ/R R AO20` (osd_bp, 1415.0 Hz DT 2.03), a ghost of
      `WA4GBT JA8ZO R-07` (−1.5 dB, 1416.6 Hz, DT 0.10: 12 symbols earlier). 512 osd_bp decodes in all.
      Fix: the `_is_ghost` filter now also checks osd_bp (`GHOST_PRONE`, C `ghost_prone`); offline it flags exactly
      that 1 of the 512.
      **`nn_v2_og` (osd_bp + ghost filter on osd_bp) — DONE, all checks pass.** Run 2 26,091 → 26,340 decodes,
      slot-confirmed 19,320 → 19,480, unconf 1 → 1 (`CQ 0V1YXG GG88`, the same one as before); run 1 29,584 → 29,929,
      slot 22,852 → 23,090, unconf 3 → 3; training recordings 16,460 → 16,621, jt9-only 463 → 405; AWGN 50%
      −22.37 → −22.52 dB (90% −21.10 → −21.28); noise 0 false / 998. osd_bp decodes: 511 run 2, 703 run 1.
      Golden C vs Python: 00_40m 698/698, 08_80m 587/587, run 2 02_80m 724/724 all fields; AWGN 430/430.
      Later: more AP types; training with slow (seconds) signal fades.
   2b. **A 4th subtraction pass — DONE as `nn_v2_p4`.** Prototype (2026-11-05, scratch `proto_maxcand.py`): on run 2,
      subtract all `nn_v2_eh` decodes, take up to 1500 candidates of the residual and decode with plain BP + OSD
      (NN metric): 180 new messages (58 exactly jt9-only), checkable 173 = 106 slot, 67 active, 0 unconf.
      Only 10 of them sit at candidate rank ≥ 600, so raising `max_cand` is not worth it (busy receivers have 580–660
      candidates per pass, so the cap does bind, but what lies beyond it rarely decodes). The other 170 are what a
      4th pass would see: `decode_slot(passes=3)` stops before the residual is exhausted.
      Change: `decode_slot(passes=4)` default, bench flag `--passes`, C `PASSES 4`. **`nn_v2_p4` vs `nn_v2_og`:** run 2
      26,340 → 26,439 decodes, slot-confirmed 19,480 → 19,550, unconf 1 → 1; run 1 29,929 → 30,116, slot 23,090 →
      23,230, unconf 3 → 3; training 16,621 → 16,638 (jt9-only 405 → 400); AWGN 50% −22.52 (unchanged); noise 0 false.
      Smaller than the prototype's +106 slot because `osd_bp` already took part of that residual. Golden C vs Python:
      00_40m 698/698, 08_80m 588/588, run 2 02_80m 725/725 all fields; AWGN 430/430.
   2c. **Context a-priori decoding (`bp_cx`/`osd_cx`) — DONE as `nn_v2_cy`.** In a QSO the
      next message from a station usually repeats known fields: the same two calls (`K1ABC W9XYZ` + a new report)
      or at least the same sender. We take every standard message (i3 = 1/2) we decoded 30 s and 60 s earlier within
      10 Hz (`CTX_DF`) and 0.5 s DT (`CTX_DDT`) of the hypothesis and pin either the pair (bits 0–57 + i3, 61 bits)
      or the sender (bits 29–57 + i3, 32 bits) at 1.01·max|LLR|, then BP, else OSD-2; the result must keep the pinned
      bits and pass CRC. Gate: expected errors < 48 (`CTX_GATE`), NN metric only, not early-DT, tried after
      OSD/osd_bp and before CQ AP. One message per sender per slot (`_drop_ctx_dupes`): a context decode is dropped
      when a normal decode or an earlier context decode in the slot has the same sender.
      Prototype on run 2: +1,117 new decodes (bound). **Shuffle test:** the same with context taken from the wrong
      station/time: 0 decodes from 211,315 tries, so a pattern alone doesn't force a CRC-valid codeword.
      Two false decodes found and fixed on the way: `OK1VKP KN4TYK QJ11` (a second hypothesis 0.65 s off a signal
      decoded normally in the same slot → one message per sender) and `9R2NKT NI8T R IC50` ×2 (DT 2.36 vs 0.13:
      a late copy of a strong neighbour that then fed itself as context → `CTX_DDT`; same-sender |ΔDT| 30/60 s later
      is p90 0.05 s, p99 0.49 s on run 2).
      **Plausibility check** (scratch `cx_check.py`): the unpinned bits of a false decode are random and then almost
      always (98.9%) unpack as a grid, while real ones carry reports/RRR/RR73/73 or the sender's known grid.
      Run 2: 2,371 cx decodes (1,481 bp_cx, 890 osd_cx) = 1,537 known grid, 828 report/RRR/RR73/73, 6 new grid that
      fits the call's country. Run 1: 2,789 = 1,797 known grid, 988 report-type, 4 new fitting grid (UA6YW LN04,
      IW3HXR JN55, each in 2 slots).
      **`nn_v2_cy` vs `nn_v2_p4`:** run 2 26,439 → 27,930 decodes, slot-confirmed 19,550 → 20,505, unconf 1 → 1
      (`CQ 0V1YXG GG88`, the same); run 1 30,116 → 31,957, slot 23,230 → 24,449, unconf 3 → 3 (none context:
      `K11BNB` dropped out, `ZT5SRQ/R UW3THX IM59` osd_er came in through a changed residual); training recordings
      16,638 → 17,586, jt9-only 400 → 309; AWGN 50% −22.52 (unchanged; no context there); noise 0 false / 998
      (no context there either, so not a test of this stage). Golden C vs Python: 00_40m 756/756, 08_80m 600/600,
      run 2 02_80m 733/733 all fields (incl. cx decodes); AWGN 430/430.
   3. **Clean C timing — DONE (2026-11-06, Mac mini M4, `nn_v2_cy`).** `ft8nn_cli --time` prints per-slot decode
      wall time to stderr. Scratch script runs the CLI strictly one after another under `caffeinate -dims`.
      **Pitfall:** the first attempt ran while the mini was asleep in "dark wake" (no full wake since the night):
      heavy CPU there triggers `Dark Wake Thermal Emergency` sleep, giving 450–550 s "slots" and inflated CPU time
      too. `caffeinate -i` alone doesn't prevent that; check `pmset -g log` for Sleep/DarkWake during any timing run.
      Clean numbers (s per slot, wall; no sleep events in the log):

      | set | dec/slot | 1 thread mean / max (> 15 s) | 2 threads mean / max | 4 threads mean / max |
      |---|---|---|---|---|
      | validation run 1 01_40m, first 40 slots | 47.6 | 15.76 / 17.08 (37 of 40) | 11.38 / 12.12 | 9.57 / 10.98 |
      | training 00_40m, first 40 slots | 36.9 | 12.26 / 15.58 (5 of 40) | 8.86 / 11.26 | 7.30 / 9.37 |
      | training, first 20 slots of each of 10 stations | 0.6–38 | 6.00 / 16.98 (10 of 200) | – | – |

      Peak RSS 330–580 MB per context. Threads only speed up ONNX Runtime: 4 threads give 1.6×, not 4×.
      Profile of 6 busy 01_40m slots (1 thread, 15.6 s): NN 10.9 s (70%), BP/OSD incl. context AP 2.6 s, fine sync
      1.1 s, candidates 0.5 s, classical 0.3 s; **each of the 4 passes costs the same ~3.9 s** (≈ 1,200 hypotheses
      each), while passes 3–4 add only 0–6 decodes. So one core no longer keeps up on the busiest bands; 2 cores do.
      **Next speedup (exact, no result change if done right):** in passes 2–4 most candidates lie in time-frequency
      regions that no subtraction touched, so their features and NN LLRs are the same as in the pass before; cache
      them and run the NN only on hypotheses that overlap a subtracted signal or are new. Needs a check that the
      candidate search and fine sync give the same hypotheses there (candidate baseline is a local median).
   3b. **Candidate reuse in passes 2–4 — DONE as `nn_v2_rs`.** Not bit-exact, because the
      candidate baseline (median over the whole band), the noise PSD and subtraction leakage are slot-wide. So
      instead of caching LLRs: a candidate (same 3.125 Hz bin, lag and track) tried in an earlier pass is skipped
      when no decode subtracted since then lies within `reuse_df` = 62.5 Hz of it (its baseband window −1.5…8.5
      tones + fine-sync range + the subtracted signal's width); it would get nearly the same LLRs and fail again.
      Python `decode_slot(reuse_df=62.5)`, bench `--reuse-df` (0 = off), C `REUSE_DF` with a per-slot tried table.
      8 busy run-2 slots: candidates processed 19,411 → 12,852 (pass 1 −7%, pass 2 −45%, pass 3 −90%: pass 0's
      ~30 decodes touch nearly the whole band), same 330 decodes; 30 Hz lost 2 of 330.
      **`nn_v2_rs` vs `nn_v2_cy`:** run 2 27,930 → 27,929 (lost `LU3JP KR4HBQ -12`, osd_bp at −22.8 dB; the same
      message decodes in the next two slots), slot 20,505 → 20,504, unconf 1 → 1; run 1 31,957 → 31,952 (5 real
      messages lost), slot 24,449 → 24,444, unconf 3 → 3 (same 3); training recordings 17,586 identical (jt9-only 309);
      AWGN 50% −22.52 / 90% −21.28 unchanged; noise 0 false. Golden C vs Python: 756/756, 600/600, run 2 733/733,
      AWGN 430/430. Python bench time: run 2 54 → 36 min, run 1 54 → 39 min.
      **C timing (2026-11-07, same slots as 7.3, no sleep events):**

      | set | 1 thread mean / max (> 15 s) | 2 threads mean / max |
      |---|---|---|
      | validation run 1 01_40m, first 40 slots | 10.91 / 12.46 (0 of 40), was 15.76 / 17.08 | 7.92 / 9.03, was 11.38 / 12.12 |
      | training 00_40m, first 40 slots | 8.12 / 9.62 (0), was 12.26 / 15.58 | – |
      | training, first 20 slots of each of 10 stations | 4.25 / 11.82 (0 of 200), was 6.00 / 16.98 | – |

      −31% on the busiest receiver: one core now keeps up on every measured slot. Peak RSS 320–530 MB.
      Timing scripts: `data/timing/` (gitignored).
   3c. **Portable build + official ONNX Runtime — DONE 2026-11-07.** Found by the architecture study
      (`docs/arch-study.md`): Homebrew's onnxruntime 1.31.1 runs our graph 2.3–3.7x slower than the
      official release of the same version (libft8nn NN stage 2.18 → 0.94 ms per hypothesis, M4, 1 thread;
      decodes byte-identical). This also explains plan 2's old "ORT 2.4 ms vs torch 0.82 ms" gap.
      `tools/get_onnxruntime.sh` downloads the official C release for the host (macOS arm64/x86_64, Linux
      x64/aarch64) into `third_party/onnxruntime` (osx-arm64 1.31.1 tgz sha256 2e1344a1…896d6f); `c/Makefile` uses it
      by default (fallback Homebrew, then /usr/local; `ORT_PREFIX=` overrides), builds `.so` on Linux and `.dylib`
      on macOS, finds headers in `include/` (official) or `include/onnxruntime/` (Homebrew), and makes
      `-march=native` opt-in (`NATIVE=1`). Code: no variable-length arrays, no `__int128` (`unpack.c` free text by
      bit-serial long division, hex from 4-bit groups: identical to the old code on 2M random payloads, all i3/n3,
      with and without hash tables), `pthread_once`/`clock_gettime`/`timegm` behind small `_WIN32` shims, wide model
      path for ORT on Windows. Clean with `-std=c11 -pedantic -Wvla`. Linux and Windows were built later in CI (below). Golden C vs Python with the official build: 756/756, 600/600, run 2 733/733, AWGN 430/430.
      Paired timing under load (fork's benchmarks running), 6 busy 01_40m slots, 1 thread: Homebrew 11.47 / 11.47 s,
      official 9.14 / 7.61 s per slot, same 280 decodes.
      **Clean timing (2026-11-07 22:31, idle, no power events), official ORT + reuse, same slots as 7.3/7.3b:**
      busiest receiver 01_40m 1 thread 6.93 / max 7.95 s (Homebrew 10.91 / 12.46; this morning's `nn_v2_cy` 15.76),
      2 threads 6.21 / 7.05; training 00_40m 5.29 / 6.49; 10-station sample 2.78 / max 7.51 (0 of 200 over 15 s);
      peak RSS 360–570 MB; decodes byte-identical to the Homebrew run.
      **Linux CI (2026-11-08):** `.github/workflows/linux.yml`, Ubuntu 24.04 x64 + arm64: apt WSJT-X (ft8code), uv,
      `tools/get_onnxruntime.sh`, `make -C c`, pytest, `tools/ci_smoke.py` (random-weight model exported to ONNX,
      8 synthetic slots × 8 strong signals, Python `run_real.py` vs C via `golden_c.py --onnx`; the classical fallback
      metrics decode them, so the whole C pipeline is compared field by field). arm64 run locally with `act` +
      Colima: all steps pass, 25 tests + 1 skip (test_onnx, no model), C == Python 64/64. Found on the way: `.gitignore`
      `data/` also hid `ft8nn/data/` (ldpc.npz, messages.json), so a fresh clone couldn't import `ft8nn`; now
      `/data/` and both files tracked. ft8code tests now skip when WSJT-X is missing. gcc: one
      -Wmaybe-uninitialized false positive in fft.c (buffer filled by the loop above).
      **Windows CI (2026-11-11):** `.github/workflows/windows.yml`, `windows-2025` x64, 2 min 52 s. MSYS2 UCRT64
      MinGW-w64 gcc (MSVC can't build libft8nn: C99 `double complex` arithmetic, `__builtin_ctzll`), official ORT
      `onnxruntime-win-x64-1.31.1.zip` via `tools/get_onnxruntime.sh`, `make -C c CC=gcc`, then native Python:
      17 tests pass + 16 skip (ft8code), `ci_smoke.py` with `models/v3` 57 decodes, C == Python 57/57 in every field.
      `ft8nn_cli.exe` imports only KERNEL32, the UCRT `api-ms-win-crt-*` DLLs and `onnxruntime.dll` (checked in CI).
      `c/Makefile` Windows branch: `build/ft8nn.dll` + import lib, `.exe`, `onnxruntime.dll` copied next to the
      programs (no rpath on Windows), `-static-libgcc`, `-D_USE_MATH_DEFINES` (MinGW hides `M_PI` under `-std=c11`).
      Fixes the first Windows build found: `<windows.h>` defines `near`/`far` empty and `decode.c` has `int near`
      (`#undef`); `ft8nn_cli` named slots by the text after the last `/`, so Windows paths (`D:\...\x.wav`) gave the
      full path as slot name - CI matched 0 of 57, and on real `yymmdd_HHMMSS.wav` slots context a-priori would have
      been silently off. Not covered: the WSJT-X sidecar (`fake_wsjtx` uses `fork()` and POSIX pipes; it and
      `ft8nn_sidecar.cpp` also split paths on `/` only, lines 216 and 161/507).
   3d. **Architecture study (2026-11-07, full log `docs/arch-study.md`; models `runs/{tf,conv,gru,tf_l2,tf_l2kd}_v{1,2}`,
      scripts and logs in gitignored `data/tmp/`, `data/logs/`):
      keep the transformer.** Same v1 → v2 recipe on the M4 (a full transformer training takes 74 min here, not 4 h).

      | model | params | AWGN 50% | fading mod / dist | training rec. | run 2 dec / slot / unconf | NN ms/hyp (official ORT) |
      |---|---|---|---|---|---|---|
      | transformer runs/v2 | 843k | −22.52 | −20.17 / −20.17 | 17,586 | 27,930 / 20,505 / 1 | 0.97 |
      | transformer retrained (noise) | 843k | −22.49 | −20.19 / −20.22 | 17,661 | 27,955 / 20,518 / 5 | – |
      | conv-only (dilated residual) | 839k | −22.38 | −20.17 / −20.08 | 17,650 | 27,953 / 20,532 / 2 | 0.66 |
      | BiGRU | 847k | −22.37 | −20.04 / −20.04 | 17,593 | 27,898 / 20,475 / 4 | 0.56 |
      | transformer 2 layers | 512k | −22.36 | −20.22 / −20.14 | 17,611 | 27,899 / 20,486 / 6 | 0.60 |
      | same, distilled from runs/v2 | 512k | −22.41 | −20.25 / −20.09 | 17,577 | 27,849 / 20,437 / 3 | 0.60 |

      Noise 0/998 for all. Training noise: thresholds ±0.03–0.05 dB, real decodes ±0.3%, but run-2 unconf 1 vs 5 for
      the same recipe (the a-priori gates were tuned on runs/v2; any new network needs them re-checked). Every
      alternative loses 0.08–0.16 dB on AWGN and a lot on synthetic close overlaps (−16 dB: 0.62–0.75 vs 0.89), and is
      level on real data. None meets "within 0.1 dB and ≥ 1.5× faster"; distillation gave nothing over training the
      same size from labels; int8 dynamic quantization is slower than float on the official ORT (0.97–1.14 vs 0.85 ms).
      With the official ORT the NN is about half of a busy slot, so a smaller network would save only ~19%.
   4. **Message plausibility rules — DONE as `nn_v2_pl`.** Planned as "make the false-decode
      gates robust to retraining", but the evidence said otherwise: 19 run-2 false decodes of the six study models,
      re-decoded with their internal evidence (scratch `data/tmp/fa_info.py`), look exactly like real weak decodes
      inside (OSD-family confirmed: hard errors p50 27, expected errors p50 33; false: 23–36 and 13–44), so no LLR
      scaling or gate can separate them. What they say can: `message.plausible` (C `plausible77`) rejects a
      standard message that a station can't send: (1) a CQ with the R flag (`CQ K1ABC R FN42`; 0 confirmed in
      ~197k decodes, 9 of 30 false), (2) a locator past RR99 or a report code above 135 (`+112`), and, for
      OSD-family decodes only (`deep`), (3) a malformed full callsign or one whose first one/two characters are no
      amateur call block in use (`ft8nn/data/call_blocks.txt` + `c/src/call_blocks.h`, 9 + 697 blocks, from
      `tools/gen_call_blocks.py` over WSJT-X's cty.dat; only the coarse blocks are stored). A rejected result lets the
      next decoder stage try. Lesson on the way: some programs pack negative reports as codes 86..135 (T77RN on
      training 08_80m: `R3GO T77RN R-10` = 126), not WSJT-X's 5..34; a first version that allowed only 5..105 lost
      5 jt9-confirmed decodes there; fixed. Rule (3) was found on run-2 false decodes, so run 1 and the training
      recordings are the independent check. Synthetic sets: 4% of the benchmark pool's random calls have unallocated
      blocks, so deep decodes of those are lost there (AWGN 90% −21.28 → −21.24, 50% unchanged).
      **`nn_v2_pl` vs `nn_v2_rs`:** run 2 27,929 → 27,928, slot 20,504 → 20,504, unconf 1 → 0 (`CQ 0V1YXG GG88`
      gone); run 1 31,952 → 31,951, unconf 3 → 2 (`CQ KC9QFS/R R DA52` gone); training recordings 17,586 → 17,583,
      jt9-corroborated 15,420 → 15,420 (removed: `CQ 0T6RPV R GR19`, `CQ HZ9YRN R QR35`, `CQ 196KYK MK66`, jt9 had
      none); noise 0/998; AWGN 50% −22.52. Offline on the study models' run-2 decodes the rules remove 14 of 21
      unconfirmed and 4 "active" decodes that are mis-decodes with a real sender (`VP4TOV WA2BCK +112`,
      `1F8PNC K2AKA DB13` from context AP, `J11WYG/R OA4ENG -22` ×2): per model 1→0, 5→4 (tf retrain), 2→1, 4→1,
      6→2, 3→0. C == Python: 301k random payloads, golden 756/756, 599/599 (and 120-slot 08_80m), run 2 733/733,
      AWGN 429/429.
      **Extended 2026-11-12 to the other message types (with `nn_v3_cc`).** `message.plausible` checked only
      standard messages; the trigger was `FN3EPRKOCCN <...> 73`, an `i3=4` (non-standard call) `osd_bp_cc`
      decode on run 2. New, for OSD-family results (deep): the 11-character call of `i3=4` messages must be
      well-formed (`_c58_ok`: < 38^11, no space inside, at most two `/` with no empty part, the longest part has
      a letter and a digit and starts with a call block in use), and the call fields of DXpedition (`i3=0
      n3=1`) and RTTY Roundup (`i3=3`) messages get the standard messages' test (`_call_ok`); for every stage, an
      EU VHF (`i3=5`) 6-character locator past RR99XX is rejected. Free text and telemetry carry nothing to check.
      **Checked on every saved payload of every tag (runs 1-4 and the training recordings, 160,295 unique),
      also with every payload treated as if OSD had produced it:** the only one removed is an `i3=3` OSD decode
      (`TJ5DBX 5P0B M`, run 4, older tags only, unscorable); no real decode is lost, jt9-corroborated ones
      included; 54% of random `i3=4` payloads fail. Nothing in `nn_v3_cc` changes, so its runs 2-4 numbers stand
      without a re-run. **It does not remove `FN3EPRKOCCN`:** that call is well-formed, and real special-event
      calls look the same (`YO1BRANCUSI`: 2 letters, digit, 8 letters, 38 decodes, 105 jt9/JTDX mentions).
      Corroboration would catch it but loses real decodes: the call in the context (30/60 s earlier) removes 122
      real deep `i3=4` decodes, the call decoded earlier in the session 29 (first decodes of `PC40EUDXF`,
      `YO1BRANCUSI`, `SX200H`, ...) and would make decoding depend on the hash table. The two other `i3=3` OSD
      decodes (`A91TDA 2W2ZUJ`, `F73CMI IL4YAN`, no RTTY Roundup on those dates) also pass; rejecting `i3=3`
      from OSD outright would remove them at no measured cost, but would cost weak contest decodes on that
      weekend (user's call). `tools/ci_smoke.py` now decodes 51, not 57: it draws fully random payloads, and 6
      of them are `i3=5` with an impossible locator. C `plausible77` mirrors it; `tests/test_plausible.py` +3 tests.
   5. **Real noise + synthetic signals — DONE, shipped as `models/v3` (tag `nn_v3`, = `runs/rn_real_seed1`).** `tools/make_noise_bank.py`
      subtracts every `nn_v2_pl` decode from each training-recording slot (never run 1 or 2) and cuts 24 segments at
      random 200..2900 Hz with the decoder's own `Baseband.downsample`: 28,728 × 3,200 complex samples
      (`data/noise/train_bank.npy`, 701 MB). Real receiver noise, AGC, carriers, interference and undecoded weak
      signals; vs the Gaussian noise it replaces: segment power p10–p99 0.37–6.99 (Gaussian 0.83–0.95), 1-s power
      fluctuation 0.35 (0.12). `synth_bb.ChannelCfg.real_noise_p` = share of training rows whose Gaussian noise is
      replaced by a bank segment (added after the band mask); `train.py --noise-bank B --real-noise-p 0.5`.
      `train.py --seed` (new): without it the worker seeds are fixed by the validation sets built just before, so two
      runs of a recipe are bit-identical. Fine-tunes from `models/v2`, lr 2e-4, 4000 steps (21 min each, M4 MPS):

      | model (run 2) | decodes | slot-confirmed | unconf | noise false | AWGN 50% / 90% |
      |---|---|---|---|---|---|
      | `nn_v2_pl` (shipped) | 27,928 | 20,504 | 0 | 0/998 | −22.52 / −21.24 |
      | control, Gaussian only: default / seed 1 / seed 2 | 27,962 / 27,975 / 27,960 | 20,521 / 20,534 / 20,520 | 0 / 0 / 2 | 0 | −22.55 / −22.53 / −22.53 |
      | real noise p 0.5, all 10 stations: default / seed 1 | 28,178 / 28,198 | 20,681 / 20,701 | 1 / 1 | 0 | −22.50 / −22.49 |
      | real noise, 3 stations (8,616 segs) / 5 stations (14,352) | 28,154 / 28,126 | 20,671 / 20,651 | 2 / 2 | 0 | −22.49 / −22.48 |

      Real noise: +146..+176 slot-confirmed over the control mean (20,525; control spread 14), on every bank size;
      AWGN ~0.04 dB worse (90% unchanged). Learning curve flat: 3 receivers already give the whole gain, so more
      recordings for the bank aren't needed. Unconfirmed 1–2 vs control 0–2: within retraining noise; the decodes:
      `HY3TBG 5V5SMW CQ25` (osd0, same frequency as a real decode 2.9 s earlier; 3 of 4 real-noise models),
      `CQ CR4UVW PI49`, `CQ UE6AJV MC49`, `CQ R75NIA/R AL38`. Training-recording counts are no longer a clean test for
      these models (their noise is in the bank); runs 1 and 2 are.
      **Full validation of `nn_rn_real_seed1` vs `nn_v2_pl`** (`tools/validate_full.sh`, `ONNX=` = its export, 1 h 28
      min with 10 workers): run 1 31,951 → 32,244 decodes, slot-confirmed 24,444 → 24,660, unconf 2 → 1
      (`CQ KR9ZUC/R AO63`); training recordings 17,583 → 17,706, jt9-only 309 → 305 (biased, see above); AWGN 50%
      −22.52 → −22.49, 90% −21.24 → −21.24; golden C == Python with its ONNX 764/764, 607/607, run 2 733/733, AWGN
      429/429. Fails the "AWGN 50% not worse" rule by 0.03 dB (~6 of ~190 signals at −22 dB), against +0.8–0.9%
      slot-confirmed on runs 1 and 2. Run 3 (12 more unseen Kiwis, another day): `nn_v2_pl` 26,535 / 19,982 / unconf 1
      → candidate 26,688 / 20,105 / 0; runs 2 + 3 pooled unconf 1 → 1. noise10k (10,000 noise-only slots): 0 false
      for both (bound < ~3 per 10,000). Shipped 2026-11-09 (user's call: accept the 0.03 dB AWGN loss for the real-data gain).
   6. **After that: low-latency decoding for operating** (agreed 2026-11-08). Throughput is fine (busiest receiver 6.9 s
      per 15 s slot on one M4 core, 2.8 s average), but replying in the next slot needs decodes ~1.5–2 s after the
      transmission ends (13.1 s into the slot). Busy 01_40m slots, per pass: 2.4 s / 28 of 45 decodes, 2.2 s / 13,
      1.3 s / 3, 0.4 s / 1. Plan: (1) hand out each pass's decodes as soon as it finishes (C API callback), (2) start an
      early pass before the slot ends (WSJT-X decodes at ~11.8 s on partial audio), (3) parallelise the per-hypothesis
      decoding across cores (today only ONNX Runtime uses threads: 2 threads 6.9 → 6.2 s).
      **Step (1), decodes streamed per pass - DONE 2026-11-11.** `ft8nn_decode_stream()` (C API) calls back after
      every subtraction pass with that pass's decodes, then once with `pass = -1` and the context a-priori decodes:
      the only thing that ever removes an earlier pass's decode is the final `drop_ctx_dupes`, and it only drops
      context decodes, so everything else is final when its pass ends (nothing ever has to be retracted - the
      protocol cannot). `ft8nn_decode`/`ft8nn_decode_at` are unchanged; texts are now unpacked per pass (decoding
      never reads the hash table, only `unpack77` does). `ft8nn_sidecar` sends each batch at once, by frequency
      within a batch; **WSJT-X needed no change**: `Ft8Sidecar::handle_line` emits each decode line on arrival into
      `readFromStdout()`, the path `jt9`'s own multi-pass output takes. `fake_wsjtx --times FILE` logs each
      reply's arrival time. Checks: golden C vs Python under MallocScribble 764/764, 607/607, 733/733, AWGN
      429/429 (as `nn_v3_dtc`); old vs new sidecar on 12 busy slots identical as sets (614 lines); all 8
      `test_sidecar.sh` checks. **Measured** (this M1, idle, `fake_wsjtx --times`, run 1 `01_40m` first 40 slots,
      1,944 decodes, identical as sets old vs new): median slot 14.05 s (slowest 15.67 s) both; **median time to
      first decodes 14.05 -> 5.26 s, mean latency per decode 14.01 -> 7.85 s.** Per pass: pass 0 1,102 decodes
      (56.7%) out at median 5.26 s, pass 1 576 (29.6%) at 10.22 s, pass 2 88 (4.5%) at 13.30 s, pass 3 13 (0.7%)
      at 14.14 s, context 165 (8.5%) at the end, 14.05 s. **Side effect, bigger than expected on this M1:** 7 of
      40 busy slots overran WSJT-X's 15 s request timeout, which drops every reply that comes later - the old
      all-at-the-end sidecar lost those slots entirely. Decodes WSJT-X would have shown within 15 s: 1,546 of
      1,944 (79.5%) -> 1,906 (98.0%). **Next:** (a) context decodes are 8.5% here and the most QSO-relevant ones
      (a station answering), yet wait for the slot's end - first measure how often the final check actually
      drops one, then decide whether sending them with their pass is worth that risk; (b) step (3),
      parallelising pass 0, which carries 57% of the decodes; (c) passes 2-3 cost ~4 s for 5.2% of the decodes.
      **Step (3), every pass on all performance cores, decodes streamed within a pass - DONE 2026-11-11.**
      Exact: the decodes are the same in every field for any thread count (checks below). How:
      - `c/src/par.c`: a pool of worker threads per context (pthreads; Windows threads under `_WIN32`), idle on a
        condition variable between decodes; `ft8nn_open(onnx, n_threads)` now means decode threads (0 = one per core,
        on macOS one per performance core: an efficiency core would hold up the last work items), ONNX Runtime runs
        1 intra-op thread per call and our threads call `Run` concurrently on the shared session. `ft8nn_threads()`.
      - Per pass: candidate search (spectrogram frames and Costas lag rows in parallel, the same additions per
        cell), fine sync per candidate in parallel, then chunks of 32 hypotheses (`NN_CHUNK`: NN batch, classical
        metrics and every decoder stage) in parallel. Each hypothesis is decoded *speculatively*, i.e. even if the
        serial loop would have skipped it, and a serial merge replays the old loop's choices in hypothesis order
        (skip a candidate once decoded, skip a hypothesis within 3 Hz / 0.05 s of a decode of this pass, drop known
        payloads). Wasted work: ~1.4% more BP runs. ONNX Runtime's CPU time per hypothesis is the same for batches
        of 16, 32 and 128, and smaller batches keep all threads busy to the end of a pass.
      - Streaming within a pass: the calling thread merges the finished prefix between its own chunks and hands
        the merged decodes to `ft8nn_decode_stream`'s callback at once (several calls per pass now; the strongest
        candidates come first), up to the first `osd_er`/`osd_bp` decode, which the end-of-pass ghost filter may
        still drop - so texts are unpacked, and the hash table learns callsigns, in exactly the old order.
      - A pass's decodes are reported *before* they are subtracted (subtraction only prepares the next pass), and
        nothing is subtracted after the last pass. Subtraction stays serial (each fit sees the previous one's
        residual), but its alignment search runs in parallel and its two 160,000-point FFTs use a parallel FFT
        (`fft_forward_par`, the top recursion levels as independent sub-transforms, then butterflies in chunks:
        bit-identical to `fft_forward` for 2-10 threads on all four sizes we use); 22 -> ~11 ms per decode
        under load. `baseband_init`'s 192,000-point FFT uses it too.
      - `ft8nn_cli --stream` (decode with `ft8nn_decode_stream`, a `batch` line per callback on stderr), and the
        sidecar defaults to `--threads 0`. `tools/golden_c.py --threads N` (default 4) / `--stream`, so CI's smoke
        test now covers the threaded path on Linux and Windows. `wsjtx/test_sidecar.sh` compares sorted lines,
        because the order *within* a slot now depends on which thread finishes first (the set does not).
      **Checks.** C before (HEAD `51b0971`) vs after, every output field, slots in time order (context a-priori,
      hashes): training 00_40m and 08_80m, run-2 02_80m, run-1 01_40m (20 slots each), all 119 slots of the two
      drifting receivers (run 3 `03_20m`, run 4 `08_12m`, so the DT re-centring and shifted windows are covered:
      1,656 decodes at DT < -1.5 s) and 60 AWGN files - 4 threads streamed and 10 threads non-streamed, 0 lines
      differ (764 / 607 / 733 / 942 / 644 / 4,147 / 429 decodes, the first three and the AWGN count equal to the
      golden counts, the drifting two equal to `nn_v3_dtc`). Golden C vs Python run 2 02_80m 733/733 all fields
      under MallocScribble, streamed and with `--threads 0`; `ci_smoke.py` 57/57; `test_sidecar.sh` all 8 checks.
      Not yet run: the Windows build of `par.c` (CI will), `validate_full.sh` (its Python references for
      `nn_v3_dtc` are not on the mini).
      **Measured** (M4 mini, 4 performance cores, while another session's 8 Python workers kept the load average at
      14-18 - so wall times are upper bounds, and CPU time is the fair comparison): `fake_wsjtx --times`, run 1
      `01_40m` first 40 slots, 1,944 decodes, identical as sets; old sidecar (1 thread) vs new (4):
      median time to first decodes **5.20 -> 0.33 s** (p90 6.34 -> 0.40), slot done 13.00 -> 4.61 s (max 16.89 ->
      5.18), mean latency per decode 7.67 -> 1.76 s, decodes within WSJT-X's 15 s 97.1% -> 100%. Work: 6 busy
      slots, instructions retired 623.5 G (old) vs 625.1 G (new, 1 thread) and 625.3 G (4 threads), +0.3%;
      cycles +6% at 4 threads (shared caches and memory bandwidth on a loaded machine), and user CPU 477.7 ->
      537.4 s in the sidecar run, which also catches more efficiency-core time when 4 threads compete with 8
      other jobs. Peak RSS 497 -> 536 MB. Not yet measured on an idle machine or on the M1 Pro (8 performance
      cores): there pass 0 should take well under a second.
      **Next:** (a) context decodes (8.5% of a busy slot) still wait for the slot's end; (b) the early pass (6a) -
      at ~5 s per busy slot an early request no longer blocks the final one, so `--early` is worth re-testing.
      **6a. Early pass: classical vs truncation-trained transformer.** At ~11.8 s the last ~9 of the 79 symbols are
      missing (for DT ≈ 0.5 s: the third Costas sync block, symbols 72–78, plus data symbols 70–71, i.e. ~6 of the 174
      code bits; more for a later DT). Compare (a) the classical nsym 1/2/3 LLRs with the missing bits at LLR 0,
      against (b) a transformer fine-tuned from `models/v3` with random tail truncation (in some training rows the
      end of the signal is zeroed or replaced by noise, so the network learns LLR ≈ 0 for missing symbols), with a
      control fine-tune without truncation as in 7.5. Measure on truncated audio: run-2 decodes at 11.8 s,
      slot-confirmed and unconfirmed, noise-only false decodes, AWGN 50% threshold at 11.8 s. Also check that (b)
      loses nothing on full-length signals (the usual A/B + AWGN); if so, ship one network for both passes.
9. **WSJT-X integration (`wsjtx/`) — first version DONE 2026-11-09.** `ft8nn_sidecar` speaks the
   fork's external-FT8-decoder protocol (`WSJTX_FT8_DECODER`, `doc/ft8-sidecar-protocol.md`): JSON lines and the
   period's audio on stdin, JSON decode lines and `done` on stdout, stderr into the WSJT-X log. Decided along the
   way: early passes (`final: false`, about 11.8 s and 13.2 s into the period) answer at once with no decodes,
   because WSJT-X skips a request that arrives while the decoder is busy and a slow early pass would cost the full
   pass (`--early` to decode them); a re-decode (`samples: 0`) replays the last period's audio; `[nfa, nfb]` is
   honoured and decodes are reported in frequency order; our CQ a-priori decodes report `ap: 1` and context ones
   `ap: 2`, which WSJT-X shows as `a1`/`a2`; a missing model exits so WSJT-X falls back to its own `jt9`. Measured
   9-12 s per slot on the plain M1 for the busiest 40 m station (M4 mini: 6.9-8.0 s), against WSJT-X's 15 s
   request timeout - the low-latency work in 7.6 matters for live use, and a pass that overruns is logged.
   Next, if wanted: a side-by-side live run in WSJT-X against `jt9` (not tried yet - no WSJT-X build here), and
   `ft8nn_set_mycall` so WSJT-X's own callsign and AP settings can be used (plan 8).
11. **Comparison against JTDX and FT8TW — DONE 2026-11-10.** Report and reproduce commands:
    `docs/decoder-comparison.md`; numbers in "In flight" above. Three things to know before touching this again:
    - **JTDX's decoder cannot read a WAV file.** JTDX deleted WSJT-X's file-reading branch from `lib/jt9.f90`
      (its header now says "Can run as the back end of JTDX only, with data placed in a shared memory region") and
      dropped the `-8` option. The decoder subroutines are untouched, so `tools/jtdx_ft8_file.f90` re-adds file
      mode: it mirrors `lib/jt9a.f90`'s `ndiskdat` branch and fills `params_block` the way `MainWindow::decode()`
      does, then calls JTDX's own `multimode_decoder`. `tools/jtdx_build.py` clones the SourceForge tree
      (`2a0e2be`, 2022-03-01, the last public commit) into `third_party/jtdx` and builds decoder + driver with no
      Qt, hamlib or libusb; it parses the source lists out of JTDX's own CMakeLists so they track the revision,
      and patches two upstream files that modern toolchains reject (`lib/{char,int}.h` Reed-Solomon prototypes,
      `lib/jplsubs.f`'s own `SPLIT` vs the F2023 intrinsic — neither on the FT8 path). It needs gfortran, cmake,
      pkg-config, FFTW and Boost headers (`brew install gcc cmake pkg-config fftw boost`). `tools/jtdx_run.py` runs
      one process per station with all its slots in chronological order, because JTDX keeps cross-slot state (an
      even/odd message memory and a callsign hash table) and loses decodes without it. It passes absolute WAV paths (the decoder runs in
      `data/jtdx-work`; relative ones failed to open before 2026-11-10).
    - **Keep JTDX's `HideFT8Dupes` on** (its own default). With duplicate suppression off it emits weak spurious
      copies of messages it just decoded at unrelated frequencies (+8% decode lines on a spot check), and
      `verify.py` scores those as slot-confirmed because they name a station that really is there.
    - **FT8TW cannot be run and cannot be measured.** Neither FT8TW nor its upstream FT8CN publishes decoder
      source: FT8TW's tree is Java with every JNI entry point commented out, and FT8CN ships the decoder only as a
      prebuilt Android arm64 `libft8cn.so`. ft8_lib (`kgoba/ft8_lib`, `9fec6ca`, `tools/ft8lib_{build.sh,run.py}`)
      stands in on FT8CN's own README grounds (it credits Karlis Goba and says it uses lightweight rather than
      deep decoding); read that row as a floor for the family, not a measurement of the app.
12. **Full-message a-priori ("repeat") stage — DONE 2026-11-11, shipped as the default (tag `nn_v3_rpf`).**
    - **What it does** (`decoder._repeat_stage`, C `repeat_stage` in `c/src/decode.c`, method `rp` /
      `FT8NN_REPEAT`): after the last pass, on the residual, for each sender of a standard context message (30/60 s
      earlier) with no decode in this slot: fine sync at the earlier decode's (f0, dt), NN LLRs clipped to ±2, and
      ρ = Σ l·s / Σ|l| for each earlier message of that sender and ~130 competitors (same calls × R flag ×
      grid/RRR/RR73/73/reports −30..+30, plus the sender's own CQ with its grid). Accept the best if it is an
      earlier message, ρ ≥ 0.40 and it beats the runner-up of the same hypothesis by 0.15. Accepted decodes join
      the slot's decodes, so they are context for the next slots. `bench/run_real.py --repeat off` disables it.
    - **Thresholds chosen on validation run 1 only** (`tools/rp_eval.py`, which emulates the stage from saved
      decodes in minutes): with every receiver given another band's context (6 offsets, all acceptances false),
      **0 of 223,643** tests pass at (0.40, 0.15); at margin 0.10, 1; at ρ 0.35 / margin 0.15, 1. Clip 2 vs 4: same.
      Clipping is needed: interference on a few symbols gave |l| ≈ 9 on bits 0–14 only (shared by every CQ).
      The CQ competitor was added after a run 3 observation (a structural fix, not a threshold; it removed 1–2
      accepts per run).
    - **Results, full decoder** (decodes / slot-confirmed / unconf, `nn_v3` → `nn_v3_rpf`): run 1 32,244 / 24,660 /
      1 → 34,295 / 26,299 / 1; run 2 28,198 / 20,701 / 1 → **29,435 / 21,646 / 1**; run 3 26,688 / 20,105 / 0 →
      27,711 / 20,883 / 1; run 4 45,744 / 33,070 / 2 → 48,008 / 34,840 / 2; **runs 2–4 pooled 100,630 / 73,876 / 3 →
      105,154 / 77,369 / 4** (+4.5% / +4.7%). The new unconf, run 3 `<...> C29IUP +35` at 161.9 Hz, is an `osd0`
      decode (SNR estimate −55.8, the band-edge normalisation issue of plan 7.1), changed in indirectly through the
      context. 4,193 `rp` decodes on runs 2–4. Training recordings 17,706 → 18,489 (jt9-only 305 → 209). AWGN
      −22.49 (no context: unchanged by construction); noise 0/998. Golden: all four PASS. CPU: run times within
      ±1 min of `nn_v3` (38.9 / 38.7 / 52.9 min, 8 workers on the M4).
    - **Stale check** (`tools/rp_stale.py`, the risk `verify.py` cannot see: a sender who moved on, read as
      repeating): of the `rp` decodes whose sender jt9, JTDX or deep JTDX also decoded in that slot, **2,022 of
      2,025 carry the same message**; the 3 others disagree only with JTDX's own `^` hypothesis stage (jt9 decoded
      none of the three), and on each our ρ favoured our message by ≥ 0.18 on the same audio — undecidable here.
    - vs JTDX (pooled): +12.6% / +11.9% over its defaults, +7.6% / +7.2% over its ceiling; JTDX-only slot-confirmed
      pairs 3,069 → 1,719. Next candidates: the 424 second signals within 10 Hz, and the 115 `verify.py` bracketed-
      call asymmetry (user's call).
    - **Combined with DT re-centring, rebased onto `main`'s threaded decoder (tag `nn_v3_dtcrp`, the shipped
      default)**: run 1 34,295 / 26,299 / 1, run 2 29,435 / 21,646 / 1, training 18,489, AWGN −22.49 — all
      identical to `nn_v3_rpf` (the re-centring gate never arms there); run 3 28,206 / 21,266 / 1, run 4 48,433 /
      35,110 / 2; **runs 2–4 pooled 106,074 / 78,022 / 4**, +5.4% / +5.6% over `nn_v3`. Stale check: 1,936 of
      1,939 checkable `rp` decodes agree (the same 3 `^`-only disputes). On the threaded C, the last pass is now
      subtracted when there is context (`main` had dropped that subtraction; the stage needs the residual), and
      repeat decodes go out in the stream's `pass = -1` callback. Golden (MallocScribble): 00_40m 785/785,
      08_80m 627/627, run 2 02_80m 749/749 at 4 threads streaming and at 1 thread, AWGN 429/429.
    - Reproduce: thresholds `uv run python tools/rp_eval.py $V/recordings --decodes data/val/run1 [--shuffle
      --shuffle-offset N] --dump F` (needs `decodes_nn_v3.txt`); A/B `WORKERS=8 tools/ab_run2.sh nn_v3_rpf`, then
      `tools/validate_full.sh nn_v3_rpf` (the stage is the default now); stale check `python3 tools/rp_stale.py
      --decodes data/val/run2 --tag nn_v3_rpf --ref "$V/run2/recordings/{st}/decodes.txt" --ref
      "data/val/run2/{st}/decodes_jtdx.txt"`.
    - **Only competitors a station could send - DONE 2026-11-12, tag `nn_v3_ccrpc` (on top of `nn_v3_cc`; Python +
      C, golden PASS, `validate_full.sh nn_v3_ccrpc` PASS).** Found by the runs 2-4 miss profile
      (`docs/miss-profile-dtcrp.md`, `tools/miss_profile.py`, `tools/miss_replay.py`): of 614 missed exact repeats,
      320 lost only on the margin with the right message scoring best (median ρ 0.64, margin 0.10), and in 226 of
      those the runner-up was a message no station sends - `_successors` built a CQ's competitors like a QSO
      message's, so `CQ K1ABC R FN42` and `CQ K1ABC RR73/73/-12`, one 15-bit field from the truth.
      Change: `decoder._sendable` (C `sendable`) drops from the competitors what `message.plausible` rejects and a
      CQ carrying RRR/RR73/73/a report; QSO messages keep every competitor, the stale-message successors included.
      A sender whose earlier messages are all CQs is then left with no competitor, and its runner-up is
      `REPEAT_SOLO` = 0.35 (the stage took `order[1]` before; in C the runner-up was -1), i.e. it needs ρ ≥ 0.50.
      `REPEAT` stays (0.40, 0.15, 2.0). `decoder.REPEAT_SENDABLE` / bench `--repeat-competitors shipped|sendable`
      (default `sendable`) keeps the plan-12 set reachable.
      **Null on run 1 only** (`tools/rp_competitors.py --shuffle 0..5`, base `nn_v3`, the same 223,643
      wrong-receiver tests as above, which it reproduces exactly: shipped set 0 / 1 / 1 at 0.40-0.15 / margin 0.10
      / ρ 0.35): sendable set with runner-up 0 - **6** false accepts, all uncontested CQs (43,557 tests, worst wrong
      CQ ρ 0.459); runner-up 0.30 - 1; **0.35 - 0, and 1 at margin 0.10**, the same as the shipped set.
      **Emulated first** (same tool, `--real`, both sets chained from saved decodes; the shipped-set emulation
      reproduces 96% of the shipped `rp` decodes): run 1 +153 / +122 slot-confirmed / unconf 1 -> 1; runs 2-4 on
      `nn_v3_dtcrp` +533 / +437 / 4 -> 4, on `nn_v3_cc` +536 / +438 / 4 -> 4 - the two stages add up.
      **A/B on top of `nn_v3_cc`** (fork 1's stage merged; with `--repeat-competitors shipped` the merged code
      reproduces `decodes_nn_v3_cc.txt` on 2,705 of 2,705 sample lines in every field): run 2 29,588 / 21,751 / 1 ->
      **29,691 / 21,820 / 1**, run 3 28,295 / 21,324 / 1 -> **28,418 / 21,421 / 1**, run 4 48,668 / 35,249 / 2 ->
      **48,999 / 35,541 / 2**; **pooled 106,551 / 78,324 / 4 -> 107,108 / 78,782 / 4 (+557 / +458)**, the same
      four unconfirmed; predicted +536 / +438, and 89-96% of the real new `rp` decodes are ones the emulation named.
      `rp` gained 578 (479 slot / 99 active / 0 unconf), lost 63 (uncontested CQs at ρ 0.40-0.50 the floor now
      refuses). Stale check on the new `rp` decodes: 319 of 319 that jt9 / JTDX / deep JTDX also decoded carry
      the same message. Noise 0 / 998 (no context there, so not a test of this stage).
      `validate_full.sh nn_v3_ccrpc` vs `nn_v3_cc`: run 1 34,585 / 26,505 / 1 -> **34,727 / 26,615 / 1** (+142 /
      +110, the same `CQ KR9ZUC/R AO63`); training recordings 18,563 -> 18,677 (jt9-only 197 -> 198); AWGN 50% -22.49 /
      90% -21.24, identical (no context: unchanged by construction); golden C == Python under MallocScribble
      00_40m 815/815, 08_80m 629/629, run 2 02_80m 754/754 (also at 1 thread and 4 threads streaming, and run 4
      02_15m 885/885, 00_20m 898/898 - 13 gained and 6 lost `rp` decodes among them), AWGN 429/429; `ci_smoke.py`
      51/51; `wsjtx/test_sidecar.sh` all checks. 1 h 21 min on the idle M4 with 8 workers, no sleep events.
      **The 12 doubtful `rp` decodes** (an `rp` decode with a reference decoding a different sender within 10 Hz /
      0.4 s and no reference decoding ours; `docs/rp-competitors.md` section 1): none false - 8 real at a different
      spot 4-9 Hz away, 1 a real two-station collision (deep JTDX decodes ours in the same slot), 2 weak `rp`
      chains. No guard: chain decodes (last CRC decode of the sender >= 90 s back) are slot-confirmed as often as
      direct ones on every run, 0 unconfirmed.
      Reproduce: `docs/rp-competitors.md`.

    The investigation that led to it: where JTDX's 3,069 slot-confirmed
    (slot, sender) pairs that `nn_v3` misses come from (runs 2–4 pooled; `tools/diag_ref_only.py --tsv
    data/logs/jtdx_cmp/jtdx_only_slot.tsv`): 75% are below −15 dB (JTDX's SNR); 62.5% come from JTDX's `^`
    tone-matching stage, 12.0% from AP (`*`), 25.5% from plain BP/OSD; 70% are not decoded by jt9 either.
    - **1,675 (55%) are repeats:** the exact text was decoded by `nn_v3` in the same period 30 or 60 s earlier
      within 10 Hz (2,182 earlier at any distance/age). Context AP should catch these and does not.
      `tools/diag_ctx_miss.py` on 300 of them (first gate that loses each): `ctx_gate` (expected bit errors ≥ 48)
      159, no candidate within `CTX_DF`/`CTX_DDT` 68, pinned BP/OSD-2 fails 66, decodes in the residual 5, early-DT 2.
    - **Full-message test** (all 77 bits known, so the whole codeword is; ρ = Σ l·s / Σ|l| between the NN LLRs and
      the hypothesis codeword s = ±1), at the earlier decode's (f0, dt) with no candidate search: right message
      median ρ 0.80, ≥ 0.40 for 90.7%; the right message on 3 receivers on other bands (station absent) max 0.36,
      0 of 899 ≥ 0.40. Projected order of magnitude: +1,500 slot-confirmed on runs 2–4 if it lands near JTDX's rate.
    - **Risk: stale messages.** A payload 1 bit away is still 40–50 codeword bits away, but a perfect signal gives
      such a neighbour ρ up to 0.54 (0.61 at 2–4 bits); wrong messages on the right audio reached 0.70. A fixed
      threshold would emit the previous over after the sender moved on (`-10` → `R-10`), and `verify.py` would
      score it slot-confirmed. The stage must beat the hypothesis's successors (report → R-report → RR73/RRR/73)
      by a margin. False-decode bound: the wrong-receiver test at context-AP scale (211,315) and `noise10k`.
    - **Scorer asymmetry (not fixed, user's call):** 115 of the 3,069 are decoded by us too — we print a resolved
      hashed call as `<SX200H>`, JTDX as `SX200H`, and `verify.py`'s `sender()` treats any `<` as uncheckable.
      Fixing it changes every recorded number slightly in our favour; `verify.py` mirrors the data repo's copy.
    - Also open: 424 (14%) are a second signal within 10 Hz of one we decode (JTDX gets both).
10. Optional later work:
   - More AP types: the a2 MyCall type needs the operator's own callsign, and that is only useful inside an app
     (an `ft8nn_set_mycall` would slot into the same pinned-bit machinery as context AP).
   - Retrain on a bigger model (cheap now: 74 min per transformer training on the M4).
   - Train with an AGC/passband-tilt impairment.

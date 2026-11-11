# Status and plan (resume here)

Last updated 2026-11-11 (**decodes now stream to WSJT-X pass by pass**, plan 6 step (1): on 40 busy slots
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
- Last commit: `wsjtx/ft8nn_sidecar`, the decoder as WSJT-X's external FT8 decoder (plan 9). Next: low-latency
  decoding (plan 7.6, with 6a: early pass, classical vs truncation-trained transformer).
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
`context=` in Python, `ft8nn_decode_at` in C; the bench passes it, see plan 7.2c).

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
| 3 (unseen sites, other day) | jt9 | 23,524 | 17,888 | 17 | – | – |
| 3 | nn_v2_pl | 26,535 (+12.8%) | 19,982 (+11.7%) | 1 | | |
| 3 | **nn_v3** | **26,688 (+13.5%)** | **20,105 (+12.4%)** | **0** | | |
| 3 | **nn_v3_dtc** † | **26,993 (+14.7%)** | **20,342 (+13.7%)** | **0** | | |
| 4 (unseen sites, high bands, afternoon) | jt9 | 38,432 | 28,053 | 29 | – | – |
| 4 | **nn_v3** | **45,744 (+19.0%)** | **33,070 (+17.9%)** | **2** | 948 | 8,259 |
| 4 | **nn_v3_dtc** † | **46,248 (+20.3%)** | **33,397 (+19.0%)** | **2** | 727 | 8,542 |
| **2 + 3 + 4 pooled (unseen)** | jt9 | 86,247 | 63,934 | 62 | – | – |
| **2 + 3 + 4 pooled (unseen)** | **nn_v3** | **100,630 (+16.7%)** | **73,876 (+15.6%)** | **3** | | |
| **2 + 3 + 4 pooled (unseen)** | **nn_v3_dtc** † | **101,439 (+17.6%)** | **74,440 (+16.4%)** | **3** | | |

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
   - Golden test `tools/golden_c.py` (uses `models/v3/ft8nn.onnx`; compares against tag `nn_v3` by default; `nn_v3`
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
   1c. **Co-channel collisions: the strong signal's subtraction destroys the weak one's sync - DIAGNOSED
      2026-11-11, not implemented.** With 1b in, the remaining jt9-only decodes were re-profiled from scratch.
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
      **Next: post-subtraction forced retry.** The decoder already knows every cell it subtracted from, so this
      needs no new detector - only a gate, since forcing hypotheses is exactly where ghosts of the subtracted
      signal live (`_is_ghost` already exists for that) and OSD on a forced hypothesis can fit CRC-valid
      garbage. Cost is the open question: the +-7 Hz / 0.5 Hz grid used above is 377 hypotheses per subtracted
      decode, ~50 decodes per busy slot, which is several times the whole current per-pass budget. Narrow it
      first - a 1 Hz df step and dt +- 1 sample is 39, and the cell can be skipped unless residual energy at
      that frequency is still above the local noise floor.

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
      path for ORT on Windows. Clean with `-std=c11 -pedantic -Wvla`. Linux and Windows not built yet (no Docker or
      Linux box here). Golden C vs Python with the official build: 756/756, 600/600, run 2 733/733, AWGN 430/430.
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
10. Optional later work:
   - More AP types: the a2 MyCall type needs the operator's own callsign, and that is only useful inside an app
     (an `ft8nn_set_mycall` would slot into the same pinned-bit machinery as context AP).
   - Retrain on a bigger model (cheap now: 74 min per transformer training on the M4).
   - Train with an AGC/passband-tilt impairment.

# Status and plan (resume here)

Last updated 2026-11-07 (portable C build + official ONNX Runtime committed, plan 7.3c; candidate reuse `nn_v2_rs`
and `collect/` committed).

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

- Last commits: `collect/` data-collection tools (docs/data-collection.md), candidate reuse `nn_v2_rs` (plan 7.3b).
- Architecture study finished 2026-11-07 (plan 7.3d): keep the transformer. Its code (`train.py --kind conv|gru`,
  `--teacher`, `model.build`) is merged; models in `runs/{tf,conv,gru,tf_l2,tf_l2kd}_v{1,2}`, scripts `data/tmp/` (gitignored).
- Loop log (finished): `nn_v2_eh` pass, `nn_v2_ob` fail (1 ghost), `nn_v2_og` pass, `nn_v2_p4` pass → run 2
  slot-confirmed 19,550 ≥ 19,500 with unconf 1, noise 0 false: loop stopped. Commits need the user's OK again.
  Next candidates: training with real noise + synthetic signals (exact labels, real AGC/QRM), retraining.

## Where we are

Best model: `runs/v2/model.pt`. It is gitignored and exists only on this machine, so back it up.
Recipe: v1 (6000 steps, `--cfg v1`), then fine-tune with
`train.py --cfg v2 --init runs/v1/model.pt --lr 5e-4 --steps 8000`. That took about 4 h on the M1 Pro and takes
74 min on the M4 mini (MPS). `--kind conv|gru` trains the study's alternatives, `--teacher CKPT` distils from a checkpoint.

Decoder defaults: `decoder.decode_slot`. These are 4 passes with subtraction, 2 fine-sync hypotheses,
NN metric with classical nsym 1/2/3 fallback (`--also-classical`), OSD order 2 gated on expected errors < 40,
CQ a-priori decoding (expected errors < 44), burst-erasure retry (expected errors < 15) with a wide-window OSD stage
(`osd_er`), OSD on summed BP posteriors after iterations 1–2 (`osd_bp`), ghost filter, candidate search 150–3100 Hz plus an early-DT track down to −2.5 s, candidate reuse in passes 2–4
(`reuse_df` 62.5 Hz, plan 7.3b),
context a-priori decoding from our own decodes 30/60 s earlier (`bp_cx`/`osd_cx`, expected errors < 48; needs
`context=` in Python, `ft8nn_decode_at` in C; the bench passes it, see plan 7.2c).

### Results vs WSJT-X jt9 (`-d 3`)

50% decode threshold, DT ≤ 0.5 s, 10 signals per slot:

| set | jt9 | v1 | **v2** |
|---|---|---|---|
| AWGN | −21.31 | −22.20 | **−22.37** |
| moderate fading (0.5 Hz / 1 ms) | −19.40 | −20.12 | **−20.12** |
| disturbed fading (1 Hz / 2 ms) | −19.19 | −19.85 | **−20.09** |
| noise only, 100 slots | – | 0 false | **0 false** |

Real KiwiSDR recordings (10 stations, 1197 slots, `../ft8-training-data/recordings`, read-only; another agent owns it):
**Current default `nn_v2_rs` decodes 17586, jt9 decodes 15729 (+11.8%).**
2166 decodes are ours only (2087 name a callsign jt9 heard at that station); 309 are jt9 only.
History: `nn_v2` 16055 (jt9-only 663) → baseline fix `nn_v2_bm` 16336 (jt9-only 512) → erasure `nn_v2_er` 16443 (jt9-only 466) → fmax 3100 `nn_v2_f31` 16452 (jt9-only 463) → early-DT + osd_er `nn_v2_eh` 16460 (463) → osd_bp `nn_v2_og` 16621 (405) → 4 passes `nn_v2_p4` 16638 (400) → context AP `nn_v2_cy` 17586 (309).
08_80m: 3477 vs jt9 3258. AWGN 50% with `nn_v2_og`/`nn_v2_p4`/`nn_v2_cy`/`nn_v2_rs`: −22.52 (90%: −21.28).

### Validation set with ground truth (`../ft8-validation-set`, added by the user 2026-11-04)

KiwiSDR recordings plus every PSK Reporter FT8 spot logged at the same time; its `README.md` explains it all.
Run 1 (`recordings/`) = 9 training-set Kiwis on other bands/times; **run 2 (`run2/recordings`) = 9 Kiwis never
used for training** (the honest generalisation test). 120 slots per station, jt9 baseline in each `decodes.txt`.
Its `tools/verify.py` puts each decode's sender into a tier: `slot` (spotted by others in that slot) > `active`
> `repeat` > `unconf`. **`unconf` ≈ false decodes** (jt9: 13 of 27,462 checkable in run 1, 16 of 23,958 in run 2).
Treat the folder as read-only; our decodes go to `data/val/run{1,2}/<station>/decodes_<tag>.txt`:

```sh
V=/Users/jps/tmp/ft8-validation-set
uv run python bench/run_real.py $V/recordings --model runs/v2/model.pt --also-classical --tag nn_v2_rs --workers 6 --out data/val/run1
uv run python bench/run_real.py $V/run2/recordings --model runs/v2/model.pt --also-classical --tag nn_v2_rs --workers 6 --out data/val/run2
python3 collect/verify.py --decodes data/val/run1 --tag nn_v2_rs --list
python3 collect/verify.py --recordings $V/run2/recordings --spots $V/run2/pskr_spots.jsonl --decodes data/val/run2 --tag nn_v2_rs --list
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
| 1 | **nn_v2_rs** | **31,952 (+15.2%)** | **24,444 (+13.7%)** | **3** | | |
| 2 (unseen sites) | jt9 | 24,291 | 17,993 | 16 | – | – |
| 2 | nn_v2_er | 25,994 (+7.0%) | 19,244 (+7.0%) | 1 | 886 | 2590 |
| 2 | nn_v2_f31 | 26,032 (+7.2%) | 19,281 (+7.2%) | 1 | 850 | |
| 2 | nn_v2_eh | 26,091 (+7.4%) | 19,320 (+7.4%) | 1 | | |
| 2 | nn_v2_og | 26,340 (+8.4%) | 19,480 (+8.3%) | 1 | | |
| 2 | nn_v2_p4 | 26,439 (+8.8%) | 19,550 (+8.7%) | 1 | | |
| 2 | nn_v2_cy | 27,930 (+15.0%) | 20,505 (+14.0%) | 1 | | |
| 2 | **nn_v2_rs** | **27,929 (+15.0%)** | **20,504 (+14.0%)** | **1** | | |

The gain is larger on unseen receivers than on training sites, and it holds up against outside spots:
false decodes are below jt9's. **Caveat for `nn_v2_cy`:** a context decode's sender is by construction a station
we decoded 30/60 s earlier, so the spot tiers can't catch a false context decode well; its false-alarm evidence is
the shuffle test and the free-bits plausibility check in plan 7.2c. Odd: jt9-only includes many strong signals (run 1: 289 at ≥ 0 dB, run 2: 114),
e.g. run 2 00_40m `E70A KU2X FN13` at +4…+9 dB missed in 3 slots. Many were above our old 3000 Hz search limit
(fixed by `nn_v2_f31`: run 2 strong jt9-only 114 → 82); the rest are plan step 7.1.

### Reproduce the numbers

```sh
uv run pytest -q                                                    # 26 tests
uv run python bench/run_ours.py data/bench/awgn --model runs/v2/model.pt --also-classical --tag nn_v2_rs --workers 6
uv run python bench/report.py data/bench/awgn --dt-max 0.5
uv run python bench/run_real.py /Users/jps/tmp/ft8-training-data/recordings --model runs/v2/model.pt --also-classical --tag nn_v2_rs --workers 6
uv run python bench/run_real.py /Users/jps/tmp/ft8-training-data/recordings --compare-only --tag nn_v2_rs
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
   `uv run python tools/export_onnx.py runs/v2/model.pt runs/v2/ft8nn.onnx` writes a 3.5 MB graph
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
   - Golden test `tools/golden_c.py` (uses `runs/v2/ft8nn.onnx`; compares against tag `nn_v2_rs` by default; last
     verified with `nn_v2_rs`: 756/756 + 600/600 real, run 2 02_80m 733/733, 430/430 AWGN; the real-data golden
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
   `tools/diag_strong_miss.py runs/v2/model.pt nn_v2_bm -9 2000` over all 98 slots with strong jt9-only misses:
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
        `uv run python tools/diag_strong_miss.py runs/v2/model.pt nn_v2_f31 0 2000 --root $V/run2/recordings --out data/val/run2`
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
8. Optional later work:
   - More AP types: the a2 MyCall type needs the operator's own callsign, and that is only useful inside an app
     (an `ft8nn_set_mycall` would slot into the same pinned-bit machinery as context AP).
   - Retrain on a bigger model (cheap now: 74 min per transformer training on the M4).
   - Train with an AGC/passband-tilt impairment.

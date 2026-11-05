# Status and plan (resume here)

Last updated 2026-11-04 (ONNX export, C library, candidate-baseline fix, burst-erasure retry done; validation set scored).

## Where we are

Best model: `runs/v2/model.pt`. It is gitignored and exists only on this machine, so back it up.
Recipe: v1 (6000 steps, `--cfg v1`), then fine-tune with
`train.py --cfg v2 --init runs/v1/model.pt --lr 5e-4 --steps 8000`. That takes about 4 h on M-series with MPS.

Decoder defaults: `decoder.decode_slot`. These are 3 passes with subtraction, 2 fine-sync hypotheses,
NN metric with classical nsym 1/2/3 fallback (`--also-classical`), OSD order 2 gated on expected errors < 40,
CQ a-priori decoding (expected errors < 44), and burst-erasure retry (expected errors < 15).

### Results vs WSJT-X jt9 (`-d 3`)

50% decode threshold, DT ≤ 0.5 s, 10 signals per slot:

| set | jt9 | v1 | **v2** |
|---|---|---|---|
| AWGN | −21.31 | −22.20 | **−22.37** |
| moderate fading (0.5 Hz / 1 ms) | −19.40 | −20.12 | **−20.12** |
| disturbed fading (1 Hz / 2 ms) | −19.19 | −19.85 | **−20.09** |
| noise only, 100 slots | – | 0 false | **0 false** |

Real KiwiSDR recordings (10 stations, 1197 slots, `../ft8-training-data/recordings`, read-only; another agent owns it):
**v2 + median baseline + burst erasure (tag `nn_v2_er`, current default) decodes 16443, jt9 decodes 15729 (+4.5%).**
1180 decodes are ours only (94% name a callsign jt9 heard at that station); 466 are jt9 only.
History: `nn_v2` 16055 (jt9-only 663) → baseline fix `nn_v2_bm` 16336 (jt9-only 512) → erasure `nn_v2_er` 16443 (jt9-only 466).
08_80m: 3314 vs jt9 3258.

### Validation set with ground truth (`../ft8-validation-set`, added by the user 2026-11-04)

KiwiSDR recordings plus every PSK Reporter FT8 spot logged at the same time; its `README.md` explains it all.
Run 1 (`recordings/`) = 9 training-set Kiwis on other bands/times; **run 2 (`run2/recordings`) = 9 Kiwis never
used for training** (the honest generalisation test). 120 slots per station, jt9 baseline in each `decodes.txt`.
Its `tools/verify.py` puts each decode's sender into a tier: `slot` (spotted by others in that slot) > `active`
> `repeat` > `unconf`. **`unconf` ≈ false decodes** (jt9: 13 of 27,462 checkable in run 1, 16 of 23,958 in run 2).
Treat the folder as read-only; our decodes go to `data/val/run{1,2}/<station>/decodes_<tag>.txt`:

```sh
V=/Users/jps/tmp/ft8-validation-set
uv run python bench/run_real.py $V/recordings --model runs/v2/model.pt --also-classical --tag nn_v2_er --workers 6 --out data/val/run1
uv run python bench/run_real.py $V/run2/recordings --model runs/v2/model.pt --also-classical --tag nn_v2_er --workers 6 --out data/val/run2
python3 $V/tools/verify.py --decodes data/val/run1 --tag nn_v2_er --list
python3 $V/tools/verify.py --recordings $V/run2/recordings --spots $V/run2/pskr_spots.jsonl --decodes data/val/run2 --tag nn_v2_er --list
```

Scores (`nn_v2_er` vs jt9 baseline, checkable = sender call is spotted-checkable):

| run | decoder | decodes | slot-confirmed | unconf (≈ false) | jt9-only | ours-only (99% callsign-corroborated) |
|---|---|---|---|---|---|---|
| 1 (training sites) | jt9 | 27,730 | 21,492 | 13 | – | – |
| 1 | **nn_v2_er** | **29,372 (+5.9%)** | **22,669 (+5.5%)** | **3** | 1543 | 3185 |
| 2 (unseen sites) | jt9 | 24,291 | 17,993 | 16 | – | – |
| 2 | **nn_v2_er** | **25,994 (+7.0%)** | **19,244 (+7.0%)** | **1** | 886 | 2590 |

The gain is larger on unseen receivers than on training sites, and it holds up against outside spots:
false decodes are below jt9's. Odd: jt9-only includes many strong signals (run 1: 289 at ≥ 0 dB, run 2: 114),
e.g. run 2 00_40m `E70A KU2X FN13` at +4…+9 dB missed in 3 slots. Not yet diagnosed (see plan step 7).

### Reproduce the numbers

```sh
uv run pytest -q                                                    # 26 tests
uv run python bench/run_ours.py data/bench/awgn --model runs/v2/model.pt --also-classical --tag nn_v2_er --workers 6
uv run python bench/report.py data/bench/awgn --dt-max 0.5
uv run python bench/run_real.py /Users/jps/tmp/ft8-training-data/recordings --model runs/v2/model.pt --also-classical --tag nn_v2_er --workers 6
uv run python bench/run_real.py /Users/jps/tmp/ft8-training-data/recordings --compare-only --tag nn_v2_er
```

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
- 5 jt9 decodes are above 3000 Hz, our search limit. `--fmax 3100` exists but was not A/B'd on its own.

## Plan

1. **ONNX export of the NN — DONE.**
   `uv run python tools/export_onnx.py runs/v2/model.pt runs/v2/ft8nn.onnx` writes a 3.5 MB graph
   (input `features` (B,79,48) float32 → output `llr` (B,174), dynamic batch, opset 17) and checks it with onnxruntime:
   max |diff| 1.9e-5 on 512 v2-mix samples, 100% sign agreement. `tests/test_onnx.py` covers it.
   `_conv_bl` switches to plain `Conv1d` under `torch.onnx.is_in_onnx_export()` (unfold doesn't export).
   Speed: ORT CPU 0.75 ms/candidate vs torch 0.35 ms (batch 128). Fine for now; try ORT transformer fusion later if needed.
2. **C library `libft8nn` — first version DONE, matches Python exactly.**
   - Code in `c/` (`make -C c` → `c/build/libft8nn.{a,dylib}`, `c/build/ft8nn_cli`). Own C for everything,
     including a mixed-radix 2/3/5 FFT; the only dependency is ONNX Runtime (`brew install onnxruntime`).
     API in `c/include/ft8nn.h`: `ft8nn_open(onnx, n_threads)`, `ft8nn_decode(ctx, audio, n, out, max)`,
     `ft8nn_reset_hashes`, `ft8nn_close`. One context per thread; the context's hash table learns callsigns.
     `ft8nn_decode` normalises audio to unit std internally (as `bench/run_real.py` does).
   - LDPC tables: `c/src/ldpc_tables.h`, generated by `tools/gen_c_tables.py`.
   - Golden test `tools/golden_c.py` (uses `runs/v2/ft8nn.onnx`; compares against tag `nn_v2_er` by default; re-verified
     after the baseline fix: 1264/1264 real, 418/418 AWGN):
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
   1. **Strong jt9-only misses on the validation set — IN PROGRESS.** Run 1: 289 at ≥ 0 dB, of which **119 are
      above 2990 Hz**; run 2: 114, of which 33. jt9 decodes up to f0 = 3052 Hz (169 decodes at 3000–3024 Hz across
      all sets); our candidate search stops at `fmax` = 3000. Spot check with `fmax=3100`: both test slots gain exactly
      the missing jt9 decode (run 2 00_40m 222900 `E70A KU2X FN13` 3014 Hz; run 1 01_40m 221800 3001 Hz).
      A/B running: tag `nn_v2_f31` (`--fmax 3100`) on validation run 2 + noise, then run 1 + training recordings.
      If kept: change the C default (`find_candidates(..., 3000.0, ...)` in `c/src/decode.c`) and Python default, golden.
      Then diagnose the remaining strong misses with `tools/diag_strong_miss.py ... --root <recordings> --out data/val/run2`
      (script now takes `--root`, `--out`, `--max-snr`).
   2. Deeper OSD or more AP types for expected-error 13–40 FAILs; training with
   slow (seconds) signal fades.
   3. Re-measure C decode time per slot on an idle machine, busiest stations (00_40m, validation 01_40m), single core
      and `--threads 2/4`. README "System requirements" quotes 18 s for one busy slot measured under load and
      recommends ≥ 2 cores; replace with clean numbers.
8. Optional later work:
   - More AP types: the a2 MyCall type needs a callsign context, and that is only useful inside an app.
   - Retrain on a bigger model.
   - Train with an AGC/passband-tilt impairment.

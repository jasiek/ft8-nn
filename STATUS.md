# Status and plan (resume here)

Last updated 2026-11-04 (ONNX export done).

## Where we are

Best model: `runs/v2/model.pt`. It is gitignored and exists only on this machine, so back it up.
Recipe: v1 (6000 steps, `--cfg v1`), then fine-tune with
`train.py --cfg v2 --init runs/v1/model.pt --lr 5e-4 --steps 8000`. That takes about 4 h on M-series with MPS.

Decoder defaults: `decoder.decode_slot`. These are 3 passes with subtraction, 2 fine-sync hypotheses,
NN metric with classical nsym 1/2/3 fallback (`--also-classical`), OSD order 2 gated on expected errors < 40,
and CQ a-priori decoding (expected errors < 44).

### Results vs WSJT-X jt9 (`-d 3`)

50% decode threshold, DT ≤ 0.5 s, 10 signals per slot:

| set | jt9 | v1 | **v2** |
|---|---|---|---|
| AWGN | −21.31 | −22.20 | **−22.37** |
| moderate fading (0.5 Hz / 1 ms) | −19.40 | −20.12 | **−20.12** |
| disturbed fading (1 Hz / 2 ms) | −19.19 | −19.85 | **−20.09** |
| noise only, 100 slots | – | 0 false | **0 false** |

Real KiwiSDR recordings (10 stations, 1197 slots, `../ft8-training-data/recordings`, read-only; another agent owns it):
**v2 decodes 16055, jt9 decodes 15729.** 989 decodes are ours only, and 93% of those name a callsign that jt9 heard at that station.
663 decodes are jt9 only, and 93 of those are jt9 AP (`a1`/`a2`) decodes.

### Reproduce the numbers

```sh
uv run pytest -q                                                    # 25 tests
uv run python bench/run_ours.py data/bench/awgn --model runs/v2/model.pt --also-classical --tag nn_v2 --workers 6
uv run python bench/report.py data/bench/awgn --dt-max 0.5
uv run python bench/run_real.py /Users/jps/tmp/ft8-training-data/recordings --model runs/v2/model.pt --also-classical --tag nn_v2 --workers 6
uv run python bench/run_real.py /Users/jps/tmp/ft8-training-data/recordings --compare-only --tag nn_v2
```

Real decode run is about 20 min with 6 workers (about 4 s/slot/core). The synthetic sets take about 5 min each.
Don't run training and benchmarks at the same time, because memory pressure stalls both.

### What we know about the remaining real-data misses

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
2. **C library `libft8nn`** (about 2–3 h for a first version that matches Python).
   - API sketch:
     `ft8nn_ctx* ft8nn_open(const char* onnx_path)`,
     `int ft8nn_decode(ctx, const float* audio12k, int n, ft8nn_decode_t* out, int max_out)`,
     `void ft8nn_close(ctx)`.
     `ft8nn_decode_t` holds `{char text[40]; float f0, dt, snr; uint8_t bits77[10]; int method;}`.
   - Port these from Python: candidate search, full-slot FFT downsampler, `fine_sync2`, `features`, BP, OSD-2,
     CRC, `unpack77` with the hash table, and subtraction.
   - The NN runs through the ONNX Runtime C API.
   - Constraint: compiling cloned third-party C was blocked earlier. Writing our own C is fine.
     User approved (2026-11-04): `brew install onnxruntime`; FFT = our own mixed-radix C (no FFTW, no vDSP).
   - Golden tests: C vs Python decodes on `data/bench/awgn` plus 20 real slots, which must give identical message sets.
3. **Local noise whitening** (about 1 h, plus a 25 min real A/B run). It targets the `00_40m` misses described above.
   Keep the change only if the real total goes up and the AWGN 50% point does not get worse.
4. Optional later work:
   - More AP types: the a2 MyCall type needs a callsign context, and that is only useful inside an app.
   - Retrain on a bigger model.
   - Train with an AGC/passband-tilt impairment.

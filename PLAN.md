# CPU Efficiency Plan (`ft8-nn`)

## 1. Current State & Fresh CPU Profile (`HEAD` = `models/v4`)

### Terminology
- **Slot:** One 15-second FT8 transmit/receive period (`180,000` audio samples at `12 kHz`).
- **Candidate:** A coarse `(f0, dt)` frequency/time-offset peak found from the 3.125 Hz spectrogram by correlating FT8's three 7-symbol Costas sync arrays (`find_candidates`).
- **DT:** Time offset (seconds) of a signal relative to the nominal `0.5 s` start into the slot.
- **Hypothesis (`SyncHyp`):** A finely synchronized `(f0, dt)` baseband waveform produced by semi-coherent Costas grid search (`fine_sync2`, up to `2` hypotheses per candidate) and evaluated by the soft demodulator.
- **LLR (Log-Likelihood Ratio):** Per-bit soft confidence $\log P(b=1)/P(b=0)$ for each of the 174 codeword bits; **expected bit errors (`exp_err`)** $= \sum_{i=0}^{173} 1/(1 + e^{|\text{LLR}_i|})$ (~70–82 on pure noise, `< 45` on weak decodable signals).
- **BP (Belief Propagation):** Iterative sum-product decoder for FT8's `LDPC(174, 91)` code (`bp_decode`, up to 30 iterations with early stopping).
- **OSD (Ordered-Statistics Decoding):** Algebraic Gaussian-elimination fallback decoder (`osd_decode`, order 2 or 3) run when BP fails on low-`exp_err` neural LLRs; **`osd_bp`** runs OSD on the summed BP posteriors of iterations 1–2.
- **AP (A-Priori Decoding):** Pinning known payload bits (`CQ` or callsigns from decodes 30–180 s earlier, `bp_cx`/`osd_cx`) before BP/OSD.
- **Slot-confirmed / Unconfirmed (`unconf`):** Independent PSK Reporter validation tiers (`collect/verify.py`): *slot-confirmed* means another receiver worldwide spotted that sender in the same 15 s slot; *unconfirmed* means the callsign was never spotted during the session and is our primary real-air false-decode proxy.

---

### Why CPU Grew Since the Mid-November Cuts
In mid-November (on `models/v3` before the final sensitivity push), 40 busy 40 m slots (`01_40m` of validation run 1) took `4.52 s/slot` at 1 thread on the Mac mini M4 (`~8.1 s/slot` on the 8-core Apple M1). Since then, five sensitivity features raised unseen validation decodes by **+5.2%** (`+1,554` on unseen run 2), at the cost of expanding hypothesis evaluations per busy slot from `~3,000` to **`~4,750–5,200`**:
1. Bandwidth-scaled candidate budget (`max_cand` raised `600 -> 900` at default `150–4000 Hz`, even when receiver audio rolls off at 3 kHz).
2. Post-stage targeted pass (`post_pass` = `pass 5`, scanning all candidates within `±62.5 Hz` whenever co-channel, refine, Chase, or repeat adds a decode, with `tried = NULL`).
3. Micro-shift 3×3 test-time augmentation (`refine_stage`, up to `64 seeds × 8 = 512` extra neural hypotheses per slot).
4. Wider burst-erasure BP gate (`erase_bp_gate` raised `15 -> 30`), 3-minute context lookback (`CTX_AGES` `2 -> 6` slots), and fallback `OSD-3` in `try_ctx` and `chase_stage`.
5. Live WSJT-X `--early` + `--final` mode running two early passes at `11.8 s` and then re-running Pass 0 (`~1,500` hypotheses) from scratch at `14.4 s`.

---

### Fresh Baseline Profile (`Apple M1`, 8 cores = 4P + 4E, idle, `models/v4/ft8nn.onnx`)
Measured on 6 consecutive busy 40 m slots of validation run 1 (`01_40m`, `261104_221745`..`221900`, averaging **54.2 decodes/slot**, context/Chase/repeat/`post_pass` active):

| Execution Mode (`ft8nn_cli`) | Mean Wall / Slot | Max Wall / Slot | Process CPU / Slot | Pass Hyps / Slot | Notes |
|---|---:|---:|---:|---:|---|
| `--threads 1` (single P-core) | **13.89 s** | **14.53 s** | **13.89 s** | 4,210 (+537 post) | Only `0.47 s` below WSJT-X's 15 s timeout on 1 core |
| `--threads 0` (4 P-cores) | **4.38 s** | **4.61 s** | ~16.8 s | 4,210 (+537 post) | 3.17× wall speedup across 4 P-cores |
| `--threads 0 --early 141696` (live sidecar sequence) | **4.92 s** (`2.19 s` early + `2.73 s` final) | **5.21 s** (`2.21 s` early) | ~18.4 s | 4,831 (+361 post) | Early pass has only `0.4 s` margin on M1 before WSJT-X's `14.4 s` final request (`2.6 s` budget; hits `2.9–3.0 s` on busiest slots) |

#### Where Every Second Goes (`--threads 1`, `13.89 s/slot` mean)

| Pipeline Component | Time / Slot (1T) | Share | Volume / Slot & Unit Cost |
|---|---:|---:|---|
| **1. Neural Net (`SoftDemod` in ONNX Runtime + `nn_features`)** | **9.85 s** | **70.9%** | `~4,780` hyps (`4,210` passes + `306` `cc` + `231` `rf` + `~33` `rp`) × **`2.06 ms/hyp`** (`2.00 ms` ORT `Run` + `0.062 ms` `nn_features`) |
| **2. Classical Fallback Metrics + Classical BP (`k = 1..3`)** | **1.65 s** | **11.9%** | `4,210` hyps × (`0.125 ms` `symbol_spectra8` + `coherent_llr(1,2,3)` = `0.53 s` + `3 × 0.089 ms` noise `bp_decode` = `1.12 s`) |
| **3. Fine Sync, Baseband & Co-Channel `data_sync`** | **1.06 s** | **7.6%** | `~2,105` cands × `0.33 ms` `fine_sync2` + `8 × 20.4 ms` `baseband_init` + `51 × 5.5 ms` `data_sync` |
| **4. Signal Subtraction (`subtract_signal`)** | **0.67 s** | **4.8%** | `~51` subtracted signals × **`13.3 ms`** (`0.9 ms` `complex_wave` + `4.5 ms` 32-shift `block_energy` + `7.9 ms` two `160k`-pt FFTs) |
| **5. Neural-Metric BP, OSD, `osd_bp`, `ctx`, `ap`, `erase`** | **0.47 s** | **3.4%** | `~4,780` NN BP runs (`16,431` total BP − `12,630` classical BP) + `45.3` OSD runs |
| **6. Coarse Candidate Search (`find_candidates`)** | **0.14 s** | **1.0%** | `4.7` passes × `30–35 ms` (`369 × 3840`-pt FFTs + `median7` Costas grid) |
| **7. Chase + Repeat Overhead (non-NN)** | **0.05 s** | **0.4%** | Cross-slot LLR matching & codeword correlation |

#### Per-Pass Yield vs Cost (`--threads 1`, 6-slot mean)
- **Pass 0:** `1,371` hyps (`3.86 s`) $\to$ **30.5 decodes** (`7.9` decodes/s)
- **Pass 1:** `1,459` hyps (`3.91 s`, only ~27 cands skipped by `reuse_df`) $\to$ **15.3 decodes** (`3.9` decodes/s)
- **Pass 2:** `884` hyps (`2.41 s`) $\to$ **4.0 decodes** (`1.7` decodes/s)
- **Pass 3:** `288` hyps (`0.81 s`) $\to$ **0.5 decodes** (`0.6` decodes/s)
- **Post-stages (`cc` + `rf` + `ch` + `rp`):** `~570` hyps (`1.67 s`) $\to$ **~3.7 decodes** (`2.2` decodes/s)
- **Pass 5 (`post_pass`):** `208` hyps (`0.58 s` avg across all 6 slots; `0.87 s` on the 4 slots where it triggers) $\to$ **0.17 decodes** (`0.3` decodes/s)

#### Inside ONNX Runtime (`ORT Run` = `2.00 ms/hyp` on M1, `~0.85 ms/hyp` on M4)
Per-operator profiling of `models/v4/ft8nn.onnx` at batch `B=32` (1 thread, M1):
- `MatMul` (`55.1%`) + `Gemm` (`5.4%`): **60.5%** — running at `98.7 GFLOP/s`, i.e., **96% of a single M1 P-core's theoretical 102.4 GFLOP/s FP32 NEON limit** (M4 hits `~0.85 ms/hyp` via higher clock + ARMv9 SME2).
- `Conv` (`/conv.0` + `/conv.2`, `k=5, 128->128`): **14.7%** + `Transpose` (`3.7%`) + unfused `Gelu` (`1.2%`) = **19.6%** of NN time for two layers that represent only 18% of FLOPs.
- `BiasGelu` (`7.8%`), `Add` (`3.7%`), `Softmax` (`3.2%`), unfused LayerNorm ops (`Sub`/`Mul`/`ReduceMean`/`Div`/`Sqrt` = `4.3%`).

#### Two Hardware / ORT Dead Ends Already Tested & Ruled Out on M1
1. **ONNX FP16 in ORT 1.31.1 CPU Execution Provider:** Microbenchmarked `(2528×128)@(128×384)` `MatMul` in ORT 1.31.1: **FP16 is slower (`2.91 ms`) than FP32 (`2.49 ms`)** on M1 CPU EP.
2. **ONNX Runtime CoreML Execution Provider (`CPU_ONLY` and `ALL`):** Tested on `models/v4/ft8nn.onnx`: PyTorch's attention export creates 44 graph partitions (`242/427` supported nodes, 0-length `Slice` tensors), running **28% slower (`2.53 ms/hyp` vs `1.98 ms/hyp`)**. `XNNPACK` EP is not compiled into Microsoft's official NuGet `libonnxruntime`.

---

## 2. Options by Risk & Payoff Tier

We organize the options into **three tiers**:
- **Tier 1 (Exact / Zero-Decode-Loss):** Pure C and ONNX-graph transformations that preserve mathematical outputs (no retraining, no threshold re-tuning; verified directly with `tools/golden_c.py`).
- **Tier 2 (Gated / Near-Lossless Pipeline Cuts):** Cutting wasted classical BP, fine-sync, `post_pass`, and live `--early` work in Python (`ft8nn/`) and C (`c/`), gated on validation run 1 and verified with `tools/ab_run2.sh` + `tools/validate_full.sh`. Speeds up **both** C and Python benchmark runs.
- **Tier 3 (Major Neural Engine Acceleration):** Directly attacking the `70.9%` NN bottleneck via either native BLAS/AMX in `c/`, a tiny ONNX pre-screener cascade, or a slimmer `v4`-recipe student.

### Summary Comparison Table

| Status | Option | Target | Est. / Measured Slot CPU Cut | Decode Impact | Touches | Effort |
|---|---|---|---:|---|---|---|
| [x] | **1A** | Layer-3 Costas token pruning (`79->58`) + `com.microsoft::Attention` + tap-wise `MatMul+BiasGelu` convs | **Measured: −5.1% (1T), −6.9% wall / −9.8% NN CPU (4T)** (`13.88->13.17s` 1T, `4.35->4.05s` 4T) | **0** (`325/325` byte-identical, ONNX diff `3.34e-5`) | `ft8nn/model.py`, `tools/export_onnx.py`, `models/v4/ft8nn.onnx` | Small |
| [x] | **1B** | O(N) 3-term `sin^2` LPF in `subtract_signal` (drop two `160k` FFTs) + `double` split-re/im align & `complex_wave` | **Measured: −2.4% incremental (`−7.5%` cum. 1T, `−9.5%` cum. 4T)** (`subtract` `−49%`: `4.04->2.05s` 1T, `1.60->0.94s` 4T) | **0** (`870/870`, `636/636`, `1134/1134` exact field match) | `c/src/subtract.c`, `c/src/ft8.c`, `c/src/internal.h` | Small |
| [x] | **1C** | Cache `fft_plan`s, shared `post_bb`, `median_d` quickselect, `N/2` real-FFT, stack IFFT, phasor `rotate`, SIMD `nn_features` | **Measured: −3.7% inc. 1T / −5.4% inc. 4T (`−10.9%` cum. 1T, `−14.4%` cum. 4T, `−15.8%` live `--early`)** | **0** (`2,640/2,640` lines byte-identical across 3 stations) | `c/src/frontend.c`, `c/src/fft.c`, `c/src/decode.c`, `c/src/internal.h` | Small–Med |
| [ ] | **2A** | Gate `--also-classical` (`nsym=1,2,3`) + its `12,600` noise BP runs per slot (or tone-SNR screen) | **Est. −10% to −11.5%** (`~1.4–1.6 s`) | **~0** (verify `bp1..3` share on run 1) | `ft8nn/decoder.py`, `c/src/decode.c` | Small |
| [ ] | **2B** | Gate 2nd fine-sync hypothesis (`h=1`) & skip speculative `h=1` NN when `h=0` is clear/strong | **Est. −14% to −22%** (`~2.0–3.0 s`) | Tuned on run 1 for **0 net loss** | `ft8nn/frontend.py`, `ft8nn/decoder.py`, `c/src/decode.c` | Medium |
| [ ] | **2C** | Narrow `post_pass` (`pass 5`) + passband/SNR-aware `reuse_df` & `scaled_max_cand` | **Est. −5% to −10%** (`~0.7–1.4 s`, more on 3 kHz audio) | Tuned on run 1 for **0 net loss** | `ft8nn/decoder.py`, `c/src/decode.c` | Medium |
| [ ] | **2D** | Live `--early` / `--final` budget & pass-1 cap (`wsjtx` live mode) | **Est. −15% to −25% live** (`early` `2.2s -> ~1.2s`) | Preserves full `final` recovery; prevents skipped `14.4s` pass on M1 | `ft8nn/decoder.py`, `c/src/decode.c`, `wsjtx/` | Small–Med |
| [ ] | **2E** | `float32` SIMD + minimax `tanh`/`atanh` in `bp_decode` | **Est. −2% to −3%** after 2A (**−6%** standalone) | Near-exact (`ab_run2.sh` check) | `ft8nn/ldpc.py`, `c/src/ldpc.c` | Small |
| [ ] | **3A** | Pure-C `SoftDemod` forward pass in `c/` using macOS `Accelerate` `cblas_sgemm` (AMX) + ORT fallback | **Est. −50% to −55%** (`13.9s -> ~6.3s` on M1; RSS `540 -> ~80 MB`) | **0** (same FP32 `models/v4` weights) | `c/src/` (~250 LOC; needs user OK on ORT policy) | Medium |
| [ ] | **3B** | Two-stage ONNX cascade: tiny `d=64, L=1` screener (`0.11 ms/hyp`) gates full `models/v4` (`exp_err < 66`) | **Est. −40% to −48%** on **all OSes/CPUs** | Tuned on run 1 (`<0.1%` miss) | `ft8nn/model.py`, `train.py`, `c/src/decode.c` | Med–Large |
| [ ] | **3C** | Retrain slimmer single `SoftDemod` (`d=96, L=3, ff=288`) with `v4` calibrated recipe | **Est. −32% to −38%** on **all OSes/CPUs** | `~0.05–0.10 dB` synthetic QRM risk; re-check gates | `ft8nn/model.py`, `train.py`, `models/` | Med–Large |

---

## 3. Detailed Option Breakdown

### Tier 1 — Exact, Zero-Decode-Loss Optimizations

#### Option 1A: Prune Dead Final-Layer Sync Tokens + Unfold `Conv1d` in ONNX (`tools/export_onnx.py`, `ft8nn/model.py`)
- **Problem:**
  1. In `SoftDemod.forward` (`ft8nn/model.py#L79-L87`), all 4 Transformer layers process all `79` FT8 symbols (`58` data + `21` Costas sync), and immediately afterward line 86 slices `h[:, self.data_pos]`, **discarding all 21 sync positions** of Layer 3's output.
  2. During ONNX export (`ft8nn/model.py#L54`), `_conv_bl` falls back to `Conv1d` wrapped in two `(B, 79, 128) <-> (B, 128, 79)` `Transpose` ops. In ORT 1.31.1, those two `Conv` nodes + `Transpose` + unfused `Gelu` take **19.6% of total NN time**, because ORT cannot fuse `BiasGelu` onto `Conv` and `Conv1d` runs slower per FLOP than `MatMul`.
- **Change:**
  1. In the final Transformer layer (`self.tf.layers[-1]`), compute `K` and `V` over all `79` tokens as today, but slice `Q` and the residual stream to `self.data_pos` (`58` tokens) *before* `Q` projection, attention, output projection, `norm2`, and the `128 -> 384 -> 128` FFN. Mathematically 100% identical outputs using existing `models/v4/model.pt` weights (`−26.6%` FLOPs in Layer 3 = `−5.5%` of total NN FLOPs).
  2. Export the two `k=5` 1-D convolutions as padded slice-concatenation `(B, 79, 640) @ (640, 128)` `MatMul + Add + Gelu`, allowing ORT to fuse `BiasGelu` and eliminating both `Transpose` nodes (`−4%` to `−8%` of NN time).
- **Expected gain:** `10–14%` faster NN inference (`~1.0–1.4 s/slot` saved on M1 1T), **zero retraining**, bit-equivalent decodes.
- **Measured result (DONE, 8-core Apple M1, 6 busy `01_40m` slots):**
  - Exported `models/v4/ft8nn.onnx` (`3.48 -> 3.38 MB`, `max |onnx - torch| = 3.34e-5`, sign agreement `1.000000`, `pytest tests/test_onnx.py` PASS) using `com.microsoft::Attention` (layers 0–2), `com.microsoft::MultiHeadAttention` on the 58 `data_pos` queries in Layer 3, tap-wise `MatMul + BiasGelu` 1-D convs, and unfused LayerNorm; mirrored Layer-3 `data_pos` pruning in `ft8nn/model.py`.
  - Paired 6-slot A/B (`--threads 1`): `83.31 s -> 79.02 s` (`13.884 -> 13.170 s/slot`, **−5.1% total slot time**, pass NN CPU `51.75 -> 47.91 s` = **−7.4%**).
  - Paired 6-slot A/B (`--threads 0`, 4 P-cores): `26.09 s -> 24.30 s` (`4.349 -> 4.049 s/slot`, **−6.9% wall time**, pass NN CPU `58.87 -> 53.11 s` = **−9.8%** from lower memory/transpose traffic).
  - Decodes: **325/325 lines byte-for-byte identical** at both `--threads 1` and `--threads 0`.

#### Option 1B: O(N) Exact `sin^2` Envelope Filter & Faster Alignment in `subtract_signal` (`c/src/subtract.c`, `c/src/ft8.c`)
- **Problem:** `subtract_signal` (`c/src/subtract.c#L92-L135`) takes `13.3 ms` per decode (`0.67 s/slot` for ~51 decodes):
  1. `conv_same` (`7.9 ms`) allocates `5.1 MB` (`b1`, `b2`) and runs **two 160,000-point complex FFTs** just to convolve `x(t) conj(ref(t))` with a `4001`-sample `sin^2` window.
  2. `align_job` (`4.5 ms`) evaluates `block_energy` over all `151,680` samples at `25` coarse + `7` fine offsets (`4.85M` double-complex multiplies, re-evaluating `d = coarse` twice).
  3. `complex_wave` (`0.92 ms` in `c/src/ft8.c#L63-L84`) calls scalar `cos(ph)` and `sin(ph)` `151,680` times.
- **Change:**
  1. Use the exact trigonometric identity $\sin^2(\pi k / M) = \frac{1}{2} - \frac{1}{2}\cos(2\pi k / M)$: the `4001`-point FIR convolution is **one sliding boxcar sum minus one sliding complex phasor sum** (re-anchored every 256 samples so rounding stays `< 1e-13`), requiring **zero 160,000-point FFTs** and **zero 5 MB heap allocations** (`7.9 ms -> < 0.4 ms`).
  2. Skip the duplicate `d = coarse` offset in the fine alignment search and accumulate `block_energy` in `float` (4-wide NEON vectorization; alignment only picks an integer sample `k0 += fine`).
  3. Precompute the ramp envelope once in `ft8_init` and use 64-sample re-anchored phasor stepping where frequency is constant or fast polynomial `sincos` in `complex_wave`.
- **Expected gain:** `subtract_signal` drops `13.3 ms -> ~2.0 ms` (`~0.55 s/slot` saved on M1 1T, `0` decode changes).
- **Measured result (DONE, 8-core Apple M1, 6 busy `01_40m` slots):**
  - Replaced both `160,000`-point complex FFTs (`5.1 MB` heap churn per decode) in `c/src/subtract.c` with the exact O(N) 3-term periodic `sin^2` sliding recurrence (`1.35e-16` max error vs direct FIR), vectorized `complex_wave_split` in `c/src/ft8.c` (`3.9e-13` max error), skipped the duplicate `d = coarse` offset in fine alignment, and kept `block_energy` in split-re/im `double` (Note: `float32` `block_energy` saved an extra `0.35 s` over 6 slots, but shifted a 1-sample alignment tie on 1 strong signal in `00_40m` slot `261104_030245`, nudging `f0`/`snr` by `0.1`; keeping `double` preserves 100% `golden_c.py` field identity).
  - Subtraction stage wall time across the 6 busy slots dropped **`4.03 s -> 2.05 s` (−49.1% at 1T)**, **`1.60 s -> 0.94 s` (−41.3% at 4T)**, and **`1.21 s -> 0.66 s` (−45.5% in live `--early` 4T)**.
  - Decodes: **100% field-for-field identical** (`870/870` on `00_40m`, `636/636` on `08_80m`, `1,134/1,134` on `01_40m`).

#### Option 1C: Front-End FFT Plan Caching, Lazy `spectrum_baseline`, Shared Post-Stage `baseband`, `select_d`, & Real-FFT (`c/src/frontend.c`, `c/src/fft.c`, `c/src/decode.c`)
- **Problem:**
  1. `baseband_init` (`c/src/frontend.c#L386-L411`, `20.4 ms`) is called **8 to 9 times per busy slot**: every pass (`0..3`, `5`) plus independently inside `cochannel_stage`, `refine_stage`, `chase_stage`, and `repeat_stage` — even when the preceding post-stage added `0` decodes and `x` did not change at all!
  2. Each `baseband_init` calls `fft_plan_new(192000)` (`2.3 ms` of single-thread `sin`/`cos` + `3 MB` malloc), sorts `44,800` bins with `qsort` instead of `select`, and **eagerly** runs `spectrum_baseline` (`73 × 3840`-pt FFTs) even when a stage emits `0` decodes (Python's `Baseband.noise_ratio` in `ft8nn/frontend.py#L267-L274` already computes `spectrum_baseline` lazily!).
  3. `find_candidates` sorts `~128,000` cells with `qsort` for a single median (`c/src/frontend.c#L163`), and all real-input FFTs (`192,000`-pt slot FFT and `369 × 3,840`-pt spectrogram FFTs) run as full complex-input FFTs with imaginary part `0`.
  4. `rotate` and `fractional_shift` (`c/src/frontend.c#L489-L596`) call scalar `sin`/`cos` `6,400` times per hypothesis (`~27M` trig calls/slot) for uniform phase ramps.
- **Change:**
  1. Cache `fft_plan`s for fixed sizes (`192000`, `3840`, `3200`) once in `frontend_init()`.
  2. Compute `spectrum_baseline` lazily on first `baseband_noise_ratio` call (matching Python), and share a single `baseband` across `cochannel_stage -> refine_stage -> chase_stage -> repeat_stage`, invalidating it only when a decode is actually subtracted.
  3. Replace `qsort` medians in `baseband_init` and `find_candidates` with exact quickselect (`select_d`), add standard `N/2` real-to-complex FFT packing (`fft_rfft`), and replace uniform-ramp `sin`/`cos` loops in `rotate` / `fractional_shift` with 64-sample re-anchored complex multiplies.
- **Expected gain:** `~0.50–0.70 s/slot` saved on M1 1T (`0` decode changes; also lowers multi-thread barrier contention).
- **Measured result (DONE, 8-core Apple M1, 6 busy `01_40m` slots):**
  - Cached global `rfft_plan` (`192,000` and `3,840`), `fft_plan` (`3,200`), and Nuttall window tables once in `frontend_init()`; added half-size real-to-complex `fft_rfft`/`fft_rfft_par` (`1.4e-13` / `1.8e-12` max diff) and stack IFFT buffers (`n <= 4096`, eliminating ~6,500 heap allocs/slot); replaced `qsort` medians (`124k` and `44.8k` doubles) with bit-identical `median_d` quickselect (`20,000/20,000` unit checks); replaced `28M` per-slot scalar `sin`/`cos` calls in `rotate` and `fractional_shift` with 64-sample re-anchored phasor steps; transposed `dft64`/`dft32` for SIMD `nn_features` and `symbol_spectra8`; and shared `post_bb` across `cochannel -> refine -> chase -> repeat` whenever `x` is unchanged.
  - Across the 6 busy slots (1T): `sync` dropped `4.40 -> 3.88 s` (**−11.8%**), `ST_NN` (via SIMD `nn_features`) dropped a further `~1.0 s` (`51.75 -> 47.03 s` = **−9.1% vs baseline**), `ST_CL` dropped `3.15 -> 2.88 s` (**−8.6%**), `cochannel` `5.89 -> 5.46 s`, `refine` `2.96 -> 2.62 s`.
  - **Cumulative Tier 1 (`1A + 1B + 1C`) 6-slot results on `01_40m` (8-core Apple M1, idle):**

    | Mode (`01_40m` 6 busy slots) | Baseline (`HEAD`) | After Tier 1 (`1A+1B+1C`) | Gain |
    |---|---:|---:|---:|
    | `--threads 1` mean / max per slot | `13.884 s` / `14.508 s` (`83.31 s` total) | **`12.373 s` / `12.975 s`** (`74.23 s` total) | **−10.9%** (`−1.51 s/slot`) |
    | `--threads 0` (4 P-cores) mean / max | `4.349 s` / `4.570 s` (`26.09 s` total) | **`3.722 s` / `3.890 s`** (`22.33 s` total) | **−14.4%** (`−0.63 s/slot`) |
    | `--threads 0 --early 141696` total wall | `4.913 s` / `5.199 s` (`29.48 s` total) | **`4.139 s` / `4.379 s`** (`24.83 s` total) | **−15.8%** (`−0.77 s/slot`) |
    | `--threads 0 --early 141696` 11.8 s early pass | `2.184 s` / `2.218 s` (`0.38 s` margin) | **`1.882 s` / `1.907 s`** (`0.69 s` margin) | **−13.8%** (`+0.31 s` margin) |

  - **Equivalence & safety checks:** `MallocScribble=1 --threads 4 --stream` before-vs-after C decode lines **100% identical in every field** on `00_40m` (`870/870`), `08_80m` (`636/636`), and `01_40m` (`1,134/1,134`), plus `01_40m` 6-slot `1T` (`325/325`), `0T` (`325/325`), and `--early 141696` (`319/319`); `wsjtx/test_sidecar.sh` all 8 checks PASS.

---

### Tier 2 — Gated / Near-Lossless Algorithmic & Pipeline Cuts

#### Option 2A: Screen or Retire Redundant `--also-classical` (`nsym=1, 2, 3`) Work (`ft8nn/decoder.py`, `c/src/decode.c`)
- **Problem:** In every pass, for **every single hypothesis** (`~4,210/slot`), `hyp_worker` (`c/src/decode.c#L752-L758`) computes `symbol_spectra8` + `coherent_llr` for `nsym = 1, 2, 3` (`0.53 s/slot`) and `decode_hyp` runs **three classical `bp_decode` passes** after NN fails (`~12,600` BP calls/slot = `1.12 s/slot`). Together, `--also-classical` consumes **`1.65 s/slot` (`11.9%` of total CPU and `73%` of all non-NN decode time)**! On modern `models/v4` checkpoints, almost 100% of real decodes come from `metric = 0` (the neural net), while 98%+ of hypotheses fed to `k = 1..3` are pure noise where classical BP has zero chance of converging.
- **Change:**
  1. Audit saved validation run 1 decode logs (or run a 5-minute run-1 tally) to count exact `bp1`/`bp2`/`bp3` exclusives under `models/v4` and record their minimum `nsym=1` `exp_err` / Costas sync / tone SNR.
  2. Move `symbol_spectra8` + `coherent_llr` inside `decode_hyp` *after* `k = 0` fails, and gate `k = 1..3` on a cheap tone-coherence / `nsym=1` reliability threshold (chosen safely below every real `bp1..3` decode on run 1, while still letting strong synthetic signals pass so `tools/ci_smoke.py`'s untrained-model test continues to work).
- **Expected gain:** Cuts `>90–95%` of `ST_CL` and classical `bp_decode` calls, saving **`~1.4–1.6 s/slot` (`−10%` to `−11.5%` total CPU)** in C and speeding up Python's batched `ldpc.bp_decode` by **~3–4×**.

#### Option 2B: Gate the 2nd Fine-Sync Hypothesis (`h=1`) & Speculative Duplicate Work (`ft8nn/frontend.py`, `ft8nn/decoder.py`, `c/src/decode.c`)
- **Problem:** `fine_sync2` unconditionally emits `N_HYP = 2` hypotheses for every candidate (`~2,105` candidates $\to$ `4,210` NN calls/slot). Two large sources of waste occur:
  1. When `h=0` has a dominant Costas peak (`M[0] >> M[1]`, e.g. strong/clean signals), `h=0` decodes immediately — yet in C's `NN_CHUNK = 32` speculative batching (`c/src/decode.c#L733-L765`), `h=1` sits adjacent in the same chunk and **still runs `SoftDemod` and `bp_decode`** before `merge_to` discards it (`done_cand[ci] = 1`).
  2. On marginal candidates near `min_score = 1.3`, the secondary peak `M[1]` in `costas_grid` is usually an unrelated noise fluctuation in the `51 × 25` grid rather than a genuine rival sync peak.
- **Change:** Measure on validation run 1 the distribution of `(M[0], M[1], M[1]/M[0], cand.score)` for all decodes won by `q = 1` (`h=1` when `h=0` failed). Emit `h=1` from `fine_sync2` only when `M[1]` is a plausible rival peak (e.g., `M[1] >= alpha * M[0]` and `M[0]` is below the unambiguous-lock ceiling where `h=0` always succeeds).
- **Expected gain:** Eliminating `h=1` on `40–70%` of candidates cuts `20–35%` of pass NN + BP evaluations = **`~2.0–3.0 s/slot` (`−14%` to `−22%` total CPU)**.

#### Option 2C: Tighten `post_pass` (`pass 5`), Passband-Aware `scaled_max_cand`, & Asymmetric `reuse_df` (`ft8nn/decoder.py`, `c/src/decode.c`)
- **Problem:**
  1. `post_pass` (`c/src/decode.c#L2013-L2028`) passes `tried = NULL` and re-evaluates *every* candidate within `±62.5 Hz` whenever `cc`, `rf`, `ch`, or `rp` subtracts even a single `-22 dB` signal (`0.87 s` on triggered slots for `0.17` decodes/slot).
  2. `scaled_max_cand` (`c/src/decode.c#L892-L899`) inflates `max_cand` from `600` to `900` whenever `[fmin, fmax]` is `150..4000 Hz`, even when the receiver's SSB filter cuts off at `2.7–3.0 kHz` (so `3.0–4.0 kHz` is silent or filter skirt).
  3. `reuse_df = 62.5 Hz` is symmetric and SNR-independent: Pass 1 skips almost nothing (`~27` of `750` candidates) because Pass 0's ~30 decodes × `125 Hz` blanket the whole band, even where weak subtractions (`< -16 dB`) barely perturb non-overlapping bins more than `±20 Hz` away.
- **Change:**
  1. Pass `tried` into `post_pass` and narrow the `post_pass` frequency radius to the true overlap window measured on run 1's `post_pass` recoveries (`~20–35 Hz`), skipping `post_pass` for ultra-weak (`< -21 dB`) subtractions if run-1 probe shows zero yield.
  2. Scale `max_cand` by **occupied passband width** (detected from `spectrum_baseline` roll-off) rather than raw `[fmin, fmax]` when the upper kHz is ≥15 dB below mid-band noise floor.
  3. Test exact asymmetric overlap `[-55 Hz, +52 Hz]` (and narrower window for weak `< -16 dB` subtractions) for `reuse_df` on run 1.
- **Expected gain:** **`~0.7–1.4 s/slot` (`−5%` to `−10%` total CPU)** on wideband slots, and up to **`−18%`** on standard 3 kHz receiver passbands.

#### Option 2D: Live WSJT-X `--early` + `--final` Mode Optimization (`ft8nn/decoder.py`, `c/src/decode.c`, `wsjtx/ft8nn_sidecar.cpp`)
- **Problem:** On the 8-core M1, live `--early 141696` takes `2.19 s` on our 6-slot sample and `2.88–3.0 s` on the busiest run-1 slots (`STATUS.md`), **overrunning WSJT-X's `2.6 s` window** between `11.8 s` and `14.4 s` (which causes WSJT-X to skip the `14.4 s` final decode request). Meanwhile, `early` Pass 1 processes `~1,260` hypotheses for ~12 decodes that `final` Pass 0 would otherwise catch 2.6 s later, and `early + final` runs `4,831` pass hypotheses (`+15%` over plain decode).
- **Change:**
  1. In `ft8nn_decode_early`, cap Pass 1 to stronger candidates (e.g., `min_score = 1.6` or `max_cand = 250`) — or skip Pass 1 on slow machines when Pass 0 exceeds `1.1 s` (clean pass-boundary check, preserving determinism when off) — plus optional priority ordering near `params.nfqso` (`rx_freq`) when WSJT-X provides it.
  2. Apply Tiers 1 + 2A/2B, which alone pull M1's 2-pass `early` down from `2.2 s` to **`~1.2–1.4 s`** (comfortably inside `2.6 s` without dropping Pass 1).
- **Expected gain:** Keeps M1 live `early` latency safely at **`~1.0–1.3 s`** (`>1.3 s` margin before `14.4 s`) and trims total live CPU by **`15–25%`**.

#### Option 2E: Fast `float32` SIMD & Minimax `tanh`/`atanh` in `bp_decode` (`ft8nn/ldpc.py`, `c/src/ldpc.c`)
- **Problem:** `bp_decode` (`c/src/ldpc.c#L57-L115`) runs scalar `double` `expm1` and `log` calls across `83 × 7` check edges per iteration.
- **Change:** Switch BP internal messages to `float` and replace `expm1`/`log` with a degree-5/7 minimax rational approximation (accurate to `<1e-6`), ported identically to `ft8nn/ldpc.py` and `c/src/ldpc.c`.
- **Expected gain:** `2.5–3×` faster `bp_decode` (`~0.9 s/slot` standalone, or `~0.3 s/slot` = **`−2%` to `−3%`** if Option 2A already removes 95% of classical BP calls).

---

### Tier 3 — Major Neural Engine Acceleration (Targeting the `70.9%` NN Share)

Even after Tiers 1 + 2 reduce hypothesis count and non-NN overhead, `SoftDemod` (`0.84M` params, `143M` FLOPs/hypothesis) remains the dominant CPU consumer. Three distinct paths can cut NN CPU by **2.5× to 6×**:

#### Option 3A: Native Pure-C `SoftDemod` Backend Using System BLAS / Apple `Accelerate` (`c/src/`)
- **Why it is so fast on Apple Silicon:** On this M1, ONNX Runtime's CPU EP uses NEON FP32 (`98.7 GFLOP/s`, `2.52 ms` for a `2528×128 @ 128×384` `MatMul`). We benchmarked macOS's built-in system framework `-framework Accelerate` (`cblas_sgemm` with `VECLIB_MAXIMUM_THREADS=1` per worker thread, which dispatches to Apple's AMX matrix coprocessor) on the exact same M1 thread and matrix shape: **`0.325 ms` (`763.7 GFLOP/s` — 7.75× faster than ORT!)**.
- **What it looks like:** `SoftDemod` has a tiny, fixed graph (`Linear -> 2×Conv1d -> 4×TransformerLayer -> LayerNorm -> Linear`). Writing a self-contained `~250`-line C11 forward pass (`c/src/nn.c`, reading weights directly from the `.onnx` initializers or a tiny flat header, using `cblas_sgemm` on macOS and keeping ONNX Runtime on Linux/Windows or behind a build flag):
  - **Zero third-party C dependencies** (`Accelerate` is part of macOS `libSystem`/SDK, just like `-lm` and `-lpthread`).
  - **Exact same `models/v4` weights and FP32 math** (`0` retraining, `0` accuracy loss, verified against ORT/PyTorch to `<1e-5`).
  - Cuts M1 NN time per hypothesis from `2.00 ms` to **`~0.35–0.45 ms`** (**~5× faster NN**), dropping M1 1-thread busy-slot time from **`13.9 s -> ~6.2 s` (`−55%` alone, or `~3.8 s` combined with Tiers 1–2!)** and cutting peak RSS from **`~540 MB -> ~80 MB`**.
- **Constraint check:** `CLAUDE.md` currently notes *"Keep ONNX Runtime for the NN"*. This option requires explicit user approval to either use `Accelerate` as a fast macOS path alongside ORT (keeping ORT for Linux/Windows CI and verification) or stay strictly inside ORT (Options 3B / 3C).

#### Option 3B: Two-Stage ONNX Cascade — Tiny Screener Net Before Full `models/v4` (100% ONNX Runtime, All Platforms)
- **Why it works everywhere:** Across a busy slot's `~4,780` NN hypotheses, **~82–88% are pure noise or sidelobes (`exp_err >= 66`)** that never decode in BP, OSD (`<40`), CQ AP (`<44`), refine (`<45`), context AP (`<48`), co-channel (`<66`), or Chase (`<68`). Running a `143M`-FLOP 4-layer Transformer just to learn `exp_err = 78` wastes >80% of NN compute.
- **What it looks like:**
  1. Train a tiny `SoftDemod(d=64, layers=1, ff=192, conv_k=5)` screener (`~8M` FLOPs = **18× fewer FLOPs**, `~0.11 ms/hyp` in ORT on M1, `~0.05 ms/hyp` on M4) distilled on `models/v4` LLRs / `exp_err`.
  2. Run the screener on every hypothesis batch; run full `models/v4` `SoftDemod` **only** on the `~15–18%` of hypotheses with `screener_exp_err < 66` (or if the screener's own LLRs already pass `bp_decode` + `crc_ok`!).
  3. Effective NN cost drops to `0.11 + 0.17 × 2.00 = 0.45 ms/hyp` (**~4.4× faster NN** on macOS, Linux x64/arm64, and Windows x64, using unmodified ONNX Runtime).
- **Trade-off:** Needs training the small screener (~15 min on MPS), shipping/loading a second small `.onnx` (~250 KB) or combined model, and tuning the screener gate on validation run 1 so `<0.1%` of decodable signals are missed.

#### Option 3C: Slimmer Single `SoftDemod` Student (`d=96, L=3, ff=288`) Trained With the `v4` Recipe (100% ONNX Runtime, Single File)
- **What it looks like:** Keep a single `.onnx` file and zero decoder changes, replacing `SoftDemod(d=128, L=4, ff=384)` (`143M` FLOPs) with `SoftDemod(d=96, L=3, ff=288)` + Option 1A's Layer-3 token pruning (`~50M` FLOPs = **2.85× fewer FLOPs**, `~0.72 ms/hyp` on M1, `~0.32 ms/hyp` on M4), trained with `models/v4`'s calibrated multipath + negative-row recipe.
- **Trade-off:** Earlier architecture study (`docs/arch-study.md`) showed `L=2, d=128` matched real-air run-2 decodes within `0.3%` (`−0.08 dB` AWGN) while hurting close synthetic overlaps (`qrmclose -16 dB`); `L=3, d=96` keeps 3 attention layers and should sit closer, but requires ~2 h MPS training + full `runs 2–4` and `noise10k` false-decode gate verification.

---

## 4. Recommended Phased Plan & Validation Protocol

Depending on how aggressively we want to cut CPU and whether platform-native `Accelerate` on macOS or cross-platform speed is the priority, here is the cleanest execution order:

### Phase 1 — Zero-Risk Exact Wins (Tier 1: Options 1A + 1B + 1C) — **DONE 2026-11-26**
- **Measured cumulative cut (8-core Apple M1, idle, no sleep events):**
  - `--threads 1`: **`−10.9%`** (`13.884 -> 12.373 s/slot` mean, `14.508 -> 12.975 s` max on busy `01_40m` slots)
  - `--threads 0` (4 P-cores): **`−14.4%`** (`4.349 -> 3.722 s/slot` mean, `4.570 -> 3.890 s` max)
  - `--threads 0 --early 141696` (live WSJT-X sidecar sequence): **`−15.8%`** total wall (`4.913 -> 4.139 s/slot`), **`−13.8%`** on the 11.8 s early pass (`2.184 -> 1.882 s`, raising headroom below WSJT-X's 2.60 s deadline from `0.38 s` to `0.69 s`)
- **Validation passed:** `MallocScribble=1 --threads 4 --stream` C before-vs-after **100% identical in every field** (`00_40m` `870/870`, `08_80m` `636/636`, `01_40m` `1,134/1,134`); `tools/ci_smoke.py` `52/52` `C == Python` in every field; `wsjtx/test_sidecar.sh` all 8 checks PASS; `pytest` `103 passed, 16 skipped` (ft8code).

### Phase 2 — High-ROI Pipeline Pruning (Tier 2: Options 2A $\to$ 2B $\to$ 2C + 2D)
- **Expected cumulative cut (with Phase 1):** **`−40%` to `−50%` total CPU** (`13.89 s -> ~7.2–8.2 s` 1T on M1; 4-core wall `4.38 s -> ~2.3 s`; live `--early` `2.19 s -> ~1.1 s`, eliminating M1 early-pass overruns; Python benchmark runs **~1.8× faster** too).
- **Validation (strictly per `CLAUDE.md`):**
  1. Measure and choose every threshold on **validation run 1 + training recordings only** (never peek at runs 2–4).
  2. Implement in Python (`ft8nn/`) first, then mirror in C (`c/`).
  3. Gate with `tools/ab_run2.sh TAG` (noise `0` false, unseen run 2 decodes & slot-confirmed preserved, `unconf` not up), followed by `tools/validate_full.sh TAG` and `tools/golden_c.py`.

### Phase 3 — Neural Inference Engine Leap (Choose One of 3A, 3B, or 3C)
- **If macOS (M1 / M4) live & bench CPU + memory is the main pain point $\to$ Option 3A (`Accelerate` AMX C backend + ORT fallback):**
  - Combined with Phases 1–2, brings M1 1-thread busy-slot CPU from **`13.89 s` down to `~3.2–3.8 s` (`−73%` to `−77%` CPU!)**, 4-core wall time to **`~1.0 s`**, and RSS from `540 MB` to `~80 MB`, with **zero model retraining or sensitivity risk**.
- **If Linux/Windows x64/arm64 CPU must drop equally without custom BLAS $\to$ Option 3B (Tiny ONNX Screener Cascade):**
  - Combined with Phases 1–2, brings 1-thread CPU down **`~65–70%` across all operating systems** while keeping ONNX Runtime everywhere.

---

## 5. Decisions Needed Before Starting Implementation

1. **Target platform priority:** Are we optimizing primarily for **Apple Silicon (M1 / M4)**, or equally for **Linux / Windows (x64 / arm64)**?
2. **ONNX Runtime vs macOS `Accelerate` (Option 3A):** Would you like to keep ONNX Runtime as the *sole* neural backend (`CLAUDE.md`'s current rule $\to$ focus on Phases 1–2 + Option 3B/3C), or are you open to a small native C `SoftDemod` path using macOS `-framework Accelerate` (with ORT kept for Linux/Windows and golden checks)?
3. **Sensitivity tolerance:** Should we require **strictly zero decode loss** (`±0.0%` on validation runs, i.e., Phase 1 + lossless gates in Phase 2 + Option 3A), or is a tiny trade (e.g. `≤ 0.2–0.3%` real decodes for a large extra hypothesis cut in Phase 2 / Option 3B–3C) acceptable?

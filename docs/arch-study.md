# Architecture study: convolution-only and BiGRU vs the transformer (2026-11-07, Mac mini M4)

Run in a separate git worktree (code at d0cdb44, before candidate reuse), then merged: the code is in `ft8nn/model.py`
and `train.py`; models in `runs/`, scripts in `data/tmp/`, logs in `data/logs/` (both gitignored). The wheel-ORT
symlink `data/tmp/ortlib` pointed into the worktree's venv and was not kept; `tools/get_onnxruntime.sh` now
provides the official build (plan 7.3c). Outcome: STATUS plan 7.3d.

Question: can a cheaper network replace `SoftDemod` (conv front end + 4 transformer layers) without losing
sensitivity? The NN is ~70% of C decode time (STATUS plan 7.3), so ms per hypothesis matters.

Same I/O for every model: features (B, 79, 48) -> llr (B, 174), so `tools/export_onnx.py`, the C library and
the golden tests need no changes. Select with `train.py --kind {tf,conv,gru}`; checkpoints store `arch` with a
`kind` key (none = transformer, so old checkpoints load unchanged); `model.build(arch)` constructs them.

## Models

| kind | layout | params |
|---|---|---|
| tf (current `runs/v2`) | Linear 48->128 + pos, 2 convs k=5, 4 pre-norm transformer layers (d 128, 4 heads, ff 384) | 842,627 |
| conv (`ConvDemod`) | Linear 48->128 + pos, 5 residual pre-norm blocks of 2 dilated convs k=5 (dilations 1,2,4,8,16 cycling: receptive field 249 symbols) | 838,787 |
| gru (`GRUDemod`) | Linear 48->128, same 2-conv front end as tf, 2-layer BiGRU hidden 152 per direction | 846,771 |

Recipe for each (as v2): `train.py --kind K --cfg v1 --steps 6000 --out runs/K_v1`, then
`train.py --kind K --cfg v2 --init runs/K_v1/model.pt --lr 5e-4 --steps 8000 --out runs/K_v2`.

## Speed: ONNX Runtime CPU, 1 thread, batch 128 (`data/tmp/speed.py`)

First look, machine busy with other jobs (untrained weights; speed doesn't depend on them):

| model | ms / hypothesis |
|---|---|
| tf (runs/v2/ft8nn.onnx) | 0.90 |
| conv | 0.54 (1.7x) |
| gru | 0.44 (2.0x) |
| tf int8 dynamic quantization (`quantize_dynamic`, QInt8 weights) | 0.97 (slower) |

ONNX export: conv and gru match torch to ~1e-6. int8: `quant_pre_process` needs `skip_symbolic_shape=True`
(symbolic shape inference asserts on the attention reshapes).

### Finding: the Homebrew ONNX Runtime build is 2.3-3.7x slower than the wheel's (same version 1.31.1)

Clean machine (08:40), 1 thread, batch 128 random features, `data/tmp/ortbench.c` (plain ORT C API, the same
calls libft8nn makes) linked against each library:

| graph | Homebrew `/opt/homebrew/opt/onnxruntime` (libft8nn today) | wheel `.venv/.../onnxruntime/capi/libonnxruntime.1.31.1.dylib` | Python wheel |
|---|---|---|---|
| tf | 2.08 ms/hyp | **0.82** | 0.90 |
| conv | 1.64 | **0.52** | 0.54 |
| gru | 1.64 | **0.42** | 0.44 |

libft8nn itself (`ft8nn_cli --threads 1`, FT8NN_PROFILE, first 6 slots of validation run 1 01_40m, 29,842
hypotheses, `data/tmp/ctime.sh`):

| ORT library | NN stage ms/hyp (incl. C features) | s per busy slot |
|---|---|---|
| Homebrew | 2.18 | 15.85 |
| wheel (`DYLD_LIBRARY_PATH` to a `libonnxruntime.1.dylib` symlink) | **0.94** | **9.67** |

Decodes are byte-identical with both libraries (280/280 lines, `data/tmp/cmp_ort.sh`). So for the *current*
transformer, linking libft8nn against an official ORT build instead of Homebrew's gives a 2.3x faster NN and a
busy slot that fits in 15 s on one core again, with no model change. (Likely cause: the Homebrew bottle is built
without the optimised Arm kernels of the official release; not investigated further.) Under Homebrew ORT the
architecture gains shrink to 1.24x (conv 1.76, gru 1.77 vs tf 2.18 ms/hyp in libft8nn); presumably the slower
kernels there cost similar time for all three graphs (not profiled). Note `caffeinate` (SIP) strips `DYLD_LIBRARY_PATH` from its children: run it alongside
(`caffeinate -dims -w $$ &`), not as a wrapper.

## Training speed on MPS

`_conv_bl` (unfold + matmul, written for ARM CPU inference) was used for every conv, also on MPS. Per training
step (fwd + bwd, batch 512, `data/tmp/mpsbench.py`): tf 326 ms unfold / 287 native conv1d; **conv 2,282 / 155**;
gru 252 / 239. `_conv_bl` now uses native conv1d off-CPU (same math; CPU inference and ONNX export unchanged; 26
tests pass). The first conv training (2.3 s/step) was stopped at step 500 and restarted.

## Runs

Driver: `data/tmp/arch_run.sh train KIND` / `eval TAG MODEL` (logs `data/logs/<KIND|TAG>/`); chained by
`data/tmp/train_rest.sh` (gru, then tf retrained with the same recipe as a training-noise baseline `tf_v2`) and
`data/tmp/eval_all.sh` (tf baseline `tf_base` = runs/v2 on the fading sets with the current decoder, then
nn_conv, nn_gru, nn_tf2). Transformer reference numbers from the main checkout, same decoder code (`nn_v2_cy`):
run 2 27,930 decodes / 20,505 slot / 1 unconf; training recordings 17,586; AWGN 50% -22.52 dB; noise 0/998.

| run | started | took | notes |
|---|---|---|---|
| conv v1 + v2 | 09:10 | 42 min (v1 18, v2 24) | 0.165 s/step; final train-val (BP rate) below tf everywhere, most on overlaps (table below) |

Final built-in validation (`train.py` evaluate: BP-only decode rate on 512 synthetic frames per set):

| set | conv_v2 | gru_v2 | tf runs/v2 (M1, Nov 3) | tf_v2 (retrained here) |
|---|---|---|---|---|
| awgn -20/-21/-22/-23 | 0.89 / 0.65 / 0.23 / 0.02 | 0.86 / 0.62 / 0.21 / 0.01 | 0.95 / 0.79 / 0.31 / 0.03 | 0.96 / 0.79 / 0.32 / 0.03 |
| fade1Hz -17/-19/-21 | 0.82 / 0.48 / 0.10 | 0.83 / 0.48 / 0.10 | 0.87 / 0.56 / 0.11 | 0.86 / 0.54 / 0.10 |
| qrm -19/-21 | 0.69 / 0.31 | 0.67 / 0.31 | 0.74 / 0.40 | 0.75 / 0.40 |
| qrmclose -12/-16 | 0.90 / 0.67 | 0.90 / 0.62 | 1.00 / 0.89 | 1.00 / 0.91 |
| strong 0/15 | 0.97 / 0.97 | 0.97 / 0.97 | 0.99 / 0.99 | 1.00 / 0.99 |
| gainburst -14/5 | 0.78 / 0.99 | 0.78 / 0.99 | 0.79 / 0.99 | 0.79 / 1.00 |

The retrained transformer reproduces runs/v2 within +-0.02 on every set, so that is the training noise; the
conv/gru gaps (0.14 at awgn -21, 0.22-0.27 at qrmclose -16) are 7-13x larger.

| tf retrain v1 + v2 | 10:45 | 74 min (v1 31, v2 42) | 0.31 s/step |
| tf_l2 v1 + v2 | 11:59 | 47 min | final val: awgn -20/-21/-22/-23 0.94/0.70/0.20/0.02, fade1Hz 0.85/0.54/0.10, qrm 0.70/0.32, qrmclose 0.97/0.71, strong 0.98/0.98, gainburst 0.77/0.99 |
| tf_l2kd v1 + v2 (distilled) | 12:46 | 58 min | final val: awgn 0.93/0.69/0.20/0.01, fade1Hz 0.84/0.51/0.10, qrm 0.69/0.30, qrmclose 0.98/0.75, strong 0.99/0.99, gainburst 0.78/0.99: same as tf_l2 within noise |

Evaluation chain started 13:45 (`data/tmp/eval_all2.sh`), then `data/tmp/final_timing.sh` (idle machine).
Each model's full evaluation takes ~94 min (8 workers).

## Evaluation (current decoder, `bench/run_ours.py` / `run_real.py` with `--also-classical`, 8 workers)

| model | AWGN 50% | moderate 50% | disturbed 50% | noise false | training rec. (jt9 15,729) | run 2 dec / slot / unconf |
|---|---|---|---|---|---|---|
| tf runs/v2 (`nn_v2_cy`, `tf_base`) | -22.52 | -20.17 | -20.17 | 0/998 | 17,586 | 27,930 / 20,505 / 1 |
| conv_v2 (`nn_conv`) | -22.38 | -20.17 | -20.08 | 0/998 | 17,650 | 27,953 / 20,532 / 2 |
| gru_v2 (`nn_gru`) | -22.37 | -20.04 | -20.04 | 0/998 | 17,593 | 27,898 / 20,475 / 4 |

| tf_v2 retrained, same recipe (`nn_tf2`) | -22.49 | -20.19 | -20.22 | 0/998 | 17,661 | 27,955 / 20,518 / 5 |
| tf 2 layers (`nn_l2`) | -22.36 | -20.22 | -20.14 | 0/998 | 17,611 | 27,899 / 20,486 / 6 |
| tf 2 layers distilled (`nn_l2kd`) | -22.41 | -20.25 | -20.09 | 0/998 | 17,577 | 27,849 / 20,437 / 3 |

**Training noise** (runs/v2 vs its same-recipe retrain): AWGN 0.03 dB, fading 0.02-0.05 dB, training recordings
+75, run 2 +25 decodes / +13 slot, and run-2 unconf 1 vs 5 (`AT3REZ W3UDO EN91`, `CQ P4HRU NP43`,
`CQ 2V0TAD/R CJ41`, `CQ TT6QCN KP62`, `A8TOO S28RCD -04`). So a few unconfirmed decodes are within training noise;
the decoder's gates were tuned on runs/v2, and any retrained network has a slightly different LLR calibration.
Against the same-machine retrain, conv is 0.11 dB behind on AWGN, 0.02 on moderate and 0.14 on disturbed fading,
and level on real data.

## Final timing (22:19-22:29, machine idle and fully awake, `data/tmp/final_timing.sh`, `data/logs/timing/`)

| model | params | train time (M4 MPS) | ORT ms/hyp Homebrew / wheel (`ortbench`) | libft8nn NN ms/hyp Homebrew / wheel | busy slot s, 1 thread, Homebrew / wheel |
|---|---|---|---|---|---|
| tf runs/v2 | 842,627 | ~4 h (M1, earlier) / 74 min retrain | 2.05 / 0.85 | 2.17 / 0.97 | 15.79 / 9.83 |
| conv | 838,787 | 42 min | 1.64 / 0.54 | 1.76 / 0.66 | 13.66 / 8.17 |
| gru | 846,771 | 53 min | 1.64 / 0.43 | 1.80 / 0.56 | 13.88 / 7.68 |
| tf 2 layers (l2, l2kd) | 511,875 | 47 / 58 min | 1.18 / 0.47 | 1.31 / 0.60 | 11.53 / 7.93 |
| tf int8 dynamic, QInt8 | – | – | 1.26 / 0.97 | – | – |
| tf int8 dynamic, QUInt8 | – | – | 1.28 / 1.14 | – | – |
| tf int8 dynamic, per-channel | – | – | 1.29 / 0.98 | – | – |

libft8nn = `ft8nn_cli --threads 1`, first 6 slots of validation run 1 01_40m (~29,800 hypotheses, the NN stage
includes the C feature extraction). int8 helps only on the slow Homebrew build (1.6x) and is slower than float on
the wheel build, so its accuracy was not measured.

## Conclusion

- No alternative clearly meets "within noise and >= 1.5x faster". All five networks are level on real data
  (run 2 27,849-27,955 decodes; training recordings 17,577-17,661; the transformer's own retrain spans 27,930-27,955
  and 1-5 unconfirmed). On the synthetic sets every smaller or non-attention model loses 0.08-0.16 dB on AWGN, and
  the synthetic overlap test (qrmclose -16 dB) shows the transformer far ahead (0.89-0.91 vs 0.62-0.75).
  Closest: the distilled 2-layer transformer (AWGN -0.08/-0.11 dB, disturbed -0.08/-0.13, run 2 -0.3/-0.4%,
  unconf 3, 1.6x faster NN); conv (AWGN -0.11/-0.14, disturbed -0.09/-0.14, run 2 level, 1.5x with the wheel build,
  only 1.23x with Homebrew).
- Distillation (same size, same recipe, 0.5 soft / 0.5 hard) gave nothing measurable over training from labels.
- **The big, free win is the ONNX Runtime build**: the same transformer runs the NN 2.2x faster in libft8nn with
  the official build (the one inside the Python wheel) than with Homebrew's, decodes byte-identical, and a busy
  slot drops from 15.8 to 9.8 s on one core. After that switch the NN is about half of the slot time, so a 1.6x
  smaller network would save only ~1.9 s per busy slot (19%) at a ~0.1 dB AWGN cost.
- Recommendation: keep the transformer; link libft8nn against the official ORT release build instead of Homebrew's
  (needs the user's OK to download/vendor that binary; the wheel's `libonnxruntime.1.31.1.dylib` works with the C
  API unchanged, tested here via a `libonnxruntime.1.dylib` symlink + `DYLD_LIBRARY_PATH`).

gru's 4 unconfirmed are all CQ a-priori garbage (`CQ 1P9Z J R LG35`, `CQ 3J2 EH R DN33`, `CQ FV2YAC DC88`,
`CQ JS5TSN/R R OQ13`).

conv's 2 unconfirmed are both CQ a-priori decodes at odd DT (`CQ JN7QPR/R DG27` osd_ap0 DT 2.04,
`CQ M95PRR R RN51` osd_ap0 DT -1.43); it doesn't produce the transformer's one (`CQ 0V1YXG GG88`). On real data
conv matches the transformer (run 2 +0.08%, training recordings +0.4%), unlike its synthetic validation gap
(qrmclose -16 dB 0.67 vs 0.89): the subtraction passes presumably take care of most real overlaps.

| gru v1 + v2 | 09:52 | 53 min (v1 23, v2 30) | 0.22 s/step |

Both alternatives lose clearly on overlaps (qrmclose -16 dB: conv 0.67, gru 0.62, tf 0.89), so cost-cutting models
were queued right away (`data/tmp/train_cut.sh`, same v1 -> v2 recipe): `tf_l2` = transformer with 2 layers
(514k params) from labels, and `tf_l2kd` = the same size distilled from runs/v2 (`train.py --teacher`, loss
0.5 x BCE on true bits + 0.5 x BCE on the teacher's P(bit=1)). All evaluations then run in one chain
(`data/tmp/eval_all2.sh`).

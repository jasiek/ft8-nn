# Repeat stage: the doubtful decodes, and competitors no station sends

Draft checkpoint, 2026-11-12. Python and offline only; there is no C port yet. Follows `docs/miss-profile-dtcrp.md`,
which ranked this change as lever 1 and listed 12 doubtful repeat-stage decodes.

Terms. The **repeat stage** (`rp`, plan 12) handles a sender we decoded 30/60 s earlier but not in this slot:
- it scores whole known messages against the residual audio with ρ = Σ l·s / Σ|l| (LLRs clipped to ±2,
  s = ±1 codeword bits; 1 = perfect agreement, about 0 = noise);
- the candidates are that sender's earlier messages and ~130 **competitors**, the messages the same calls could
  send next;
- it accepts an earlier message only if it scores best with ρ ≥ 0.40 and beats the runner-up by ≥ 0.15
  (the **margin**).

**Slot-confirmed / active / unconfirmed** are `collect/verify.py`'s PSK Reporter tiers for the sending callsign:
- slot-confirmed = someone spotted it in that slot;
- active = spotted on the band within 10 min;
- unconfirmed = never spotted, roughly a false decode.

The tiers check only the sender, not the text.

## 1. The 12 doubtful `rp` decodes: none is false

These were `rp` decodes where a reference decoder (jt9, JTDX or deep JTDX) decodes a *different* sender within
10 Hz / 0.4 s and no reference decodes our `rp` sender in that slot. There are 11 distinct `rp` decodes; one sits
next to two references. Each was examined in the audio (`scratch/rp_doubt.py` → `data/logs/miss_dtcrp/rp_doubt.txt`):
- our message's ρ and margin under both competitor sets;
- the reference's message scored at our hypothesis;
- both messages at both spots, plus a forced CRC decode at each;
- the context's age and stage;
- what our decoder and all three references heard from our sender 90 s either side.

| run, receiver, slot | our `rp` decode | reference's decode there | verdict and evidence |
|---|---|---|---|
| 2 `00_40m` 223215 | `NA6Y DL2KDF JO30` 2068.8 Hz, ρ 0.71 / margin 0.16 | `CQ N9JOW EN51` 2060 Hz (all three) | **real, different spot.** 8.8 Hz apart; each message scores only at its own spot (theirs at ours 0.03, ours at theirs −0.03). DL2KDF sent the same text at −60/−30 s (all three references) |
| 2 `03_20m` 225630 | `LU6WAV WI0O EN35` 1051.1, 0.75 / 0.23 | `CQ N6BCC CM97` 1044 | **real, different spot.** Same text at −90/−60/−30 s (references too); WI0O slot-confirmed |
| 3 `00_15m` 054930 | `CQ BG0HT NL59` 1513.8, emulated 0.34 | `CQ BH8BMZ OM21` 1512, DT 0.4 s later (all three) | **weak; a repeat-stage chain.** BH8BMZ is a separate signal that CRC-decodes at its spot (E 15); ours scores only 0.27 at our spot. Context is our own `rp` decode at −60 s (from a `bp_cx` at −90 s); no reference ever decodes BG0HT here, though BG0HT is slot-confirmed (on the air). Not provably false |
| 3 `11_15m` 060115 | `KH8A RZ9OC -10` 1343.5, 0.57 / 0.17 | `YB2BTH YB4DCE -08` 1340 (jt9) | **real, different spot.** Same text at −60/−30 s (all three references); each message scores only at its own spot |
| 4 `00_20m` 151930 | `SP9JBT K1JAK FN53` 1699.0, 0.61 / 0.35 | `<LZ911TD> RM1B KO59` 1691 (all three) | **real, different spot.** K1JAK sends this text from −60 to +90 s (jt9, deep JTDX, ours); RM1B CRC-decodes at its own spot |
| 4 `02_15m` 151945 (×2) | `CT7BRV LW2EDM -18` 1514.2, 0.70 / 0.22 | `CQ EA1CM IN63` 1505 and `CQ EC3A JN00` 1509 (jt9) | **probably real; a long chain.** Different spots (theirs score ≤ 0.10 at ours). But our only LW2EDM decodes for ±90 s are `rp` (−60, 0, +60 s); LW2EDM is active, not slot-confirmed. A station repeating a report it never got answered is normal, and ρ 0.70 is far above the wrong-receiver null (max 0.36) |
| 4 `02_15m` 151430 | `SP3DOF WW4OCF R-06` 2123.8, 0.56 / 0.34 | `K5VYT IK8PGE 73` 2117 (JTDX) | **real, different spot; a chain** seeded by an `osd0` at −60 s, `rp` since. Waiting for RR73 |
| 4 `04_40m` 151415 | `YC8BUL N7ECV DN07` 1051.2, 0.73 / 0.30 | `CQ YG2AUI OI42` 1051, same DT (jt9, `a1` a-priori) | **real; a true collision.** Deep JTDX decodes `YC8BUL N7ECV DN07` in this very slot, and all three references decode it at −30/+30/+60 s. YG2AUI is slot-confirmed too: two stations on one frequency |
| 4 `04_40m` 151945 | `<...> HA7TM R-07` 1537.5, 0.56 / 0.24 | `W1BTB KI5DXY +12` 1531 (JTDX) | **real, different spot**, weak (SNR −31 as estimated). Context an `osd0` at −30 s; HA7TM slot-confirmed |
| 4 `05_12m` 152430 | `M0KZC F4LNM JN09` 816.5, 0.65 / 0.38 | `CQ LY5A KO15` 822, DT 0.4 s later (deep) | **real, different spot.** F4LNM decoded by us at −60/−30/+30 s; slot-confirmed |
| 4 `09_15m` 152100 | `KA0BOJ OP7B RR73` 459.4, 0.86 / 0.38 | `CQ EA7GNO IM86` 455 (JTDX) | **real**: our message also *CRC-decodes* at our spot (E 27), as all three references decoded it at −30 s |

- **9 are real.** In 8 the reference's station is a separate signal 4–9 Hz away. My 10 Hz / 0.4 s window for
  "co-channel" was wider than "the same spot".
- **1 is a genuine two-station collision.**
- **2 are repeat-stage chains with only weak evidence:** BG0HT and the LW2EDM decode that sits next to two
  references.
- **None shows our message read off the other station's signal.** At our hypothesis the reference's message
  scores ρ ≤ 0.19 in every case except the collision.

**Guards considered, none proposed.**
- *No accept when a CRC decode of a different sender sits at that f/DT.* Among the 12 it touches only the
  N7ECV/YG2AUI collision, which is real on both sides. No evidence for it.
- *No accept from a chain* (the sender's last non-`rp` decode is 90 s or more back), from `scratch/rp_chain.py`
  → `data/logs/miss_dtcrp/rp_chain.txt`:

  | run | `rp` decodes (checkable sender) | CRC decode in the context (30/60 s) | chain, 90–120 s | chain, 150–300 s | > 300 s |
  |---|---|---|---|---|---|
  | 1 | 1,913 | 1,675, 80.5% slot-confirmed | 186, 82.3% | 52, 82.7% | 0 |
  | 2 | 1,159 | 1,023, 76.0% | 113, 81.4% | 23, 91.3% | 0 |
  | 3 | 1,015 | 894, 76.3% | 82, 76.8% | 29, 93.1% | 10, 80.0% |
  | 4 | 2,008 | 1,800, 78.6% | 162, 85.8% | 43, 81.4% | 3, 100% |

  0 unconfirmed in any cell. Chain decodes are slot-confirmed as often as direct ones, so stopping them would cost
  run 1 238 of its 1,913 `rp` decodes (12%) and buys nothing measurable. The tiers cannot see a *stale* chain
  (the sender still on the air but sending something else); the stale check below covers that where a reference
  decoded the sender.

## 2. Lever 1: only competitors a station could send

### The problem

For every earlier message, `_successors` built the same ~130 competitors, both R flags × grid / RRR / RR73 / 73 /
reports −30…+30, CQs included. For a CQ that yields `CQ K1ABC R FN42`, `CQ K1ABC -12`, `CQ K1ABC RR73` and so on:
messages no station sends (WSJT-X never sends a CQ with the R flag, and `message.plausible` rejects it). Each sits
one 15-bit field away from the truth, so on a weak signal it scores close behind it. On runs 2–4, 320 repeats lost
only on the margin with the right message scoring best, and in 226 of them the runner-up was such a CQ
(`docs/miss-profile-dtcrp.md`).

### The change (Python, `ft8nn/decoder.py`)

- **`_sendable(c)`:** false if `message.plausible` rejects `c` (the CQ + R case), or if `c` is a CQ (call field a
  CQ token) carrying RRR / RR73 / 73 / a report instead of a locator. A CQ with no locator stays allowed.
- **`_successors(b, grids, sendable_only=True)`** keeps only sendable payloads. `sendable_only=False` gives the
  plan-12 set, which the emulator uses for side-by-side scoring.
- **Unchanged for QSO messages:** for any non-CQ earlier message the competitor set is identical, 128–131
  payloads. That includes every stale-message successor (report → R-report → RR73 / RRR / 73, the sender's own CQ
  with its grids). `tests/test_repeat.py` checks this.
- **A CQ-only sender is left with no competitor** (the set collapses from 131 to the CQ itself). The stage took the
  runner-up as `order[1]`, which would have crashed. Its runner-up is now **`REPEAT_SOLO` = 0.35**, the noise
  ceiling, so an uncontested sender needs ρ ≥ 0.50 instead of 0.40. The value comes from the null below.
- `REPEAT` (0.40, 0.15, clip 2) is unchanged.
- New tests (`tests/test_repeat.py`, 3): the CQ set has no unsendable payload; the QSO set is unchanged; a −12 dB
  synthetic CQ repeat is accepted with no competitor left, and pure noise is not. `uv run pytest -q`: 36 passed.

### The false-accept null (run 1 only)

`tools/rp_competitors.py --shuffle 0 1 2 3 4 5` on validation run 1 with base `nn_v3`, exactly as plan 12 ran
`rp_eval.py`: every receiver gets the context of another-band receiver of the same run, so every acceptance is
false. Both competitor sets are scored from the same ρ matrix. 223,643 tests, the same number plan 12 counted
(`data/logs/miss_dtcrp/rpc_null_run1_summary.txt`):

| set | 0.40 / 0.15 | 0.40 / 0.10 | 0.35 / 0.15 | 0.45 / 0.15 |
|---|---|---|---|---|
| shipped (plan 12) | **0** | 1 | 1 | 0 |
| sendable, uncontested runner-up 0 (ρ ≥ 0.40) | **6** | 6 | 37 | 1 |
| sendable, uncontested runner-up 0.30 (ρ ≥ 0.45) | **1** | 6 | – | – |
| **sendable, uncontested runner-up 0.35 (ρ ≥ 0.50)** | **0** | **1** | – | – |

- The shipped row reproduces plan 12's numbers exactly (0, 1, 1), so the emulator matches `rp_eval.py`.
- Dropping the nonsense competitors alone fails the null. All 6 false accepts are uncontested CQs: 43,557 of the
  tests have no competitor left, and their best wrong ρ reaches 0.459 (`CQ WB3FAA FN21`; p99 0.264, p99.9 0.342).
- At runner-up 0.35 the null is back to plan 12's (0 at the operating point, 1 at margin 0.10), with 0.04 between
  the worst wrong uncontested CQ and the new 0.50 bar.

### Gains: run 1 for tuning, runs 2–4 as measurement only

`tools/rp_competitors.py --real` emulates the stage on every slot in time order:
- it subtracts the slot's non-`rp` decodes of the base tag (the passes' output, plus the co-channel stage's on the
  `nn_v3_cc` base, which runs before the repeat stage in fork 1's decoder);
- it chains each competitor set's own acceptances as context for later slots;
- it writes `decodes_nn_v3_dtcrp_emu` (shipped set) and `decodes_nn_v3_rpc_emu` (sendable set + 0.35 floor),
  which `collect/verify.py` then scores.

The shipped-set emulation reproduces the shipped `rp` decodes to 96% on every run (run 1: 1,848 of 1,920; the
rest differ by subtraction rounding). Its totals are within 12 decodes of the real `nn_v3_dtcrp`, a noise floor
of ~0.1% against the change's effect.

On base `nn_v3_dtcrp` (shipped default). Decodes / slot-confirmed / active / unconfirmed, then the `rp` decodes
gained and lost (slot / active / unconfirmed):

| run | shipped-set emulation | **sendable set** | change | `rp` gained | `rp` lost |
|---|---|---|---|---|---|
| 1 (tuning) | 34,287 / 26,289 / 7,657 / 1 | **34,440 / 26,411 / 7,688 / 1** | +153 / **+122** / +31 / 0 | 185 (149 / 36 / 0) | 32 (27 / 5 / 0) |
| 2 | 29,432 / 21,649 / 7,419 / 1 | **29,548 / 21,729 / 7,455 / 1** | +116 / **+80** / +36 / 0 | 132 (95 / 37 / 0) | 16 (15 / 1 / 0) |
| 3 | 28,194 / 21,258 / 6,527 / 1 | **28,314 / 21,352 / 6,553 / 1** | +120 / **+94** / +26 / 0 | 136 (108 / 28 / 0) | 16 (14 / 2 / 0) |
| 4 | 48,435 / 35,119 / 12,331 / 2 | **48,732 / 35,382 / 12,365 / 2** | +297 / **+263** / +34 / 0 | 339 (301 / 38 / 0) | 42 (38 / 4 / 0) |
| **2–4 pooled** | 106,061 / 78,026 / 4 | **106,594 / 78,463 / 4** | **+533 / +437 / 0 unconf** | 607 (504 / 103 / 0) | 74 (67 / 7 / 0) |

- **Gained:** mostly CQ repeats (`CQ LU1DG GF05` ρ 0.74, `CQ 9A5MP JN75`, `CQ E70A JN94`).
- **Lost:** uncontested CQs between ρ 0.40 and 0.50 that the shipped set accepted (on a margin over the nonsense
  CQs) and the 0.35 floor now refuses. All are slot-confirmed or active; that is the price of keeping the null at
  zero.
- **Against the other decoders** (pooled, emulated; `compare_rpc.txt`): +23.6% decodes / +22.7% slot-confirmed
  over jt9 (shipped +23.0% / +22.0%). JTDX-only slot-confirmed pairs 1,599 → **1,400**, deep-JTDX-only
  2,066 → 1,844, jt9-only 789 → 776.

**Stale check** (`tools/rp_stale.py` on the gained `rp` decodes only, `decodes_nn_v3_rpc_gain`). This asks: where
a reference decoded the same sender in that slot, did it decode the same message?
- Run 1 (jt9 only): 3 of 3 the same.
- Runs 2–4 (jt9, JTDX and deep JTDX): 62 + 80 + 180 = **322 of 322 the same, 0 different**.

The tiers cannot see this risk, and it is the one that matters for an uncontested CQ: a station that stopped
calling and now answers someone with the same locator. The 0.50 bar and these 0 stale results say it does not
happen at this rate.

**On base `nn_v3_cc`** (fork 1's co-channel stage, decodes for runs 2–4): the same emulation gives +114 / +78,
+120 / +94, +302 / +266, pooled **106,541 / 78,330 / 4 → 107,077 / 78,768 / 4 (+536 / +438)**. Stale: 322 of
322 the same. The two changes are additive: the co-channel stage takes senders the repeat stage then skips, but the
CQ repeats this change adds are a different population.

## 3. A/B plan (after fork 1's runs; tag `nn_v3_rpc`)

**Base.** Measure on top of **`nn_v3_cc`** if the co-channel stage ships first, otherwise on `nn_v3_dtcrp`. The
emulation says the gain is the same either way (+437 vs +438 slot-confirmed on runs 2–4), so the order is
free. The A/B must still run on the combined decoder, because both stages work on the same residual and repeat-
stage decodes are among the co-channel neighbours. Concretely: merge fork 1's `_cochannel_stage` and this
`_successors` / `REPEAT_SOLO` change in `ft8nn/decoder.py` (separate hunks), tag the result `nn_v3_rpc`, and
compare against `nn_v3_cc`.

**Steps and times.** Python decoder on the idle M4 with 8 workers, wrapped in `caffeinate -dims`; times are from
fork 1's `nn_v3_cc` runs this morning (`data/logs/nn_v3_cc/`):
1. `WORKERS=8 tools/ab_run2.sh nn_v3_rpc`: 100 noise slots, then run 2 — about **50 min**. Expected from the
   emulation: run 2 +114 decodes / +78 slot-confirmed, unconfirmed 1 → 1.
   - Noise 0 false is a vacuous check here: noise slots carry no context, so the repeat stage never runs. The
     false-accept bound is the run-1 shuffle above.
2. Runs 3 and 4 (`bench/run_real.py` + `collect/verify.py`): about **45 + 54 min**.
3. Run 1 (tuning set): about **45 min**. No `nn_v3_cc` run 1 exists yet, so this needs a baseline run too
   (another ~45 min), or compare against `nn_v3_dtcrp` if the A/B is done on that base.
4. `tools/rp_stale.py` on the `rp` decodes `nn_v3_rpc` has and the base does not, and `tools/compare_decoders.py`.
   Minutes.

Total about **3 h 15 min** (4 h with the run-1 baseline); shaky by ±15 min, depending on the busiest run-4 stations.

**Accept if** run 2 decodes and slot-confirmed rise and unconfirmed do not, and runs 3–4 confirm with 0 new
unconfirmed and 0 stale.

**Then:**
- **C port.** `_sendable` / `REPEAT_SOLO` in `c/src/decode.c` `repeat_stage`; the C runner-up must also take the
  floor when there is one candidate. Then `tools/golden_c.py`.
- **`tools/validate_full.sh nn_v3_rpc`:** run 1, training recordings, AWGN, golden — about 2 h 20 min.
- **`noise10k`** is not on this machine; regenerating it (seed 19, README) bounds nothing for this stage either,
  as it has no context.

## 4. A/B on top of the co-channel stage (tag `nn_v3_ccrpc`, Python)

**Merge.** Fork 1's co-channel stage was taken from its worktree as it produced `nn_v3_cc`:
- `ft8nn/decoder.py`: `COCHANNEL*`, `_nn_only`, `_cochannel_stage`, `decode_slot(cochannel=)`;
- `ft8nn/frontend.py`: `data_sync`;
- `ft8nn/nn_metric.py`: `llr_fn.nn`;
- `bench/run_real.py` and `bench/run_ours.py`: `--cochannel`.

It contained nothing of the new plausibility rule; `message.py` was untouched. The merged `decoder.py` differs from
fork 1's only by this change. New switch: `bench --repeat-competitors shipped|sendable` (default `sendable`), which
sets `decoder.REPEAT_SENDABLE`.

**Proof.** With `--repeat-competitors shipped`, the merged code decoded the first 20 slots, in time order, of run 2
`00_40m` and `01_15m` and run 4 `02_15m_s57bit`. That is 2,705 decode lines, including 24 co-channel and 120
repeat-stage decodes, and all are **identical in every field** to `decodes_nn_v3_cc.txt` (`scratch/cmp_decodes.py`).
Run 4 had been decoded with fork 1's last code before its 10:27 edit, so that edit changed nothing.

**Run.** `scratch/ab_ccrpc.sh` under `nohup caffeinate -dims`, after the full-wake check (last wake 07:48 FullWake,
no sleep since). `tools/ab_run2.sh nn_v3_ccrpc` with 8 workers, then runs 3 and 4 once run 2 passed. Logs are in
`data/logs/nn_v3_ccrpc/`. Fork 1's `validate_full.sh nn_v3_cc` ran from about 11:09 on the same machine, with a
load average of 90–230, so the wall times are not timings.

**Results** (`collect/verify.py`; decodes / slot-confirmed / unconfirmed). "Predicted" is the emulation's change
from `nn_v3_cc_emu` to `nn_v3_ccrpc_emu`:

| run | `nn_v3_cc` | **`nn_v3_ccrpc`** | change | predicted | `rp` gained (slot / active / unconf) | `rp` lost | stale (same / different) |
|---|---|---|---|---|---|---|---|
| 2 | 29,588 / 21,751 / 1 | **29,691 / 21,820 / 1** | +103 / +69 | +114 / +78 | 114 (78 / 36 / 0) | 15 | 58 / **0** |
| 3 | 28,295 / 21,324 / 1 | **28,418 / 21,421 / 1** | +123 / +97 | +120 / +94 | 129 (102 / 27 / 0) | 17 | 79 / **0** |
| 4 | 48,668 / 35,249 / 2 | **48,999 / 35,541 / 2** | +331 / +292 | +302 / +266 | 335 (299 / 36 / 0) | 31 | 182 / **0** |
| **pooled** | 106,551 / 78,324 / 4 | **107,108 / 78,782 / 4** | **+557 / +458** | +536 / +438 | 578 (479 / 99 / 0) | 63 | 319 / **0** |

- Noise: 0 false of 998.
- **The 4 unconfirmed decodes are the same ones as `nn_v3_cc`:** run 2 `CQ CR4UVW PI49`, run 3 `<...> C29IUP +35`,
  run 4 `CQ U87GYB/R LC30` and `TNX 73 GL`.
- **The emulation predicted individual decodes, not just totals:** of the real gained `rp` decodes, 90% / 96% / 89%
  (runs 2 / 3 / 4) are exactly ones it predicted.
- **Against the other decoders** (`compare_decoders.py`, `data/logs/nn_v3_ccrpc/compare_decoders.txt`): pooled
  +24.2% decodes / +23.2% slot-confirmed over jt9 (`nn_v3_cc` +23.5% / +22.5%).
  - Pairs only the other decoder has, slot-confirmed: JTDX 1,330, deep JTDX 1,745, jt9 715.
  - `nn_v3_cc` vs `nn_v3_ccrpc`: 64 pairs only in `nn_v3_cc` (55 slot / 9 active), 618 only in `nn_v3_ccrpc`
    (513 / 105), 0 unconfirmed either way.
- **Wall times** (heavily loaded): noise + run 2 10:59:58–12:13:10 (73 min), run 3 12:13–13:18 (65 min), run 4
  13:18–14:14 (56 min), 3 h 15 min in all. `pmset -g log`: no Sleep, DarkWake or Wake event from 10:00 to 15:00.

**Rebased onto `main` `8ed894a`** (the co-channel stage committed, with the extended plausibility rules). The
`shipped` proof was repeated there: again 2,705 of 2,705 lines identical to `nn_v3_cc`.

**C port** (`c/src/decode.c`): a `sendable()` filter in `repeat_stage`'s competitor loop (`plausible77` and the
CQ-with-report/RRR/RR73/73 rule), and `REPEAT_SOLO` 0.35 as the runner-up when no competitor is left (it was −1).

Golden C == Python under MallocScribble:

| sample | 1 thread | 4 threads streaming | `rp` changes it covers |
|---|---|---|---|
| run 2 `02_80m` | 754 / 754 | 754 / 754 | 2 gained, 1 lost |
| run 4 `02_15m` | – | 885 / 885 | 7 gained, 5 lost |
| run 4 `00_20m` | 898 / 898 | – | 6 gained, 1 lost |

`tools/ci_smoke.py` 51 / 51; `wsjtx/test_sidecar.sh` all checks.

**`tools/validate_full.sh nn_v3_ccrpc` PASS**, 17:11–18:32 on the idle M4 with 8 workers, no sleep events:
- run 1: 34,585 / 26,505 / 1 → **34,727 / 26,615 / 1** (+142 / +110; the same unconfirmed decode; the emulation had
  predicted +153 / +122);
- training recordings: 18,563 → **18,677** (jt9-only 197 → 198);
- AWGN: 50% −22.49 / 90% −21.24, identical;
- golden: 00_40m 815 / 815, 08_80m 629 / 629, run 2 02_80m 754 / 754, AWGN 429 / 429.

## Files

Changed, uncommitted: `ft8nn/decoder.py`:
- `_sendable`, new;
- `_successors(sendable_only=True)`;
- `REPEAT_SOLO = 0.35`;
- the single-competitor runner-up in `_repeat_stage`.

New, uncommitted:
- `tests/test_repeat.py`;
- `tools/rp_competitors.py` (null and chained emulation, both competitor sets);
- `docs/rp-competitors.md` (this file);
- `scratch/`: `rp_doubt.py`, `rp_chain.py`, `null_sum.py`, `rpc_gain.py`.

Outputs, all new, under `data/`:
- In `data/logs/miss_dtcrp/`:
  - `rp_doubt.txt`, `rp_chain.txt`;
  - `rpc_null_run1.log`, the `rpc_null_run1.tsv.d/` folder, `rpc_null_run1_summary.txt`;
  - `rpc_real_run{1,2,3,4}.log`, `rpc_cc_run{2,3,4}.log`;
  - `rpc_gain_run{1..4}.txt`, `rpc_gain_cc_run{2,3,4}.txt`;
  - `rpc_stale_run{1..4}.txt`, `rpc_stale_cc.txt`;
  - `verify_run*_{dtcrp_emu,rpc_emu,cc_emu,ccrpc_emu}.txt`;
  - `compare_rpc.txt`.
- Per station, under `data/val/run{1..4}/<station>/`:
  - decode files `decodes_nn_v3_dtcrp_emu`, `decodes_nn_v3_rpc_emu`, `decodes_nn_v3_rpc_gain`;
  - on runs 2–4 also `decodes_nn_v3_cc_emu`, `decodes_nn_v3_ccrpc_emu`, `decodes_nn_v3_ccrpc_gain`;
  - the matching `verified_*.tsv`.

Reproduce:

```sh
V=<main>/datasets/ft8-validation-set
uv run python tools/rp_competitors.py $V/recordings --decodes data/val/run1 --base nn_v3 --shuffle 0 1 2 3 4 5 \
  --dump data/logs/miss_dtcrp/rpc_null_run1.tsv --workers 6                     # null, ~22 min at 2 workers under load
uv run python scratch/null_sum.py data/logs/miss_dtcrp/rpc_null_run1.tsv
uv run python tools/rp_competitors.py $V/recordings --decodes data/val/run1 --base nn_v3_dtcrp --real \
  --tag-old nn_v3_dtcrp_emu --tag-new nn_v3_rpc_emu --workers 6                # runs 2-4: $V/runN/recordings
python3 collect/verify.py --decodes data/val/run1 --tag nn_v3_rpc_emu          # runs 2-4: --recordings, --spots
python3 scratch/rpc_gain.py 1 nn_v3_dtcrp_emu nn_v3_rpc_emu --diff-tag nn_v3_rpc_gain
python3 tools/rp_stale.py --decodes data/val/run2 --tag nn_v3_rpc_gain --ref "$V/run2/recordings/{st}/decodes.txt" \
  --ref "data/val/run2/{st}/decodes_jtdx.txt" --ref "data/val/run2/{st}/decodes_jtdx_deep.txt"
```

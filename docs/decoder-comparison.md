# ft8nn vs WSJT-X `jt9`, JTDX, JTDX deep, and `ft8_lib`

## 1. Current comparison — `models/v4` (`libft8nn`) on the v2 multi-receiver validation set (2026-11-28)

Measured 2026-11-28 on the v2 validation set (`datasets/ft8-validation-set` `6a95807`: `runs/r02` and `runs/r05`–`r14`, 60 unseen KiwiSDRs across all 10 HF bands `160–10 m` and all 6 continents, disjoint from training and v1 runs 1–4, paired with simultaneous PSK Reporter spots and live OpenWebRX decode streams) on an **8-core Apple M1** (`6` workers, `caffeinate -dims`, 0 sleep/DarkWake events), using `tools/run_set.py`, `collect/verify.py`, and `tools/compare_decoders.py`.

Each block in `sets/small.txt` (16 blocks = 128 scored + 32 lead-in slots) and `sets/third.txt` (265 blocks = 2,120 scored + 530 lead-in slots) is decoded as its own 10-slot chronological sequence so stateful decoders (`ft8nn`, `JTDX`) have identical context/memory warm-up.

### Pooled results — `sets/third.txt` (265 blocks, 2,120 scored slots across 11 runs)

| Decoder | Decodes | vs `jt9` | Checkable | `slot` (>50 km) | `xconf` (≥50 km) | **Confirmed (`slot+xconf`)** | vs `jt9` | Conf % | `active` | `repeat` | `unconf` | `n/a` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ft8_lib` (`9fec6ca` `decode_ft8`) | 29,780 | −44.2% | 29,175 | 22,029 | 6,527 | 28,556 | −43.6% | 97.9% | 618 | 1 | **0** | 605 |
| `jt9 -8 -d 3` (WSJT-X 3.0.2 Improved) | 53,322 | – | 52,301 | 39,445 | 11,228 | 50,673 | – | 96.9% | 1,577 | 7 | **44** | 1,021 |
| `JTDX 2.2.x` (`2a0e2be`, defaults) | 56,642 | +6.2% | 56,120 | 41,862 | 12,168 | 54,030 | +6.6% | 96.3% | 2,081 | 4 | **5** | 522 |
| `JTDX` deep (`-w -c 3 -s 2`, ceiling) | 59,340 | +11.3% | 58,800 | 43,704 | 12,665 | 56,369 | +11.2% | 95.9% | 2,403 | 8 | **20** | 540 |
| **`ft8nn` (`models/v4`, `c/build/ft8nn_cli` `3A`)** | **65,919** | **+23.6%** | **64,783** | **48,169** | **13,649** | **61,818** | **+22.0%** | **95.4%** | **2,951** | **8** | **6** | 1,136 |

- **`ft8nn` vs `jt9 -8 -d 3`:** **`+12,597` decodes (`+23.6%`)** and **`+11,145` confirmed (`+22.0%`)**, ahead on **all 11 runs**.
- **`ft8nn` vs `JTDX` defaults:** **`+9,277` decodes (`+16.4%`)** and **`+7,788` confirmed (`+14.4%`)**, while running **faster per core** (`4.04 s` vs `4.39 s`).
- **`ft8nn` vs `JTDX` deep (`-w -c 3 -s 2`):** **`+6,579` decodes (`+11.1%`)** and **`+5,449` confirmed (`+9.7%`)**, at **`8.2×` lower CPU** (`29.77 min` vs `244.02 min` wall).
- **`unconf` inspection on `sets/third.txt`:**
  - **`ft8nn` (`6` `unconf`, only `2` actual false decodes = `0.003%`):** 4 of the 6 are genuine on-air transmissions (`CQ D01GMW JO31` on `r11` 160 m and `CQ DX IN80CE` on `r12` 40 m, both decoded via plain BP by all four deep decoders; `<UA1AAP> R8QBH/1` on `r02` 15 m, an active QSO partner also decoded by `jt9` and `JTDX` deep; and `DL1NB JO30` on `r02` 15 m, a free-text transmission from active station `DL1NB` where `verify.py` parsed `JO30` as the 2nd-word callsign). Only `CQ WP2IOF FN44` and `CQ MA3FML/R LH68` (`osd_ap0`) are false decodes.
  - **`JTDX` defaults (`5` `unconf`):** 2 shared real (`D01GMW`, `IN80CE`), 1 plausible (`IQ5PJ EA5TA -14`), 2 random (`I05RYL 9I9YDL MR90`, `8Q7NES/P XE8ERK GM75`).
  - **`JTDX` deep (`20` `unconf`):** 3 shared real (`R8QBH/1`, `D01GMW`, `IN80CE`) and **17 visibly random false decodes**, including **4 self-consistent double hallucinations** at identical slot & audio frequency (`DE TP9PLX TU` + `KV3CQB TP9PLX R 589 0110`; `DE H74KVQ TU` + `DP5SSO H74KVQ R 579 7856`; `DE SI5WYG TU` + `O53NXD SI5WYG R 599 2606`; `DE ZE2TAQ TU` + `AJ2VKS ZE2TAQ R 599 6485`).
  - **`jt9` (`44` `unconf`):** 3 shared real (`R8QBH/1`, `D01GMW`, `IN80CE`) and **41 random false decodes**.

### Strict exclusive confirmation (`verify.py --others`, 4 Hz / 0.2 s on another SDR ≥ 50 km away)

| Decoder | Exclusive decodes (`third`) | `xconf-strict` | Strict % | `unconfirmable` | `unconfirmable` base tiers |
|---|---:|---:|---:|---:|---|
| `ft8_lib` | 5 | 3 | 60.0% | 2 | `slot`: 1, `active`: 1 |
| `JTDX` defaults | 96 | 45 | 46.9% | 51 | `slot`: 26, `active`: 24, `unconf`: 1 |
| `jt9 -8 -d 3` | 167 | 88 | 52.7% | 79 | `slot`: 27, `active`: 11, `unconf`: 41 |
| `JTDX` deep | 516 | 186 | 36.0% | 330 | `slot`: 173, `xconf`: 1, `active`: 140, `unconf`: 16 |
| **`ft8nn` (`models/v4`)** | **7,102** | **4,590** | **64.6%** | **2,512** | **`slot`: 1,658, `xconf`: 8, `active`: 843, `unconf`: 3** |

### Per-run breakdown (`sets/third.txt`, `decodes / confirmed / unconf`)

| Run (bands, region) | `jt9 -8 -d 3` | `ft8_lib` | `JTDX` defaults | `JTDX` deep | **`ft8nn` (`models/v4`)** |
|---|---:|---:|---:|---:|---:|
| `r02` (15/17 m, EU) | 8,338 / 7,883 / 8 | 4,725 / 4,550 / 0 | 8,748 / 8,305 / 0 | 9,140 / 8,639 / 3 | **9,901 / 9,212 / 3** |
| `r05` (40/20 m, EU) | 5,195 / 5,019 / 3 | 2,735 / 2,661 / 0 | 5,546 / 5,335 / 1 | 5,852 / 5,592 / 1 | **6,488 / 6,148 / 0** |
| `r06` (10 m, NA) | 1,081 / 914 / 4 | 722 / 630 / 0 | 1,185 / 1,002 / 0 | 1,217 / 1,032 / 1 | **1,359 / 1,141 / 0** |
| `r07` (40/80 m, EU) | 8,252 / 7,983 / 6 | 4,098 / 4,008 / 0 | 8,867 / 8,554 / 0 | 9,306 / 8,948 / 2 | **10,541 / 10,048 / 0** |
| `r08` (20/30 m, NA+SA) | 5,964 / 5,720 / 4 | 3,114 / 3,018 / 0 | 6,581 / 6,271 / 0 | 7,040 / 6,682 / 1 | **7,918 / 7,479 / 0** |
| `r09` (20/30 m, EU+OC) | 2,769 / 2,426 / 1 | 1,795 / 1,592 / 0 | 2,915 / 2,603 / 0 | 2,994 / 2,665 / 0 | **3,288 / 2,862 / 0** |
| `r10` (20/17 m, EU+AS) | 6,916 / 6,584 / 3 | 4,245 / 4,077 / 0 | 7,309 / 7,057 / 0 | 7,600 / 7,311 / 5 | **8,317 / 7,869 / 0** |
| `r11` (160/80 m, EU) | 8,241 / 7,980 / 5 | 4,693 / 4,547 / 0 | 8,442 / 8,285 / 3 | 8,756 / 8,561 / 5 | **9,353 / 8,950 / 2** |
| `r12` (30/40 m, EU+AF) | 3,983 / 3,760 / 6 | 1,843 / 1,768 / 0 | 4,335 / 4,099 / 1 | 4,629 / 4,350 / 1 | **5,647 / 5,254 / 1** |
| `r13` (12 m, NA) | 1,785 / 1,620 / 2 | 1,160 / 1,065 / 0 | 1,888 / 1,709 / 0 | 1,969 / 1,771 / 1 | **2,233 / 2,000 / 0** |
| `r14` (60 m, EU dawn) | 798 / 784 / 2 | 650 / 640 / 0 | 826 / 810 / 0 | 837 / 818 / 0 | **874 / 855 / 0** |

### Timing & agreement with `sets/small.txt` (8-core Apple M1, `6` workers)

| Decoder | `small` (16 blocks) dec / conf / unconf | `small` wall / core-s per slot | `third` (265 blocks) dec / conf / unconf | `third` wall / core-s per slot |
|---|---|---|---|---|
| `ft8_lib` | 1,545 / 1,475 / 0 (−44.6%) | 0.02 min / 0.04 s | 29,780 / 28,556 / 0 (−44.2%) | 0.17 min / 0.02 s |
| `jt9 -8 -d 3` | 2,789 / 2,633 / 2 (–) | 1.91 min / 3.48 s | 53,322 / 50,673 / 44 (–) | 27.05 min / 3.66 s |
| `JTDX` defaults | 2,970 / 2,819 / 0 (+6.5%) | 2.34 min / 4.30 s | 56,642 / 54,030 / 5 (+6.2%) | 32.44 min / 4.39 s |
| `JTDX` deep | 3,128 / 2,953 / 2 (+12.2%) | 18.29 min / 31.85 s | 59,340 / 56,369 / 20 (+11.3%) | 244.02 min / 33.05 s |
| **`ft8nn` (`models/v4`)** | **3,396 / 3,154 / 0 (+21.8%)** | **1.99 min / 3.22 s** | **65,919 / 61,818 / 6 (+23.6%)** | **29.77 min / 4.04 s** |

`small` (19–25 min total) and `third` (~5.6 h total) rank all five decoders in the exact same order, with relative margins agreeing within ~1–2 percentage points.

### Pairwise overlaps on `sets/third.txt` (`(run, station, slot, sender)` pairs)

| Pair (`A` vs `B`) | Both | `A`-only (`confirmed [slot+xconf] / active / repeat / unconf`) | `B`-only (`confirmed [slot+xconf] / active / repeat / unconf`) |
|---|---:|---|---|
| `jt9` vs `ft8nn` (`nn_v4`) | 51,416 | 769 (`694 [552+142] / 33 / 1 / 41`) | **13,211 (`11,808 [9,267+2,541] / 1,398 / 2 / 3`)** |
| `jtdx` vs `ft8nn` (`nn_v4`) | 54,265 | 1,675 (`1,473 [895+578] / 199 / 1 / 2`) | **10,362 (`9,241 [7,219+2,022] / 1,113 / 4 / 4`)** |
| `jtdx_deep` vs `ft8nn` (`nn_v4`) | 56,414 | 2,150 (`1,829 [1,180+649] / 307 / 1 / 13`) | **8,213 (`7,287 [5,691+1,596] / 922 / 1 / 3`)** |
| `jtdx` vs `jtdx_deep` | 55,474 | 466 (`414 [324+90] / 51 / 0 / 1`) | 3,090 (`2,713 [2,136+577] / 361 / 3 / 13`) |
| `jt9` vs `jtdx` | 50,067 | 2,118 (`1,919 [1,484+435] / 154 / 3 / 42`) | 5,873 (`5,250 [3,874+1,376] / 620 / 0 / 3`) |
| `jt9` vs `ft8_lib` | 29,076 | 23,109 (`22,103 [17,432+4,671] / 956 / 6 / 44`) | 52 (`50 [40+10] / 2 / 0 / 0`) |

---

## 2. Historical comparison — `models/v3` on archived v1 validation runs 2–4 (2026-11-10)

Measured 2026-11-10 on archived v1 validation runs 2, 3 and 4 — 36 KiwiSDR receivers never used for
training, 4,302 slots of 15 s — and scored with legacy `collect/verify.py` (`--near-km 0 --xconf off`) against the PSK Reporter spots
captured during the same sessions. `nn_v3` and `jt9` columns are the measurements already recorded
in `STATUS.md`, on the same slots with the same scorer, so the rows are directly comparable.

Terms: a **slot** is one 15 s FT8 interval; **slot-confirmed** means another receiver on the network
spotted that sender in that same slot within 50 Hz, so the decode is corroborated from outside;
**unconf** ("unconfirmed") means the sender's callsign was never spotted on that band during the
session and we heard it only once, which is the closest thing we have to a count of false decodes.
**AP** is a-priori decoding (feeding the decoder bits it expects, e.g. `CQ`); **OSD** is ordered
statistics decoding, a last-resort algebraic decoder; **BP** is the belief-propagation LDPC decoder.

## Results, runs 2 + 3 + 4 pooled (unseen receivers)

| decoder | decodes | vs jt9 | slot-confirmed | vs jt9 | unconf (≈ false) |
|---|---|---|---|---|---|
| ft8_lib (stand-in for FT8TW, see below) | 49,835 | −42.2% | 37,113 | −42.0% | 4 |
| `jt9 -d 3` (WSJT-X, the recorded baseline) | 86,247 | – | 63,934 | – | 62 |
| **JTDX 2.2.x, its own defaults** | **93,392** | **+8.3%** | **69,159** | **+8.2%** | **11** |
| `nn_v3` (ours, as compared below) | 100,630 | +16.7% | 73,876 | +15.6% | 3 |
| `nn_v3_rpf` (+ repeat stage) | 105,154 | +21.9% | 77,369 | +21.0% | 4 |
| **`nn_v3_dtcrp` (ours, now shipped: + repeat stage + DT re-centring)** | **106,074** | **+23.0%** | **78,022** | **+22.0%** | **4** |

**Ours decodes 7.8% more than JTDX (6.8% more slot-confirmed), with 3 unconfirmed decodes against
JTDX's 11.** JTDX in turn beats stock `jt9` by 8.3%, so roughly half of our margin over WSJT-X is
margin JTDX also finds, and half is ours alone. Against JTDX at its most sensitive settings the
margin is +3.0% / +2.4%, and JTDX then makes 44 unconfirmed decodes to our 3 (section below).

**Update 2026-11-11: the repeat stage** (`STATUS.md` plan 12), built from what this comparison found
(section "What the JTDX-only confirmed decodes are"), tests a station's message from 30/60 s earlier
as a whole known codeword on the audio left after every other decode, much as JTDX's `^` stage does.
With it the decoder is +12.6% decodes / +11.9% slot-confirmed over default JTDX and +7.6% / +7.2% over
JTDX at its ceiling (97,711 / 72,160 / 44), and the slot-confirmed pairs JTDX gets and we miss fall from
3,069 to 1,719. Together with `main`'s DT re-centring (receivers whose clock drifts), the shipped
`nn_v3_dtcrp` is **+13.6% / +12.8% over default JTDX and +8.6% / +8.1% over its ceiling**, with
JTDX-only slot-confirmed pairs at 1,595 (2,063 against deep JTDX, from 3,780). The rest of this
document compares against `nn_v3`, the decoder before that change.

Per run (decodes / slot-confirmed / unconf):

| run | receivers, band mix | jt9 | ft8_lib | JTDX | `nn_v3` | ours vs JTDX |
|---|---|---|---|---|---|---|
| 2 | 9, mixed day/night | 24,291 / 17,993 / 16 | 13,649 / 10,115 / 4 | 26,293 / 19,435 / 1 | 28,198 / 20,701 / 1 | +7.2% / +6.5% |
| 3 | 12, another day and time | 23,524 / 17,888 / 17 | 14,751 / 11,279 / 0 | 25,279 / 19,125 / 3 | 26,688 / 20,105 / 0 | +5.6% / +5.1% |
| 4 | 15, high bands, afternoon | 38,432 / 28,053 / 29 | 21,435 / 15,719 / 0 | 41,820 / 30,599 / 7 | 45,744 / 33,070 / 2 | +9.4% / +8.1% |

Our margin is largest on run 4, the busiest audio in the set (up to 45 decodes per slot), and
smallest on run 3, the quietest.

The JTDX column is its own defaults. Its ceiling is measured separately, below.

## JTDX at its most sensitive settings

JTDX has three sensitivity knobs above its defaults, and all three were run together: `nft8cycles` 3
(nine passes with signal subtraction instead of three), FT8 sensitivity 2 (`lft8subpass`, a further
subtraction pass on top of the already-low sync thresholds) and SWL mode (51 half-symbol blocks
instead of 49, and a sync threshold of 1.225 instead of 1.5). There is nothing more sensitive left
to turn on, so this is JTDX's ceiling.

All three runs in full (all 36 receivers; first measured with run 4 at 13 of 15 receivers, completed
and re-measured end to end on 2026-11-10 — runs 2 and 3 reproduced the earlier numbers exactly). Cells
are decodes / slot-confirmed / unconf.

| run | JTDX defaults | JTDX `-w -c 3 -s 2` | `nn_v3` (ours) | deep vs default | ours vs deep |
|---|---|---|---|---|---|
| 2 (9 receivers) | 26,293 / 19,435 / 1 | 27,417 / 20,202 / **16** | 28,198 / 20,701 / 1 | +4.3% / +3.9% | +2.8% / +2.5% |
| 3 (12 receivers) | 25,279 / 19,125 / 3 | 26,321 / 19,855 / **9** | 26,688 / 20,105 / 0 | +4.1% / +3.8% | +1.4% / +1.3% |
| 4 (15 receivers) | 41,820 / 30,599 / 7 | 43,973 / 32,103 / **19** | 45,744 / 33,070 / 2 | +5.1% / +4.9% | +4.0% / +3.0% |
| **2 + 3 + 4 pooled** | 93,392 / 69,159 / 11 | 97,711 / 72,160 / **44** | 100,630 / 73,876 / 3 | **+4.6% / +4.3%** | **+3.0% / +2.4%** |

(The two run-4 receivers missing from the first measurement, `13_15m` and `14_12m`, are quiet ones:
853 and 925 deep decodes, no unconfirmed.)

The gain is consistent and modest: **+4 to +5% decodes** over JTDX's own defaults, for **5–6× the
CPU** (on the Mac mini M4, 20.5 s per slot per core against 4.0, 235.7 min wall clock for the three
runs against 46.4; on the 8-core M1 that took the first measurement, 29.4 against 4.9). It closes about
60% of our margin: pooled over all three runs we go from +7.8% / +6.8% ahead of default JTDX to
**+3.0% / +2.4%** ahead of deep JTDX — least on run 3, the quietest (+1.4% / +1.3%), most on run 4,
the busiest (+4.0% / +3.0%).

**But it is bought almost entirely out of the false-decode budget.** Unconfirmed decodes multiply on
every run: 1 → 16, 3 → 9, 7 → 19. Pooled, deep JTDX's **44** unconfirmed decodes are much closer to
stock `jt9`'s 62 than to its own defaults' 11 — while ours is **3**. Read the ceiling that way: JTDX
can find about 4.6% more signals than its defaults do, and the price is a false-decode rate four
times its own and about fifteen times ours.

### Where the extra decodes and the new false decodes come from (run 2)

Every one of run 2's 16 unconfirmed decodes comes from the ordinary BP/OSD path (blank service tag).
None comes from the `^` tone-matching stage, which stays at **0 unconfirmed in 15,219** decodes even
here. The blank path's unconfirmed rate goes from 0.009% to 0.14%. The decodes themselves look like
what they are:

```
DE 2D3BRU TU                  80 m, 22:53:15, 2181 Hz
3C9JDH 2D3BRU R 539 6833      same slot, same frequency
Y11MLR/R HZ6GLY OC68          40 m, 22:39:30,  374 Hz
CB9VJO HG3MGZ LC41            15 m, 22:35:00, 1878 Hz
```

Implausible callsigns, and in three of the sixteen cases a `DE <call> TU` free-text line *plus* a
contest-style report naming the same invented callsign at the identical slot and frequency — one
noise peak decoded twice into a self-consistent pair.

The gain is also not monotone. Counted as (slot, sending callsign), run 2's deep pass finds 27,139
callsign-slots against the default's 26,050, but **290 of the default's are missing from the deep
run**: the additional subtraction passes remove signal that other candidates needed.

## How much do JTDX and jt9 actually hear differently?

Counted as (slot, sending callsign) pairs — the identity `verify.py` scores — which is immune to
JTDX resolving a hashed callsign that jt9 prints as `<...>`, and to jt9's `a1`…`a7` AP markers:

| | jt9 | JTDX | ft8_lib |
|---|---|---|---|
| checkable callsign-slots | 84,389 | 92,187 | 48,825 |

JTDX and jt9 share 80,940 of them. JTDX adds 11,247 that jt9 never gets (137 of them at ≥ 0 dB) and
misses 3,449 that jt9 gets (228 at ≥ 0 dB). ft8_lib finds only **227** callsign-slots that JTDX
misses, i.e. it is very nearly a subset of JTDX.

So JTDX is not a different kind of receiver from WSJT-X: it contains 96% of what jt9 hears and digs
out about 13% more. The 228 strong decodes it misses are the same class of miss we have open as plan
step 7.1 — worth a look, because they are strong signals that two independent decoders disagree on.

## What ours and JTDX each hear that the other doesn't

The same (slot, sending callsign) count, pooled over runs 2–4, with each side's exclusive pairs split
by `verify.py` tier — slot-confirmed / active (spotted on the band in the session, not in that slot)
/ repeat / unconf. Slot-confirmed exclusives are real signals the other decoder missed.
`tools/compare_decoders.py` prints this.

| | both | JTDX only | `nn_v3` only |
|---|---|---|---|
| `nn_v3` vs JTDX defaults | 87,232 | 4,955 (3,069 / 1,870 / 7 / 9) | 11,413 (7,840 / 3,567 / 4 / 2) |
| `nn_v3` vs JTDX `-w -c 3 -s 2` | 90,375 | 6,035 (3,780 / 2,208 / 9 / 38) | 8,270 (5,577 / 2,687 / 4 / 2) |
| JTDX defaults vs deep | 91,157 | 1,030 (741 / 283 / 0 / 6) defaults only | 5,253 (3,716 / 1,500 / 2 / 35) deep only |

So this is not a superset relation either way. Even against default JTDX there are **3,069
slot-confirmed signals JTDX decodes and we do not** (3.3% of what JTDX hears), against 7,840 the other
way; against deep JTDX it is 3,780 against 5,577. Those JTDX-only confirmed decodes are the clearest
list of recoverable misses this project has: real, externally corroborated signals that a classical
decoder already gets out of the same audio. Of the false-decode budget, 38 of deep JTDX's 44
unconfirmed decodes are ones we do not make, while 2 of our 3 are ones it does not.

Deep JTDX also drops 1,030 pairs its own defaults find (741 slot-confirmed), the non-monotone effect
seen in run 2 above: the extra subtraction passes remove signal that other candidates needed.

## Where JTDX's decodes come from

JTDX tags every decode with a service character (`decoder.f90`'s `ft8_decoded` callback), and those
tags say which stage produced it. Across all 93,534 raw decode lines:

| tag | count | share | stage | unconf rate |
|---|---|---|---|---|
| `^` | 50,365 | 53.8% | its `ft8sd1` / `ft8mf1` / `ft8mfcq` tone-matching stages | 0.0000% |
| (blank) | 40,862 | 43.7% | ordinary BP + OSD on the LDPC codeword | 0.0247% |
| `*` | 1,848 | 2.0% | a-priori decoding (`iaptype` ≠ 0) | 0.0000% |
| `1` | 416 | 0.4% | non-standard-callsign messages (`i3` = 1) | – (sender not checkable) |
| `.` | 43 | 0.0% | free text away from the RX frequency | 7.7% |

The striking part: **over half of JTDX's output comes from a stage that never passes the LDPC CRC.**
`ft8sd1` and friends match the received tone sequence against constructed message hypotheses and, on
a match, set `nbadcrc = 0` and hand the message straight out. That sounds like a false-decode engine,
and it is not: of 50,350 checkable `^` decodes, PSK Reporter contradicted **none**. The 11
unconfirmed decodes JTDX does produce come almost entirely from its *ordinary* BP/OSD path.

Its decoder stack is otherwise recognisably WSJT-X's with more of everything: 3, 6 or 9 passes with
signal subtraction (`nft8cycles`), `bpdecode174_91` plus `osd174_91` at `ndeep` 3–5, AP decoding,
candidate thinning, and a wall of plausibility checks (`chkcall`, `chkmsg`, `chkfalse8`, `chkhist`,
`chkgrid`, `chklong8`, `chkspecial8`, `chkflscall`). It also keeps state across slots — an even/odd
message memory and a callsign hash table — which is why it can print `IZ2FWJ EK/RX3DPK -08` where
jt9 only manages `IZ2FWJ <...> -08`.

## Running JTDX at all required a driver

JTDX's decoder binary **cannot read a WAV file**. JTDX deleted WSJT-X's file-reading branch from
`lib/jt9.f90`, whose header now says: "Decoder for JT9. Can run as the back end of JTDX only, with
data placed in a shared memory region." Only the `jt9a()` shared-memory path its GUI drives is left,
and the `-8` (FT8) option is gone from the command line.

The decoder subroutines are untouched, so `tools/jtdx_ft8_file.f90` re-adds file mode: it mirrors
`lib/jt9a.f90`'s `ndiskdat` branch (audio into `dd8`, `nzhsym` = 49 half-symbol blocks, which is
also what live JTDX uses for FT8 — `mainwindow.cpp`'s `m_hsymStop`) and fills every `params_block`
field the way `MainWindow::decode()` does, then calls JTDX's own `multimode_decoder`. JTDX prints
the decode lines itself. Each station's 120 slots go through one process in chronological order, so
the cross-slot state above behaves as it does live.

Settings are JTDX's own defaults: `ndepth` 3 (deepest), `nft8cycles` 1 (3 passes), FT8 sensitivity 1
(low sync thresholds), `ncandthin` 100 (no thinning), SWL mode off, RX-frequency filter off, contest
and telemetry hiding off, no `MyCall`/`DxCall` (so MyCall-directed AP is inert, matching how the jt9
baseline was taken). Search range 150–3100 Hz, the same window our decoder searches; every jt9-only
decode in the set falls inside it, so the range is not costing JTDX anything.

**One setting matters more than it looks.** With JTDX's duplicate suppression off, it emits weak
spurious copies of messages it has just decoded — `EA3CCI K2ZJ RR73` at 335 Hz +2 dB and again at
216 Hz −14 dB in the same slot. `HideFT8Dupes` defaults to **true** in JTDX, so that is how it was
run; left off, a four-slot spot check produced 159 decode lines instead of 147 (+8%), and those
extra lines are artefacts `verify.py` would have scored as slot-confirmed, because they name a
station that really is there.

Two upstream files need help from a modern toolchain, neither on the FT8 path: `lib/{char,int}.h`
declare the Reed-Solomon codec with prototypes that disagree with the definitions in
`lib/ftrsd/*.c`, which clang 15+ rejects (corrected copies are generated into a `compat/` dir), and
`lib/jplsubs.f` (JPL ephemeris, for the astro window) calls its own `SPLIT`, which gfortran 14+
resolves to the F2023 intrinsic (dropped; the link proves the FT8 path never refers to it).

## FT8TW cannot be run, and what stands in for it

[FT8TW](https://github.com/danleetw/FT8TW) is an Android app, a fork of
[FT8CN](https://github.com/N0BOY/FT8CN) by BG7YOZ. **Neither repository publishes decoder source.**
FT8TW's published tree is 242 Java files with every JNI entry point commented out —
`InitDecoder`, `DecoderFt8FindSync`, `DecoderFt8Analysis`, `setDecodeMode(isDeep)` — and no C or C++
directory at all. Upstream FT8CN ships the decoder only as a prebuilt `libft8cn.so` for Android
arm64, which cannot run on macOS and cannot be fed 4,302 WAV files.

So the row above is **ft8_lib** ([kgoba/ft8_lib](https://github.com/kgoba/ft8_lib), `9fec6ca`), used
as a stand-in on two grounds from FT8CN's own README: it credits Karlis Goba (YL3JG) for the
implementation its code follows, and it states the app uses "lightweight operations instead of deep
decoding" because of phone CPU and battery limits. Read the row as a floor for that family, not as a
measurement of the app: `decode_ft8` is a single pass with no subtraction and no OSD.

The result is unambiguous anyway — 49,835 decodes against our 100,630, a factor of 2.0 — and the gap
is the decoder, not a configuration ceiling: its 50-message-per-slot cap never binds (its busiest
slot holds 32 decodes where jt9 finds 54), and its 200–3000 Hz search covers the recordings' audio.
What it does find is clean (4 unconfirmed in 49,835), and 48,598 of its 48,825 callsign-slots are
also found by JTDX. It is, however, extremely cheap: about 0.03 s per slot against JTDX's ~4.9 s and
our decoder's ~4 s, which is the trade the app is making deliberately.

### What the JTDX-only confirmed decodes are

`tools/diag_ref_only.py` breaks the 3,069 slot-confirmed pairs JTDX gets and `nn_v3` misses down
(`STATUS.md` plan 12 has the detail): three quarters are below −15 dB, 62.5% come from JTDX's `^`
stage, and **1,675 (55%) are exact repeats** of a message we decoded ourselves 30 or 60 s earlier
within 10 Hz — the case our context a-priori stage exists for, which loses them mostly at its
confidence gate (53%) or for lack of a candidate (23%). Testing the whole known message against the
audio at the earlier decode's frequency and time recovers about 90% of them in a 300-signal replay
(`tools/diag_ctx_miss.py`). 424 (14%) are a second signal within 10 Hz of one we decode. And 115 (4%)
are not misses: we decode them too, but print the resolved hashed callsign in brackets (`<SX200H>`),
which `verify.py` does not count as a checkable sender while JTDX's unbracketed `SX200H` is — so the
margins in this document slightly understate ours.

## Caveats

- **JTDX keeps cross-slot state here; the jt9 baseline did not.** The baseline ran one `jt9`
  process per slot (`collect/val.py`), so jt9 had no callsign-hash or message memory, while JTDX and
  our decoder (context AP) both carry information between slots. That is how both programs really
  run, but it is not a per-slot-identical comparison.
- **The +7.8% / +6.8% headline margin is against default JTDX**, which is how JTDX ships; against
  its ceiling it is +3.0% / +2.4% (section above).
- **`^` reliability is measured against PSK Reporter coverage**, which is what bounds every
  false-decode number in this project: an unconfirmed decode means no outside spot, not proof of
  error, and 0 unconfirmed means no contradiction was found.
- ALLCALL7.TXT is empty in this setup (JTDX's user-maintained callsign list, not shipped with the
  source), i.e. fresh-install behaviour.

## Reproduce

```sh
V=datasets/ft8-validation-set

python3 tools/jtdx_build.py                      # clones JTDX, builds decoder + file driver
sh tools/ft8lib_build.sh                         # clones and builds ft8_lib's decode_ft8

for r in 2 3 4; do
  python3 tools/jtdx_run.py   --recordings $V/run$r/recordings --out data/val/run$r \
      --tag jtdx   --workers 6
  python3 tools/ft8lib_run.py --recordings $V/run$r/recordings --out data/val/run$r \
      --tag ft8lib --workers 4
  for t in jtdx ft8lib; do
    python3 collect/verify.py --recordings $V/run$r/recordings \
        --spots $V/run$r/pskr_spots.jsonl --decodes data/val/run$r --tag $t --list
  done
done
```

JTDX over all three runs took 58 min with 6 workers on an 8-core M1 (17.2 + 17.2 + 23.9 min, about
4.9 s per slot per core); ft8_lib took under a minute for the same 4,302 slots; `verify.py` about a
minute per run. JTDX at its most sensitive settings cost 132.5 min for run 2 and 122.6 min for run 3 on the same 6
workers (29.4 s per slot per core, 6× the default's CPU). Per-station time tracks decodes per slot,
not slot count: run 2's busiest 40 m receiver alone took 95 min for its 120 slots.

```sh
for r in 2 3 4; do
  python3 tools/jtdx_run.py --recordings $V/run$r/recordings --out data/val/run$r \
      --tag jtdx_deep --workers 6 -- -w -c 3 -s 2
done
```

On the Mac mini M4 with `--workers 8`, the whole comparison took 7 h: JTDX defaults 46.4 min (16.9 +
15.1 + 14.4), `nn_v3` (`bench/run_real.py`, commands in `STATUS.md`) 2 h 11 min (37.8 + 40.4 + 53.0),
deep JTDX 235.7 min (71.8 + 74.2 + 89.7). Every receiver's file is written when it finishes. Then, with
`verify.py` run for each tag:

```sh
python3 tools/compare_decoders.py --tags jt9 jtdx jtdx_deep nn_v3   # totals per run + pairwise overlap
```

Building needs gfortran, cmake, pkg-config, FFTW3 and the Boost headers (`brew install gcc cmake
pkg-config fftw boost`).

JTDX source: `git://git.code.sf.net/p/jtdx/code` at `2a0e2be` (2022-03-01, JTDX 2.2.x — the last
public commit of that tree).

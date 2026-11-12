# What other decoders still find that `nn_v3_dtcrp` misses (runs 2–4)

Draft, 2026-11-12. A re-run of plan 12's "where the JTDX-only confirmed decodes come from" against the shipped
default `nn_v3_dtcrp`, which is model v3 + DT re-centring + the repeat stage. It is widened to every reference
decoder we have: stock WSJT-X `jt9`, JTDX at its defaults, and JTDX at its ceiling ("deep JTDX", `-w -c 3 -s 2`).

## Terms

- **Decode:** a message our decoder (or a reference decoder) prints for a slot (one 15 s FT8 period of one
  receiver).
- **(slot, sender) pair:** the identity `tools/compare_decoders.py` counts. One sending callsign in one slot,
  whatever text each decoder prints for it.
- **Slot-confirmed** (`collect/verify.py` tier `slot`): someone else spotted that sender on PSK Reporter in the
  same slot, so the signal was real. **Active**: spotted on the band within 10 min. **Unconfirmed**: never
  spotted, which is roughly a false decode.
- **Miss:** a pair that at least one reference decoder reports at tier `slot` and that we report at no tier.
  "JTDX-only" etc. below always means *slot-confirmed* pairs JTDX has and we don't.
- **Candidate:** a (frequency, time offset) cell where the Costas-sync search thinks a signal starts. **DT:** the
  signal's time offset from the nominal start (+0.5 s). **Fine sync** (`fine_sync2`) refines a start point to
  the best two hypotheses within ±2.5 Hz and ±0.06 s.
- **LLR:** log-likelihood ratio, the network's soft confidence per code bit. **E** (expected bit errors) =
  Σ 1/(1+e^|LLR|) over the 174 bits. **BP / OSD:** belief propagation and ordered-statistics decoding of the
  LDPC code, both checked by the 14-bit CRC.
- **A-priori (AP):** pinning message bits we expect, then decoding the rest. **Context AP** (`bp_cx`/`osd_cx`)
  pins the call pair or the sender of a message we decoded 30/60 s earlier within 10 Hz / 0.5 s.
- **Repeat stage** (`rp`, plan 12): for a sender we decoded 30/60 s earlier but not in this slot, it tests whole
  known messages against the residual audio. ρ = Σ l·s / Σ|l| between the clipped LLRs and the codeword (s = ±1),
  so 1 means perfect agreement and about 0 means noise. It accepts an *earlier* message only if ρ ≥ 0.40 and ρ
  beats the runner-up by ≥ 0.15. The runner-up competitors are the ~130 messages the same calls could send next
  ("successors").
- **JTDX stage markers:** `^` tone-matching hypothesis (no LDPC/CRC), `*` a-priori, `-` plain BP/OSD.

## Summary

- **Size.** Pooled over runs 2–4, **2,495 slot-confirmed (slot, sender) pairs** are decoded by jt9, JTDX or deep
  JTDX and missed by `nn_v3_dtcrp` at every tier. That is 2.4% of our 106,074 decodes, down from 4,408 against
  `nn_v3`. JTDX-only: **1,595**. After the repeat stage alone (`nn_v3_rpf`) it was the 1,719 STATUS quotes,
  confirmed. Deep-JTDX-only 2,063, jt9-only 786.
- **They are weak:** median −19 dB on the reference's scale, 73% below −15 dB.
- **Mechanisms.** About 41% have no workable LLRs (expected errors ≥ 40 even with a candidate). 35% have no
  search candidate at all. Yet the reference's own message matches the audio (ρ ≥ 0.4) for 89%. The information
  is there for a known-message test, not for an unconstrained decode.
- **Co-channel** (a different sender of ours within 10 Hz / 0.4 s) is **350 misses (14.0%)**, the
  third-largest class after "heard at other times" (31.5%) and "exact repeat" (19.8%). As a lever it is
  smaller: a decode forced at the true spot succeeds for 130 (a bound), and all but 12 of those are also reached
  by the look-back stage, extra lag tracks or the budget/order fix below. The other fork's grid needs are in
  "Co-channel geometry".
- **Union of all levers:** 1,270 of the 2,495. 1,150 of them are decoder changes; the other 120 are a scorer fix.

### Ranking (step 3): recoverable slot-confirmed misses × plausibility of the fix

Counts are pooled over runs 2–4. "Unique" means not reached by any other lever in this table (`levers.txt`).
Every count is a replay emulation or a bound, not a decoder run, and every fix needs its thresholds and null set
on run 1 (and `noise10k`) before runs 2–4 judge it.

| # | lever | recoverable (run 2 / 3 / 4) | unique | example | fix | effort, risk |
|---|---|---|---|---|---|---|
| 1 | **Repeat stage without nonsense competitors** | **224** (36 / 47 / 141), exact emulation | 208 | `CQ PB1PCH JO22` ρ 0.55 lost to `CQ PB1PCH +27` 0.44 | `_successors`: no R flag and no report/RRR/RR73/73 for a CQ prior (Python + C, a few lines) | **small**; redo the run-1 wrong-receiver shuffle; live-usable |
| 2 | **More lag tracks per frequency** in the candidate search | **396** decode when forced at the true lag (144 / 77 / 175); the true lag is top-4 at its frequency for 282 | 215 | `CT1CHT OZ5AGJ JO47` +3 dB, E = 0.0 forced, no candidate | top-k lag peaks per bin (or 2-D maxima in a small window), maybe only in passes 2–4 | medium; CPU and the sidelobe budget problem; A/B as usual; live-usable |
| 3 | **Look-back repeat stage** (context from 30/60 s *later*) | **564** at the shipped thresholds (192 / 123 / 249) | 372 | `CQ AD8LV EN91` ρ 0.88 vs 0.26 (first CQ before we had context) | run `_repeat_stage` on slot t−30/t−60 once slot t is decoded (keep two residuals per period) | medium; **not live** (decodes 30–60 s late), so score it separately; backwards shuffle null |
| 4 | `verify.py` scorer fix (`<CALL>` resolved hash) | 120 (26 / 29 / 65) | 120 | `SP2ILQ <SX200H> -16` we decode; JTDX prints `SX200H` | treat a resolved `<CALL>` as checkable (mirrors the data repo's copy) | small; your call (plan 12); moves every recorded number slightly in our favour |
| 5 | Budget / order: a candidate exists and decodes in the final residual | 106 (36 / 23 / 47) | 43 | – | find out which rule skips them (candidate reuse, `max_cand`, a 5th pass) | small to look, unknown to fix |
| 6 | **Co-channel post-subtraction forced retry** (the other fork) | class 350; BP + OSD-2 at the true spot 130 (56 / 20 / 54) | 12 | `EA4IUJ HA7TM R-14` under `EA7GNO W9AH EN64`, Δf +2.6, ΔDT −0.10 | forced hypotheses around each subtracted decode | medium to large; ghosts. On its own ≤ 130; of those, 69 also come from #2, 38 from #5, 55 from #3. In the residual the class is NOCAND 128 / FAIL 182 (113 at E ≥ 40, i.e. damaged by the neighbour's subtraction, which a retry does not fix) / OK 38 |
| 7 | Accept a successor that clearly wins | ≤ 66 (23 / 11 / 32) | 24 | `D44PX F5PHW R-15` ρ 0.90 vs 0.42 | let the repeat stage accept a competitor at ρ ≥ 0.40 / margin ≥ 0.15 | small; stale-message risk is exactly what the margin guards |
| 8 | DT window / band edge | 61 + 30 | – | deep JTDX at DT +3.5…+3.6 or −2.7…−3.0 | wider DT failed its A/B before (`nn_v2_dt25`); edges are a noise-normalisation (metric) problem | low value |
| – | the rest | ~1,225 | – | mostly E ≥ 40 with no context | model / metric work (training) | long |

**Why this order.** #1 is the cheapest, live-usable, and turns an existing stage's mostly-correct "no" into
"yes" for messages that already score best. #2 is a real search defect with clean, often strong examples. #3 is
the biggest single number, but it only helps offline use. #6 shrinks once #2 and #3 exist; on its own, in live
use, it is worth about as much as #5.

## Overlaps (step 1)

`tools/compare_decoders.py --tags jt9 jtdx jtdx_deep nn_v3 nn_v3_rpf nn_v3_dtcrp`, runs 2–4. Rows are
decodes / slot-confirmed / unconfirmed:

| run | jt9 | JTDX defaults | deep JTDX | `nn_v3` | `nn_v3_rpf` | **`nn_v3_dtcrp`** |
|---|---|---|---|---|---|---|
| 2 | 24,291 / 17,993 / 16 | 26,293 / 19,435 / 1 | 27,417 / 20,202 / 16 | 28,198 / 20,701 / 1 | 29,435 / 21,646 / 1 | **29,435 / 21,646 / 1** |
| 3 | 23,524 / 17,888 / 17 | 25,279 / 19,125 / 3 | 26,321 / 19,855 / 9 | 26,688 / 20,105 / 0 | 27,711 / 20,883 / 1 | **28,206 / 21,266 / 1** |
| 4 | 38,432 / 28,053 / 29 | 41,820 / 30,599 / 7 | 43,973 / 32,103 / 19 | 45,744 / 33,070 / 2 | 48,008 / 34,840 / 2 | **48,433 / 35,110 / 2** |
| pooled | 86,247 / 63,934 / 62 | 93,392 / 69,159 / 11 | 97,711 / 72,160 / 44 | 100,630 / 73,876 / 3 | 105,154 / 77,369 / 4 | **106,074 / 78,022 / 4** |

Pairs only the reference has, by tier (slot / active / repeat / unconf), pooled:

| reference | vs `nn_v3` | vs `nn_v3_rpf` | **vs `nn_v3_dtcrp`** | ours-only vs that reference (`nn_v3_dtcrp`) |
|---|---|---|---|---|
| jt9 | 1,894 (1,373 / 459 / 0 / 62) | 1,322 (906 / 354 / 0 / 62) | **1,138 (786 / 290 / 0 / 62)** | 20,830 (14,889 / 5,934 / 3 / 4) |
| JTDX defaults | 4,955 (3,069 / 1,870 / 7 / 9) | 3,310 (1,719 / 1,575 / 7 / 9) | **3,109 (1,595 / 1,498 / 7 / 9)** | 15,003 (10,517 / 4,479 / 4 / 3) |
| deep JTDX | 6,035 (3,780 / 2,208 / 9 / 38) | 4,180 (2,269 / 1,864 / 9 / 38) | **3,874 (2,063 / 1,764 / 9 / 38)** | 11,545 (8,011 / 3,527 / 4 / 3) |

**The 1,719 JTDX-only slot-confirmed pairs after the repeat stage are confirmed (`nn_v3_rpf`); with DT
re-centring on top (`nn_v3_dtcrp`, the shipped default) it is 1,595.** Slot-confirmed misses per run:

| | run 2 | run 3 | run 4 | pooled |
|---|---|---|---|---|
| jt9-only | 266 | 179 | 341 | 786 |
| JTDX-only | 450 | 373 | 772 | 1,595 |
| deep-JTDX-only | 575 | 480 | 1,008 | 2,063 |
| **union** (any of the three) | **711** | **580** | **1,204** | **2,495** |

The union of the three is what the rest of this note profiles: **2,495 slot-confirmed (slot, sender) pairs, 2.4%
of our 106,074 decodes.** That is down from 4,408 against `nn_v3`. Most union misses are found by more than one
reference:

| who has it | JTDX + deep | deep only | jt9 + JTDX + deep | jt9 only | JTDX only | jt9 + deep | jt9 + JTDX |
|---|---|---|---|---|---|---|---|
| pairs | 932 | 588 | 455 | 224 | 189 | 88 | 19 |

## What the misses look like (JTDX-only, as in plan 12)

`tools/diag_ref_only.py --ours nn_v3_dtcrp` (output `diag_ref_only_jtdx.txt`), the same breakdown plan 12 made for
`nn_v3`:

| | `nn_v3` (plan 12, 3,069) | **`nn_v3_dtcrp` (1,595)** |
|---|---|---|
| below −15 dB (JTDX's SNR) | 75% | **71%** (37.6% below −20) |
| JTDX `^` tone-matching / `*` AP / `-` BP-OSD | 62.5% / 12.0% / 25.5% | **43.2% / 18.7% / 38.1%** |
| jt9 decodes it too | 30% | **30%** |
| we decode a different message within 10 Hz (any DT) | 14% (424) | **23% (364)** |
| CQ / grid / report / RR73-RRR-73 | – | 55% / 23% / 14% / 8% |

The repeat stage removed mostly `^` decodes: JTDX's tone matching is itself a known-message test. The remainder
is weak (median −18 dB) and spread over all stations and frequencies.

## Mechanism classes (step 2)

`tools/miss_profile.py` flags each of the 2,495 union misses from saved decodes alone (`misses.tsv`).
`tools/miss_replay.py` then replays all 1,714 slots involved (`replay.tsv`): it emulates the repeat stage for the
missed sender, runs the candidate search on the residual, and forces a decode at the reference's own (frequency,
DT). The classes below are exclusive, assigned in the table's order:

| class | what it means | pooled | share | run 2 | run 3 | run 4 | jt9 has it | JTDX `^`/`*`/`-` | < −15 dB | decoded by us verbatim 30/60 s **later** |
|---|---|---|---|---|---|---|---|---|---|---|
| scorer | we decode it; `verify.py` can't see it (our `<CALL>` vs their `CALL`) | 120 | 4.8% | 26 | 29 | 65 | 1 | 0/0/120 | 12 | 18 |
| DT window | reference DT outside our window (even after re-centring) | 61 | 2.4% | 10 | 26 | 25 | 10 | 17/6/37 | 27 | 22 |
| band edge | < 200 or > 3000 Hz (inside our 150–3100 Hz search; outside the noise-estimate band) | 30 | 1.2% | 15 | 7 | 8 | 16 | 13/2/15 | 10 | 1 |
| **co-channel** | a different sender of ours within 10 Hz / 0.4 s | **350** | **14.0%** | 129 | 56 | 165 | 128 | 131/54/123 | 269 | 135 |
| **repeat** | exact text decoded by us 30/60 s earlier within 10 Hz | **494** | **19.8%** | 100 | 141 | 253 | 54 | 421/28/37 | 416 | 264 |
| pair in context | same two calls 30/60 s earlier within 10 Hz / 0.5 s (context AP pins them) | 107 | 4.3% | 34 | 11 | 62 | 33 | 86/0/12 | 82 | 10 |
| sender in context | sender 30/60 s earlier within 10 Hz / 0.5 s | 201 | 8.1% | 54 | 53 | 94 | 112 | 12/59/102 | 122 | 88 |
| same text, other time | exact text decoded by us 30–300 s earlier at any frequency | 203 | 8.1% | 53 | 40 | 110 | 69 | 62/87/41 | 177 | 65 |
| **heard** | sender decoded by us at some other time on this receiver | **787** | **31.5%** | 266 | 191 | 330 | 315 | 168/171/345 | 575 | 295 |
| never heard | sender never decoded by us on this receiver | 142 | 5.7% | 24 | 26 | 92 | 48 | 46/46/30 | 128 | 0 |
| **all** | | **2,495** | | 711 | 580 | 1,204 | 786 | 956/453/862 | 1,818 | **898** |

### Search or metric? (replay of every miss slot)

All our decodes of the slot are subtracted, then two tests are run on what remains:
- **Candidate test:** look for a search candidate within 5 Hz / 0.2 s of the reference's signal. NOCAND = none
  exists. With a candidate, run NN → BP + OSD-2 (FAIL by expected errors E, or OK). OTHER = decoded to a
  different message.
- **Forced test:** the same decode started straight at the reference's (frequency, DT), with no search. ρ is that
  of the reference's own message there.

| class | n | SNR med | NOCAND | FAIL E < 13 | FAIL E 13–40 | FAIL E ≥ 40 | OK | forced OK | NOCAND but forced OK | forced ρ ≥ 0.4 |
|---|---|---|---|---|---|---|---|---|---|---|
| scorer | 120 | −10 | 105 | 0 | 0 | 15 | 0 | 0 | 0 | 8 |
| DT window | 61 | −15 | 59 | 0 | 0 | 1 | 1 | 40 | 38 | 58 |
| band edge | 30 | −12 | 1 | 0 | 15 | 14 | 0 | 0 | 0 | 28 |
| co-channel | 350 | −19 | 128 | 3 | 66 | 113 | 38 | 130 | 69 | 317 |
| repeat | 494 | −21 | 118 | 18 | 77 | 272 | 7 | 41 | 24 | 421 |
| pair in context | 107 | −19 | 39 | 2 | 18 | 41 | 7 | 44 | 27 | 98 |
| sender in context | 201 | −17 | 99 | 0 | 30 | 68 | 4 | 108 | 78 | 197 |
| same text, other time | 203 | −21 | 55 | 0 | 37 | 102 | 8 | 28 | 17 | 197 |
| heard | 787 | −19 | 229 | 4 | 210 | 306 | 38 | 226 | 129 | 756 |
| never heard | 142 | −21 | 44 | 0 | 14 | 81 | 3 | 23 | 14 | 134 |
| **all** | **2,495** | **−19** | **877** | 27 | 467 | **1,013** | 106 | **640** | **396** | 2,214 |

(5 decode to a different message, OTHER, not shown.) The scorer row is the control: those signals are ours
and were subtracted, so the residual has no candidate and nothing to force (0 forced OK).

What this says:
- **41% (1,013) are below what an unconstrained decode can do.** E ≥ 40 means the network is unsure of 40+ of the
  174 bits, so BP + OSD-2 cannot find the codeword even with a candidate.
- **Yet the information is there for a known-message test:** the reference's own message scores ρ ≥ 0.4 at its
  location for 89% of all misses, median ρ 0.79 among the look-back ones. That is why JTDX's `^` hypothesis stage
  and our repeat stage reach signals plain decoding doesn't.
- **ρ ≥ 0.4 is not a decision rule here:** plan 12 measured wrong messages on the right audio up to 0.70. A test
  needs the runner-up margin and a wrong-receiver null.
- **The search loses 396 misses (16%) that decode when forced** (`scratch/nocand_kind.py` → `nocand_kind.txt`):
  - **Usually not because nothing scores there.** There is a candidate within 5 Hz but at another DT: 257 (65%)
    in the original audio that pass 0 sees, and 92% in the residual. In 80, pass 0 did propose the right cell and
    the decode failed then.
  - **The true lag ranks high among the lag peaks** of the search's own score (median of the other seven tones) at
    its frequency ±1 bin. In the residual it is top 1 in 95, top 2 in 163, **top 4 in 282 (71%)**, top 8 in 351.
    In the original audio: 96 / 152 / 216 / 308.
  - **Why it is dropped:** `find_candidates` keeps three lags per frequency (best; second best ≥ 0.4 s away;
    best in the typical DT window). It then peak-picks along frequency, comparing each bin's own best lag with its
    neighbours' best lags, so a stronger lag at the next bin suppresses the true one.
  - **Worked example,** run 2 `06_160m` 223015, `CT1CHT OZ5AGJ JO47` at +3 dB. All three references decode it,
    and we decode it at E = 0.0 when forced. At 750.0 Hz the true lag (DT −0.08) scores 110, while lags 5, 9 and
    13 symbols later score 170, 141 and 131. There the strong signal's data tones happen to sit on the Costas
    tones while the other tones hold only the quiet 160 m noise floor. The tracks take +0.72 / +1.36 and the
    early track −2.0, and fine sync (±0.06 s) cannot walk back to −0.08.
  - Other strong examples: `ND1X YO8BJK KN37` +3 dB, `ZS1RU JA8GWA QN03` +2 dB, `D44PX EC7KW IM77` +2 dB.
  - **Fix direction:** keep the top ~4 lag peaks per frequency, or 2-D local maxima within a small lag window,
    at least in passes 2–4 where candidate reuse keeps the count down. The cost is candidates (CPU) and the
    sidelobe budget problem the code comment records.
- **106 have a candidate and decode in the final residual** but were not decoded in the real run: order, budget
  or candidate-reuse losses (plan 7.2b found the same thing).

## Co-channel geometry (for the post-subtraction forced retry)

`scratch/cochan.py` → `cochan.txt`. Selection: a miss with a decode of ours of a *different* sender within
10 Hz and 0.4 s in the same slot (369 flagged, 350 when the classes are made exclusive). For 340 of the 369
(92%) a reference decoded **both** signals, so these are genuine second signals, not two readings of one.

- **Δf** (miss − our neighbour) is spread evenly over the window, so it is not concentrated in the shared
  3.125 Hz cell. |Δf| p10 0.4, p25 1.7, **median 3.9**, p75 7.0, p90 8.6 Hz. Counts by |Δf| (Hz):
  [0,1) 62, [1,2) 45, [2,3.1) 43, [3.1,5) 80, [5,7.5) 62, [7.5,10] 77.
- **ΔDT** is mostly small: |ΔDT| p10 0.02, p25 0.04, **median 0.09**, p75 0.19, p90 0.28 s. One FT8 symbol is
  0.16 s.
- **SNR gap** (miss − neighbour, both from the same reference, n = 348): p10 −19, p25 −14, **median −7.5**,
  p75 −3, p90 0 dB. By gap (dB): < −20: 29, −20 to −10: 90, −10 to −5: 90, −5 to 0: 77, ≥ 0: 62. So 18% of the
  missed signals are as strong as or stronger than the neighbour we did decode. The missed signal itself has
  median −18 dB (p90 −12).
- **Neighbour's stage:** plain BP 266, `rp` 51, OSD 25, `bp_cx` 17, other 10. Per run: 134 / 58 / 177.
- **Grid coverage.** A forced `fine_sync2` start reaches ±2.5 Hz / ±0.06 s on its own:

  | starts around the neighbour's (f0, DT) | reach | co-channel misses covered |
  |---|---|---|
  | 1 (the neighbour's own) | ±2.5 Hz, ±0.06 s | 15% |
  | 3: f ± 5 Hz | ±7.5 Hz, ±0.06 s | 31% |
  | 9: f {−5, 0, +5} × DT {−0.12, 0, +0.12} | ±7.5 Hz, ±0.18 s | 58% |
  | 15: f {−5, 0, +5} × DT {0, ±0.12, ±0.24} | ±7.5 Hz, ±0.30 s | 73% |
  | 15: f {0, ±4, ±8} × DT {0, ±0.12} | ±10.5 Hz, ±0.18 s | 74% |
  | 25: f {0, ±4, ±8} × DT {0, ±0.12, ±0.24} | ±10.5 Hz, ±0.30 s | 92% |

  Coverage is relative to the 10 Hz / 0.4 s selection window. It also assumes `fine_sync2`'s top two peaks land on
  the weak signal rather than on what remains of the subtracted one.
- Examples where a reference decoded both: run 4 `12_17m` 151430 1716 Hz, miss `KI8L YO4RST KN35` (+1 dB) under
  ours `KI8L R5AV -18` (+7 dB), Δf +0.2 Hz, ΔDT −0.03 s (two stations answering the same caller on one
  frequency); run 2 `00_40m` 224215 675 Hz, miss `ON1AEY KA8CLN EN82` (−8) under `DL3MIB VE3FSN -22` (−7),
  Δf −0.7, ΔDT +0.01; run 2 `00_40m` 224845 556 Hz, miss `EA4IUJ HA7TM R-14` (−8) under `EA7GNO W9AH EN64`
  (+18), Δf +2.6, ΔDT −0.10.

## Repeats the repeat stage still misses

Flagged: 614 misses whose exact text we decoded on this receiver 30 or 60 s earlier within 10 Hz (494 in the
exclusive class; 88 are co-channel and 32 sit in earlier classes). The earlier copy is usually 30 s before (565)
at nearly the same spot: |Δf| median 0.4 Hz, p90 1.2 Hz; |ΔDT| median 0.03 s. Its stage was BP 301, `bp_cx` 68,
`rp` 65, OSD 64, `osd_cx` 53, `osd_ap` 32, other 31. These are weak (median −21 dB, p10 −24, p90 −12) and mostly
CQs (426), then grid messages (138) and reports (44). JTDX gets 81% of them with `^`.

Emulating `decoder._repeat_stage` exactly for each missed sender (`replay.tsv`, column `rp_state`), with our
decodes of 30/60 s earlier as context:

| outcome | flagged repeats (614) | what it means |
|---|---|---|
| **margin** | **320** | the right message scores best (320 of 323) with ρ ≥ 0.40, median ρ 0.64, but within 0.15 of the runner-up (median margin 0.10) |
| successor | 199 | a competitor (a "next message" the stage only scores, never accepts) beats the earlier message |
| low ρ | 37 | best < 0.40 (median 0.35) |
| have | 21 | the stage sees that sender's call already decoded in the slot |
| no context | 19 | no standard (i3 = 1/2) message of that sender in the context (hashed/compound calls) |
| accept | 18 | the emulation accepts; the real chain lost it elsewhere (context differences, duplicate drop) |

**What beats them on the margin** (rerun with the runner-up logged, `replay_repeat_sane.tsv` →
`runnerup_summary.txt`):
- **Most runner-ups are messages no station sends.** Of the 320, the runner-up is a CQ carrying a report,
  RRR, RR73 or 73 in place of its grid (`CQ PB1PCH +27` 0.44 against `CQ PB1PCH JO22` 0.55) in **198**, and a
  CQ with the R flag (`CQ K3YQA R EN90`) in **28**.
- Only 94 lose to a sensible competitor: same calls with another last word 73, R flag toggled 21
  (`YU6DX WP3UX R FK68` 0.818 against 0.832).
- **Why the nonsense ones exist:** `_successors` builds the "next message" competitors (both R flags × grid /
  RRR / RR73 / 73 / reports) for every earlier message, CQs included. For a CQ these are one 15-bit field away
  from the truth, so on a weak signal they score close behind it.
- **Without them** (ρ against the best sensible competitor), **224 of the 320 pass** at ρ ≥ 0.40 / margin
  ≥ 0.15, and 220 at margin 0.20. Per run that is 36 / 47 / 141.
- This is a structural change to the competitor set, not a threshold. The nonsense competitors act as an implicit
  sharpness test, so the run-1 wrong-receiver shuffle (`tools/rp_eval.py --shuffle`, plan 12's 0 of 223,643)
  must be redone with it before anything is trusted.

Over **all** 2,495 misses the stage has something to test for 1,118. In 749 of those the reference's message is
among the ~130 messages it scores, and in 463 it is the top score. The 735 "successor" outcomes are mostly
senders known from context who sent something new: the 308 pair/sender-in-context misses give 302 of them, and
for most the right message is not among the competitors (a new partner) or is not the top score.
**Accepting a successor** when it is the top score with ρ ≥ 0.40 and margin ≥ 0.15 would take at most **66**
(run 2: 23, run 3: 11, run 4: 32), e.g. `D44PX F5PHW R-15` ρ 0.90 vs 0.42 and `CQ CT1FIU IN50` 0.88 vs 0.23.

## New: a station's first overs (look-back)

**898 of the 2,495 misses (36%) are the exact text we decode ourselves 30 or 60 s *later* within 10 Hz.** 583 of
them are not already earlier-repeats. Per run: 246 / 231 / 421. By class: heard 295, repeat 264, co-channel
135, sender in context 88, same text other time 65, other 51. The pattern behind the "heard" class says the same
(`heard_gap.txt`): in 614 of 787 our nearest decode of the sender comes *after* the miss, median 60 s later,
at the same frequency (median |Δf| 1 Hz). These are a station's first transmissions, made before we have any
context for it. JTDX and jt9 get some of them by plain decoding, and JTDX by `^`.

**Emulated exactly** (`tools/miss_replay.py --lookback --rp-only`, `replay_lookback.tsv`, `lookback_summary.txt`).
The setup:
- the unchanged repeat stage, at the shipped thresholds (ρ ≥ 0.40, margin ≥ 0.15);
- context = our decodes of the same receiver and period 30 and 60 s **later**;
- tested on the slot's residual after all its decodes, the repeat stage's included, are subtracted.

Results:
- **It accepts 564 of the 2,495 slot-confirmed misses** (run 2: 192, run 3: 123, run 4: 249).
- By class: heard 255, co-channel 106, sender in context 74, repeat 56, same text other time 55, pair 10, other 8.
- 491 of the 564 are not earlier-repeats.
- Who has them: JTDX + deep 163, all three 152, deep only 116, jt9 only 58, other 75. JTDX stage `-` 248, `^` 130,
  `*` 128. Median SNR −18 dB.
- Among the 898 later-repeats: 538 accepted, 190 lost on the margin (the right message best, median ρ 0.64,
  margin 0.10, the same picture as forward), 118 to a competitor, 19 on low ρ.

**26 of the 564 accepts carry a different message than the reference printed** (`lookback_wrong.txt`):
- 5 are jt9's unresolved hash `<...>` against our resolved one: the same message.
- 5 are a second stream or a moved station: the later copy is 160–550 Hz away, and the reference's message also
  scores high at its own spot. An example is DL0GSH sending a CQ at 1924 Hz and a report elsewhere at once.
- 16 sit on the same spot as the later copy and are all JTDX `^` texts (tone matching, no CRC). The audio
  favours our message there: e.g. `JH4CPC BD4WRP OM92` ρ 0.87 against JTDX's `CQ BD4WRP OM92` 0.37, which is
  roughly what the shared call and grid alone give.
- In one of the 16, jt9's CRC-checked decode of that slot is our message (`R6OF AD8IF EN72`, where JTDX printed
  `CQ AD8IF EN72`).

So JTDX's `^` stage emits stale texts sometimes, and the look-back accepts look right. `verify.py` would score
both as slot-confirmed, because only the sender is checked.

**Limits:**
- **Not usable live.** A look-back decode exists 30–60 s after its slot. WSJT-X's sidecar protocol answers
  per request and cannot place a decode into an earlier period. It is worth having for offline decoding, logging
  and spotting, and it should be scored separately from what an operator sees, or the comparison stops being
  like-for-like with the live decoders.
- **The false-decode bound must be redone backwards.** That means the wrong-receiver shuffle on run 1, plus
  `noise10k` (no context there, so it bounds nothing for this stage).
- **The 564 count only misses we were scored against.** It does not show what else the stage would add (active-
  tier decodes) or how many unconfirmed ones.

## What changed since the `nn_v3` analysis

These are the same definitions run against `nn_v3`: `tools/miss_profile.py --ours nn_v3`, then
`xtab_nn_v3.txt`. Union of the three references' slot-confirmed misses: **4,408 → 2,495 (−43%)**.

| class | `nn_v3` | `nn_v3_dtcrp` | change |
|---|---|---|---|
| exact repeat 30/60 s earlier (10 Hz) | 1,747 | 494 | −1,253: the repeat stage |
| heard at other times | 932 | 787 | −145 |
| co-channel (different sender of ours within 10 Hz / 0.4 s) | 549 | 350 | −199: the repeat stage tests at the earlier location without sync, so the shared-sync problem does not stop it |
| same text at any frequency, 30–300 s earlier | 382 | 203 | −179 |
| sender in context | 224 | 201 | −23 |
| never heard | 164 | 142 | −22 |
| DT outside the window | 135 | 61 | −74: DT re-centring (run 3: 87 → 26) |
| pair in context | 120 | 107 | −13 |
| scorer (`<CALL>`) | 120 | 120 | unchanged (needs a `verify.py` fix, not a decoder change) |
| band edge | 35 | 30 | −5 |

JTDX-only slot-confirmed: 3,069 → 1,719 (repeat stage) → 1,595 (DT re-centring). Deep JTDX-only: 3,780 → 2,269 →
2,063. jt9-only: 1,373 → 906 → 786.

## Caveats

- **Measurement only, on the honest runs.** Nothing here was tuned. Any threshold a fix needs (a forced-retry
  gate, a successor-acceptance rule) must be chosen on run 1, the way the repeat stage's were, and checked on
  `noise10k` and with the wrong-receiver shuffle. Runs 2–4 then judge it.
- **Slot-confirmed means the sender was on the air in that slot,** spotted by someone within 50 Hz. It does not
  check the reference's text. JTDX's `^` decodes have no CRC behind them; plan 11 found PSK Reporter
  contradicting none of 50,350 checkable ones.
- **The classes are correlations, and exclusive by priority.** The order is scorer > DT window > band edge >
  co-channel > repeat > pair > sender > any-age repeat > heard > never heard. A co-channel miss that is also an
  exact repeat (88 such) counts as co-channel. Flags overlap a lot, as `xtab.txt` shows: repeat & pair 603,
  co-channel & sender-in-context 172.
- **The replay approximates the decoder.**
  - It subtracts our saved decodes at their printed frequency (0.1 Hz) and DT (0.01 s), all at once, not pass by
    pass.
  - The candidate search runs without DT re-centring's window shift, so run 3 `03_20m` and run 4 `08_12m` can
    show NOCAND that the real decoder would not.
  - The candidate and forced tests run plain BP + OSD-2 only, with no a-priori, erasure, `osd_bp` or context
    stage, so they bound what the metric gives without help.
  - The repeat-stage emulation is exact per sender (the stage scores each sender on its own), given our final
    decodes of 30/60 s earlier as context, which is what the chain carries.
- **Packing.** Reference texts were packed with `ft8code`. Where a decoder printed a resolved hashed call without
  brackets, the variant with that call in `<>` was used when it packs back to the same text. 19 of 2,495 still
  pack to a different text (mostly jt9's unresolved `<...>`), so their ρ values mean nothing.
- **SNR scales differ.** On the same signals jt9 reports 3.8 dB above our estimate, and JTDX and deep JTDX 1.5 dB
  above. SNRs above are the reference's own: JTDX where it has the pair, else deep JTDX, else jt9.
- **Repeat-stage decodes that overlap.** 277 of our 4,183 `rp` decodes on runs 2–4 (6.6%) sit within 3 Hz /
  0.2 s of another decode of ours (`scratch/rp_overlap.py`), 7 of them on another `rp` decode. Their tiers look
  like all `rp` decodes (72% slot-confirmed against 78%, 0 unconfirmed), so this is mostly co-channel pairs the
  stage recovers because it needs no sync. 12 `rp` decodes sit where a reference decodes a *different* sender and
  no reference hears the `rp` sender (`rp_neighbours.txt`). One example is run 4 `04_40m` 151415: `YC8BUL N7ECV
  DN07` and `CQ NL8F AO63` (both `rp`, 1.4 Hz apart) on jt9's `CQ YG2AUI OI42`, which is slot-confirmed. The
  plan-12 stale check only covered `rp` senders a reference also decoded; these 12 are the place to extend it.

## Reproduce

Needs `data/val/run{2,3,4}/<station>/{decodes,verified}_{nn_v3_dtcrp,jtdx,jtdx_deep}` (and `nn_v3`, `nn_v3_rpf`
for the comparison), the recordings' `verified_jt9.tsv`, WSJT-X's `ft8code` on PATH (to pack reference texts) and
`models/v3/model.pt`. Run from the main checkout, or set `V=<main checkout>/datasets/ft8-validation-set` and link
`data/` there. `scratch/nocand_kind.py` reads `datasets/` directly.

```sh
python3 tools/compare_decoders.py --runs 2 3 4 --tags jt9 jtdx jtdx_deep nn_v3 nn_v3_rpf nn_v3_dtcrp
python3 tools/diag_ref_only.py --ours nn_v3_dtcrp --tsv data/logs/miss_dtcrp/jtdx_only_slot.tsv
uv run python tools/miss_profile.py --tsv data/logs/miss_dtcrp/misses.tsv            # 13 s, flags from saved decodes
uv run python tools/miss_profile.py --ours nn_v3 --tsv data/logs/miss_dtcrp/misses_nn_v3.tsv
uv run python tools/miss_replay.py data/logs/miss_dtcrp/misses.tsv --out-tsv data/logs/miss_dtcrp/replay.tsv --workers 4
uv run python scratch/xtab.py data/logs/miss_dtcrp/misses.tsv                         # classes x runs/references
uv run python scratch/cochan.py data/logs/miss_dtcrp/misses.tsv                       # co-channel geometry
uv run python scratch/replay_sum.py data/logs/miss_dtcrp/replay.tsv                   # replay tables
```

Wall times (4 workers on the M4 mini, while another session's runs kept the load average at 8–29): the full replay
was 1,714 slots in 14 min; the look-back emulation (`--lookback --rp-only`) 1,098 slots in 22 min; the two repeat
reruns (`--only repeat --rp-only`) 531 slots in 5 min each; `nocand_kind.py` 355 slots in about 1 min. Further
commands:

```sh
uv run python tools/miss_replay.py data/logs/miss_dtcrp/misses.tsv --out-tsv data/logs/miss_dtcrp/replay_lookback.tsv --lookback --rp-only
uv run python tools/miss_replay.py data/logs/miss_dtcrp/misses.tsv --out-tsv data/logs/miss_dtcrp/replay_repeat_sane.tsv --only repeat --rp-only
uv run python scratch/lookback_sum.py data/logs/miss_dtcrp/replay_lookback.tsv data/logs/miss_dtcrp/misses.tsv
uv run python scratch/lookback_wrong.py data/logs/miss_dtcrp/replay_lookback.tsv data/logs/miss_dtcrp/replay.tsv
uv run python scratch/runnerup_sum.py data/logs/miss_dtcrp/replay_repeat_sane.tsv
uv run python scratch/nocand_kind.py data/logs/miss_dtcrp/replay.tsv
uv run python scratch/levers.py                                                       # overlaps for the ranking
```

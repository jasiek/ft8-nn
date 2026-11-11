# `nn_v3` vs JTDX and FT8TW on the validation set

Measured 2026-11-10 on validation runs 2, 3 and 4 — the 36 KiwiSDR receivers never used for
training, 4,302 slots of 15 s — and scored with `collect/verify.py` against the PSK Reporter spots
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
| **`nn_v3` (ours, shipped)** | **100,630** | **+16.7%** | **73,876** | **+15.6%** | **3** |

**Ours decodes 7.8% more than JTDX (6.8% more slot-confirmed), with 3 unconfirmed decodes against
JTDX's 11.** JTDX in turn beats stock `jt9` by 8.3%, so roughly half of our margin over WSJT-X is
margin JTDX also finds, and half is ours alone. Against JTDX at its most sensitive settings the
margin is +3.0% / +2.4%, and JTDX then makes 44 unconfirmed decodes to our 3 (section below).

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

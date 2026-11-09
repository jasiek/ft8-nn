# `ft8nn_sidecar` — this decoder inside WSJT-X

`ft8nn_sidecar` makes WSJT-X decode FT8 with this project's decoder (`libft8nn`, `models/v3`)
instead of its own `jt9`. It is WSJT-X's *external FT8 decoder*: a long-running process that WSJT-X
starts at startup and hands each FT8 decode pass to. WSJT-X then turns the answers into `jt9` output
lines and runs them through its usual path, so nothing downstream notices — false-decode filtering,
`ALL.TXT`, the Band Activity and Rx Frequency windows, UDP, PSK Reporter and auto-sequencing all
work as before.

The decoder lives here, in this project; WSJT-X is only told how to start it. It needs the WSJT-X
with the sidecar interface (`WSJTX_FT8_DECODER`, `doc/ft8-sidecar-protocol.md` in that tree).

## The interface

One compact JSON object per line, plus the audio as raw bytes:

| | |
|---|---|
| WSJT-X → us, stdin | `{"type":"hello","v":1,…}` once at startup; `{"type":"decode",…}` per pass, **immediately followed by `samples` × 2 bytes** of mono signed-16-bit little-endian 12 kHz audio (sample 0 = start of the T/R period); `{"type":"quit","v":1}` at shutdown |
| us → WSJT-X, stdout | one `{"seq":N,"snr":…,"dt":…,"freq":…,"msg":"…"}` per decode, then exactly one `{"seq":N,"done":true}`, flushed line by line |
| stderr | free-form text, copied into the WSJT-X log |

A request carries `utc` (`hhmmss`), `pass` (`nzhsym`, how much audio was captured), `final` (the
last pass of this period), `disk`, `period_start`, `dial_freq`, `special_op` and **every** field of
`dec_data.params` under its C name. WSJT-X won't send another request until `done` arrives, and
gives up on one that takes longer than `max(5 s, T/R period)` — 15 s for FT8.

Only FT8 comes here. Every other mode, and Fox and SuperFox FT8, stay with WSJT-X's own `jt9`, and
FT8 goes back to `jt9` if this process exits or crashes — so there is nothing for us to forward and
no other mode to implement. `doc/ft8-sidecar-protocol.md` in the WSJT-X tree is the full spec and
`Decoder/Ft8Sidecar.cpp` its implementation.

The decode lines WSJT-X builds from our replies are `jt9`'s FT8 output lines
(`lib/decoder.f90` format 1001):

```
(i6.6, i4, f5.1, i5, ' ~ ', 1x, a37, 1x, a2)
030200 -12 -0.3  213 ~  M9DNC K4WH EM95                         
     hhmmss  snr   dt  freq         message (37 cols)        a<n>
```

The column positions matter: WSJT-X reads `snr`, `dt` and `freq` back out by column
(`Decoder/decodedtext.{h,cpp}`), so our numbers have to stay inside those field widths, and the
tests check every line against them.

## Building

```bash
tools/get_onnxruntime.sh      # once: the official ONNX Runtime build, into third_party/
make -C c                     # libft8nn
make -C wsjtx                 # build/ft8nn_sidecar and build/fake_wsjtx
```

No other dependencies: the protocol is JSON lines and raw audio on a pipe, so there is no Qt and no
shared memory here (`json.h` is a 200-line reader for the handful of fields we use).

## Using it with WSJT-X

```bash
WSJTX_FT8_DECODER="$HOME/github/ft8-nn/wsjtx/build/ft8nn_sidecar --model $HOME/github/ft8-nn/models/v3/ft8nn.onnx" \
  /Applications/wsjtx.app/Contents/MacOS/wsjtx
```

The command line is split into words with quoting honoured, but no shell runs it: put variables or
pipes in a wrapper script. On macOS, an app started from Finder or the Dock sees no shell
environment, so either start WSJT-X from a terminal as above or

```bash
launchctl setenv WSJTX_FT8_DECODER "$HOME/github/ft8-nn/wsjtx/build/ft8nn_sidecar --model $HOME/github/ft8-nn/models/v3/ft8nn.onnx"
```

and restart the app. Without `--model`, `$FT8NN_MODEL` is used, then `ft8nn.onnx` next to the
binary, in `$WSJTX_DATA_DIR` or `$WSJTX_SHARE_DIR` (WSJT-X exports both), or in `../models/v3` of a
source tree. If no model can be opened the process exits at once: WSJT-X logs that, says so in the
status bar and falls back to its own `jt9`, which is more useful than a session that decodes nothing.

To see what it is doing, watch the WSJT-X log: `[ft8nn] model …/ft8nn.onnx, 1 ONNX thread(s)` and
`[ft8nn] hello from WSJT-X 3.0.0` appear at startup, and `-v` adds a line per pass
(`seq 7: utc 123015 pass 50 (172800 samples) -> 37 decode(s) in 2.8 s`).

## What it does and does not do

- **Early passes answer empty.** WSJT-X asks for a decode at about 11.8 s and 13.2 s into the period
  as well as at the end, and a pass that arrives while we are still busy is skipped — so a slow
  early pass costs the *full* pass. This decoder needs seconds per slot, so a request with
  `final: false` is answered at once with no decodes; the full pass finds those signals anyway.
  `--early` decodes them too (use it with `--threads` on a fast machine).
- **Decode speed matters.** WSJT-X gives up on a request after 15 s for FT8, and the next period's
  request arrives at the same time. The main `README.md` has the measurements: 2.8 s per slot on
  average on the M4 mini, 6.9–8.0 s on the busiest receiver tested (40 m, about 48 decodes per
  slot). A plain M1 under light other load takes 9–12 s on those same busy slots, which leaves
  little headroom, so prefer the faster machine for a live station, keep early passes off, and watch
  the `-v` timings — a pass that overran is reported on stderr, and so in the WSJT-X log.
- **Re-decodes are honoured.** When the user clicks Decode, WSJT-X sends `samples: 0`, meaning "the
  audio you were sent last, again"; we keep the last period's samples for that.
- **`[nfa, nfb]` is honoured** — decodes outside the frequency range of WSJT-X's waterfall settings
  are dropped (`--no-range` keeps them). Decodes are reported in frequency order.
- **A-priori annotations:** `"ap":1` marks our CQ a-priori decodes and `"ap":2` the context ones
  (the calls of a message the same station sent 30 or 60 s earlier — `jt9` has no equivalent);
  WSJT-X shows them as `a1` and `a2`. We never set `lowconf`.
- **Ignored `params` fields:** `ndepth`, `nfqso`/`ntol`, `lft8apon`/`lapcqonly`/`napwid`,
  `nQSOProgress`, `mycall`/`hiscall`/`hisgrid`, the `ft8md` options — this decoder has its own
  settings and its own a-priori decoding, and does not take WSJT-X's.

## Other ways to run it

```bash
wsjtx/build/ft8nn_sidecar --model models/v3/ft8nn.onnx slot1.wav slot2.wav   # like 'jt9 -8 file.wav'
```

With `.wav` files on the command line it decodes those instead of talking to WSJT-X, printing the
same `jt9` lines WSJT-X would build. Slots named `yymmdd_HHMMSS.wav` also get context a-priori
decoding, as in `c/tools/ft8nn_cli`. That is how the tests compare the protocol path against a plain
decode of the same audio.

## Tests

```bash
make -C wsjtx test                       # needs datasets/ (docs/data-collection.md)
```

`fake_wsjtx` is WSJT-X's half of the protocol: it starts the decoder, sends `hello`, feeds slots
with their audio, reads the replies, formats each one into the `jt9` line WSJT-X would build from it
(`Ft8Sidecar::format_jt9_line`), and sends `quit`. `test_sidecar.sh` then checks that

1. the protocol path gives exactly the decodes a plain decode of the same audio gives,
2. every line WSJT-X builds from our replies has `jt9`'s FT8 column layout,
3. `[nfa, nfb]` keeps exactly the decodes inside the range,
4. an early pass (`final: false`) decodes nothing by default, and does decode with `--early`,
5. a re-decode (`samples: 0`) decodes the same audio again, line for line,
6. a line that is not JSON is logged and skipped instead of breaking the stream,
7. `quit` makes the decoder exit within 1 s (`fake_wsjtx` kills it, and fails, if it does not),
8. a missing model is a clean non-zero exit, so WSJT-X falls back to its own `jt9`.

Every run also checks that each reply echoes its request's `seq`: `fake_wsjtx` fails on a reply
whose `seq` is missing or belongs to another request, which is what WSJT-X would drop.

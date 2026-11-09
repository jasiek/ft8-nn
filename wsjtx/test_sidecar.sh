#!/bin/sh
# Smoke test for ft8nn_sidecar: drive it through WSJT-X's FT8 sidecar protocol with fake_wsjtx (the
# WSJT-X half: hello, a decode line per pass followed by the audio, replies read until "done", then
# quit) and check that
#
#   1. the sidecar path gives exactly the decodes a plain decode of the same audio gives,
#   2. every line WSJT-X builds from our replies has jt9's FT8 column layout,
#   3. [nfa, nfb] keeps exactly the decodes inside WSJT-X's frequency range,
#   4. an early pass (final=false) decodes nothing by default, and does decode with --early,
#   5. a re-decode (samples=0, the user clicking Decode) decodes the same audio again,
#   6. a line that is not JSON is ignored instead of breaking the stream,
#   7. "quit" makes the decoder exit within 1 s (fake_wsjtx kills it, and fails, if it does not),
#   8. a missing model is a clean exit, so WSJT-X logs it and falls back to its own jt9.
#
# Every run also checks that each reply echoes the request's seq: fake_wsjtx fails on a reply whose
# seq is missing or belongs to another request, as WSJT-X would drop it.
#
# Needs 15 s slot WAVs: the training recordings (datasets/, see docs/data-collection.md) by default,
# or any other directory of 12 kHz mono WAVs in $TEST_SHIM_SLOTS (CI uses the synthetic slots
# tools/ci_smoke.py writes).
# Usage: ./test_sidecar.sh [NSLOTS]
set -eu
cd "$(dirname "$0")"
slots=${1:-4}
model=${FT8NN_MODEL:-../models/v3/ft8nn.onnx}
station=${TEST_SHIM_SLOTS:-$(ls -d ../datasets/ft8-training-data/recordings/*/slots 2>/dev/null | head -1 || true)}
[ -n "$station" ] || { echo "no slots: unpack ../datasets (docs/data-collection.md) or set TEST_SHIM_SLOTS"; exit 2; }
[ -f "$model" ] || { echo "no model at $model (set FT8NN_MODEL)"; exit 2; }
wavs=$(ls "$station"/*.wav 2>/dev/null | head -"$slots")
[ -n "$wavs" ] || { echo "no .wav files in $station"; exit 2; }
nwav=$(echo "$wavs" | wc -l | tr -d ' ')
echo "test_sidecar: $nwav slots from ${station%/slots}, model $model"

sidecar=./build/ft8nn_sidecar
fake=./build/fake_wsjtx
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

first=$(echo "$wavs" | head -1)

# 1. a plain decode of the files, and the same audio through the protocol. --no-range turns off the
#    [nfa, nfb] filter, which check 3 exercises on its own.
# shellcheck disable=SC2086
"$sidecar" --model "$model" $wavs > "$tmp/file.txt"
# shellcheck disable=SC2086
"$fake" --decoder "$sidecar --model $model --no-range" $wavs > "$tmp/side.txt"
if ! diff -u "$tmp/file.txt" "$tmp/side.txt" > "$tmp/diff.txt"; then
  echo "FAIL: the sidecar path and a plain decode of the same audio differ"; cat "$tmp/diff.txt"; exit 1
fi
ndec=$(grep -vc "<DecodeFinished>" "$tmp/side.txt" || true)
nfin=$(grep -c "<DecodeFinished>" "$tmp/side.txt" || true)
echo "ok: $ndec decodes, $nfin <DecodeFinished> lines, identical to a plain decode"
[ "$nfin" -eq "$nwav" ] || { echo "FAIL: one <DecodeFinished> per request expected"; exit 1; }
[ "$ndec" -gt 0 ] || { echo "FAIL: no decodes at all"; exit 1; }

# 2. jt9's FT8 line layout, field by field (1-based columns):
#      1-6 hhmmss | 7-10 snr | 11-15 dt | 16-20 freq | 21-23 ' ~ ' | 24 space
#      25-61 message (37 chars, '?' in 61 = low confidence) | 62 space | 63-64 'a<n>' or blank
#    (lib/decoder.f90 format 1001 '(i6.6,i4,f5.1,i5," ~ ",1x,a37,1x,a2)', which WSJT-X rebuilds from
#    our JSON in Ft8Sidecar::format_jt9_line; Decoder/decodedtext.{h,cpp} then reads snr, dt and
#    freq back out by column, so these positions are the interface.)
grep -v "<DecodeFinished>" "$tmp/side.txt" | awk '
  function bad(why) { printf "  %s: %s\n", why, $0; n++ }
  { if (length($0) != 64) bad("length " length($0) " not 64")
    else if (substr($0,1,6) !~ /^[0-9][0-9][0-9][0-9][0-9][0-9]$/) bad("hhmmss")
    else if (substr($0,7,4) !~ /^ *-?[0-9]+$/) bad("snr")
    else if (substr($0,11,5) !~ /^ *-?[0-9]+\.[0-9]$/) bad("dt")
    else if (substr($0,16,5) !~ /^ *[0-9]+$/) bad("freq")
    else if (substr($0,21,4) != " ~  ") bad("mode marker")
    else if (substr($0,25,1) == " ") bad("message not left-justified in 25..61")
    else if (substr($0,62,3) !~ /^ (  |a[1-9])$/) bad("ap annotation")
  }
  END { exit n > 0 }' > "$tmp/badlines.txt" || {
  echo "FAIL: line(s) without jt9's FT8 layout:"; head -5 "$tmp/badlines.txt"; exit 1; }
echo "ok: all $ndec lines have jt9's FT8 column layout"

# 3. with the filter on (fake_wsjtx asks for 200..4000 Hz), exactly the decodes inside the range
#    survive; the frequency is columns 16..20 of the line.
# shellcheck disable=SC2086
"$fake" --decoder "$sidecar --model $model" $wavs > "$tmp/range.txt"
awk '/<DecodeFinished>/ {print; next} {f=substr($0,16,5)+0; if (f>=200 && f<=4000) print}' \
    "$tmp/file.txt" | sed 's/<DecodeFinished>.*//' > "$tmp/want.txt"
sed 's/<DecodeFinished>.*//' "$tmp/range.txt" > "$tmp/got.txt"
if ! diff -u "$tmp/want.txt" "$tmp/got.txt" > "$tmp/diff2.txt"; then
  echo "FAIL: the [nfa, nfb] filter did not keep exactly the decodes in 200..4000 Hz"
  head -20 "$tmp/diff2.txt"; exit 1
fi
echo "ok: [nfa, nfb] honoured ($(grep -vc '^$' "$tmp/got.txt" || true) of $ndec decodes in 200..4000 Hz)"

# 4. WSJT-X's early passes (final=false, at about 11.8 s and 13.2 s into the period) are answered at
#    once with no decodes -- see README: a slow early pass makes WSJT-X skip the full one -- and
#    --early makes them decode. --live also exercises dating the slot from period_start rather than
#    params.yymmdd.
"$fake" --decoder "$sidecar --model $model" --live --early --pass 41 "$first" > "$tmp/early.txt"
[ "$(grep -vc '<DecodeFinished>' "$tmp/early.txt" || true)" -eq 0 ] \
  || { echo "FAIL: an early pass should decode nothing by default"; exit 1; }
"$fake" --decoder "$sidecar --model $model --early" --live --early --pass 41 "$first" > "$tmp/early2.txt"
nearly=$(grep -vc '<DecodeFinished>' "$tmp/early2.txt" || true)
[ "$nearly" -gt 0 ] || { echo "FAIL: --early should decode the early pass"; exit 1; }
echo "ok: early pass skipped by default ($nearly decodes with --early)"

# 5. a re-decode: WSJT-X sends newdat=false and samples=0 when the user clicks Decode, meaning "the
#    audio you were sent last, again", so the second answer must repeat the first.
"$fake" --decoder "$sidecar --model $model" --redecode "$first" > "$tmp/redecode.txt"
nfin2=$(grep -c "<DecodeFinished>" "$tmp/redecode.txt" || true)
[ "$nfin2" -eq 2 ] || { echo "FAIL: expected two answers, got $nfin2"; exit 1; }
awk 'BEGIN {n=0} /<DecodeFinished>/ {n++; next} {print > ("'"$tmp"'/pass" n ".txt")}' "$tmp/redecode.txt"
if ! diff -u "$tmp/pass0.txt" "$tmp/pass1.txt" > "$tmp/diff3.txt"; then
  echo "FAIL: the re-decode did not reproduce the first pass"; head -20 "$tmp/diff3.txt"; exit 1
fi
echo "ok: a re-decode (samples=0) decoded the same audio again ($(wc -l < "$tmp/pass0.txt" | tr -d ' ') decodes)"

# 6. a line that is not JSON (WSJT-X never sends one, but a future field or a stray write would
#    look like this): logged and skipped, with the request after it answered normally.
"$fake" --decoder "$sidecar --model $model" --garbage "$first" > "$tmp/garbage.txt" 2> "$tmp/garbage.err"
grep -q "not JSON" "$tmp/garbage.err" || { echo "FAIL: no complaint about the junk line"; cat "$tmp/garbage.err"; exit 1; }
grep -q "<DecodeFinished>" "$tmp/garbage.txt" || { echo "FAIL: no answer after a junk line"; exit 1; }
[ "$(grep -vc '<DecodeFinished>' "$tmp/garbage.txt" || true)" -gt 0 ] \
  || { echo "FAIL: no decodes after a junk line"; exit 1; }
echo "ok: a non-JSON line was ignored and the next request answered"

# 7. every run above ended with "quit"; fake_wsjtx exits non-zero if the decoder had to be killed,
#    so reaching this line means it exits on quit and echoes seq on every reply.
echo "ok: the decoder exits on quit, and every reply carried the request's seq"

# 8. no model: exit instead of answering nothing all session, so WSJT-X says so and uses its jt9.
if "$sidecar" --model "$tmp/not-a-model.onnx" < /dev/null > "$tmp/nomodel.txt" 2> "$tmp/nomodel.err"; then
  echo "FAIL: a missing model should be a non-zero exit"; exit 1
fi
grep -q "ft8nn_open failed" "$tmp/nomodel.err" || { echo "FAIL: no message about the missing model"; cat "$tmp/nomodel.err"; exit 1; }
echo "ok: a missing model exits with a message, leaving FT8 to WSJT-X's jt9"

echo "test_sidecar: all checks passed"

#!/bin/sh
# Build kgoba/ft8_lib's decode_ft8, our stand-in for the FT8CN / FT8TW decoder
# (that one ships only as a prebuilt Android arm64 .so; FT8CN credits Karlis
# Goba's code as its implementation reference).  See docs/decoder-comparison.md.
#
#   sh tools/ft8lib_build.sh        -> third_party/ft8_lib/decode_ft8
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC=$REPO/third_party/ft8_lib
[ -d "$SRC" ] || git clone --depth 1 https://github.com/kgoba/ft8_lib "$SRC"
cd "$SRC"
git log -1 --format='ft8_lib %h %ad %s'
# demo/decode_ft8.c calls snprintf without including stdio.h, which clang 15+
# treats as an error; everything else builds with the project's own flags.
make decode_ft8 CFLAGS="-O3 -DHAVE_STPCPY -I. -Wno-implicit-function-declaration"
echo "built $SRC/decode_ft8"

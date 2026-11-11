#!/bin/sh
# Download the official ONNX Runtime C/C++ release for this host into third_party/onnxruntime-<ver>-<platform>
# and point third_party/onnxruntime at it (the default ORT_PREFIX of c/Makefile).
#
#   tools/get_onnxruntime.sh [VERSION]      (default 1.31.1, the version the golden tests were run with)
#
# Why not Homebrew: its onnxruntime 1.31.1 runs this model 2.3x slower than the official build of the same version
# (2.18 vs 0.94 ms per hypothesis in libft8nn, M4, 1 thread; decodes byte-identical).
set -eu
VER=${1:-1.31.1}
EXT=tgz
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  PLAT=osx-arm64 ;;
  Darwin-x86_64) PLAT=osx-x86_64 ;;
  Linux-x86_64)  PLAT=linux-x64 ;;
  Linux-aarch64) PLAT=linux-aarch64 ;;
  # Windows: an MSYS2 shell (UCRT64 / MINGW64 report MINGW64_NT-*, UCRT64_NT-*); the release is a zip
  MINGW*-x86_64 | UCRT64*-x86_64 | MSYS*-x86_64) PLAT=win-x64; EXT=zip ;;
  *) echo "no prebuilt ONNX Runtime for $(uname -s) $(uname -m); see https://onnxruntime.ai" >&2; exit 1 ;;
esac
cd "$(dirname "$0")/../third_party" 2>/dev/null || { mkdir -p "$(dirname "$0")/../third_party"; cd "$(dirname "$0")/../third_party"; }
NAME=onnxruntime-$PLAT-$VER
if [ ! -d "$NAME" ]; then
  curl -fL --retry 3 -o "$NAME.$EXT" "https://github.com/microsoft/onnxruntime/releases/download/v$VER/$NAME.$EXT"
  shasum -a 256 "$NAME.$EXT" 2>/dev/null || sha256sum "$NAME.$EXT"
  if [ "$EXT" = zip ]; then unzip -q "$NAME.zip"; else tar xzf "$NAME.tgz"; fi
  rm "$NAME.$EXT"
fi
# MSYS2's ln -s makes a copy unless native symlinks are allowed, and ln -n won't replace a copied directory
[ "$EXT" = zip ] && rm -rf onnxruntime
ln -sfn "$NAME" onnxruntime
echo "third_party/onnxruntime -> $NAME"

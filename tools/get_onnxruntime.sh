#!/bin/sh
# Download the official ONNX Runtime C/C++ build for this host into third_party/onnxruntime-<platform>-<ver> (include/
# + lib/) and point third_party/onnxruntime at it (the default ORT_PREFIX of c/Makefile).
#
#   tools/get_onnxruntime.sh [VERSION]      (default 1.31.1, the version the golden tests were run with)
#
# Source: Microsoft's NuGet package Microsoft.ML.OnnxRuntime (one zip with every platform's native library). The
# GitHub releases we used before went away with github.com/microsoft/onnxruntime (404 since 2026-11-18); the NuGet
# osx-arm64 library has the same code and data, section for section, as the GitHub release (only the signature
# differs). The package is checked against a pinned SHA-256: another VERSION needs ORT_SHA256=<its hash>.
#
# Why not Homebrew: its onnxruntime 1.31.1 runs this model 2.3x slower than the official build of the same version
# (2.18 vs 0.94 ms per hypothesis in libft8nn, M4, 1 thread; decodes byte-identical).
set -eu
VER=${1:-1.31.1}
case "$VER" in
  1.31.1) SHA=6871ccbc97f3d5bdb4095a1dcec33aa78d56f6ed8de5980ef5fa1d9f2e7c679a ;;
  *) SHA=${ORT_SHA256:-} ;;
esac
[ -n "$SHA" ] || { echo "no pinned SHA-256 for ONNX Runtime $VER: set ORT_SHA256 (check the package first)" >&2; exit 1; }
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  PLAT=osx-arm64 ;;
  Linux-x86_64)  PLAT=linux-x64 ;;
  Linux-aarch64) PLAT=linux-arm64 ;;
  # Windows: an MSYS2 shell (UCRT64 / MINGW64 report MINGW64_NT-*, UCRT64_NT-*)
  MINGW*-x86_64 | UCRT64*-x86_64 | MSYS*-x86_64) PLAT=win-x64 ;;
  *) echo "no prebuilt ONNX Runtime for $(uname -s) $(uname -m) in the NuGet package; see https://onnxruntime.ai" >&2; exit 1 ;;
esac
cd "$(dirname "$0")/../third_party" 2>/dev/null || { mkdir -p "$(dirname "$0")/../third_party"; cd "$(dirname "$0")/../third_party"; }
NAME=onnxruntime-$PLAT-$VER
if [ ! -d "$NAME" ]; then
  PKG=microsoft.ml.onnxruntime.$VER.nupkg
  curl -fL --retry 3 -o "$PKG" "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$VER/$PKG"
  GOT=$( (shasum -a 256 "$PKG" 2>/dev/null || sha256sum "$PKG") | cut -d' ' -f1)
  [ "$GOT" = "$SHA" ] || { echo "SHA-256 mismatch for $PKG: got $GOT, want $SHA" >&2; rm -f "$PKG"; exit 1; }
  rm -rf "$NAME.tmp" && mkdir "$NAME.tmp"
  unzip -q "$PKG" 'build/native/include/*' "runtimes/$PLAT/native/*" -d "$NAME.tmp"
  mkdir -p "$NAME.tmp/include" "$NAME.tmp/lib"
  mv "$NAME.tmp"/build/native/include/* "$NAME.tmp/include/"
  mv "$NAME.tmp/runtimes/$PLAT/native/"* "$NAME.tmp/lib/"
  rm -rf "$NAME.tmp/build" "$NAME.tmp/runtimes" "$PKG"
  # the names the library records for itself (install name / SONAME), as the release archives had them
  case $PLAT in
    osx-*)   ln -s libonnxruntime.dylib "$NAME.tmp/lib/libonnxruntime.1.dylib" ;;
    linux-*) ln -s libonnxruntime.so "$NAME.tmp/lib/libonnxruntime.so.1" ;;
  esac
  mv "$NAME.tmp" "$NAME"
fi
# A folder that is already there is reused without the package check above, so for osx-arm64 1.31.1 check the
# library itself on every run. Two builds are good, with the same code and data bytes (checked 2026-11-22):
# the NuGet one (Microsoft's Developer ID signature) and the old GitHub release's (ad-hoc signature).
if [ "$PLAT-$VER" = osx-arm64-1.31.1 ]; then
  GOT=$(shasum -a 256 "$NAME/lib/libonnxruntime.dylib" | cut -d' ' -f1)
  case $GOT in
    2685dc296147c47b0dbd8e0615547b12fc0c445d158d1f374cbd0705d9626fdc) ;;  # NuGet
    17500d6b1e37e70f43a2c57d94233a1bc29fbcb297be0a5f8c7c13184f3307ba) ;;  # GitHub release
    *) echo "$NAME/lib/libonnxruntime.dylib: unknown SHA-256 $GOT; not using it" >&2; exit 1 ;;
  esac
fi
# MSYS2's ln -s makes a copy unless native symlinks are allowed, and ln -n won't replace a copied directory
case $PLAT in win-*) rm -rf onnxruntime ;; esac
ln -sfn "$NAME" onnxruntime
echo "third_party/onnxruntime -> $NAME"

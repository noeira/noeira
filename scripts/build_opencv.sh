#!/usr/bin/env bash
# Build the OpenCV shim that `noeira/vision/opencv/` binds to.
#
#   pixi run build-opencv           # build if stale
#   pixi run build-opencv --force   # rebuild unconditionally
#
# ⚠ NOTHING IS VENDORED.  Unlike the ImGui shim, which clones Dear ImGui into
# third_party/, OpenCV is ALREADY in every pixi env as a full C++ build —
# libopencv_{core,imgproc,calib,objdetect,videoio,imgcodecs}.dylib plus headers
# under include/opencv5.  This script only compiles our own shim against it.
#
# ⚠ THE ARTIFACT IS NOT TRACKED.  Anything importing `noeira.vision.opencv`
# fails at RUNTIME (dlopen abort), not at compile time, if this has not been
# run.  `opencv_shim_available()` exists so a caller can degrade with a message.
set -euo pipefail

ROOT="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="$ROOT/noeira/vision/opencv/opencv_shim.cpp"
OUTDIR="$ROOT/noeira/vision/opencv"

case "$(uname -s)" in
Darwin) LIB="$OUTDIR/libmojo_cv.dylib" ;;
Linux)  LIB="$OUTDIR/libmojo_cv.so" ;;
*) echo "build_opencv.sh: unsupported OS $(uname -s)" >&2; exit 1 ;;
esac

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if [ "$FORCE" -eq 0 ] && [ -f "$LIB" ] && [ "$SRC" -ot "$LIB" ]; then
    echo "opencv shim up to date: $LIB"
    exit 0
fi

# ⚠ CONDA_PREFIX IS THE PIXI ENV, AND IT IS THE ONLY CORRECT ANSWER HERE.
# A system OpenCV (homebrew) would compile and link, then be loaded alongside
# the env's copy at runtime by anything that also imports cv2 — two OpenCVs in
# one process, which is exactly how a bit-equality gate stops being about our
# marshalling.  Fail loudly rather than fall back.
PREFIX="${CONDA_PREFIX:-}"
if [ -z "$PREFIX" ] || [ ! -d "$PREFIX/include/opencv5" ]; then
    echo "build_opencv.sh: no OpenCV headers in \$CONDA_PREFIX." >&2
    echo "  Run this through pixi:  pixi run build-opencv" >&2
    exit 1
fi

# Groups A-F.
# ⚠ solvePnP lives in libopencv_GEOMETRY in OpenCV 5, not libopencv_calib --
# the header moved to geometry/3d.hpp and the symbol moved with it.

LIBS="-lopencv_core -lopencv_videoio -lopencv_imgcodecs -lopencv_objdetect -lopencv_geometry -lopencv_calib -lopencv_imgproc"

CXX="${CXX:-c++}"
echo "building opencv shim from $SRC"
echo "  against $PREFIX (OpenCV $(basename "$(ls -1 "$PREFIX"/lib/libopencv_core.*.dylib "$PREFIX"/lib/libopencv_core.*.so 2>/dev/null | head -1)" | sed 's/libopencv_core\.//;s/\.dylib//;s/\.so//'))"

# ⚠ -rpath IS NOT OPTIONAL.  Without it the dylib links fine and then fails to
# find libopencv_core at the FIRST CALL, as a dlopen abort with no useful
# message — the ImGui failure mode, one level deeper.
# ⚠ SOURCE BEFORE $LIBS: see build_http.sh — Linux `--as-needed` drops a
# library named before its first reference, and the link still succeeds.
NO_UNDEF=""
[ "$(uname -s)" = Linux ] && NO_UNDEF="-Wl,--no-undefined"
"$CXX" -O2 -std=c++17 -fPIC -shared \
    -I "$PREFIX/include/opencv5" \
    -o "$LIB" "$SRC" \
    -L "$PREFIX/lib" $LIBS \
    -Wl,-rpath,"$PREFIX/lib" $NO_UNDEF

echo "  $LIB"

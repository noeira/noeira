#!/bin/bash
# Build the Dear ImGui shim that `noeira/render/imgui/` binds to.
#
#   pixi run build-imgui          # clone ImGui if missing, then build if stale
#   pixi run build-imgui --force  # rebuild unconditionally
#
# Vendoring: ImGui is cloned into third_party/ (gitignored) rather than
# committed. It is a build-time dependency of ONE optional viewer, and the
# clone is a single shallow command that this script runs for you.
#
# ⚠ THE ARTIFACT IS NOT TRACKED. Anything importing `noeira.render.imgui`
# fails at RUNTIME (dlopen abort), not at compile time, if this has not been
# run. That is the cost of an FFI dependency; `nra_imgui_available()` in
# imgui.mojo exists so a caller can degrade instead of dying.
set -euo pipefail

ROOT="${PIXI_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMGUI_DIR="$ROOT/third_party/imgui"
# ⚠ ImGuizmo IS A SECOND VENDORED LIBRARY IN THE SAME DYLIB. It is not a
# separate artifact: it draws through ImGui's draw list and shares its
# context, so linking it anywhere else would give it a SECOND ImGui context
# and a gizmo that never sees the mouse. `nra_gz_*` lives beside `nra_ig_*`
# in one shim for that reason.
GZ_DIR="$ROOT/third_party/ImGuizmo"
SRC="$ROOT/noeira/render/imgui/imgui_shim.cpp"

case "$(uname -s)" in
    Darwin) LIB="$ROOT/noeira/render/imgui/libmojo_imgui.dylib" ;;
    *)      LIB="$ROOT/noeira/render/imgui/libmojo_imgui.so" ;;
esac

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# ── vendor ImGui on first use ────────────────────────────────────────────────
if [ ! -d "$IMGUI_DIR" ]; then
    echo "[imgui] cloning Dear ImGui into third_party/imgui ..."
    mkdir -p "$ROOT/third_party"
    git clone --depth 1 --branch docking \
        https://github.com/ocornut/imgui.git "$IMGUI_DIR"
fi

if [ ! -d "$GZ_DIR" ]; then
    echo "[imgui] cloning ImGuizmo into third_party/ImGuizmo ..."
    git clone --depth 1 \
        https://github.com/CedricGuillemet/ImGuizmo.git "$GZ_DIR"
fi

if [ ! -f "$GZ_DIR/src/ImGuizmo.cpp" ]; then
    echo "[imgui] ERROR: $GZ_DIR has no src/ImGuizmo.cpp." >&2
    echo "[imgui] Delete third_party/ImGuizmo and re-run to re-clone." >&2
    exit 1
fi

if [ ! -f "$IMGUI_DIR/backends/imgui_impl_sdlgpu3.cpp" ]; then
    echo "[imgui] ERROR: $IMGUI_DIR has no SDL_GPU3 backend." >&2
    echo "[imgui] Delete third_party/imgui and re-run to re-clone." >&2
    exit 1
fi

# ── staleness: the shim, or any ImGui translation unit, newer than the lib ───
if [ "$FORCE" = "0" ] && [ -f "$LIB" ]; then
    NEWER=$(find "$SRC" "$IMGUI_DIR"/*.cpp "$IMGUI_DIR"/backends/imgui_impl_sdl3.cpp \
                 "$IMGUI_DIR"/backends/imgui_impl_sdlgpu3.cpp \
                 "$GZ_DIR"/src/ImGuizmo.cpp "$GZ_DIR"/src/ImGuizmo.h \
                 -newer "$LIB" 2>/dev/null | head -1)
    if [ -z "$NEWER" ]; then
        echo "[imgui] $LIB is up to date"
        exit 0
    fi
fi

# SDL3 comes from the pixi env — headers AND the dylib. No system SDL is
# involved, and the rpath below is what lets the shim find libSDL3 at load
# time without the caller exporting DYLD_LIBRARY_PATH.
#
# ⚠ THE ACTIVE ENV FIRST, NOT `default`. `pixi run` sets $CONDA_PREFIX to the
# env it activated — `jetson` on the Orin, where `default` may not even be
# installed — and the viewer loads libSDL3 from that same env at run time
# (`noeira/render/sdl`). `default` stays as the fallback for a bare shell.
SDL_PREFIX=""
for c in "${CONDA_PREFIX:-}" "$ROOT/.pixi/envs/default"; do
    [ -n "$c" ] && [ -f "$c/include/SDL3/SDL.h" ] && { SDL_PREFIX="$c"; break; }
done
if [ -z "$SDL_PREFIX" ]; then
    echo "[imgui] ERROR: SDL3 headers not found in \$CONDA_PREFIX or .pixi/envs/default." >&2
    echo "[imgui] Run 'pixi install' (or 'pixi install -e jetson') first." >&2
    exit 1
fi

# ⚠ `c++`, NOT `clang++`: the other shims use the platform's default compiler,
# and a stock Jetson has g++ only. Override with CXX=... .
CXX="${CXX:-c++}"

echo "[imgui] building $LIB ..."
# imgui_demo.cpp is deliberately NOT compiled: it is ~500 KB of artifact for a
# gallery nothing here calls. Add it back if you want ShowDemoWindow().
"$CXX" -O2 -std=c++17 -fPIC -shared \
    -I "$IMGUI_DIR" -I "$IMGUI_DIR/backends" \
    -I "$GZ_DIR/src" \
    -I "$SDL_PREFIX/include" \
    "$SRC" \
    "$GZ_DIR/src/ImGuizmo.cpp" \
    "$IMGUI_DIR/imgui.cpp" \
    "$IMGUI_DIR/imgui_draw.cpp" \
    "$IMGUI_DIR/imgui_tables.cpp" \
    "$IMGUI_DIR/imgui_widgets.cpp" \
    "$IMGUI_DIR/backends/imgui_impl_sdl3.cpp" \
    "$IMGUI_DIR/backends/imgui_impl_sdlgpu3.cpp" \
    -L "$SDL_PREFIX/lib" -lSDL3 \
    -Wl,-rpath,"$SDL_PREFIX/lib" \
    -o "$LIB"

echo "[imgui] built $LIB ($(du -h "$LIB" | cut -f1))"

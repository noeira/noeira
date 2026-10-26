"""Spike DR.6 — does `std.utils.Variant` help with multi-input Modules?

DR.2 verdict was: variadic `*inputs: TileTensor[DT, L, O]` REJECTED — each
TileTensor's `MutOrigin` is per-source distinct and value-variadic packs
need homogeneous types. The question: does Variant offer a workaround?

Variant is a runtime tagged union: `Variant[T1, T2, ..., Tn]` is one
type holding any of T1..Tn at runtime, discriminant tracked.

Three concrete probes:

Probe A: trivial sanity — does Variant compile and dispatch?
    Probe with `Variant[Int, Float64]` to confirm baseline functionality.

Probe B: can Variant wrap distinct TileTensor types?
    The hard question. TileTensor with a concrete layout cannot be
    written at type-parameter position (`row_major[N, M]()` returns
    `Layout` not `TensorLayout`; same blocker we hit in DR.3 with
    `out_view()`). So even constructing `Variant[Tile1, Tile2]` may
    be impossible.

Probe C: can Variant wrap distinct LayoutTensor types?
    LayoutTensor (the older, layout-not-tensor-layout-parameterized
    type) accepts `Layout.row_major(...)` at type-param positions. If
    LayoutTensor is variant-friendly, we'd have a separate workaround
    — but at the cost of mixing two tensor abstractions in nn.
"""

from std.utils import Variant
from layout import TileTensor, TensorLayout, row_major
from noeira.core.tiles import RowMajor

from noeira.nn.constants import DT


# ──────────────────────────────────────────────────────────────────────
# Probe A — Variant[Int, Float64] baseline sanity.
# ──────────────────────────────────────────────────────────────────────


def probe_a_baseline() raises:
    print("--- Probe A: Variant[Int, Float64] baseline ---")
    comptime IntOrFloat = Variant[Int, Float64]
    var x = IntOrFloat(42)
    var y = IntOrFloat(Float64(3.14))
    print("  x.isa[Int]()       =", x.isa[Int]())
    print("  x.isa[Float64]()   =", x.isa[Float64]())
    print("  y.isa[Int]()       =", y.isa[Int]())
    print("  y.isa[Float64]()   =", y.isa[Float64]())
    if x.isa[Int]() and y.isa[Float64]():
        print("  Probe A: PASSED — Variant baseline works")
    else:
        print("  Probe A: FAILED")


# ──────────────────────────────────────────────────────────────────────
# Probe B — Variant of TileTensors with concrete layouts.
# ──────────────────────────────────────────────────────────────────────
# Concrete TileTensor types CANNOT be written at type positions because
# `row_major[N, M]()` returns `Layout` and TileTensor's `LayoutType`
# parameter requires `TensorLayout`. This is the same blocker that
# defeated DR.3's typed `out_view()`.
#
# Verify by attempting the type alias and watching the compiler reject:
#     comptime Tile_2x2 = TileTensor[DT, row_major[2, 2](), MutAnyOrigin]
# Expected error:
#     'TileTensor' parameter 'LayoutType' has 'TensorLayout' type, but
#     value has type 'Layout[*?, *?]'
#
# So `Variant[Tile_2x2, Tile_2x2]` is unconstructible at the type level.


# ──────────────────────────────────────────────────────────────────────
# Probe C — Variant of LayoutTensors (uses Layout, not TensorLayout).
# ──────────────────────────────────────────────────────────────────────


def probe_c_layouttensor_variant() raises:
    """Can Variant wrap two LayoutTensors of different layouts?
    LayoutTensor accepts `Layout.row_major(...)` in its type params, so
    we *can* write the concrete type."""
    print("--- Probe C: Variant[LayoutTensor[Layout1], LayoutTensor[Layout2]] ---")

    comptime L1 = RowMajor[2, 3]
    comptime L2 = RowMajor[2, 5]
    comptime Tile1 = TileTensor[DT, L1, MutAnyOrigin]
    comptime Tile2 = TileTensor[DT, L2, MutAnyOrigin]

    var a = List[Scalar[DT]](length=6, fill=Scalar[DT](1.0))
    var b = List[Scalar[DT]](length=10, fill=Scalar[DT](2.0))
    var t1 = Tile1(a.unsafe_ptr())
    var t2 = Tile2(b.unsafe_ptr())

    comptime V = Variant[Tile1, Tile2]
    var v0 = V(t1)
    var v1 = V(t2)
    print("  v0.isa[Tile1]() =", v0.isa[Tile1]())
    print("  v0.isa[Tile2]() =", v0.isa[Tile2]())
    print("  v1.isa[Tile1]() =", v1.isa[Tile1]())
    print("  v1.isa[Tile2]() =", v1.isa[Tile2]())
    var ok = v0.isa[Tile1]() and v1.isa[Tile2]() and not v0.isa[Tile2]() and not v1.isa[Tile1]()
    if ok:
        print("  Probe C: PASSED — Variant[LayoutTensor, LayoutTensor] works")
        print("           BUT LayoutTensor != TileTensor — using this in nn")
        print("           would require switching tensor abstractions.")
    else:
        print("  Probe C: FAILED")


def main() raises:
    print("=" * 70)
    print("DR.6 — Variant for multi-input Module dispatch?")
    print("=" * 70)
    probe_a_baseline()
    probe_c_layouttensor_variant()
    print()
    print("Probe B (`Variant[TileTensor, TileTensor]` with concrete layouts):")
    print("  STRUCTURALLY IMPOSSIBLE — TileTensor's `LayoutType` param wants")
    print("  `TensorLayout`, but `row_major[N, M]()` returns `Layout`. The")
    print("  concrete TileTensor type cannot be spelled at type positions.")
    print("  Same blocker DR.3 hit on `out_view()`. nn always uses generic")
    print("  `L: TensorLayout` arguments at call sites; concrete TileTensor")
    print("  types are never written down. Variant cannot wrap them.")
    print("=" * 70)

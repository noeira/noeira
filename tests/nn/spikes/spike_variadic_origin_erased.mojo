"""Spike DR.2-v2 — variadic `*inputs: TileTensor` with origin erasure.

The 2026-05-20 spike concluded `*inputs: TileTensor[DT, L, O]` is rejected:
per-source `MutOrigin`s don't unify across pack elements. This v2 spike
probes whether pinning the origin slot to `MutAnyOrigin` (the Discord
"origin-erasure" pattern) homogenizes the pack.

Attempts:
  A. Baseline partial-spec, no origin pinning — for regression check.
  B. Origin-pinned: `*inputs: TileTensor[..., origin=MutAnyOrigin, ...]`.
  C. Pointer-pack erasure (fallback): drop TileTensor at the variadic
     boundary, pass raw `*ptrs: Pointer[Scalar[DT], MutAnyOrigin]`.

Per-attempt feasibility is comptime-gated: if a body fails to compile,
the entire test binary refuses to build. So each attempt is in its own
section and we comment out the failing ones as we identify them.

Body fixes vs v1 spike:
  - `len(inputs)` is dynamic; use `inputs.__len__()` or static count.
  - `output[b, d]` needs `comptime assert output.flat_rank == 2` first.
"""

from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major
from std.memory import Pointer

from noeira.nn.constants import DT


# ══════════════════════════════════════════════════════════════════════
# Attempt A — confirmed REJECTED (per-source origins still don't unify
# even with partial-spec). Commenting out to keep the file compileable.
# Error: "cannot be converted from 'TileTensor[..., origin_of(b)]'
#         to 'TileTensor[..., origin_of(a)]'"
# ══════════════════════════════════════════════════════════════════════
#
# def sumN_partial_spec[
#     BATCH: Int,
#     DIM: Int,
# ](
#     var *inputs: TileTensor[
#         dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
#     ],
#     mut output: TileTensor[
#         mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
#         element_size=1, ...,
#     ],
# ) raises:
#     ...


# ══════════════════════════════════════════════════════════════════════
# Attempt B — origin pinned to MutAnyOrigin.
# ══════════════════════════════════════════════════════════════════════


def sumN_origin_erased[
    BATCH: Int,
    DIM: Int,
](
    var *inputs: TileTensor[
        dtype=DT,
        address_space=AddressSpace.GENERIC,
        element_size=1,
        origin=MutAnyOrigin,
        ...,
    ],
    output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
) raises:
    """Sum N inputs, all `origin=MutAnyOrigin`. Caller rebinds before
    passing. If this compiles, the variadic pack is homogeneous by
    construction (every element shares the pinned origin)."""
    comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
    for b in range(BATCH):
        for d in range(DIM):
            output[b, d] = Scalar[DT](0.0)
    # Homogeneous variadic → VariadicList (runtime length).
    # The element TileTensor's LayoutType is opaque inside the body, so
    # `inputs[k][b, d]` can't prove its rank. Workaround: extract `.ptr`
    # (already origin-erased) and rebuild a typed rank-2 view per iter.
    for k in range(len(inputs)):
        var elem_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
            inputs[k].ptr
        )
        var view = TileTensor(elem_ptr, row_major[BATCH, DIM]())
        for b in range(BATCH):
            for d in range(DIM):
                output[b, d] = output[b, d] + view[b, d]


# ══════════════════════════════════════════════════════════════════════
# Attempt C — pointer-pack erasure (fallback).
# ══════════════════════════════════════════════════════════════════════


def sumN_pointer_pack[
    BATCH: Int,
    DIM: Int,
](
    var *input_ptrs: Pointer[Scalar[DT], MutAnyOrigin],
    output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
) raises:
    """Sum N inputs passed as raw pointers; reconstruct TileTensor
    internally."""
    comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
    for b in range(BATCH):
        for d in range(DIM):
            output[b, d] = Scalar[DT](0.0)
    for k in range(len(input_ptrs)):
        var view = TileTensor(input_ptrs[k], row_major[BATCH, DIM]())
        for b in range(BATCH):
            for d in range(DIM):
                output[b, d] = output[b, d] + view[b, d]


# ══════════════════════════════════════════════════════════════════════
# Attempt D — mutable variadic (`mut *grad_inputs`), required for vjp.
# ══════════════════════════════════════════════════════════════════════


def scatterN_grad[
    BATCH: Int,
    DIM: Int,
](
    grad_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
    mut *grad_inputs: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
) raises:
    """vjp prototype: scatter grad_output to N mutable grad_inputs.

    Probes whether `mut *grad_inputs: TileTensor[..., origin=MutAnyOrigin]`
    compiles. If yes, the unified `NaryModule.vjp` signature is feasible.
    """
    comptime assert grad_output.flat_rank == 2
    for k in range(len(grad_inputs)):
        var elem_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
            grad_inputs[k].ptr
        )
        var view = TileTensor(elem_ptr, row_major[BATCH, DIM]())
        for b in range(BATCH):
            for d in range(DIM):
                view[b, d] = grad_output[b, d]


# ══════════════════════════════════════════════════════════════════════
# Attempt E — mock unified NaryModule with comptime-ARITY dispatch.
# ══════════════════════════════════════════════════════════════════════


struct MockNaryBinarySub[DIM: Int]:
    """Stand-in for a unified NaryModule leaf with declared comptime
    arity. Forward dispatches on Self.ARITY at comptime to give each
    branch typed access to its known number of inputs."""

    comptime ARITY: Int = 2
    comptime OUT_DIM: Int = Self.DIM

    def __init__(out self): pass

    def forward[
        BATCH: Int,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert output.flat_rank == 2
        comptime if Self.ARITY == 2:
            var in0_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                inputs[0].ptr
            )
            var in1_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                inputs[1].ptr
            )
            var in0 = TileTensor(in0_ptr, row_major[BATCH, Self.DIM]())
            var in1 = TileTensor(in1_ptr, row_major[BATCH, Self.DIM]())
            for b in range(BATCH):
                for d in range(Self.DIM):
                    output[b, d] = in0[b, d] - in1[b, d]
        else:
            raise Error("MockNaryBinarySub: ARITY must be 2")


# ══════════════════════════════════════════════════════════════════════
# Attempt F — TRAIT method with variadic param (critical for Phase 4.6).
# ══════════════════════════════════════════════════════════════════════
#
# The previous attempts used `def` on a struct or free function. Phase 4.6
# wants `trait NaryModule` with an abstract `forward[...](var *inputs: ...)`
# method. Probe whether Mojo accepts this signature in a TRAIT body.


trait NaryModuleProto(Defaultable & Movable & Deinitable):
    comptime ARITY: Int
    comptime OUT_DIM: Int

    def forward[
        BATCH: Int,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        ...


struct MockTraitConformingBinarySub[DIM: Int](NaryModuleProto):
    """Probes whether a struct can conform to a trait whose method
    signature has a variadic. Production orchestrators dispatch on
    concrete types (comptime-static), so we test that path — not the
    generic-trait-wrapper path."""

    comptime ARITY: Int = 2
    comptime OUT_DIM: Int = Self.DIM

    def __init__(out self): pass

    def forward[
        BATCH: Int,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        # Rebind output to a typed rank-2 view first (LayoutType is opaque
        # from the variadic-style partial-spec signature).
        var o_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](output.ptr)
        var out_view = TileTensor(o_ptr, row_major[BATCH, Self.DIM]())
        comptime if Self.ARITY == 2:
            var in0_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                inputs[0].ptr
            )
            var in1_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                inputs[1].ptr
            )
            var in0 = TileTensor(in0_ptr, row_major[BATCH, Self.DIM]())
            var in1 = TileTensor(in1_ptr, row_major[BATCH, Self.DIM]())
            for b in range(BATCH):
                for d in range(Self.DIM):
                    out_view[b, d] = in0[b, d] - in1[b, d]
        else:
            raise Error("MockTraitConformingBinarySub: ARITY must be 2")


# ══════════════════════════════════════════════════════════════════════
# Smoke drivers.
# ══════════════════════════════════════════════════════════════════════


def smoke_B() raises:
    print("--- Attempt B: variadic origin-erased ---")
    var a = List[Scalar[DT]](length=4, fill=Scalar[DT](1.0))
    var b = List[Scalar[DT]](length=4, fill=Scalar[DT](2.0))
    var c = List[Scalar[DT]](length=4, fill=Scalar[DT](3.0))
    var out = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))

    # Erase origin via rebind at the variadic boundary.
    var a_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](a.unsafe_ptr())
    var b_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())
    var c_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](c.unsafe_ptr())
    var o_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        out.unsafe_ptr()
    )

    var ta = TileTensor(a_ptr, row_major[2, 2]())
    var tb = TileTensor(b_ptr, row_major[2, 2]())
    var tc = TileTensor(c_ptr, row_major[2, 2]())
    var to = TileTensor(o_ptr, row_major[2, 2]())

    sumN_origin_erased[2, 2](ta, tb, tc, output=to)
    var ok = out[0] == 6.0 and out[1] == 6.0 and out[2] == 6.0 and out[3] == 6.0
    if ok:
        print("  B PASSED — origin-erased variadic WORKS")
    else:
        print("  B FAILED — got", out[0], out[1], out[2], out[3])
    _ = a^; _ = b^; _ = c^; _ = out^


def smoke_C() raises:
    print("--- Attempt C: pointer-pack erasure ---")
    var a = List[Scalar[DT]](length=4, fill=Scalar[DT](1.0))
    var b = List[Scalar[DT]](length=4, fill=Scalar[DT](2.0))
    var c = List[Scalar[DT]](length=4, fill=Scalar[DT](3.0))
    var out = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))

    var a_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](a.unsafe_ptr())
    var b_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())
    var c_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](c.unsafe_ptr())
    var o_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        out.unsafe_ptr()
    )
    var to = TileTensor(o_ptr, row_major[2, 2]())

    sumN_pointer_pack[2, 2](a_ptr, b_ptr, c_ptr, output=to)
    var ok = out[0] == 6.0 and out[1] == 6.0 and out[2] == 6.0 and out[3] == 6.0
    if ok:
        print("  C PASSED — pointer-pack erasure WORKS")
    else:
        print("  C FAILED — got", out[0], out[1], out[2], out[3])
    _ = a^; _ = b^; _ = c^; _ = out^


def smoke_D() raises:
    print("--- Attempt D: mutable variadic (mut *grad_inputs) ---")
    var go = List[Scalar[DT]](length=4, fill=Scalar[DT](7.0))
    var gi0 = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))
    var gi1 = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))
    var gi2 = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))

    var go_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        go.unsafe_ptr()
    )
    var gi0_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        gi0.unsafe_ptr()
    )
    var gi1_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        gi1.unsafe_ptr()
    )
    var gi2_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        gi2.unsafe_ptr()
    )

    var gov = TileTensor(go_ptr, row_major[2, 2]())
    var gi0v = TileTensor(gi0_ptr, row_major[2, 2]())
    var gi1v = TileTensor(gi1_ptr, row_major[2, 2]())
    var gi2v = TileTensor(gi2_ptr, row_major[2, 2]())

    scatterN_grad[2, 2](gov, gi0v, gi1v, gi2v)
    var ok = (gi0[0] == 7.0 and gi1[0] == 7.0 and gi2[0] == 7.0
              and gi0[3] == 7.0 and gi1[3] == 7.0 and gi2[3] == 7.0)
    if ok:
        print("  D PASSED — mut variadic WORKS (vjp signature feasible)")
    else:
        print("  D FAILED — got gi0[0]=", gi0[0], " gi1[0]=", gi1[0])
    _ = go^; _ = gi0^; _ = gi1^; _ = gi2^


def smoke_E() raises:
    print("--- Attempt E: NaryModule mock with comptime-ARITY dispatch ---")
    var a = List[Scalar[DT]](length=4, fill=Scalar[DT](10.0))
    var b = List[Scalar[DT]](length=4, fill=Scalar[DT](3.0))
    var out = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))

    var a_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](a.unsafe_ptr())
    var b_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())
    var o_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        out.unsafe_ptr()
    )
    var ta = TileTensor(a_ptr, row_major[2, 2]())
    var tb = TileTensor(b_ptr, row_major[2, 2]())
    var to = TileTensor(o_ptr, row_major[2, 2]())

    var sub = MockNaryBinarySub[2]()
    sub.forward[BATCH=2](ta, tb, output=to)
    var ok = out[0] == 7.0 and out[1] == 7.0 and out[2] == 7.0 and out[3] == 7.0
    if ok:
        print("  E PASSED — comptime-ARITY dispatch with typed views WORKS")
    else:
        print("  E FAILED — got", out[0], out[1], out[2], out[3])
    _ = a^; _ = b^; _ = out^


def smoke_F() raises:
    print("--- Attempt F: TRAIT method with variadic (Phase 4.6 critical) ---")
    var a = List[Scalar[DT]](length=4, fill=Scalar[DT](10.0))
    var b = List[Scalar[DT]](length=4, fill=Scalar[DT](3.0))
    var out = List[Scalar[DT]](length=4, fill=Scalar[DT](0.0))

    var a_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](a.unsafe_ptr())
    var b_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())
    var o_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        out.unsafe_ptr()
    )
    var ta = TileTensor(a_ptr, row_major[2, 2]())
    var tb = TileTensor(b_ptr, row_major[2, 2]())
    var to = TileTensor(o_ptr, row_major[2, 2]())

    var sub = MockTraitConformingBinarySub[2]()
    # Direct concrete-type dispatch (matches production orchestrator pattern).
    sub.forward[BATCH=2](ta, tb, output=to)
    var ok = out[0] == 7.0 and out[1] == 7.0 and out[2] == 7.0 and out[3] == 7.0
    if ok:
        print("  F PASSED — trait+variadic+generic dispatch WORKS")
    else:
        print("  F FAILED — got", out[0], out[1], out[2], out[3])
    _ = a^; _ = b^; _ = out^


def main() raises:
    print("=" * 70)
    print("DR.2-v2 — variadic TileTensor with origin erasure")
    print("Mojo nightly: 1.0.0b2.dev2026052006")
    print("=" * 70)
    print("Attempt A (partial-spec, no pinning): REJECTED (origin_of(a)/(b)")
    print("  do not unify) — confirmed reproduction of 2026-05-20 blocker.")
    smoke_B()
    smoke_C()
    smoke_D()
    smoke_E()
    smoke_F()
    print("=" * 70)

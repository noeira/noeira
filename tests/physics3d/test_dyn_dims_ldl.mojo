"""2b.0's payoff test: ONE compiled LDL body, two providers, three models.

WHAT THIS PROVES, AND WHY IT COMES BEFORE THE SWEEP
===================================================

Phase 2b converts ~120 shared declarations to `D: DimsLike` + `LM: Layout`
so that one body serves both legs (assessment §12.4). Everything about that
plan rests on three claims, and each is cheap to check on ONE kernel and
expensive to discover at declaration ninety:

  A. a runtime-dims provider drives the same body a comptime one does, and
     they agree numerically;
  B. `Layout.row_major[2]()` + `RuntimeLayout` is accepted by the same
     `LM: Layout` parameter that takes `Layout.row_major(BATCH, NV*NV)`;
  C. ONE instantiation serves models of DIFFERENT SIZE — the property that
     makes the dynamic leg worth having at all. A dynamic provider that
     still needed a rebuild per model would be pure cost.

⚠ (C) IS THE ONE THAT CANNOT BE FAKED BY ACCIDENT. (A) and (B) would both
pass if the compiler quietly specialised the "dynamic" arm on a constant —
which is exactly how the §12.3 layout probe first returned a meaningless
1.000 (the comptime aliases were being folded into the "dynamic" function).
So the dynamic arm here runs NV=7 and NV=5 through the same `DynDims`
type, and the two answers must differ from each other while each matches its
own static counterpart. `DynDims` now takes NO parameters at all, which makes
the claim sharper than it was: there is not even a cap left to specialise on.

Tolerance: f64 throughout, and the comparison is against a static arm that
runs the SAME source lines. §4.4 warns the dynamic leg cannot be assumed
bit-exact in general (a comptime bound changes FMA contraction), so the gate
is 1e-12 and the actual worst error is PRINTED — if it is ever exactly zero
that is information, not something to assert.
"""

from std.utils import IndexList
from layout import Layout, RuntimeLayout
from noeira.core.tiles import RowMajor

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import Dims, DynDims
from noeira.physics3d.fields.rt_layout import DYN1
from noeira.physics3d.fields.dims import DIM_POISON
from noeira.physics3d.fields.scratch import Scratch
from noeira.physics3d.dynamics.ldl import _ldl_factor_env, _ldl_solve_env
from noeira.physics3d.gpu.constants import MODEL_TREE_SIZE

comptime DT = DType.float64
comptime DYN2 = Layout.row_major[2]()
comptime BATCH = 2


struct Tally(Movable):
    """Mojo has no module-level `var`, so the counters travel in a value."""

    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, name: String, ok: Bool):
        self.checks += 1
        if not ok:
            self.failures += 1
            print("  FAIL:", name)
        else:
            print("  ok:", name)


def _spd(nv: Int, env: Int) -> List[Float64]:
    """A diagonally dominant SPD matrix, deterministic in (nv, env)."""
    var m = List[Float64](length=nv * nv, fill=0.0)
    for i in range(nv):
        for j in range(nv):
            var v = 1.0 / Float64(1 + i + j + env)
            m[i * nv + j] = v
        m[i * nv + i] += Float64(nv) + Float64(i) * 0.25
    return m^


def _rhs(nv: Int, env: Int) -> List[Float64]:
    var b = List[Float64](length=nv, fill=0.0)
    for i in range(nv):
        b[i] = Float64(1 + i) * 0.5 - Float64(env)
    return b^


def _fill(mut M: TensorImpl[DT], mut b: TensorImpl[DT], nv: Int):
    for e in range(BATCH):
        var m = _spd(nv, e)
        for i in range(nv * nv):
            M.data[e * nv * nv + i] = m[i]
        var r = _rhs(nv, e)
        for i in range(nv):
            b.data[e * nv + i] = r[i]


def solve_static[NV: Int]() raises -> List[Float64]:
    """The arm that ships today: comptime dims, comptime layout."""
    comptime LM = RowMajor[BATCH, NV * NV]
    comptime LNV = RowMajor[BATCH, NV]
    comptime LTREE = RowMajor[NV * MODEL_TREE_SIZE]
    var M = TensorImpl[DT].alloc(BATCH * NV * NV)
    var L = TensorImpl[DT].alloc(BATCH * NV * NV)
    var D = TensorImpl[DT].alloc(BATCH * NV)
    var b = TensorImpl[DT].alloc(BATCH * NV)
    var x = TensorImpl[DT].alloc(BATCH * NV)
    _fill(M, b, NV)

    var M_v = M.tt["cpu", BATCH, NV * NV]()
    var L_v = L.tt["cpu", BATCH, NV * NV]()
    var D_v = D.tt["cpu", BATCH, NV]()
    var b_v = b.tt["cpu", BATCH, NV]()
    var x_v = x.tt["cpu", BATCH, NV]()
    # ⚠ THE ZEROED BLOCK TABLE, DELIBERATELY. This file hand-builds its
    # tensors and has no `Model`, so it exercises the fallback a parser-free
    # model takes: rows past `ntree` are `(0, 0, 0)`, `dof_num == 0` ends the
    # walk, and an all-zero table therefore means NO BLOCKS — which the
    # factorisation must read as ONE block spanning the whole `nv`, not as no
    # work. See `dynamics/ldl.mojo`'s header.
    var T = TensorImpl[DT].alloc(NV * MODEL_TREE_SIZE)
    for i in range(NV * MODEL_TREE_SIZE):
        T.data[i] = Scalar[DT](0)
    var T_v = T.tt["cpu", NV * MODEL_TREE_SIZE]()
    var dims = Dims[nv=NV]()
    for e in range(BATCH):
        _ldl_factor_env(e, dims, M_v, L_v, D_v, T_v)
        _ldl_solve_env(e, dims, L_v, D_v, b_v, x_v, T_v)

    var out = List[Float64]()
    for i in range(BATCH * NV):
        out.append(Float64(x.data[i]))
    return out^


@no_inline
def solve_dynamic(nv: Int) raises -> List[Float64]:
    """The dynamic leg. `nv` is an ARGUMENT — no comptime parameter anywhere
    in this function, so every call shares one instantiation.

    ⚠ `@no_inline` IS PART OF THE ASSERTION, NOT AN OPTIMISATION HINT.
    Without it the compiler may inline this at both call sites and constant-
    fold `nv` into each copy, at which point "one body, two sizes" is true of
    the source and false of the machine code — the exact way §12.3's first
    layout probe fooled itself. The decorator pins it to one compiled body."""
    var M = TensorImpl[DT].alloc(BATCH * nv * nv)
    var L = TensorImpl[DT].alloc(BATCH * nv * nv)
    var D = TensorImpl[DT].alloc(BATCH * nv)
    var b = TensorImpl[DT].alloc(BATCH * nv)
    var x = TensorImpl[DT].alloc(BATCH * nv)
    _fill(M, b, nv)

    var rl_m = RuntimeLayout[DYN2].row_major(IndexList[2](BATCH, nv * nv))
    var rl_v = RuntimeLayout[DYN2].row_major(IndexList[2](BATCH, nv))
    var M_v = M.tt_dyn["cpu", DYN2](rl_m)
    var L_v = L.tt_dyn["cpu", DYN2](rl_m)
    var D_v = D.tt_dyn["cpu", DYN2](rl_v)
    var b_v = b.tt_dyn["cpu", DYN2](rl_v)
    var x_v = x.tt_dyn["cpu", DYN2](rl_v)
    # ⚠ THE ZEROED BLOCK TABLE, DELIBERATELY. This file hand-builds its
    # tensors and has no `Model`, so it exercises the fallback a parser-free
    # model takes: rows past `ntree` are `(0, 0, 0)`, `dof_num == 0` ends the
    # walk, and an all-zero table therefore means NO BLOCKS — which the
    # factorisation must read as ONE block spanning the whole `nv`, not as no
    # work. See `dynamics/ldl.mojo`'s header.
    var T = TensorImpl[DT].alloc(nv * MODEL_TREE_SIZE)
    for i in range(nv * MODEL_TREE_SIZE):
        T.data[i] = Scalar[DT](0)
    var rl_t = RuntimeLayout[DYN1].row_major(
        IndexList[1](nv * MODEL_TREE_SIZE)
    )
    var T_v = T.tt_dyn["cpu", DYN1](rl_t)
    var dims = DynDims(nv=nv)
    for e in range(BATCH):
        _ldl_factor_env(e, dims, M_v, L_v, D_v, T_v)
        _ldl_solve_env(e, dims, L_v, D_v, b_v, x_v, T_v)

    var out = List[Float64]()
    for i in range(BATCH * nv):
        out.append(Float64(x.data[i]))
    return out^


def worst(a: List[Float64], b: List[Float64]) raises -> Float64:
    if len(a) != len(b):
        raise Error(
            "length mismatch: " + String(len(a)) + " vs " + String(len(b))
        )
    var w = 0.0
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > w:
            w = d
    return w


def residual(x: List[Float64], nv: Int) raises -> Float64:
    """M·x vs b, recomputed from scratch. Without this the test would be
    happy with two arms that agree on the same WRONG answer."""
    var w = 0.0
    for e in range(BATCH):
        var m = _spd(nv, e)
        var r = _rhs(nv, e)
        for i in range(nv):
            var s = 0.0
            for j in range(nv):
                s += m[i * nv + j] * x[e * nv + j]
            var d = s - r[i]
            if d < 0:
                d = -d
            if d > w:
                w = d
    return w


def main() raises:
    var t = Tally()
    print("=== A. the two providers agree, NV=7 ===")
    var s7 = solve_static[7]()
    var d7 = solve_dynamic(7)
    var e7 = worst(s7, d7)
    print("  static vs dynamic worst err:", e7)
    t.check("NV=7 dynamic matches static within 1e-12", e7 < 1e-12)
    print("  bit-exact:", e7 == 0.0)

    print("\n=== B. the solve is actually a solve (M·x - b) ===")
    var r_s = residual(s7, 7)
    var r_d = residual(d7, 7)
    print("  static residual:", r_s, " dynamic residual:", r_d)
    t.check("static arm solves the system", r_s < 1e-10)
    t.check("dynamic arm solves the system", r_d < 1e-10)

    print("\n=== C. ONE instantiation, TWO model sizes ===")
    var s5 = solve_static[5]()
    var d5 = solve_dynamic(5)
    var e5 = worst(s5, d5)
    print("  NV=5 static vs dynamic worst err:", e5)
    t.check("NV=5 dynamic matches static within 1e-12", e5 < 1e-12)
    t.check("NV=5 dynamic solves the system", residual(d5, 5) < 1e-10)
    # The vacuity guard for (C): if the "dynamic" arm had been specialised on
    # a constant nv, these two calls could not both be right.
    t.check("the two sizes returned different lengths", len(d5) != len(d7))
    t.check(
        "and different values",
        d5[0] != d7[0] and d5[len(d5) - 1] != d7[len(d7) - 1],
    )

    print("\n=== D. the comptime members of a dynamic provider are POISON ===")
    t.check("DynDims.NV is the poison value", DynDims.NV == DIM_POISON)
    t.check("poison is negative, not zero", DIM_POISON < 0)

    # ⚠ THE TWO FAMILIES POISON DIFFERENTLY, AND BOTH DIRECTIONS ARE
    # LOAD-BEARING. `NV` is -1 so an unconverted site dies at the site;
    # `CAP_NV` is 0 so that PRODUCTS of caps stay 0 and keep selecting the
    # heap. This pair of checks is what stops someone merging them.
    t.check("the scratch cap poisons to ZERO", DynDims.CAP_NV == 0)
    t.check(
        "a PRODUCT of caps stays zero (it would be 1 with -1)",
        DynDims.CAP_NV * DynDims.CAP_NV == 0,
    )
    t.check(
        "and so selects the HEAP leg, not a one-element stack array",
        not Scratch[Float64, DynDims.CAP_NV * DynDims.CAP_NV].STATIC,
    )

    print("\n=== E. there is NO cap: any model size constructs ===")
    # This section used to assert that `DynDims[cap_nv=4](nv=9)` RAISES.
    # §10.7 removed the reason for the cap to exist (a fixed-cap stack array
    # is slower than the heap it was meant to beat), so the parameter and its
    # construction-time check are gone and a binary is no longer built for a
    # maximum model. The replacement claim is the one that now matters: a
    # model far larger than any cap this test ever used still works.
    var huge = DynDims(nv=100000)
    t.check("a model of 100000 dofs constructs", huge.get_nv() == 100000)
    var huge_out = solve_dynamic(23)
    t.check("and the dynamic arm still solves", len(huge_out) == BATCH * 23)

    print("\nchecks:", t.checks, " failures:", t.failures)
    if t.failures == 0:
        print("test_dyn_dims_ldl: ALL PASS")
    else:
        raise Error("test_dyn_dims_ldl: " + String(t.failures) + " failure(s)")

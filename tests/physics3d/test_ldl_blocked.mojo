"""Block-restricted LDL / M^-1 / solve == the dense ones, BIT FOR BIT.

P2's gate, extended by F1 to `_ldl_solve_env` — the third kernel in the module
and, after P2 sped its two siblings up, the LARGEST of them (66% of `ldl_pair`
at k=13).

`_ldl_factor_env` and `_m_inv_col_env` restrict their loops to the kinematic
tree containing each column. The claim is that this is BIT-EXACT on every
model, because `M`'s cross-tree entries are STRUCTURALLY exactly `0.0`:

  * the treewalk CRBA zeroes `M` and writes only ancestor pairs
    (`mass_matrix.mojo:592-600`);
  * the dense CRBA accumulates only over bodies in BOTH subtrees (`:219-223`),
    which no cross-tree pair has;
  * between CRBA and the factorisation only `_armature_env` runs, and it
    touches the diagonal.

A sequential accumulation that drops exact zeros returns the identical bits.

⚠⚠ WHY THIS FILE EXISTS RATHER THAN LEANING ON THE SUITE. Almost every model
in the tree is SINGLE-TREE — walker2d, hopper, ant, cartpole, the arms — and on
one tree the restriction is a NO-OP. The whole existing suite would pass on a
completely broken partition. This is the only place a multi-tree `M` is
factored both ways.

THE A/B NEEDS NO TOGGLE. A zeroed tree table means "no table", which
`_dof_block` renders as ONE block spanning `[0, nv)` — i.e. exactly the dense
code path, byte for byte. So arm B feeds the same `M` with the table cleared
and the two results must agree.

  A  two trees vs no table: `L`, `D`, `m_inv` and `x = M^-1 b` bit-identical.
  B  `m_inv`'s cross-block entries are exactly 0 in BOTH arms — the property
     that lets a consumer keep treating it as dense.
  C  ⚠ THE NEGATIVE CONTROL. Plant a cross-block entry in `M` — which CRBA can
     never produce — and the two arms MUST diverge. Without it, arm A passes on
     a harness where the partition does nothing.

Run: pixi run mojo run -I . tests/physics3d/test_ldl_blocked.mojo
"""

from noeira.core.tiles import RowMajor
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import Dims
from noeira.physics3d.dynamics.ldl import (
    _ldl_factor_env, _m_inv_env, _ldl_solve_env,
)
from noeira.physics3d.gpu.constants import (
    MODEL_TREE_SIZE, TREE_IDX_DOF_ADR, TREE_IDX_DOF_NUM,
)

comptime DT = DType.float64
comptime NV = 12
comptime SPLIT = 5          # trees [0,5) and [5,12) — deliberately uneven
comptime BATCH = 1
comptime LM = RowMajor[BATCH, NV * NV]
comptime LNV = RowMajor[BATCH, NV]
comptime LT = RowMajor[NV * MODEL_TREE_SIZE]


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _fill_M(mut M: TensorImpl[DT], coupled: Bool):
    """Block-diagonal SPD `M` over [0,SPLIT) and [SPLIT,NV) — the shape CRBA
    produces on a two-tree model."""
    for i in range(NV * NV):
        M.data[i] = Scalar[DT](0)
    for i in range(NV):
        for j in range(NV):
            if (i < SPLIT) != (j < SPLIT):
                continue
            M.data[i * NV + j] = Scalar[DT](
                1.0 / Float64(1 + (i - j) * (i - j))
            )
        M.data[i * NV + i] = Scalar[DT](4.0 + 0.25 * Float64(i))
    if coupled:
        # ⚠ CRBA CANNOT PRODUCE THIS. It exists only so the control can show
        # the two arms are capable of disagreeing.
        M.data[2 * NV + 9] = Scalar[DT](0.75)
        M.data[9 * NV + 2] = Scalar[DT](0.75)


def _table(mut T: TensorImpl[DT], two_trees: Bool):
    for i in range(NV * MODEL_TREE_SIZE):
        T.data[i] = Scalar[DT](0)
    if not two_trees:
        return                      # zeroed = "no table" = one whole-nv block
    T.data[0 * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR] = Scalar[DT](0)
    T.data[0 * MODEL_TREE_SIZE + TREE_IDX_DOF_NUM] = Scalar[DT](SPLIT)
    T.data[1 * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR] = Scalar[DT](SPLIT)
    T.data[1 * MODEL_TREE_SIZE + TREE_IDX_DOF_NUM] = Scalar[DT](NV - SPLIT)


def _run(coupled: Bool, two_trees: Bool) raises -> Tuple[
    List[Float64], List[Float64], List[Float64], List[Float64]
]:
    var M = TensorImpl[DT].alloc(BATCH * NV * NV)
    var L = TensorImpl[DT].alloc(BATCH * NV * NV)
    var D = TensorImpl[DT].alloc(BATCH * NV)
    var MI = TensorImpl[DT].alloc(BATCH * NV * NV)
    var T = TensorImpl[DT].alloc(NV * MODEL_TREE_SIZE)
    var bb_ = TensorImpl[DT].alloc(BATCH * NV)
    var xx = TensorImpl[DT].alloc(BATCH * NV)
    for i in range(BATCH * NV):
        bb_.data[i] = Scalar[DT](1.0 + 0.5 * Float64(i))
        # ⚠ POISONED like `m_inv`: `x` is written for every row today, so a
        # restriction that skipped rows would show up rather than hide.
        xx.data[i] = Scalar[DT](-999.0)
    _fill_M(M, coupled)
    _table(T, two_trees)
    # ⚠ POISONED, NOT ZEROED. `m_inv` is reused scratch in the engine, so a
    # block-restricted write that skipped the off-block rows would leave the
    # PREVIOUS step's values there. Seeding with a value that cannot be a
    # result turns that from invisible into a failure.
    for i in range(BATCH * NV * NV):
        MI.data[i] = Scalar[DT](-999.0)

    var Mv = M.tt["cpu", BATCH, NV * NV]()
    var Lv = L.tt["cpu", BATCH, NV * NV]()
    var Dv = D.tt["cpu", BATCH, NV]()
    var MIv = MI.tt["cpu", BATCH, NV * NV]()
    var Tv = T.tt["cpu", NV * MODEL_TREE_SIZE]()
    var dims = Dims[nv=NV]()
    var bv = bb_.tt["cpu", BATCH, NV]()
    var xv = xx.tt["cpu", BATCH, NV]()
    _ldl_factor_env(0, dims, Mv, Lv, Dv, Tv)
    _m_inv_env(0, dims, Lv, Dv, MIv, Tv)
    _ldl_solve_env(0, dims, Lv, Dv, bv, xv, Tv)

    var lo = List[Float64]()
    var do_ = List[Float64]()
    var mo = List[Float64]()
    for i in range(NV * NV):
        lo.append(Float64(L.data[i]))
        mo.append(Float64(MI.data[i]))
    var xo = List[Float64]()
    for i in range(NV):
        do_.append(Float64(D.data[i]))
        xo.append(Float64(xx.data[i]))
    return (lo^, do_^, mo^, xo^)


def _diff(a: List[Float64], b: List[Float64]) -> Int:
    var d = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            d += 1
    return d


def main() raises:
    var t = Tally()
    print("=== block-restricted LDL / M^-1 vs dense, bit for bit (P2) ===")

    # ── A: two trees vs no table, on a block-diagonal M ──────────────────
    print("--- A: two trees vs one whole-nv block ---")
    var seg = _run(False, True)
    var den = _run(False, False)
    t.truth(_diff(seg[0], den[0]) == 0,
            String("L identical over ", NV * NV, " entries (",
                   _diff(seg[0], den[0]), " differ)"))
    t.truth(_diff(seg[1], den[1]) == 0,
            String("D identical over ", NV, " entries (",
                   _diff(seg[1], den[1]), " differ)"))
    t.truth(_diff(seg[2], den[2]) == 0,
            String("m_inv identical over ", NV * NV, " entries (",
                   _diff(seg[2], den[2]), " differ)"))
    t.truth(_diff(seg[3], den[3]) == 0,
            String("x = M^-1 b identical over ", NV, " entries (",
                   _diff(seg[3], den[3]), " differ) — F1"))
    # ⚠ AND `x` WAS ACTUALLY WRITTEN. It is seeded to -999; a restriction that
    # skipped rows outside the block would leave that value in place, and a
    # bit-identical comparison against a dense arm that ALSO skipped them would
    # still pass.
    var poisoned = 0
    for i in range(NV):
        if seg[3][i] == -999.0:
            poisoned += 1
    t.truth(poisoned == 0,
            String("x fully written: ", poisoned, " of ", NV,
                   " rows still hold the -999 poison"))

    # ── B: m_inv's cross-block entries are exactly zero, in BOTH arms ─────
    print("--- B: m_inv is exactly 0 across the block boundary ---")
    var nz_seg = 0
    var nz_den = 0
    var checked = 0
    for i in range(NV):
        for j in range(NV):
            if (i < SPLIT) == (j < SPLIT):
                continue
            checked += 1
            if seg[2][i * NV + j] != 0.0:
                nz_seg += 1
            if den[2][i * NV + j] != 0.0:
                nz_den += 1
    t.truth(checked > 0, String("cross-block entries checked: ", checked))
    t.truth(nz_seg == 0,
            String("segmented: ", nz_seg, " of ", checked, " nonzero"
                   " (the -999 poison would show here)"))
    t.truth(nz_den == 0,
            String("dense:     ", nz_den, " of ", checked, " nonzero"))

    # ── C: the negative control ──────────────────────────────────────────
    print("--- C: a cross-block M entry MUST make the arms diverge ---")
    var cseg = _run(True, True)
    var cden = _run(True, False)
    var cd = (_diff(cseg[0], cden[0]) + _diff(cseg[2], cden[2])
              + _diff(cseg[3], cden[3]))
    t.truth(cd > 0,
            String("L+m_inv+x differ in ", cd, " entries — the partition is not"
                   " a no-op, so arm A's agreement means something"))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_ldl_blocked: " + String(t.fails) + " failed")

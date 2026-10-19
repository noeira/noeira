"""`build_dof_segments` — the partition PN2 factors `H` over. PN2a's gate.

The function is pure and takes flat arrays, so it is gated here on hand-built
cases with known answers rather than through the solver. Nothing uses it yet.

⚠⚠ CASE E IS THE ONE THAT MATTERS. The merge flags and the per-dof output
share a buffer, and the writes are indexed by DOF while the flags are indexed
by TREE — so a forward walk destroys the flags for trees 1..5 while writing the
dofs of tree 0. That bug yields a CORRECT partition on every uncoupled scene
(including the park scenes this work is for) and only shows up when a coupling
exists between trees whose flag index collides with an earlier tree's dofs.
Case E is exactly that scene. Without it the gate would have passed on the
broken version.

Run: pixi run mojo run -I . tests/physics3d/test_newton_blocks.mojo
"""

from noeira.core.tiles import RowMajor
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.solver.newton_blocks import build_dof_segments
from noeira.physics3d.gpu.constants import (
    MODEL_TREE_SIZE, TREE_IDX_DOF_ADR, TREE_IDX_DOF_NUM, TREE_IDX_KIND,
)

comptime DT = DType.float64
comptime NV = 60
comptime NTREE = 10
comptime ME = 8
comptime L_T = RowMajor[NV * MODEL_TREE_SIZE]
comptime L_J = RowMajor[ME * NV]
comptime L_S = RowMajor[NV]


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


struct Case:
    """Ten trees of six dofs — the k=9 park scene's shape."""
    var trees: TensorImpl[DT]
    var je: TensorImpl[DT]
    var s0: TensorImpl[DT]
    var s1: TensorImpl[DT]
    var ntree: Int

    def __init__(out self, ntree: Int = NTREE) raises:
        self.ntree = ntree
        self.trees = TensorImpl[DT].alloc(NV * MODEL_TREE_SIZE)
        self.je = TensorImpl[DT].alloc(ME * NV)
        self.s0 = TensorImpl[DT].alloc(NV)
        self.s1 = TensorImpl[DT].alloc(NV)
        for i in range(NV * MODEL_TREE_SIZE):
            self.trees.data[i] = Scalar[DT](0)
        for i in range(ME * NV):
            self.je.data[i] = Scalar[DT](0)
        for t in range(ntree):
            self.trees.data[t * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR] = (
                Scalar[DT](t * 6)
            )
            self.trees.data[t * MODEL_TREE_SIZE + TREE_IDX_DOF_NUM] = (
                Scalar[DT](6)
            )
            self.trees.data[t * MODEL_TREE_SIZE + TREE_IDX_KIND] = Scalar[DT](0)

    def row(mut self, e: Int, dof: Int):
        self.je.data[e * NV + dof] = Scalar[DT](1)

    def run(mut self, num_edges: Int) raises -> Int:
        return build_dof_segments[DT](
            NV, self.ntree, num_edges,
            self.trees.tt["cpu", NV * MODEL_TREE_SIZE](),
            self.je.tt["cpu", ME * NV](),
            self.s0.tt["cpu", NV](),
            self.s1.tt["cpu", NV](),
        )

    def seg(self, i: Int) -> String:
        return String("[", Int(self.s0.data[i]), ",", Int(self.s1.data[i]), ")")

    def spans(self, i: Int, lo: Int, hi: Int) -> Bool:
        return Int(self.s0.data[i]) == lo and Int(self.s1.data[i]) == hi


def main() raises:
    var t = Tally()
    print("=== build_dof_segments (PN2a) ===")

    # ── A. no rows at all: every tree is its own segment ──────────────────
    print("--- A: ten trees, no constraint rows ---")
    var a = Case()
    var na = a.run(0)
    t.truth(na == 10, String("segments = ", na, " (want 10)"))
    t.truth(a.spans(0, 0, 6) and a.spans(5, 0, 6),
            String("dof 0 and 5 -> ", a.seg(0), " ", a.seg(5)))
    t.truth(a.spans(54, 54, 60) and a.spans(59, 54, 60),
            String("dof 54 and 59 -> ", a.seg(54), " ", a.seg(59)))

    # ── B. the MEASURED park case: 6 friction rows, all on dofs 0..5 ──────
    print("--- B: the k=9 park scene — 6 rows, all in tree 0 ---")
    var b = Case()
    for e in range(6):
        b.row(e, e)                      # FRICTION_DOF row on dof e
    var nb = b.run(6)
    t.truth(nb == 10, String("segments = ", nb, " (want 10 — rows in ONE tree"
                             " couple nothing)"))
    t.truth(b.spans(3, 0, 6), String("dof 3 -> ", b.seg(3), " (the arm)"))
    t.truth(b.spans(30, 30, 36), String("dof 30 -> ", b.seg(30),
                                        " (a parked prop, untouched)"))

    # ── C. a row spanning two ADJACENT trees ─────────────────────────────
    print("--- C: one row touching trees 0 and 1 ---")
    var c = Case()
    c.row(0, 2)
    c.row(0, 8)
    var nc = c.run(1)
    t.truth(nc == 9, String("segments = ", nc, " (want 9)"))
    t.truth(c.spans(0, 0, 12) and c.spans(11, 0, 12),
            String("dofs 0 and 11 -> ", c.seg(0), " ", c.seg(11)))
    t.truth(c.spans(12, 12, 18), String("dof 12 -> ", c.seg(12), " (separate)"))

    # ── D. a NON-CONTIGUOUS component becomes a SPAN ─────────────────────
    print("--- D: one row touching trees 0 and 3 — the span rule ---")
    var d = Case()
    d.row(0, 1)
    d.row(0, 19)                         # tree 3
    var nd = d.run(1)
    t.truth(nd == 7, String("segments = ", nd, " (want 7: trees 0-3 spanned)"))
    t.truth(d.spans(0, 0, 24) and d.spans(23, 0, 24),
            String("dofs 0 and 23 -> ", d.seg(0), " ", d.seg(23)))
    t.truth(d.spans(13, 0, 24),
            String("dof 13 -> ", d.seg(13), " (tree 2, swept up by the span)"))
    t.truth(d.spans(24, 24, 30), String("dof 24 -> ", d.seg(24), " (free)"))

    # ── E. ⚠⚠ THE ALIASING CASE. Coupling between HIGH trees only. ───────
    # A forward walk writes tree 0's dofs (0..5) over the merge flags of
    # trees 1..5 before reading them. Here the coupling is trees 7-8, whose
    # flag lives at index 7 — inside tree 1's dof range — so a forward walk
    # loses it and reports 10 segments instead of 9.
    print("--- E: coupling between trees 7 and 8 (the aliasing case) ---")
    var e = Case()
    e.row(0, 44)                         # tree 7
    e.row(0, 50)                         # tree 8
    var ne = e.run(1)
    t.truth(ne == 9, String("segments = ", ne,
                            " (want 9 — 10 means the merge flag was clobbered)"))
    t.truth(e.spans(42, 42, 54) and e.spans(53, 42, 54),
            String("dofs 42 and 53 -> ", e.seg(42), " ", e.seg(53)))
    t.truth(e.spans(54, 54, 60), String("dof 54 -> ", e.seg(54), " (free)"))

    # ── F. a row coupling everything: exactly today's behaviour ──────────
    print("--- F: one row touching tree 0 and tree 9 ---")
    var f = Case()
    f.row(0, 0)
    f.row(0, 59)
    var nf = f.run(1)
    t.truth(nf == 1, String("segments = ", nf, " (want 1 — the dense case)"))
    t.truth(f.spans(0, 0, 60) and f.spans(59, 0, 60),
            String("dofs 0 and 59 -> ", f.seg(0), " ", f.seg(59)))

    # ── G. the degenerate table: ntree == 0 -> ONE segment, not zero ─────
    print("--- G: no tree table (a Model built without the parser) ---")
    var g = Case(ntree=0)
    var ng = g.run(0)
    t.truth(ng == 1, String("segments = ", ng, " (want 1)"))
    t.truth(g.spans(0, 0, 60) and g.spans(59, 0, 60),
            String("dofs 0 and 59 -> ", g.seg(0), " ", g.seg(59)))

    # ── H. a table that does not tile [0, nv) -> the same fallback ───────
    print("--- H: a table with a GAP ---")
    var h = Case()
    h.trees.data[3 * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR] = Scalar[DT](30)
    var nh = h.run(0)
    t.truth(nh == 1, String("segments = ", nh,
                            " (want 1 — a malformed table must not partition)"))

    # ── non-vacuity ─────────────────────────────────────────────────────
    print("--- the cases were not all the same ---")
    t.truth(na != nf and nd != na and ne != na,
            String("distinct segment counts observed: A=", na, " D=", nd,
                   " E=", ne, " F=", nf))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_newton_blocks: " + String(t.fails) + " failed")

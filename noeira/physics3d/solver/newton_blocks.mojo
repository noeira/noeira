"""The Newton Hessian's DIAGONAL BLOCKS — `H = M + sum D*J^T J`, segmented.

PN2a. This module computes the partition and NOTHING USES IT YET.

⚠⚠ WHY THIS IS NOT JUST `Model.trees`. `M`'s blocks are the kinematic trees and
that is a MODEL-TIME fact (`gpu/constants.MODEL_TREE_SIZE`). `H`'s are not: a
constraint row couples every tree its Jacobian touches, and which rows exist is
a RUNTIME property of the step. A contact between the arm and a prop merges two
trees that `Model.trees` lists apart.

WHAT IT IS WORTH, MEASURED. P0 on `so101_park_k9` (RTX 5090):

    newton  33.3 of 47.6 ms/step   70% of GPU time, 78% of the parked-slot cost
    nv = 60, ncon = 0, nefc = 6 — six FRICTION_DOF rows, all on dofs 0..5
    trees containing a constraint row: [0]  of 10

Nine of the ten trees carry no row at all, so their blocks of `H` are their
blocks of `M` — which P1's classifier already calls COMPACT, i.e. diagonal —
and they are being folded into one dense 60x60 Cholesky. One 6^3 plus nine
diagonals is 270 operations against 216,000.

⚠ SEGMENTS, NOT COMPONENTS, AND THE DIFFERENCE IS DELIBERATE. A connected
component can be non-contiguous in dof space — the arm (tree 0) gripping the
fourth prop (tree 3) is the component {0, 3}. Factoring a non-contiguous index
set needs a permutation, an indirection in the innermost loop, and a second
addressing scheme to get wrong. Instead a row's trees are merged as a SPAN:
`{0, 3}` becomes the segment `trees[0..3]`, which sweeps up trees 1 and 2 as
well. That is:

  * always CONTIGUOUS, so the Cholesky change is a loop bound and nothing else;
  * always a SUPERSET of the true coupling, so it can never drop a nonzero —
    the extra entries it factors are exact zeros;
  * still the whole win where it matters: at k=9 an arm holding one prop gives
    one segment of at most 24 dofs and six untouched blocks of 6, not 60.

⚠ IT USES `Je`'s SPARSITY, NOT THE ROW STATES. A row's state flips between
iterations (`SROW_QUADRATIC` or not), so a partition derived from the ACTIVE
set would have to be rebuilt every iteration and would change under the
factorisation. `Je` is built once, before the loop; keying on it gives one
partition for the whole solve that is a superset of every iteration's coupling.

⚠ A DEGENERATE TABLE MEANS ONE SEGMENT, NEVER ZERO. `ntree == 0` (a `Model`
built without the parser leaves `trees` zeroed) and any table that does not
tile `[0, nv)` exactly both fall back to a single segment spanning every dof —
which is today's behaviour, bit for bit.
"""

from ..fields.scratch import Scratch
from layout import Layout, LayoutTensor
from max.gpu.memory import AddressSpace

from ..gpu.constants import (
    MODEL_TREE_SIZE,
    TREE_IDX_DOF_ADR,
    TREE_IDX_DOF_NUM,
)


@always_inline
def build_dof_segments[
    DTYPE: DType,
    LT: Layout,
    LJ: Layout,
    LS: Layout,
    # ⚠ THE OPERANDS LIVE IN DIFFERENT ADDRESS SPACES. In the blocked kernel
    # `Je` is SHARED or GLOBAL depending on whether it fit (`JE_AS`), the
    # segment arrays are threadgroup memory, and `trees` is a plain model
    # tensor. Defaults keep a CPU caller — and the gate — writing none of this.
    T_AS: AddressSpace = AddressSpace.GENERIC,
    J_AS: AddressSpace = AddressSpace.GENERIC,
    S_AS: AddressSpace = AddressSpace.GENERIC,
](
    nv: Int,
    ntree: Int,
    num_edges: Int,
    trees: LayoutTensor[DTYPE, LT, MutAnyOrigin, address_space=T_AS],
    Je: LayoutTensor[DTYPE, LJ, MutAnyOrigin, address_space=J_AS],
    seg_start: LayoutTensor[DTYPE, LS, MutAnyOrigin, address_space=S_AS],
    seg_end: LayoutTensor[DTYPE, LS, MutAnyOrigin, address_space=S_AS],
) -> Int:
    """Per-dof segment bounds for `H`. Returns the segment count.

    `seg_start[i]` / `seg_end[i]` are the half-open dof range of the segment
    containing dof `i`, so a Cholesky restricts to `[seg_start[j], j)` and
    `[j+1, seg_end[j])` and changes nothing else.

    ⚠ EVERY OPERAND IS FLAT. `trees` is `[t*MODEL_TREE_SIZE + col]` and `Je` is
    `[e*nv + i]` — matching `Je_sh` in the blocked kernel. A 2-D `LayoutTensor`
    given ONE index returns a ROW rather than an element, which is a mismatch
    this tree has already paid for once (`fields/model.mojo`'s `L_CAM` note).
    """
    return build_dof_segments_p[
        DTYPE, T_AS=T_AS, J_AS=J_AS, S_AS=S_AS
    ](nv, ntree, num_edges, trees.ptr, Je.ptr, seg_start.ptr, seg_end.ptr)


@always_inline
def build_dof_segments_p[
    TO: MutOrigin,
    JO: MutOrigin,
    SO: MutOrigin,
    EO: MutOrigin, //,
    DTYPE: DType,
    T_AS: AddressSpace = AddressSpace.GENERIC,
    J_AS: AddressSpace = AddressSpace.GENERIC,
    S_AS: AddressSpace = AddressSpace.GENERIC,
    # `SPARSE`: the caller already holds each row's nonzero dof list
    # (`je_n[e]` entries at `je_ix[e*nv ..]`, ascending) — the CPU Newton
    # does — so a row's tree range is its first and last entry, not a scan
    # of all `nv` (PERFORMANCE.md §13.24: this second scan was half of the
    # Newton's `setup` on dog). Same `lo`/`hi`, same segments, bit-exact.
    SPARSE: Bool = False,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
](
    nv: Int,
    ntree: Int,
    num_edges: Int,
    trees: Pointer[Scalar[DTYPE], TO, address_space=T_AS],
    Je: Pointer[Scalar[DTYPE], JO, address_space=J_AS],
    seg_start: Pointer[Scalar[DTYPE], SO, address_space=S_AS],
    seg_end: Pointer[Scalar[DTYPE], EO, address_space=S_AS],
    je_n: Scratch[Int, N_CAP] = Scratch[Int, N_CAP](1, fill=0),
    je_ix: Scratch[Int, IX_CAP] = Scratch[Int, IX_CAP](1, fill=0),
) -> Int:
    """Pointer form of `build_dof_segments` — THE body; the `LayoutTensor`
    spelling above owns no arithmetic and delegates here.

    It exists so the per-env CPU solver, whose rows live in a `Scratch`, can
    share the one implementation with the blocked kernel, whose rows live in
    threadgroup memory — the same split `chol_solve_seg` / `chol_solve_seg_p`
    already makes, for the same reason: a rule written twice drifts.
    """

    # ⚠ THREE PHASES, ONE BODY EACH (2026-09-26). The blocked kernel runs the
    # middle one on every thread — it is the `num_edges * nv` scan of `Je`,
    # which spills to global memory, and it was ~37 us of every env's setup
    # on so101_tower at 1024 lanes serial on thread 0 — while this driver and
    # the CPU solver run all three in order. Each row's marks are writes of
    # the same value, so the rows can be taken in any order and by any
    # thread.
    var nt = dof_segments_init_p[DTYPE, T_AS=T_AS, S_AS=S_AS](
        nv, ntree, trees, seg_start, seg_end
    )
    if nt <= 0:
        return 1
    dof_segments_mark_p[
        DTYPE, J_AS=J_AS, S_AS=S_AS, SPARSE=SPARSE, N_CAP=N_CAP, IX_CAP=IX_CAP
    ](nv, 0, 1, num_edges, Je, seg_start, seg_end, je_n, je_ix)
    return dof_segments_finish_p[DTYPE, T_AS=T_AS, S_AS=S_AS](
        nt, trees, seg_start, seg_end
    )


@always_inline
def dof_segments_init_p[
    TO: MutOrigin,
    SO: MutOrigin,
    EO: MutOrigin, //,
    DTYPE: DType,
    T_AS: AddressSpace = AddressSpace.GENERIC,
    S_AS: AddressSpace = AddressSpace.GENERIC,
](
    nv: Int,
    ntree: Int,
    trees: Pointer[Scalar[DTYPE], TO, address_space=T_AS],
    seg_start: Pointer[Scalar[DTYPE], SO, address_space=S_AS],
    seg_end: Pointer[Scalar[DTYPE], EO, address_space=S_AS],
) -> Int:
    """Phase 1 of `build_dof_segments_p`: tree id per dof into `seg_start`,
    merge flags cleared in `seg_end`. Returns the tree count, or 0 after
    writing the single-segment fallback (nothing left to do)."""

    @always_inline
    def one_segment() {imm} -> Int:
        for i in range(nv):
            seg_start[unsafe_offset=i] = Scalar[DTYPE](0)
            seg_end[unsafe_offset=i] = Scalar[DTYPE](nv)
        return 0

    if ntree <= 0 or nv <= 0:
        return one_segment()

    # ── tree id per dof, parked in `seg_start` ───────────────────────────
    #
    # ⚠ AND VALIDATED WHILE BUILDING. The table must tile `[0, nv)` exactly:
    # a gap would leave a dof with no tree and an overlap would give it two,
    # and either way a segment bound computed from it is meaningless. Rather
    # than trust it, walk it and fall back on anything unexpected.
    var covered = 0
    var nt = 0
    for t in range(ntree):
        var adr = Int(
            trees[unsafe_offset = t * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR]
        )
        var num = Int(
            trees[unsafe_offset = t * MODEL_TREE_SIZE + TREE_IDX_DOF_NUM]
        )
        # Self-terminating: rows past `ntree` are (0, 0, 0).
        if num <= 0:
            break
        if adr != covered or adr + num > nv:
            return one_segment()
        for i in range(adr, adr + num):
            seg_start[unsafe_offset=i] = Scalar[DTYPE](t)
        covered = adr + num
        nt = t + 1
    if covered != nv or nt <= 0:
        return one_segment()

    # ── merge flags, parked in `seg_end`: does tree t join tree t+1? ──────
    for t in range(nt):
        seg_end[unsafe_offset=t] = Scalar[DTYPE](0)
    return nt


@always_inline
def dof_segments_mark_p[
    JO: MutOrigin,
    SO: MutOrigin,
    EO: MutOrigin, //,
    DTYPE: DType,
    J_AS: AddressSpace = AddressSpace.GENERIC,
    S_AS: AddressSpace = AddressSpace.GENERIC,
    SPARSE: Bool = False,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
](
    nv: Int,
    e_first: Int,
    e_step: Int,
    num_edges: Int,
    Je: Pointer[Scalar[DTYPE], JO, address_space=J_AS],
    seg_start: Pointer[Scalar[DTYPE], SO, address_space=S_AS],
    seg_end: Pointer[Scalar[DTYPE], EO, address_space=S_AS],
    je_n: Scratch[Int, N_CAP] = Scratch[Int, N_CAP](1, fill=0),
    je_ix: Scratch[Int, IX_CAP] = Scratch[Int, IX_CAP](1, fill=0),
):
    """Phase 2: rows `e_first, e_first + e_step, ...` mark the trees they
    couple (`seg_end[t] = 1` for t in [lo, hi)). Every write is the same
    value, so threads may split the rows between them."""
    for e in range(e_first, num_edges, e_step):
        var lo = -1
        var hi = -1
        comptime if SPARSE:
            var n_e = je_n[e]
            if n_e > 0:
                # Ascending list, and `seg_start` is monotone in the dof
                # index, so the first and last entries bound the trees.
                lo = Int(seg_start[unsafe_offset=je_ix[e * nv]])
                hi = Int(seg_start[unsafe_offset=je_ix[e * nv + n_e - 1]])
        else:
            for i in range(nv):
                if Je[unsafe_offset=e * nv + i] != 0:
                    var t = Int(seg_start[unsafe_offset=i])
                    if lo < 0 or t < lo:
                        lo = t
                    if t > hi:
                        hi = t
        # A row that touches nothing couples nothing. Not a defect: a limit
        # row whose Jacobian is a single dof still has lo == hi.
        if lo < 0:
            continue
        for t in range(lo, hi):
            seg_end[unsafe_offset=t] = Scalar[DTYPE](1)



@always_inline
def dof_segments_finish_p[
    TO: MutOrigin,
    SO: MutOrigin,
    EO: MutOrigin, //,
    DTYPE: DType,
    T_AS: AddressSpace = AddressSpace.GENERIC,
    S_AS: AddressSpace = AddressSpace.GENERIC,
](
    nt: Int,
    trees: Pointer[Scalar[DTYPE], TO, address_space=T_AS],
    seg_start: Pointer[Scalar[DTYPE], SO, address_space=S_AS],
    seg_end: Pointer[Scalar[DTYPE], EO, address_space=S_AS],
) -> Int:
    """Phase 3: runs of merged trees become per-dof bounds. Returns the
    segment count."""
    # ── runs of merged trees -> per-dof bounds, WALKED BACKWARDS ─────────
    #
    # ⚠⚠ REVERSE ORDER IS A CORRECTNESS REQUIREMENT, NOT A STYLE CHOICE.
    # `seg_end[0 .. nt)` currently holds the merge flags indexed by TREE,
    # while the writes below are indexed by DOF — and dof indices start at 0
    # too. Forwards, the very first run (trees 0..0, dofs 0..5 on the park
    # scene) writes `seg_end[0..5] = 6` and destroys the flags for trees 1..5
    # before they are read: every later tree then reads `6 != 1` and is
    # silently treated as unmerged. The bug produces a plausible partition —
    # it would even be RIGHT on any scene with no coupling — which is exactly
    # the kind that survives a weak gate.
    #
    # Backwards it cannot happen. A run ending at tree `t1` starts at tree
    # `t0` and writes dofs from `d0` upwards, and every tree holds at least
    # one dof, so `d0 >= t0`. The flags still to be read live at indices
    # `<= t0 - 2`, which is strictly below anything this run writes.
    var nseg = 0
    var t1 = nt - 1
    while t1 >= 0:
        var t0 = t1
        while t0 - 1 >= 0 and Int(seg_end[unsafe_offset=t0 - 1]) == 1:
            t0 -= 1
        var d0 = Int(
            trees[unsafe_offset = t0 * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR]
        )
        var d1 = Int(
            trees[unsafe_offset = t1 * MODEL_TREE_SIZE + TREE_IDX_DOF_ADR]
        ) + Int(
            trees[unsafe_offset=t1 * MODEL_TREE_SIZE + TREE_IDX_DOF_NUM]
        )
        for i in range(d0, d1):
            seg_start[unsafe_offset=i] = Scalar[DTYPE](d0)
            seg_end[unsafe_offset=i] = Scalar[DTYPE](d1)
        nseg += 1
        t1 = t0 - 1
    return nseg

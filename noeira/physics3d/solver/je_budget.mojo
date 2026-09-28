"""Size of the blocked-Newton constraint Jacobian, and whether it must spill.

⚠⚠ ONE SOURCE OF TRUTH, ON PURPOSE. Two places need this number and they MUST
agree exactly: `ContactScratch` allocates the spill buffer, and the blocked
Newton kernel indexes it. If the allocation is a single scalar smaller than
what the kernel writes, the overrun lands in whatever tensor was allocated
next — silent corruption, not a crash. Hence this module rather than the
formula written twice.

WHY A SPILL EXISTS. `Je_sh` is `ME * NV` scalars of THREADGROUP memory and is
usually the largest single array in the blocked kernel. Measured on NVIDIA
(2026-08-10), humanoid_CMU asked for 169,820 B against a 101,376 B limit and
`ptxas` refused to compile the kernel at all. Models past the budget put `Je`
in a dedicated global buffer instead; models under it keep the threadgroup
array and are untouched, bit for bit.

⚠⚠ THE BUDGET IS THE TOTAL, AND WAS `Je` ALONE UNTIL P4. That is the defect
this file existed with: at the k=12 park scene `Je` is 54 KB — comfortably
under the old 64 KB — so it declined to spill, while the three `NV*NV` matrices
put the block at 136,212 B and `ptxas` refused. `Je` is the biggest array on a
high-CONTACT model, which is what this was tuned on; it is not the biggest on a
high-nv LOW-contact one, which is exactly the shape a fixed scene budget
produces. `newton_shared_elems` below counts all eleven arrays.

Measured sizes (float32). ⚠ THE TOTAL IS WHAT DECIDES; `Je` is shown only
because it is the term that used to:

    model            NV   ME    Je      TOTAL    spills?
    quadruped        22  156   13 KB    25 KB    no
    humanoid         27  199   21 KB    37 KB    no
    quadruped_fetch  28  340   37 KB    59 KB    no   (at its real condim 6)
    humanoid_CMU     62  432  105 KB   166 KB    YES
    dog              79  491  152 KB   244 KB    YES
    dog_fetch        85  539  179 KB   285 KB    YES
    so101_park k=9   60  154   36 KB    85 KB    no   <- the ceiling before P4
    so101_park k=10  66  162   42 KB   100 KB    YES  <- would NOT COMPILE before
    so101_park k=12  78  178   54 KB   134 KB    YES  <- would NOT COMPILE before

None of the first six changes its answer under the new rule — gated in
`tests/physics3d/test_newton_shared_budget.mojo`, arm E, because widening a
budget can silently start spilling models that ran fine, and a spilled `Je` is
re-read from global on every Newton iteration.

⚠ SPILLING REACHES k=13, NOT FURTHER. Past that the three `NV*NV` arrays are
the binding term and the fix is a different one (move the Hessian to global, as
mujoco_warp does, or pack the triple by block).
"""

from std.sys.info import size_of

from ..types import ConeType
from ..constraints.elliptic_layout import ell_nt


# ⚠⚠ THE LAUNCH-SHAPE KNOB — F3 step 2, and it is an OCCUPANCY experiment, not
# a work-division one. The blocked kernel holds ~88 KB of shared memory per
# block against a per-SM budget of ~99-228 KB, so exactly ONE block is resident
# per SM. At `MAX_CONTACTS = 16` that is 16 active threads per SM out of 2048 —
# 0.8% occupancy, against the ~25-50% a GPU needs to hide memory latency. This
# multiplier does NOT change shared memory, so blocks/SM stays 1 while
# threads/SM goes 16 -> 32 -> 64 -> 128.
#
# ⚠ `NEWTON_COOP_DIV` PRICED A DIFFERENT MECHANISM AND ITS ANSWER STANDS. It
# varied the cooperative STRIDE at a fixed thread count, so it measured how much
# work DIVIDES (2.25 ms) and nothing about how many warps are resident. The two
# are independent; do not quote that ceiling against this.
#
# ⚠ THE RISK THIS SWEEP EXISTS TO MEASURE is local memory, and it is the thing
# this kernel was built to avoid — see its header. About fifteen per-thread
# `Scratch` arrays of `NV` (qacc, Ma, f_smooth, jar, search, Mv, the five
# `old_*`, ...) are reserved by EVERY thread even though only tid 0 uses most of
# them: ~5 KB/thread at NV=84, so 8x threads is ~645 KB per block instead of
# ~80 KB. If that spills hard enough it cancels the occupancy win, and the sweep
# will show it as a curve that turns over rather than one that keeps improving.
#
# MEASURED, k=13 park scene (NV=84, MC=16), newton excess over k=0, controls
# flat at 1.000-1.001 across all four legs:
#
#     threads  occupancy   newton excess   speedup   env-steps/s
#        16      0.78%       20.258 ms      1.000       30,026
#        32      1.56%       18.720         1.082       31,433
#        64      3.12%       16.281         1.244       33,861   <- best
#       128      6.25%       17.303         1.171       32,719   <- turns over
#
# ⚠⚠ AND THE TURNOVER IS THE LOCAL-MEMORY TAX THIS SWEEP EXISTED TO FIND. At
# k=0 128 threads is 0.70x — 43% SLOWER — where there is no work to amortise
# the ~5 KB/thread of per-thread `Scratch` (645 KB/block at 128 vs 80 KB at
# 16). The tax scales with THREADS * NV, so it bites hardest on wide models,
# which are exactly the ones the extra warps were bought for.
#
# Decomposing against `NEWTON_COOP_DIV`'s 2.25 ms of divisible work: at 64
# threads work-division explains 1.688 ms of the 3.977 saved and the residual
# 2.289 is occupancy. That residual is BIGGER than the mechanism COOP_DIV
# priced, which is why reopening F3 on the occupancy argument was right and why
# its 2.25 ms ceiling was never the whole story.
#
# ⚠ A FLOOR, NOT A MULTIPLIER, and that is a deliberate change from what was
# swept. A `4x` multiplier reproduces 64 threads at MC=16 but would give 128 at
# `so101_tabletop`'s MC=32 — squarely in the regressing region. An absolute
# floor gives every model the thread count that was actually measured good, and
# never more.
#
# ⚠ ONLY MC=16 / NV=84 WAS SWEPT. 64 is the measured optimum there and a
# defensible default elsewhere because it is LOWER than a multiplier would give;
# it is not known to be optimal for another shape.
comptime NEWTON_THREADS_FLOOR: Int = 64

# ⚠⚠ AND 64 IS WRONG FOR A SMALL BLOCK — ONE WARP WINS WHEN SHARED MEMORY IS
# NOT WHAT BINDS. The sweep above ran at 1 block per SM, where the SM was
# SHARED-bound and extra threads were free warps. A small block is the other
# regime: at 130-170 registers a 64-thread block is REGISTER-bound at 6 blocks
# per SM on the RTX 5090 (`cuOccupancyMaxActiveBlocksPerMultiprocessor`), and
# since the block IS one env, every thread it does not need is an env that
# cannot be resident. Halving it to one warp buys residency exactly when shared
# memory admits a 7th block, and nothing otherwise.
#
# MEASURED 2026-09-25, RTX 5090, Newton launch at 32 threads / 64 threads.
# `bytes` is `newton_shared_elems` (the compiled block carries ~1 KB more):
#
#     model            NV  MC   bytes   blocks/SM 32 : 64   newton 32/64
#     park k=0          6  16    5244        12 : 6            0.617
#     half_cheetah      9  20    8292        10 : 6            0.77  (4096 lanes)
#     park k=3         24  16    6808        12 : 6            0.758
#     park k=6         42  16   12928         7 : 6            0.759
#     park k=9         60  16   21640         4 : 4            1.020
#     park k=13        84  16   37288         2 : 2            1.119
#
#     (park at 1024 lanes, `examples/so101/so101_park_attrib_probe.mojo`)
#
# The win at k=6 is ONE block per SM: 7 x 170 = 1190 envs per wave against
# 6 x 170 = 1020, so 1024 lanes run in one wave instead of two. Where both
# sizes hold the same blocks (k=9, k=13) the half block only loses cooperative
# workers. HalfCheetah's obs checksums are BIT-IDENTICAL at 32 and 64 at every
# batch from 1 to 4096; the cooperative phases partition work, they do not
# reorder sums.
#
# ⚠ THE BOUNDARY IS THE 5090'S, AND IT IS CONSERVATIVE ELSEWHERE. 13 KB is
# where a 7th block stops fitting in 100 KB of shared memory per SM. An SM
# with more (A100/Orin 164 KB, H100 228 KB) would admit the extra block at
# larger footprints, so this under-uses one warp there, never over-uses it.
# ⚠ At a batch that fits in one wave either way, one warp is ~2% slower
# (HalfCheetah N=1: 52.8 vs 51.9 µs) — the rule does not see the batch.
comptime NEWTON_THREADS_FLOOR_SMALL_BLOCK: Int = 32
comptime NEWTON_SMALL_BLOCK_MAX_BYTES: Int = 13 * 1024


def newton_block_threads[
    DTYPE: DType,
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
    CONE_TYPE: Int,
]() -> Int:
    """Threads per block for `_newton_blocked_fields_kernel`.

    One warp when the block's threadgroup footprint is small enough that an
    SM can hold more one-warp blocks than register-bound 64-thread ones
    (`NEWTON_SMALL_BLOCK_MAX_BYTES`), else `NEWTON_THREADS_FLOOR`; never below
    `MAX_CONTACTS`. The footprint is `newton_shared_elems` with the kernel's
    own spill decision, so the rule sees the same block the kernel allocates.

    ⚠⚠ ONE SOURCE FOR TWO PLACES THAT MUST NOT DISAGREE — the kernel's
    cooperative stride (`comptime THREADS`) and the launch's `block_dim`. They
    were two independent spellings of `_max_one[MAX_CONTACTS]()`, which is
    exactly the shape of `_a_rule_written_inline_twice_drifts`: numerically
    equal today, and a silent out-of-range thread the moment one moves.

    ⚠ IT MUST NEVER RETURN LESS THAN `MAX_CONTACTS`. The contact phases map one
    slot to one thread, so a smaller block leaves the tail slots
    UNINITIALISED — `_init_common_normal_ws` never runs for them and the
    workspace keeps the previous step's values. More is safe (every such phase
    is now guarded `< MC` or `< nc`); fewer is a wrong answer.

    ⚠ NEVER LESS THAN `MAX_CONTACTS` — see the note on the contact phases above
    — so the floor only ever raises the count, never lowers it.
    """
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime JE_IN_SHARED = not je_spills[
        DTYPE, NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM,
        CONE_TYPE,
    ]()
    comptime BYTES = newton_shared_elems[
        NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM,
        JE_IN_SHARED, CONE_TYPE,
    ]() * size_of[Scalar[DTYPE]]()
    comptime FLOOR = (
        NEWTON_THREADS_FLOOR_SMALL_BLOCK if BYTES
        <= NEWTON_SMALL_BLOCK_MAX_BYTES else NEWTON_THREADS_FLOOR
    )
    return MC if MC > FLOOR else FLOOR


# ⚠⚠ TWO NUMBERS, AND THEY WERE ONE UNTIL 2026-09-07. The LIMIT is what
# `ptxas` accepts per block; the BUDGET is the footprint above which `Je`
# leaves threadgroup memory. They were the same constant, so `Je` spilled
# only when the block would otherwise not COMPILE — which optimised for the
# wrong thing, because a block's footprint decides how many blocks an SM
# holds, and this kernel is thread-0 LATENCY-bound: it needs co-resident
# blocks to overlap, not a full threadgroup.
#
# THE LIMIT, an NVIDIA number on purpose (`solve_newton` routes PYRAMIDAL +
# NVIDIA here and everything else to the per-env kernel, which never reads
# this file). 0x18c00 is what `ptxas` itself reports on an RTX 5090:
#
#     ptxas error : Entry function 'noeira_physics3d_solver_newt...' uses
#                   too much shared data (0x21414 bytes, 0x18c00 max)
comptime SOLVER_SHARED_LIMIT: Int = 0x18C00

# THE BUDGET — the spill POLICY, and it is a measurement, not a portability
# figure. RTX 5090, the parked-slot probe (BLOCK_DIAGONAL_..., 2026-09-07,
# "Experiment 3"), Newton µs per launch with `Je` in threadgroup memory
# against `Je` spilled to its per-env global buffer, every other kernel
# at 1.00 and the answer bit-identical:
#
#     k   nv   footprint with Je     spilled     ratio
#     3   24        22.5 KB           157/204    1.30x faster
#     6   42        50.2 KB           508/556    1.09x
#     9   60        90.1 KB          1282/1308   1.02x
#    12   78   (already spilled: 84 KB was over the LIMIT)
#
# A spilled `Je` is re-read from global on every Newton iteration and that
# was the reason not to spill; the reads turn out to cost less than the
# blocks-per-SM the array was buying, at every k measured. 16 KB keeps the
# k=0 scene (6 KB) and nothing else in threadgroup memory. ⚠ The ledger's
# earlier "no spill penalty at k=12/13" was the same fact seen from above.
#
# ⚠ THE OLD 64 KB WAS INCOHERENT, WHICH IS WHY IT IS GONE. It was justified as
# "the widely-supported opt-in floor", so that a model fitting everywhere kept
# the fast path — but it was compared against `Je` ALONE while the kernel's
# TOTAL was already 87 KB at k=9 and compiling fine. A budget that guards one
# array against a portability figure the whole block has already blown is not
# protecting portability; it is just failing to predict `ptxas`.
comptime SOLVER_SHARED_BUDGET: Int = 0x4000


def _max_one[N: Int]() -> Int:
    """`max(N, 1)` — a zero-sized dimension is a crash, not an empty tensor."""
    return N if N > 0 else 1


def je_edge_rows[
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
]() -> Int:
    """`ME` — the blocked solver's constraint-row count.

    ⚠ MUST MATCH `newton_solve.solve_newton_blocked`'s `ME` EXACTLY. The terms,
    in the order that file derives them:

        NE       = 2*(MAX_CONDIM-1)   pyramidal edges per contact
        MAX_LIM  = max(1, 2*NJOINT)   joint limits (lo + hi)
        MAX_FRIC = max(1, NV)         one dry-friction row per dof
        MAX_TLIM = 2*NTENDON          tendon limits (lo + hi)
        MAX_TEQ  = NTENDON            one bilateral row per equality tendon
        MAX_WELD = 6*NEQUALITY        connect (3) / weld (6) rows

    ⚠ The friction and tendon terms were MISSING from the blocked path until
    2026-07-31, so a model with `frictionloss` or a limited tendon silently had
    no such rows. `MAX_WELD` arrived 2026-08-12 with the defect-29a conversion
    of connect/weld from a post-pass into rows. Growing this function grows the
    spill buffer with it — that is the point of routing both through here.

    ⚠ NEQUALITY IS A PARAMETER, not a term folded into another. It was
    tempting to reuse NTENDON's slot since both are "equality" counts; they are
    different models' dimensions and a model can have either without the other.
    """
    return (
        2 * (MAX_CONDIM - 1) * _max_one[MAX_CONTACTS]()
        + _max_one[2 * NJOINT]()
        + _max_one[NV]()
        + 3 * NTENDON
        + 6 * NEQUALITY
    )


def je_elems[
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
]() -> Int:
    """Scalars in `Je` for ONE env: `ME * V_SIZE`."""
    return (
        je_edge_rows[
            NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
        ]()
        * _max_one[NV]()
    )


def newton_elliptic_extra_elems[MAX_CONTACTS: Int, MAX_CONDIM: Int]() -> Int:
    """Scalars of THREADGROUP memory the blocked kernel's ELLIPTIC leg adds
    on top of the pyramidal list (2026-09-15), in the order it declares them.

        fr_e_sh                   ME            `con->friction[t]` per contact row
        mu_sh/ntc_sh/cact_sh/cs_sh 4 * MC       per-contact cone data and state
        cmask_sh                  MC            per-contact dof mask (2026-09-26)
        hb_sh                     MC * (NT+1)^2  the cone Hessian block per contact

    ⚠ ZERO FOR THE PYRAMIDAL CONE — the kernel sizes these arrays at 1 there
    (the same `1 when spilled` trick `Je_sh` uses), so every pyramidal
    footprint `test_newton_shared_budget` pins against `ptxas` is unchanged.
    """
    comptime MC = _max_one[MAX_CONTACTS]()
    comptime HN = (ell_nt[MAX_CONDIM]() + 1) * (ell_nt[MAX_CONDIM]() + 1)
    return 5 * MC + MC * HN


def newton_shared_elems[
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
    JE_IN_SHARED: Bool,
    CONE_TYPE: Int = ConeType.PYRAMIDAL,
]() -> Int:
    """Scalars of THREADGROUP memory `_newton_blocked_fields_kernel` asks for.

    ⚠⚠ MUST MATCH THE KERNEL'S `stack_allocation()` LIST EXACTLY — this is the
    same one-source-of-truth contract `je_elems` has with `ME`, and it is the
    thing the old budget got wrong by counting a single array. In the order the
    kernel declares them (`newton_solve.mojo:3640+`):

        L_sh (H built in, in place)   1 * max(1, NV*NV)   (stage 1; was 3)
        seg0_sh, seg1_sh          2 * max(1, NV)      (PN2c)
        grad_sh                   1 * max(1, NV)      (F3b)
        Je_sh                     ME * max(1, NV), or 1 when spilled
        De/bias_e/force/kind_e/
        R_e/floss_e/state_e/
        Jv_e/jar                  9 * ME
        search/Mv/qacc/qfrc       4 * max(1, NV)
        ctrl_sh                   3
        + the ELLIPTIC leg's extras (`newton_elliptic_extra_elems`, and one
          more `ME` for `fr_e_sh`) when `CONE_TYPE` is ELLIPTIC — 0 otherwise

    ⚠ VERIFIED AGAINST `ptxas` ON FOUR POINTS, not derived and hoped for — see
    `tests/physics3d/test_newton_shared_budget.mojo`, which pins it to the byte
    counts the k=6/9/10/12 park scenes produced.
    """
    return (
        # ONE dense array since stage 1 (2026-09-07): `L_sh` holds the Hessian
        # and is factored in place; `M_sh` and `H_sh` are gone.
        1 * _max_one[NV * NV]()
        # 2 seg + 1 grad + search/Mv/qacc/qfrc
        + 7 * _max_one[NV]()
        + (
            je_elems[
                NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
            ]() if JE_IN_SHARED else 1
        )
        + 9
        * je_edge_rows[
            NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
        ]()
        + 3
        + (
            (
                je_edge_rows[
                    NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
                ]()
                + newton_elliptic_extra_elems[MAX_CONTACTS, MAX_CONDIM]()
            ) if CONE_TYPE == ConeType.ELLIPTIC else 0
        )
    )


def je_spills[
    DTYPE: DType,
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
    CONE_TYPE: Int = ConeType.PYRAMIDAL,
]() -> Bool:
    """Does the kernel's TOTAL threadgroup footprint force `Je` out?

    ⚠⚠ THE TOTAL, NOT `Je`. This compared `Je` alone against 64 KB until P4,
    and the failure mode is on record: at the k=12 park scene `Je` is 54 KB —
    comfortably under — so it declined to spill, while the three `NV*NV`
    matrices put the block at 136,212 B against a 101,376 B limit and `ptxas`
    refused to compile the kernel at all. Budgeting one array out of eleven
    cannot predict that, and the models it WAS tuned on (humanoid_CMU, dog)
    hid it because they are high-nv AND high-contact, so `Je` dominated. A
    fixed scene budget produces the shape it was never tuned for: high nv, LOW
    contact count.
    """
    return (
        newton_shared_elems[
            NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM, True,
            CONE_TYPE,
        ]()
        * size_of[Scalar[DTYPE]]()
    ) > SOLVER_SHARED_BUDGET


def je_ws_size[
    DTYPE: DType,
    NV: Int,
    NJOINT: Int,
    NTENDON: Int,
    NEQUALITY: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
    # ⚠ THE CONE IS PART OF THE FOOTPRINT since the ELLIPTIC leg landed in
    # the blocked kernel (2026-09-15): its extra shared arrays can tip a
    # model over the budget. Every integrator passes its `CONE_TYPE`; the
    # default keeps pyramidal callers and their pins exactly as they were.
    CONE_TYPE: Int = ConeType.PYRAMIDAL,
]() -> Int:
    """Per-env spill-buffer size: `ME*NV` when spilling, else 0.

    `ContactScratch` allocates `BATCH * max(this, 1)`; a model that does not
    spill pays one scalar per env, not a buffer.
    """
    comptime if je_spills[
        DTYPE, NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM,
        CONE_TYPE,
    ]():
        return je_elems[
            NV, NJOINT, NTENDON, NEQUALITY, MAX_CONTACTS, MAX_CONDIM
        ]()
    return 0

"""`je_spills` budgets the WHOLE threadgroup footprint. P4's gate.

WHY THIS EXISTS
===============
`je_spills` compared **`Je` alone** against a 64 KB constant. At the k=12 park
scene `Je` is 54 KB — comfortably under — so it declined to spill, while the
kernel's three `NV*NV` matrices put the block at 136,212 B against a 101,376 B
limit and `ptxas` refused to compile it:

    ptxas error : Entry function 'noeira_physics3d_solver_newt...' uses
                  too much shared data (0x21414 bytes, 0x18c00 max)

Budgeting one array out of eleven cannot predict that. The models it WAS tuned
on — humanoid_CMU, dog — hid it, because they are high-nv AND high-contact so
`Je` dominates the total. A fixed scene budget produces the shape it was never
tuned for: high nv, LOW contact count.

⚠⚠ THE FORMULA IS PINNED TO `ptxas`, NOT TO ITSELF. `newton_shared_elems` is a
transcription of the kernel's `stack_allocation()` list, and a transcription
that drifts is the whole failure being fixed here. Arm A checks it against the
four byte counts `ptxas` actually printed for the k=6/9/10/12 park scenes —
48,372 / 86,676 / 101,940 / 136,212 — which is an EXTERNAL oracle, obtained
before any of this code existed.

⚠ THOSE FOUR NUMBERS PREDATE PN2c, which added `seg0_sh`/`seg1_sh` — `2 * NV`
scalars — and F3b, which added `grad_sh` — `1 * NV`. So the expectation is
`recorded + 12*NV` bytes, and the test spells the delta out rather than folding
it in: if someone adds another shared array, this arm fails and names the size,
instead of the model failing to compile later. (It did exactly that for
`grad_sh`.)

Run: pixi run mojo run -I . tests/physics3d/test_newton_shared_budget.mojo
"""

from std.sys.info import size_of
from noeira.physics3d.solver.je_budget import (
    newton_shared_elems, je_spills, je_elems, SOLVER_SHARED_BUDGET,
    SOLVER_SHARED_LIMIT, newton_block_threads,
)
from noeira.physics3d.types import ConeType

comptime DT = DType.float32          # the park probe's dtype
comptime MC = 16                     # PARK_MAX_CONTACTS
comptime CONDIM = 3


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


def _bytes[NV: Int, NJOINT: Int]() -> Int:
    return newton_shared_elems[
        NV, NJOINT, 0, 0, MC, CONDIM, True
    ]() * size_of[Scalar[DT]]()


def _je_bytes[NV: Int, NJOINT: Int]() -> Int:
    return je_elems[NV, NJOINT, 0, 0, MC, CONDIM]() * size_of[Scalar[DT]]()


def main() raises:
    var t = Tally()
    print("=== the Newton kernel's shared footprint (P4) ===")
    print("  limit:", SOLVER_SHARED_LIMIT, "B (ptxas 0x18c00 on an RTX 5090)  spill budget:", SOLVER_SHARED_BUDGET, "B")

    # ── A: the formula reproduces ptxas, to the byte ─────────────────────
    # (k, nv, njoint, the bytes ptxas reported BEFORE PN2c's two seg arrays)
    print("--- A: vs the four byte counts ptxas printed ---")
    # PN2c's seg0/seg1 (2*NV) plus F3b's grad_sh (1*NV) = 3*NV scalars =
    # 12*NV bytes. Spelled out, not folded in, so the NEXT array to arrive
    # fails this arm and names its size instead of failing `ptxas` later.
    # Stage 1 (2026-09-07) took TWO of the three NV*NV arrays away — `M_sh`
    # (the matvecs read global M) and `H_sh` (the Hessian is factored in
    # place in `L_sh`) — so the same recorded counts now carry `- 2*NV*NV*4`.
    # ⚠ The recorded numbers stay the pre-stage-1 ptxas print; the
    # post-stage-1 print is the box's to confirm (`p0_kernel_shape.py`).
    var seg_delta_42 = 12 * 42 - 2 * 42 * 42 * 4
    var seg_delta_60 = 12 * 60 - 2 * 60 * 60 * 4
    var seg_delta_66 = 12 * 66 - 2 * 66 * 66 * 4
    var seg_delta_78 = 12 * 78 - 2 * 78 * 78 * 4
    t.truth(_bytes[42, 12]() == 48372 + seg_delta_42,
            String("k=6  nv=42: ", _bytes[42, 12](), " == 48372 + ",
                   seg_delta_42, " (ptxas + PN2c's 2*NV + F3b's grad_sh"
                   " - stage 1's two NV*NV arrays)"))
    t.truth(_bytes[60, 15]() == 86676 + seg_delta_60,
            String("k=9  nv=60: ", _bytes[60, 15](), " == 86676 + ",
                   seg_delta_60))
    t.truth(_bytes[66, 16]() == 101940 + seg_delta_66,
            String("k=10 nv=66: ", _bytes[66, 16](), " == 101940 + ",
                   seg_delta_66))
    t.truth(_bytes[78, 18]() == 136212 + seg_delta_78,
            String("k=12 nv=78: ", _bytes[78, 18](), " == 136212 + ",
                   seg_delta_78))

    # ── B: the decision flips where ptxas does ───────────────────────────
    # ⚠ THE POLICY FLIPPED ON 2026-09-07: the budget is 16 KB, an
    # OCCUPANCY figure measured on the RTX 5090 (je_budget's table: Newton
    # 1.30x / 1.09x / 1.02x faster at k=3/6/9 with `Je` spilled), no longer
    # the ptxas limit. Only the k=0 scene keeps `Je` in threadgroup memory.
    print("--- B: k=0 keeps threadgroup Je, every wider leg spills ---")
    t.truth(not je_spills[DT, 6, 6, 0, 0, MC, CONDIM](),
            "k=0  does NOT spill (6 KB, under the 16 KB budget)")
    t.truth(je_spills[DT, 24, 9, 0, 0, MC, CONDIM](),
            "k=3  SPILLS (22.5 KB over the budget) — measured 1.30x faster")
    t.truth(je_spills[DT, 42, 12, 0, 0, MC, CONDIM](),
            "k=6  SPILLS (48,708 B) — measured 1.09x faster")
    t.truth(je_spills[DT, 60, 15, 0, 0, MC, CONDIM](),
            "k=9  SPILLS (87,156 B) — measured 1.02x faster")
    t.truth(je_spills[DT, 66, 16, 0, 0, MC, CONDIM](),
            "k=10 SPILLS (102,468 B over the LIMIT) — was a COMPILE FAILURE")
    t.truth(je_spills[DT, 78, 18, 0, 0, MC, CONDIM](),
            "k=12 SPILLS (136,836 B over the LIMIT) — was a COMPILE FAILURE")

    # ── C: ⚠ THE OLD RULE WOULD HAVE GOT B WRONG. Without this the gate
    # only says the new code agrees with itself.
    print("--- C: the old `Je`-alone rule, on the same models ---")
    var je10 = _je_bytes[66, 16]()
    var je12 = _je_bytes[78, 18]()
    t.truth(je10 <= 64 * 1024,
            String("k=10 Je alone = ", je10, " B — UNDER the old 64 KB, so the"
                   " old rule did not spill and the kernel did not compile"))
    t.truth(je12 <= 64 * 1024,
            String("k=12 Je alone = ", je12, " B — likewise"))

    # ── D: after spilling, the block actually fits ───────────────────────
    print("--- D: spilling is enough — the rest fits ---")
    var r10 = newton_shared_elems[66, 16, 0, 0, MC, CONDIM, False]() * 4
    var r12 = newton_shared_elems[78, 18, 0, 0, MC, CONDIM, False]() * 4
    t.truth(r10 <= SOLVER_SHARED_LIMIT,
            String("k=10 with Je spilled: ", r10, " B fits the LIMIT"))
    t.truth(r12 <= SOLVER_SHARED_LIMIT,
            String("k=12 with Je spilled: ", r12, " B fits the LIMIT"))
    # ⚠ AND WHERE IT STOPS BEING ENOUGH, so nobody reads "P4 unblocks k" as
    # unbounded. Past this the ONE remaining NV*NV array (`L_sh`) is the
    # binding term. Before stage 1 the three of them stopped the reach at
    # k=13 (k=14 was 74,320 B over 0x18C00); with one, k=14 fits and the
    # ceiling moves out to about k=24.
    var r14 = newton_shared_elems[90, 20, 0, 0, MC, CONDIM, False]() * 4
    t.truth(r14 <= SOLVER_SHARED_LIMIT,
            String("k=14 with Je spilled: ", r14, " B fits the LIMIT —"
                   " stage 1 moved the ceiling past k=13"))
    var r25 = newton_shared_elems[156, 31, 0, 0, MC, CONDIM, False]() * 4
    t.truth(r25 > SOLVER_SHARED_LIMIT,
            String("k=25 with Je spilled: ", r25, " B still OVER the LIMIT —"
                   " the reach is bounded by `L_sh` now"))

    # ── E: ⚠ NO SHIPPED MODEL CHANGES ITS MIND. Widening the budget from
    # "Je vs 64 KB" to "the total vs the device limit" could easily have made
    # models that run today start spilling — a straight perf regression, since
    # a spilled `Je` is re-read from global across every Newton iteration.
    # The six models `je_budget`'s own table records must keep their answer.
    # ⚠ E USED TO PIN "no shipped model changes its mind", because a spilled
    # `Je` was assumed to be a perf regression. It is the opposite at every
    # k measured (je_budget's table), so the pin now says what the policy
    # says: every model over 16 KB spills, and the spilled path is what the
    # golden fingerprint and the free-joint oracle run on Metal.
    print("--- E: the six models in je_budget's table follow the policy ---")
    t.truth(je_spills[DT, 22, 78, 0, 0, 16, CONDIM](),
            "quadruped       (nv 22, 25 KB) spills under the 16 KB budget")
    t.truth(je_spills[DT, 27, 96, 0, 0, 32, CONDIM](),
            "humanoid        (nv 27, 37 KB) spills")
    t.truth(je_spills[DT, 28, 156, 0, 0, 24, 6](),
            "quadruped_fetch (nv 28, 59 KB) spills")
    t.truth(je_spills[DT, 62, 185, 0, 0, 64, CONDIM](),
            "humanoid_CMU    (nv 62) still SPILLS")
    t.truth(je_spills[DT, 79, 206, 0, 0, 24, CONDIM](),
            "dog             (nv 79) still SPILLS")
    t.truth(je_spills[DT, 85, 227, 0, 0, 28, CONDIM](),
            "dog_fetch       (nv 85) still SPILLS")

    # ── F: the ELLIPTIC leg's footprint, vs the PTX (2026-09-15) ─────────
    # The blocked kernel's elliptic leg adds `fr_e_sh` (ME), four MC arrays
    # and the cone Hessian blocks (`MC * (NT+1)^2`). Pinned to an EXTERNAL
    # oracle the same way arm A is: `mojo build --target-accelerator sm_120
    # --emit asm` of `examples/libero/libero_demo_batched.mojo` (LIBERO goal:
    # nv 37, njoint 17, `LIBERO_GOAL_MAX_CONTACTS` 144, the model def's
    # default `max_condim` 3, no tendon or equality) declared 24 `.shared`
    # arrays totalling 39,892 B for the elliptic kernel. The formula counts
    # one scalar more: the spilled `Je`'s one-element threadgroup backing,
    # which nothing reads once `Je` lives in global memory and which the
    # PTX therefore does not declare. Same dims, PYRAMIDAL: the extras are
    # exactly zero, which is what keeps arms A-E as they were.
    print("--- F: the elliptic leg's extras, vs the LIBERO PTX ---")
    var ell_libero = newton_shared_elems[
        37, 17, 0, 0, 144, 3, False, ConeType.ELLIPTIC
    ]() * size_of[Scalar[DT]]()
    var pyr_libero = newton_shared_elems[
        37, 17, 0, 0, 144, 3, False, ConeType.PYRAMIDAL
    ]() * size_of[Scalar[DT]]()
    # ⚠ `cmask_sh` (MC scalars, 2026-09-26) arrived after that PTX was read:
    # it is spelled out as `+ 144 * 4` rather than folded into the recorded
    # number, the same way arm A carries its later arrays.
    t.truth(ell_libero == 39892 + 4 + 144 * 4,
            String("LIBERO goal, ELLIPTIC: ", ell_libero, " == 39892 (the"
                   " sm_120 PTX's 24 .shared arrays) + 4 (the elided Je"
                   " backing scalar) + 576 (cmask_sh)"))
    t.truth(ell_libero - pyr_libero
            == (10 * 647 - 9 * 647 + 5 * 144 + 144 * 9) * size_of[Scalar[DT]](),
            String("the elliptic extras are fr_e_sh (ME=647) + 5*MC +"
                   " MC*(NT+1)^2 = ", ell_libero - pyr_libero, " B"))
    t.truth(newton_shared_elems[42, 12, 0, 0, MC, CONDIM, True]()
            == newton_shared_elems[
                42, 12, 0, 0, MC, CONDIM, True, ConeType.PYRAMIDAL
            ](),
            "the CONE_TYPE default is PYRAMIDAL, so every pin above reads"
            " the same formula it always did")

    # ── G: threads per block, per shape (2026-09-25) ────────────────────
    # One warp where the block is small enough that an SM holds more of them
    # than register-bound 64-thread blocks; 64 where shared memory binds both
    # the same. Every row is a shape measured on the RTX 5090 (the table in
    # `je_budget.NEWTON_SMALL_BLOCK_MAX_BYTES`): 32 won at the first four,
    # tied or lost at the last two. A rule that moves the boundary across a
    # measured row fails here and names it.
    print("--- G: newton_block_threads, the measured shapes ---")
    comptime PYR = ConeType.PYRAMIDAL
    var hc_bytes = newton_shared_elems[9, 9, 0, 0, 20, 3, True]() * size_of[
        Scalar[DT]
    ]()
    t.truth(hc_bytes == 8292,
            String("half_cheetah footprint ", hc_bytes, " == 8292 (the"
                   " compiled kernel's 9316 minus the uncounted 1 KB)"))
    t.truth(newton_block_threads[DT, 9, 9, 0, 0, 20, 3, PYR]() == 32,
            "half_cheetah (nv 9, MC 20, 8292 B) -> 32")
    t.truth(newton_block_threads[DT, 6, 6, 0, 0, MC, CONDIM, PYR]() == 32,
            "park k=0 (nv 6) -> 32")
    t.truth(newton_block_threads[DT, 24, 9, 0, 0, MC, CONDIM, PYR]() == 32,
            "park k=3 (nv 24, Je spilled) -> 32")
    t.truth(newton_block_threads[DT, 42, 12, 0, 0, MC, CONDIM, PYR]() == 32,
            "park k=6 (nv 42, 12928 B) -> 32")
    t.truth(newton_block_threads[DT, 60, 15, 0, 0, MC, CONDIM, PYR]() == 64,
            "park k=9 (nv 60, 21640 B) -> 64")
    t.truth(newton_block_threads[DT, 84, 19, 0, 0, MC, CONDIM, PYR]() == 64,
            "park k=13 (nv 84) -> 64")
    # Never below MAX_CONTACTS: LIBERO goal's 144 contacts set the block.
    t.truth(newton_block_threads[
                DT, 37, 17, 0, 0, 144, 3, ConeType.ELLIPTIC
            ]() == 144,
            "LIBERO goal (MC 144, elliptic) -> 144, MAX_CONTACTS wins")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_newton_shared_budget: " + String(t.fails) + " failed"
        )

"""The FB losses, as pure functions of activations.

Both losses take `[BATCH, D]` activation buffers and write gradients back onto
them. Nothing here knows about networks, optimizers or replay — the trainer
forwards its nets, calls these, and backpropagates the returned gradients. That
split exists so the mathematically delicate part is testable on its own, which
`docs/BFM_ZERO_SHOT_RL.md` §11 asks for by name.

## The measure loss

    L_FB = E_ij[ ( F(s_i,a_i,z_i)·B(s+_j) - gamma·Fbar(s'_i,a'_i,z_i)·Bbar(s+_j) )^2 ]
           - 2·E_i[ F(s_i,a_i,z_i)·B(s'_i) ]

⚠⚠ **The second term is not a regulariser.** It is what is left when the square
of the successor measure is expanded, and it is the only term that ties `F` to
the immediate transition. Drop it and `L_FB` still descends — the first term
alone is minimised by driving `F·B` towards its own bootstrapped target, which
zero satisfies perfectly. The model then "runs" and encodes nothing.
`test_fb_loss.mojo` pins this by ablation rather than by comment: it checks that
removing the anchor admits a collapsed solution that the full loss rejects.

Note the shapes. The first term pairs EVERY i with EVERY j — that is the
`[BATCH, BATCH]` successor-measure matrix, and why `PairwiseDot` exists. The
second pairs i with i only, so it uses `RowDot`: taking the diagonal of a
`PairwiseDot` would compute BATCH² dot products to keep BATCH.

⚠ THIS PARAGRAPH USED TO ARGUE THE OPPOSITE, AND WAS WRONG (§12.17). It said
`s+` must be a SECOND, INDEPENDENT draw because "if `s+` were the batch's own
next-states, the matrix would only ever be evaluated on pairs one step apart".
It would not: for `i != j`, `B(s'_j)` is an arbitrary state from the batch, not
one step from `s_i`. The objection applies only to the DIAGONAL, which is
exactly the entry the reference splits out of the square and gives the linear
anchor. So there is ONE `B` tensor, `B(goal)` with `goal = next_obs`
(`agent.py:190`), used for the matrix, its diagonal and the ortho alike.

The two-tensor form ran `anchor/quad` at 4.04 where the reference holds 0.41
flat for 200 M steps — a ~10x balance error, measured at the same timestep.

## The orthonormality regulariser

    L_ortho = 0.5·E_{i!=j}[ (B(s+_i)·B(s+_j))^2 ] - E_i[ ||B(s+_i)||^2 ]

ONE batch, against ITSELF. Both indices range over the SAME tensor, and that is
load-bearing — see the warning below.

Pushes `E[B B^T]` towards the identity. Without it `B` collapses: a constant `B`
makes the measure loss trivially satisfiable, and — this is the trap
§11 flags — the loss curve looks the same either way. On `point_mass` the
collapse is detectable by hand; on walker it is not, which is why the milestone
validates there first.

⚠ **THIS TERM WAS ITSELF A COLLAPSE OBJECTIVE until §12.13 of
`docs/BFM_ZERO_G1_REPRODUCTION.md`.** It ran `E_ij[(B(s_i)·B(s+_j))^2]` across
TWO INDEPENDENT batches. With two tensors there is no self-pairing, so nothing
anchors any direction's variance: the minimiser is not "B isotropic" but "let
the two batches span ORTHOGONAL SUBSPACES" — a rank-≤d/2 collapse of BOTH, the
exact failure this term exists to prevent. Following the gradient on free rows
(d=32, BATCH=128, the run's 4:1 ratio): the two-tensor form drives rank
25.3 → 11.3 with cross-batch mean|cos| → 1e-4; the one-tensor form reaches
rank = d = 32.000 in 200 steps and holds there. It killed the first G1 run.

`tests/deep_agents/test_fb_ortho_fixed_point.mojo` is the gate, and it FOLLOWS
THE GRADIENT to the fixed point. Both older ortho gates PASS on the broken
objective: the finite-difference check confirms the gradient matches the loss
(and the loss was the bug), and "a rank-1 B scores worse than a spread one"
compares two hand-built points and never visits the real minimiser.

BFM-Zero masks the diagonal (`references/BFM-Zero-main/humanoidverse/agents/fb/
agent.py:250`). We do not need to: `|B|` is pinned to √d by the net's own sphere
projection, so the i=j term is a constant whose gradient is purely radial and
the projection removes it. Measured — including it still converges to rank = d
exactly. It costs a reported offset of `D^2/BATCH` and nothing else.

## Gradients

All three are exact, not approximations:

    dL_FB/dM     = 2(M - Mtarget)/BATCH^2          M = F·B(s+)^T
    dL_FB/dF    += -2/BATCH · B(s')                 (anchor)
    dL_FB/dB(s') = -2/BATCH · F
    dL_ortho/dO  = O/(BATCH·(BATCH-1)), ZERO on the diagonal   O = B(s+)·B(s+)^T
    dL_ortho/dB(s+) += -2/BATCH · B(s+)

`B(s+)` appears on BOTH sides of `O`, so its total derivative is the SUM of the
pairwise-dot vjp's two input gradients. They are equal here (`O` is symmetric),
but summing is the statement that does not rely on that.

`Mtarget` carries no gradient — it is built from the target networks and is
passed in already scaled by `gamma`.
"""

from std.math import abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB, TPB_REDUCE
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.pairwise_dot import PairwiseDot, RowDot

from .kernels import (
    fb_diag_override_kernel,
    diag_sumsq_kernel,
    fb_diag_stats_kernel,
    residual_grad_kernel,
    sq_diff_reduce_kernel,
    sum_reduce_kernel,
    sumsq_reduce_kernel,
    fill_kernel,
    axpy_kernel,
    scale_kernel,
)


struct FBLossWorkspace[D: Int, BATCH: Int](Movable & Deinitable):
    """Persistent scratch for one FB loss evaluation.

    The `[BATCH, BATCH]` matrices are the reason this exists. At BATCH = 1024
    each is 4 MB and three are live per step; allocating them per call — which
    is what the plain `fb_measure_loss` entry point does — is acceptable for a
    gate and not for 2 M training steps.

    `acc` is a 1-element device scalar the reduction kernels write into. It is
    downloaded ONLY when the caller asks for the loss value: a per-step D2H
    read would serialise the whole pipeline behind a device sync, and the loss
    is diagnostics, not part of the update.
    """

    var pd: PairwiseDot[Self.D, Self.BATCH]
    var rd: RowDot[Self.D]
    var m: Tensor
    var go: Tensor
    var r_out: Tensor
    var r_go: Tensor
    var acc: Tensor
    var acc2: Tensor
    var ga: Tensor
    var gb2: Tensor

    def __init__(out self):
        self.pd = PairwiseDot[Self.D, Self.BATCH]()
        self.rd = RowDot[Self.D]()
        self.m = Tensor()
        self.go = Tensor()
        self.r_out = Tensor()
        self.r_go = Tensor()
        self.acc = Tensor()
        self.acc2 = Tensor()
        self.ga = Tensor()
        self.gb2 = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.pd = move.pd^
        self.rd = move.rd^
        self.m = move.m^
        self.go = move.go^
        self.r_out = move.r_out^
        self.r_go = move.r_go^
        self.acc = move.acc^
        self.acc2 = move.acc2^
        self.ga = move.ga^
        self.gb2 = move.gb2^

    def prepare[target: StaticString](
        mut self, ctx: Optional[DeviceContext] = None
    ) raises:
        comptime NN = Self.BATCH * Self.BATCH
        comptime ND = Self.BATCH * Self.D
        comptime if target == "cpu":
            self.m.ensure(NN)
            self.go.ensure(NN)
            self.r_out.ensure(Self.BATCH)
            self.r_go.ensure(Self.BATCH)
            self.acc.ensure(1)
            self.acc2.ensure(2)
            self.ga.ensure(ND)
            self.gb2.ensure(ND)
        else:
            var c = ctx.value()
            self.m.ensure_gpu(c, NN)
            self.go.ensure_gpu(c, NN)
            self.r_out.ensure_gpu(c, Self.BATCH)
            self.r_go.ensure_gpu(c, Self.BATCH)
            self.acc.ensure(1)
            self.acc.ensure_gpu(c, 1)
            self.acc2.ensure(2)
            self.acc2.ensure_gpu(c, 2)
            self.ga.ensure_gpu(c, ND)
            self.gb2.ensure_gpu(c, ND)


def pairwise_matrix[
    D: Int, BATCH: Int
](ref a: Tensor, ref c: Tensor, mut m: Tensor) raises:
    """`m[i,j] = sum_k a[i,k]·c[j,k]`, on the CPU. No gradient bookkeeping.

    Used for the TARGET matrix, which must not carry gradient. Going through
    the same primitive as the differentiable path is deliberate: a
    hand-inlined target would be one edit away from disagreeing with the online
    one, and that disagreement is invisible in the loss value.
    """
    var op = PairwiseDot[D, BATCH].make["cpu", Deterministic](None)
    var ins = TensorPack[2]()
    ins[0].ensure(BATCH * D)
    ins[1].ensure(BATCH * D)
    for i in range(BATCH * D):
        ins[0].data[i] = a.data[i]
        ins[1].data[i] = c.data[i]
    op.forward["cpu", BATCH](TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), m, None)


def fb_measure_loss[
    D: Int, BATCH: Int
](
    ref f: Tensor,
    ref b_goal: Tensor,
    ref m_target: Tensor,
    mut g_f: Tensor,
    mut g_b_goal: Tensor,
    with_anchor: Bool = True,
) raises -> Float64:
    """`L_FB` and its gradients. See the module docstring for the formula.

    ONE `B(goal)` tensor: `M = F·B(goal)^T`, the quadratic on its OFF-DIAGONAL
    and the anchor on its DIAGONAL residual. Inputs are `[BATCH, D]` except
    `m_target`, which is `[BATCH, BATCH]` and already multiplied by `gamma`.

    `with_anchor=False` drops the diagonal term, leaving the diagonal out of
    the loss entirely. That switch exists ONLY so the gate can demonstrate what
    its absence permits — without it `F = 0` is a perfect global minimum — and
    it is never what a trainer wants.
    """
    var op = PairwiseDot[D, BATCH].make["cpu", Deterministic](None)
    var ins = TensorPack[2]()
    ins[0].ensure(BATCH * D)
    ins[1].ensure(BATCH * D)
    for i in range(BATCH * D):
        ins[0].data[i] = f.data[i]
        ins[1].data[i] = b_goal.data[i]
    var m = Tensor.alloc(BATCH * BATCH)
    op.forward["cpu", BATCH](TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), m, None)

    var inv_od = 1.0 / (Float64(BATCH) * (Float64(BATCH) - 1.0))
    var loss = Float64(0)
    var go = Tensor.alloc(BATCH * BATCH)
    for i in range(BATCH * BATCH):
        var r = Float64(m.data[i]) - Float64(m_target.data[i])
        loss += r * r
        go.data[i] = Scalar[DT](2.0 * r * inv_od)
    # the diagonal leaves the square and takes the linear anchor instead
    var diag_sum = Float64(0)
    for i in range(BATCH):
        var r = (
            Float64(m.data[i * BATCH + i])
            - Float64(m_target.data[i * BATCH + i])
        )
        loss -= r * r
        diag_sum += r
        go.data[i * BATCH + i] = Scalar[DT](
            -2.0 / Float64(BATCH) if with_anchor else 0.0
        )
    loss *= inv_od
    if with_anchor:
        loss += -2.0 * diag_sum / Float64(BATCH)

    var grads = TensorPack[2]()
    op.vjp["cpu", BATCH](
        TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), go, TensorRefs[2, MutAnyOrigin](grads[0], grads[1]),
        None,
    )
    g_f.ensure(BATCH * D)
    g_b_goal.ensure(BATCH * D)
    for i in range(BATCH * D):
        g_f.data[i] = grads[0].data[i]
        g_b_goal.data[i] = grads[1].data[i]
    return loss


def fb_rank_eff_from_ortho[D: Int, BATCH: Int](ortho: Float64) -> Float64:
    """Participation-ratio effective rank of `C = B^T B / BATCH`, from
    `L_ortho` alone. EXACT — gated against a rank computed from B itself.

    `|B|` is pinned to sqrt(d) by the net's sphere projection, so it is
    STRUCTURALLY incapable of showing a directional collapse — which is what
    killed run 1 (§12.12). This is the number that can, and it comes free from
    the ortho loss because every row's norm is known:

        L_ortho  = 0.5·E_{i!=j}[(B_i·B_j)^2] - E_i[||B_i||^2] = 0.5·E_off - d
        E_off    = 2·(L_ortho + d)
        tr(C)    = d
        tr(C^2)  = (1/n^2)·sum_ij (B_i·B_j)^2 = (d^2 + (n-1)·E_off) / n
        rank_eff = tr(C)^2 / tr(C^2) = d^2·n / (d^2 + (n-1)·E_off)

    ⚠ `BATCH` is NOT optional. The `i==j` self-pairs contribute `d^2/n` to
    `tr(C^2)` — at d 256 / n 1024 that is 64 against an `E_off` near 260, a
    fifth of the total. The first version of this helper dropped it and read
    32.2 where the truth was 25.9; `test_fb_ortho_fixed_point` [0] caught it
    by computing the same quantity from B directly.

    ⚠⚠ THE SCALE CHANGED HERE. `bfm_zero_train_gpu.mojo` used to compute
    `d^2 / E_off` inline, which reads ~341 at true isotropy rather than d, so
    every `b_rank_eff` in a CSV before docs §12.31 is on that other scale:
    the runs logging 248-256 were near 200 on this one, and so was the
    reference. RELATIVE comparisons between them still hold — both sides went
    through the same formula — but the absolute numbers do not transfer.

    The conversion lives beside `L_ortho` because it depends on its exact
    definition: it was inline in the driver, correct only for the pre-§12.28
    form, and when the loss moved to the reference's scale that site was
    missed (`_a_rule_written_inline_twice_drifts`).

    The reference sits at ~204 (its `orth_loss` -127.83 at 11.1 M).
    """
    var e_off = 2.0 * (ortho + Float64(D))
    var d2 = Float64(D) * Float64(D)
    var tr_c2 = (d2 + (Float64(BATCH) - 1.0) * e_off) / Float64(BATCH)
    if tr_c2 <= 1e-12:
        return 0.0
    return d2 / tr_c2


def fb_ortho_loss[
    D: Int, BATCH: Int
](
    ref b: Tensor,
    mut g_b: Tensor,
) raises -> Float64:
    """`L_ortho` and its gradient.

    ONE batch against itself. Passing two independent batches here is a
    COLLAPSE objective, not a regulariser — see the module header.

    OVERWRITES `g_b` rather than accumulating. `B(s+)` receives gradient from
    both losses, and making the trainer add them explicitly keeps that visible
    at the call site — an accumulating signature would let a forgotten zeroing
    silently double one contribution.
    """
    var op = PairwiseDot[D, BATCH].make["cpu", Deterministic](None)
    var ins = TensorPack[2]()
    ins[0].ensure(BATCH * D)
    ins[1].ensure(BATCH * D)
    for i in range(BATCH * D):
        ins[0].data[i] = b.data[i]
        ins[1].data[i] = b.data[i]
    var o = Tensor.alloc(BATCH * BATCH)
    op.forward["cpu", BATCH](TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), o, None)

    # OFF-DIAGONAL only, normalised by the reference's `off_diag_sum`
    # (`agent.py:249`): `0.5*(Cov*off_diag).pow(2).sum()/off_diag_sum`.
    var inv_od = 1.0 / (Float64(BATCH) * (Float64(BATCH) - 1.0))
    var loss = Float64(0)
    var go = Tensor.alloc(BATCH * BATCH)
    for i in range(BATCH * BATCH):
        var v = Float64(o.data[i])
        if (i // BATCH) == (i % BATCH):
            go.data[i] = Scalar[DT](0.0)   # the diagonal is masked out
        else:
            loss += v * v
            go.data[i] = Scalar[DT](v * inv_od)
    loss *= 0.5 * inv_od

    var grads = TensorPack[2]()
    op.vjp["cpu", BATCH](
        TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), go, TensorRefs[2, MutAnyOrigin](grads[0], grads[1]),
        None,
    )
    # `B` is BOTH inputs of `O`, so its total derivative is the SUM of the two
    # input gradients.
    g_b.ensure(BATCH * D)
    for i in range(BATCH * D):
        g_b.data[i] = grads[0].data[i] + grads[1].data[i]

    # -mean_i ||B(s+_i)||^2, gradient -2/BATCH · B(s+)
    # (`agent.py:248`: `orth_loss_diag = -Cov.diag().mean()`).
    var sq = Float64(0)
    for i in range(BATCH * D):
        var v = Float64(b.data[i])
        sq += v * v
    loss += -sq / Float64(BATCH)
    var c = -2.0 / Float64(BATCH)
    for i in range(BATCH * D):
        g_b.data[i] = Scalar[DT](
            Float64(g_b.data[i]) + c * Float64(b.data[i])
        )
    return loss


# ══════════════════════════════════════════════════════════════════════
# Target-parameterized implementations (CPU + GPU).
#
# The plain entry points above allocate per call and are CPU-only; they are
# what `test_fb_loss.mojo` gates and what the M1 example uses. These take a
# persistent workspace and run on either device — the M2 path.
#
# The MATH is not duplicated: the CPU branch here is the same sequence of
# operations, expressed once. What differs is only where the elementwise work
# and the reductions run.
# ══════════════════════════════════════════════════════════════════════


def fb_measure_loss_into[
    target: StaticString, D: Int, BATCH: Int
](
    mut ws: FBLossWorkspace[D, BATCH],
    ref [MutAnyOrigin] f: Tensor,
    ref [MutAnyOrigin] b_goal: Tensor,
    ref [MutAnyOrigin] m_target: Tensor,
    ref [MutAnyOrigin] g_f: Tensor,
    ref [MutAnyOrigin] g_b_goal: Tensor,
    mut out_quad: Float64,
    mut out_anchor: Float64,
    want_loss: Bool = True,
    ctx: Optional[DeviceContext] = None,
) raises -> Float64:
    """`L_FB` and its gradients, on `target`. See the module docstring.

    ONE `B` tensor, `B(goal)` with `goal = next_obs` (`agent.py:190`), used for
    the matrix AND its diagonal — not two. `M = F.B(s')^T`:

        out_quad   = sum_{i!=j} (M-Mt)^2 / (BATCH*(BATCH-1))
        out_anchor = -2 * mean_i (M_ii - Mt_ii)

    the reference's `fb_offdiag` / `fb_diag` PER HEAD. `online.mojo` averages
    the two heads and that is the whole conversion — `agent.py:243` sums `diff`
    over both heads while dividing by one matrix's `off_diag_sum`, so its
    leading 0.5 already is that mean. ⚠ This docstring used to claim "our
    uniform 2x scale, so halve them"; that halving was spurious and put both
    metrics at HALF the reference's (docs §12.27). Both are 0 when `want_loss`
    is False.

    THE DIAGONAL IS EXCLUDED FROM THE SQUARE and carries the linear anchor
    instead, so its upstream gradient is the constant `-2/BATCH`. That is the
    whole point of the split: the successor measure has a Dirac at the actual
    successor, and fitting it with a bootstrapped square would fight it.

    ⚠ This used TWO B tensors until §12.17 — `M = F.B(s+)^T` on an independent
    draw, quadratic over ALL pairs, anchor `-2*E[F.B(s')]` on a second tensor
    with no target subtracted. Measured against the reference at the same
    timestep that ran `anchor/quad` 4.04 where the reference holds 0.41, flat
    for 200 M steps. The header above this file argued for the two-tensor form
    — "if `s+` were the batch's own next-states, the matrix would only ever be
    evaluated on pairs one step apart" — and that is wrong: for `i != j`,
    `B(s'_j)` is an arbitrary state from the batch, not one step from `s_i`.
    The objection applies only to the DIAGONAL, which is exactly the entry
    split out here.

    `want_loss=False` skips the two reduction kernels AND the device readback,
    leaving the return value at 0. The gradients are unaffected — the loss
    value never enters the update. Use it on the steps you do not log: the D2H
    read is a full device sync, and at 2 M steps paying it every step is the
    difference between a few hours and most of a day.
    """
    ws.prepare[target](ctx)
    out_quad = 0.0
    out_anchor = 0.0
    comptime NN = BATCH * BATCH
    comptime ND = BATCH * D
    # off-diagonal count, the reference's `off_diag_sum`
    var inv_od = Scalar[DT](
        1.0 / (Float64(BATCH) * (Float64(BATCH) - 1.0))
    )
    var diag_go = Scalar[DT](-2.0 / Float64(BATCH))

    # M = F · B(goal)^T
    ws.pd.forward[target, BATCH](TensorRefs[2, MutAnyOrigin](f, b_goal), ws.m, ctx)

    var loss = Float64(0)
    var diag_sum = Float64(0)
    var diag_sumsq = Float64(0)
    comptime if target == "cpu":
        var all_sq = Float64(0)
        for i in range(NN):
            var r = Float64(ws.m.data[i]) - Float64(m_target.data[i])
            ws.go.data[i] = Scalar[DT](2.0 * r * Float64(inv_od))
            if want_loss:
                all_sq += r * r
        for i in range(BATCH):
            var r = (
                Float64(ws.m.data[i * BATCH + i])
                - Float64(m_target.data[i * BATCH + i])
            )
            # the diagonal leaves the square and takes the linear anchor
            ws.go.data[i * BATCH + i] = diag_go
            if want_loss:
                diag_sum += r
                diag_sumsq += r * r
        if want_loss:
            all_sq -= diag_sumsq
            out_quad = all_sq * Float64(inv_od)
            out_anchor = -2.0 * diag_sum / Float64(BATCH)
            loss = out_quad + out_anchor
    else:
        var c = ctx.value()
        c.enqueue_function[residual_grad_kernel[NN]](
            ws.go.dev.value().unsafe_ptr(),
            ws.m.dev.value().unsafe_ptr(),
            m_target.dev.value().unsafe_ptr(),
            Scalar[DT](inv_od),
            grid_dim=(NN + TPB - 1) // TPB,
            block_dim=TPB,
        )
        c.enqueue_function[fb_diag_override_kernel[BATCH]](
            ws.go.dev.value().unsafe_ptr(), diag_go,
            grid_dim=(BATCH + TPB - 1) // TPB,
            block_dim=TPB,
        )
        if want_loss:
            c.enqueue_function[sq_diff_reduce_kernel[NN]](
                ws.m.dev.value().unsafe_ptr(),
                m_target.dev.value().unsafe_ptr(),
                ws.acc.dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            ws.acc.download(c)
            # the kernel returns the MEAN over NN
            var all_sq = Float64(ws.acc.data[0]) * Float64(NN)
            c.enqueue_function[fb_diag_stats_kernel[BATCH]](
                ws.m.dev.value().unsafe_ptr(),
                m_target.dev.value().unsafe_ptr(),
                ws.acc2.dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            ws.acc2.download(c)
            diag_sum = Float64(ws.acc2.data[0])
            diag_sumsq = Float64(ws.acc2.data[1])
            out_quad = (all_sq - diag_sumsq) * Float64(inv_od)
            out_anchor = -2.0 * diag_sum / Float64(BATCH)
            loss = out_quad + out_anchor

    # dF, dB(goal). ONE vjp: the anchor is the DIAGONAL of this same matrix
    # now, so the separate RowDot pass over a second B tensor is gone — with
    # it the `r_out`/`r_go`/`ga` scratch and one B forward per step.
    ws.pd.vjp[target, BATCH](
        TensorRefs[2, MutAnyOrigin](f, b_goal), ws.go,
        TensorRefs[2, MutAnyOrigin](g_f, g_b_goal), ctx,
    )
    return loss


def fb_ortho_loss_into[
    target: StaticString, D: Int, BATCH: Int
](
    mut ws: FBLossWorkspace[D, BATCH],
    ref [MutAnyOrigin] b: Tensor,
    ref [MutAnyOrigin] g_b: Tensor,
    want_loss: Bool = True,
    ctx: Optional[DeviceContext] = None,
) raises -> Float64:
    """`L_ortho` and its gradient, on `target`. OVERWRITES `g_b`.

    ONE batch against itself — see the module header for why two independent
    batches make this a collapse objective.
    """
    ws.prepare[target](ctx)
    comptime NN = BATCH * BATCH
    comptime ND = BATCH * D
    # the reference's `off_diag_sum` — the quadratic runs over i != j only
    var inv_od = Scalar[DT](
        1.0 / (Float64(BATCH) * (Float64(BATCH) - 1.0))
    )

    ws.pd.forward[target, BATCH](TensorRefs[2, MutAnyOrigin](b, b), ws.m, ctx)

    var loss = Float64(0)
    comptime if target == "cpu":
        var all_sq = Float64(0)
        for i in range(NN):
            var v = Float64(ws.m.data[i])
            if want_loss:
                all_sq += v * v
            ws.go.data[i] = Scalar[DT](v * Float64(inv_od))
        var diag_sq = Float64(0)
        for i in range(BATCH):
            var v = Float64(ws.m.data[i * BATCH + i])
            if want_loss:
                diag_sq += v * v
            ws.go.data[i * BATCH + i] = Scalar[DT](0.0)
        if want_loss:
            loss = 0.5 * (all_sq - diag_sq) * Float64(inv_od)
    else:
        var c = ctx.value()
        # The ortho target is ZERO, so the residual IS O and the gradient is
        # just a scaling: `go = O/off_diag_sum`, then the diagonal is masked
        # to zero. Reusing `residual_grad_kernel` here would mean passing `go`
        # as both its output and its zeroed `mt` input — safe per-thread, but
        # an aliased read/write the next person has to re-derive. One scale.
        c.enqueue_function[scale_kernel[NN]](
            ws.go.dev.value().unsafe_ptr(),
            ws.m.dev.value().unsafe_ptr(),
            Scalar[DT](inv_od),
            grid_dim=(NN + TPB - 1) // TPB,
            block_dim=TPB,
        )
        c.enqueue_function[fb_diag_override_kernel[BATCH]](
            ws.go.dev.value().unsafe_ptr(), Scalar[DT](0.0),
            grid_dim=(BATCH + TPB - 1) // TPB,
            block_dim=TPB,
        )
        if want_loss:
            c.enqueue_function[sumsq_reduce_kernel[NN]](
                ws.m.dev.value().unsafe_ptr(),
                ws.acc.dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            ws.acc.download(c)
            # the kernel returns the MEAN over NN
            var all_sq = Float64(ws.acc.data[0]) * Float64(NN)
            c.enqueue_function[diag_sumsq_kernel[BATCH]](
                ws.m.dev.value().unsafe_ptr(),
                ws.acc2.dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            ws.acc2.download(c)
            loss = 0.5 * (all_sq - Float64(ws.acc2.data[0])) * Float64(inv_od)

    # `b` is BOTH inputs of `O`, so its total derivative is the SUM of the two
    # input gradients — `ws.gb2` catches the mirrored half.
    ws.pd.vjp[target, BATCH](
        TensorRefs[2, MutAnyOrigin](b, b), ws.go, TensorRefs[2, MutAnyOrigin](g_b, ws.gb2), ctx
    )

    # -mean_i ||B(s+_i)||^2 ; gradient -2/BATCH · B(s+).
    var c4 = Scalar[DT](-2.0 / Float64(BATCH))
    comptime if target == "cpu":
        if want_loss:
            var sq = Float64(0)
            for i in range(ND):
                var v = Float64(b.data[i])
                sq += v * v
            loss += -sq / Float64(BATCH)
        for i in range(ND):
            g_b.data[i] = Scalar[DT](
                Float64(g_b.data[i]) + Float64(ws.gb2.data[i])
                + Float64(c4) * Float64(b.data[i])
            )
    else:
        var c = ctx.value()
        if want_loss:
            c.enqueue_function[sumsq_reduce_kernel[ND]](
                b.dev.value().unsafe_ptr(),
                ws.acc.dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            ws.acc.download(c)
            # the kernel returns the MEAN over ND, so scale back to per-ROW
            loss += -Float64(ws.acc.data[0]) * Float64(D)
        c.enqueue_function[axpy_kernel[ND]](
            g_b.dev.value().unsafe_ptr(),
            ws.gb2.dev.value().unsafe_ptr(),
            Scalar[DT](1.0),
            grid_dim=(ND + TPB - 1) // TPB,
            block_dim=TPB,
        )
        c.enqueue_function[axpy_kernel[ND]](
            g_b.dev.value().unsafe_ptr(),
            b.dev.value().unsafe_ptr(),
            c4,
            grid_dim=(ND + TPB - 1) // TPB,
            block_dim=TPB,
        )
    return loss

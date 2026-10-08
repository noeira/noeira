"""LeWM planning rollout, reference-exact: `stable_worldmodel` 0.1.1
`LeWM.rollout` + `criterion` (`references/stable-worldmodel-0.1.1/.../lewm.py`).

docs/LEWM_REOPEN_PLAN.md P3. With the eval config's `history_size: 1` the
context starts at ONE encoded frame and grows to the predictor's 3:

    emb[0] = encode(start);  act_emb[k] = ActionEmbedder(candidate block k)
    for t in 0 ..< HORIZON:
        lo = max(0, t + 1 - 3);  L = t + 1 - lo           # 1, 2, 3, 3, 3
        emb[t+1] = Predictor(emb[lo..t] + pos[0..L-1],  act_emb[lo..t])[L-1]
    cost = Σ_d (emb[HORIZON] - goal)²                       # last step only
                                                    # (`PlanCost` for others)

Variable L on a fixed 3-token predictor: the real L tokens sit at positions
0..L-1 and the tail is zero padding. Attention is CAUSAL and every other op
(AdaLN modulation, the LayerNorms, the FFN, pred_proj with BN in EVAL mode) is
per token, so output L-1 never sees the padding and equals the reference's
L-token result — and the position embedding is the reference's `pos[:, :L]`.
(The old port replicated the current latent into all 3 slots: audit A.5.)

⚠ Activations round-trip the host between module calls (`_up` / `_down`).
Fine for the gates and the semantics; the first thing to move on device when
the full CEM budget (300 x 30 per replan) runs in P5.
"""

from std.math import sqrt
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn import BiasAdd, RepeatConditional, Tokenwise, LayerNorm
from .ref_model import (
    ActionEmbedderRef,
    ConditionalTransformerBlockRef,
    ProjectorRef,
    LeWMEncoderRef,
    TORCH_LN_EPS,
)
from .ref_load import load_ref


# the published model (config.json of quentinll/lewm-pusht)
comptime REF_EMB = 192
comptime REF_ACT = 10       # frameskip 5 x 2
comptime REF_CTX = 3        # predictor num_frames
comptime REF_DEPTH = 6
comptime REF_PROJ_H = 2048

comptime RefEncoder = LeWMEncoderRef[3, 224, 14, 192, 3, 12, REF_EMB, REF_PROJ_H]


def _up[target: StaticString](
    vals: List[Scalar[DT]], ctx: Optional[DeviceContext]
) raises -> Tensor:
    var t = Tensor.alloc(len(vals))
    for i in range(len(vals)):
        t.data[i] = vals[i]
    comptime if target == "gpu":
        t.upload(ctx.value())
    return t^


def _down[target: StaticString](
    mut t: Tensor, n: Int, ctx: Optional[DeviceContext]
) raises -> List[Scalar[DT]]:
    comptime if target == "gpu":
        ctx.value().synchronize()
        t.download(ctx.value())
    var out = List[Scalar[DT]](capacity=n)
    for i in range(n):
        out.append(t.data[i])
    return out^


def encode_ref[target: StaticString, N: Int](
    mut enc: RefEncoder, pixels: List[Scalar[DT]], ctx: Optional[DeviceContext]
) raises -> List[Scalar[DT]]:
    """N ImageNet-normalised CHW frames -> N x EMB (BN in eval mode)."""
    enc.set_attr["training"](Scalar[DT](0.0))
    var x = _up[target](pixels, ctx)
    var out = Tensor.alloc(N * REF_EMB)
    enc.forward[target, N](TensorRefs[1](x), out, ctx)
    return _down[target](out, N * REF_EMB, ctx)


struct LeWMRefRollout[target: StaticString, S: Int, HORIZON: Int, ACT: Int = REF_ACT]:
    """S candidate action sequences of HORIZON blocks, rolled out from one
    start embedding. The predictor and action embedder of the published
    model, loaded by the loss graph's walk names (`ref_load`). `ACT` is the
    action block's width (PushT 10; the SO-101 model 30)."""

    comptime D = REF_EMB
    comptime H = REF_CTX

    var ae: ActionEmbedderRef[Self.HORIZON, Self.ACT, REF_EMB]
    var pe: BiasAdd[REF_CTX * REF_EMB]
    var pred: RepeatConditional[
        REF_DEPTH,
        ConditionalTransformerBlockRef[REF_EMB, 16, REF_CTX, 2048, 64],
    ]
    var ln: Tokenwise[REF_CTX, LayerNorm[REF_EMB, DT, TORCH_LN_EPS]]
    var pp: Tokenwise[REF_CTX, ProjectorRef[REF_EMB, REF_PROJ_H, REF_EMB]]
    var ctx: Optional[DeviceContext]

    def __init__(out self, dump_dir: String, ctx: Optional[DeviceContext]) raises:
        self.ctx = ctx
        self.ae = ActionEmbedderRef[Self.HORIZON, Self.ACT, REF_EMB].make[
            Self.target, Kaiming
        ](ctx)
        self.pe = BiasAdd[REF_CTX * REF_EMB].make[Self.target, Kaiming](ctx)
        self.pred = RepeatConditional[
            REF_DEPTH,
            ConditionalTransformerBlockRef[REF_EMB, 16, REF_CTX, 2048, 64],
        ].make[Self.target, Kaiming](ctx)
        self.ln = Tokenwise[REF_CTX, LayerNorm[REF_EMB, DT, TORCH_LN_EPS]].make[
            Self.target, Kaiming
        ](ctx)
        self.pp = Tokenwise[REF_CTX, ProjectorRef[REF_EMB, REF_PROJ_H, REF_EMB]].make[
            Self.target, Kaiming
        ](ctx)
        _ = load_ref[Self.target](self.ae, dump_dir, String("act_emb."), ctx)
        _ = load_ref[Self.target](self.pe, dump_dir, String("x_pe."), ctx)
        _ = load_ref[Self.target](self.pred, dump_dir, String("pred_raw."), ctx)
        _ = load_ref[Self.target](self.ln, dump_dir, String("pred_ln."), ctx)
        _ = load_ref[Self.target](self.pp, dump_dir, String("pred."), ctx)
        self.pp.set_attr["training"](Scalar[DT](0.0))

    def sync_from[V: ParamVisitor](mut self, mut fill_pred: V, mut fill_ae: V, mut fill_pe: V,
                                   mut fill_ln: V, mut fill_pp: V) raises:
        """Re-load the planner's predictor side after a test-time update (one
        visitor per module, each carrying that module's name prefix). Without
        this the CEM keeps planning on the weights it was built with."""
        self.pred.for_each_param[Self.target](fill_pred, self.ctx)
        self.ae.for_each_param[Self.target](fill_ae, self.ctx)
        self.pe.for_each_param[Self.target](fill_pe, self.ctx)
        self.ln.for_each_param[Self.target](fill_ln, self.ctx)
        self.pp.for_each_param[Self.target](fill_pp, self.ctx)
        self.pp.for_each_state[Self.target](fill_pp, self.ctx)
        self.pp.set_attr["training"](Scalar[DT](0.0))

    def _predict_last(
        mut self,
        x_ctx: List[Scalar[DT]],
        c_ctx: List[Scalar[DT]],
        L: Int,
    ) raises -> List[Scalar[DT]]:
        """One predictor pass over (S, 3, D) left-aligned contexts; returns
        the output token L-1 of every row, (S, D)."""
        comptime HD = REF_CTX * REF_EMB
        var x = _up[Self.target](x_ctx, self.ctx)
        var xp = Tensor.alloc(Self.S * HD)
        self.pe.forward[Self.target, Self.S](TensorRefs[1](x), xp, self.ctx)
        var ins = TensorPack[2]()
        var xh = _down[Self.target](xp, Self.S * HD, self.ctx)
        ins[0].ensure(Self.S * HD)
        ins[1].ensure(Self.S * HD)
        for q in range(Self.S * HD):
            ins[0].data[q] = xh[q]
            ins[1].data[q] = c_ctx[q]
        comptime if Self.target == "gpu":
            ins[0].upload(self.ctx.value())
            ins[1].upload(self.ctx.value())
        var y = Tensor.alloc(Self.S * HD)
        self.pred.forward[Self.target, Self.S](TensorRefs[2](ins[0], ins[1]), y, self.ctx)
        var z = Tensor.alloc(Self.S * HD)
        self.ln.forward[Self.target, Self.S](TensorRefs[1](y), z, self.ctx)
        var p = Tensor.alloc(Self.S * HD)
        self.pp.forward[Self.target, Self.S](TensorRefs[1](z), p, self.ctx)
        var ph = _down[Self.target](p, Self.S * HD, self.ctx)
        var out = List[Scalar[DT]](capacity=Self.S * Self.D)
        for s in range(Self.S):
            for d in range(Self.D):
                out.append(ph[s * HD + (L - 1) * Self.D + d])
        return out^

    def embed_actions(mut self, actions: List[Scalar[DT]]) raises -> List[Scalar[DT]]:
        """(S, HORIZON, ACT) z-scored blocks -> (S, HORIZON, D) action
        embeddings. Per token, so a shorter sequence can be zero-padded."""
        var a = _up[Self.target](actions, self.ctx)
        var ae_t = Tensor.alloc(Self.S * Self.HORIZON * Self.D)
        self.ae.forward[Self.target, Self.S](TensorRefs[1](a), ae_t, self.ctx)
        return _down[Self.target](ae_t, Self.S * Self.HORIZON * Self.D, self.ctx)

    def predict_ctx(
        mut self, x_ctx: List[Scalar[DT]], c_ctx: List[Scalar[DT]], L: Int
    ) raises -> List[Scalar[DT]]:
        """One predictor step on (S, 3, D) left-aligned embedding / action-
        embedding contexts of length L (tail zero): output token L − 1, (S, D).
        The building block of a rollout whose actions are chosen step by step
        (`intact.IntactDirect`)."""
        return self._predict_last(x_ctx, c_ctx, L)

    def rollout(
        mut self, start_emb: List[Scalar[DT]], actions: List[Scalar[DT]]
    ) raises -> List[Scalar[DT]]:
        """start_emb (D), actions (S, HORIZON, ACT) z-scored ->
        predicted embeddings (S, HORIZON + 1, D); entry 0 is the start."""
        return self.rollout_ctx(start_emb, 1, actions)

    def rollout_ctx(
        mut self, obs_embs: List[Scalar[DT]], n_obs: Int, actions: List[Scalar[DT]]
    ) raises -> List[Scalar[DT]]:
        """`rollout` from `n_obs` OBSERVED embeddings (n_obs, D), shared by
        every row: entries 0 ..< n_obs of the result are those, the rest are
        predicted. `actions` (S, HORIZON, ACT): block k is the one taken after
        frame k, so blocks 0 ..< n_obs − 1 are the executed ones (identical
        across rows) and the rest are the candidates'."""
        comptime D = Self.D
        comptime T1 = Self.HORIZON + 1
        var act_emb = self.embed_actions(actions)

        var embs = List[Scalar[DT]](length=Self.S * T1 * D, fill=Scalar[DT](0))
        for s in range(Self.S):
            for k in range(n_obs):
                for d in range(D):
                    embs[(s * T1 + k) * D + d] = obs_embs[k * D + d]
        for t in range(n_obs - 1, Self.HORIZON):
            var lo = max(0, t + 1 - Self.H)
            var L = t + 1 - lo
            var x = List[Scalar[DT]](length=Self.S * Self.H * D, fill=Scalar[DT](0))
            var c = List[Scalar[DT]](length=Self.S * Self.H * D, fill=Scalar[DT](0))
            for s in range(Self.S):
                for j in range(L):
                    for d in range(D):
                        x[(s * Self.H + j) * D + d] = embs[(s * T1 + lo + j) * D + d]
                        c[(s * Self.H + j) * D + d] = act_emb[
                            (s * Self.HORIZON + lo + j) * D + d
                        ]
            var nxt = self._predict_last(x, c, L)
            for s in range(Self.S):
                for d in range(D):
                    embs[(s * T1 + t + 1) * D + d] = nxt[s * D + d]
        return embs^

    def cost(
        self, embs: List[Scalar[DT]], goal_emb: List[Scalar[DT]],
        step_w: List[Float64] = List[Float64](),
    ) -> List[Scalar[DT]]:
        """`criterion`: Σ_d (emb[HORIZON] - goal)² per row. With `step_w`
        (HORIZON weights, `PlanCost.weights`): Σ_t w[t] Σ_d (emb[t+1] - goal)²."""
        comptime D = Self.D
        comptime T1 = Self.HORIZON + 1
        var out = List[Scalar[DT]](capacity=Self.S)
        for s in range(Self.S):
            var acc = Float64(0)
            if len(step_w) == 0:
                for d in range(D):
                    var diff = Float64(embs[(s * T1 + Self.HORIZON) * D + d]) - Float64(goal_emb[d])
                    acc += diff * diff
            else:
                for t in range(Self.HORIZON):
                    if step_w[t] == 0.0:
                        continue
                    var st = Float64(0)
                    for d in range(D):
                        var diff = Float64(embs[(s * T1 + t + 1) * D + d]) - Float64(goal_emb[d])
                        st += diff * diff
                    acc += step_w[t] * st
            out.append(Scalar[DT](acc))
        return out^


comptime COST_LAST = 0
comptime COST_ALL = 1
comptime COST_STAGED = 2


@fieldwise_init
struct PlanCost(Copyable, Movable, Writable):
    """The planning cost over the predicted steps.

      last       LeWM's `criterion`: the final step only (the default);
      all:b      Σ_t b^t·‖emb[t+1] − goal‖² over every predicted step, the
                 weights normalised (AdaJEPA's `objective_fn_all`, base 2) —
                 rewards plans that get close EARLY, so a planner that executes
                 one block per replan is not paid to procrastinate;
      staged:b   AdaJEPA's default `mode: staged`: `last` while the replan
                 index is < HORIZON, `all:b` from then on
                 (references/adajepa-main/planning/objectives.py — its `step`
                 is the MPC iteration).

    The start embedding's term is constant across candidates and dropped."""

    var kind: Int
    var base: Float64

    @staticmethod
    def parse(spec: String) raises -> Self:
        var parts = spec.split(":")
        var base = Float64(String(parts[1])) if len(parts) == 2 else 2.0
        if parts[0] == "last" and len(parts) == 1:
            return Self(COST_LAST, base)
        if parts[0] == "all":
            return Self(COST_ALL, base)
        if parts[0] == "staged":
            return Self(COST_STAGED, base)
        raise Error("unknown --cost " + spec + " (last | all[:base] | staged[:base])")

    def weights(self, horizon: Int, replan: Int) -> List[Float64]:
        """Per-step weights for replan `replan`; empty = `last` (the exact
        reference criterion)."""
        if self.kind == COST_LAST or (self.kind == COST_STAGED and replan < horizon):
            return List[Float64]()
        var w = List[Float64](capacity=horizon)
        var tot = 0.0
        for t in range(horizon):
            w.append(self.base ** Float64(t + 1))
            tot += w[t]
        for t in range(horizon):
            w[t] /= tot
        return w^


@fieldwise_init
struct CEMStep(Movable):
    """One CEM iteration's result (`cem_step`)."""

    var candidates: List[Scalar[DT]]  # (S, HORIZON, ACT)
    var costs: List[Scalar[DT]]       # (S)
    var elite: List[Int]              # K indices, ascending cost
    var mean: List[Scalar[DT]]        # (HORIZON, ACT)
    var std: List[Scalar[DT]]         # (HORIZON, ACT), unbiased


def cem_step[
    target: StaticString, S: Int, HORIZON: Int, K: Int
](
    mut roll: LeWMRefRollout[target, S, HORIZON],
    start_emb: List[Scalar[DT]],
    goal_emb: List[Scalar[DT]],
    mean: List[Scalar[DT]],
    std: List[Scalar[DT]],
    noise: List[Scalar[DT]],
    step_w: List[Float64] = List[Float64](),
) raises -> CEMStep:
    """`CEMSolver.solve`'s loop body (stable_worldmodel 0.1.1 solver/cem.py),
    one env:

        candidates = noise * var + mean;  candidates[0] = mean
        costs = get_cost(candidates)
        elite = topk(costs, K, largest=False)          (ascending)
        mean, var = elite.mean(0), elite.std(0)        (torch std: unbiased)

    `var` in the reference IS a standard deviation (it multiplies the noise).
    `noise` is the caller's (S, HORIZON, ACT) draw, so a gate can replay
    torch's generator exactly. `step_w`: `LeWMRefRollout.cost`."""
    comptime A = HORIZON * REF_ACT
    var cand = List[Scalar[DT]](length=S * A, fill=Scalar[DT](0))
    for s in range(S):
        for i in range(A):
            cand[s * A + i] = mean[i] if s == 0 else noise[s * A + i] * std[i] + mean[i]
    var embs = roll.rollout(start_emb, cand)
    var costs = roll.cost(embs, goal_emb, step_w)
    # top-K smallest, ascending (partial selection; K << S)
    var taken = List[Bool](length=S, fill=False)
    var elite = List[Int](capacity=K)
    for _ in range(K):
        var best = -1
        for s in range(S):
            if not taken[s] and (best < 0 or costs[s] < costs[best]):
                best = s
        taken[best] = True
        elite.append(best)
    var m = List[Scalar[DT]](length=A, fill=Scalar[DT](0))
    var v = List[Scalar[DT]](length=A, fill=Scalar[DT](0))
    for i in range(A):
        var acc = Float64(0)
        for k in range(K):
            acc += Float64(cand[elite[k] * A + i])
        var mu = acc / Float64(K)
        var ss = Float64(0)
        for k in range(K):
            var d = Float64(cand[elite[k] * A + i]) - mu
            ss += d * d
        m[i] = Scalar[DT](mu)
        v[i] = Scalar[DT](sqrt(ss / Float64(K - 1)))
    return CEMStep(cand^, costs^, elite^, m^, v^)

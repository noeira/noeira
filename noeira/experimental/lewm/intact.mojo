"""INTACT's intent-to-action actor and Direct planning on the reference LeWM.

docs/LEWM_REOPEN_PLAN.md P8, `references/INTACT-JEPA-main` (arXiv
2607.26056, MIT). INTACT is LeWM plus one MLP actor trained jointly; Direct
mode reads actions from the latent and evaluates no cost:

    for k in 0 ..< HORIZON:
        a_k   = actor_mean(ẑ_k, z_goal − ẑ_k)              # [z, m, z ⊙ m]
        ẑ_k+1 = predictor(context ≤ 3 frames, actions)     # LeWM's own step

At one block per replan only a_0 is executed: the encoder and the actor, no
rollout — the part P7 found robust to visual shifts.

The actor is the history-free release (`INTACT-no-previous-action`: input
3 × 192 = 576): `Linear 576→1024, LN, GELU` ×3 (1024→1024 after the first),
`Linear 1024→20` = (mean, log σ) of a 5 × 2 block in the training
normalisation. torch `nn.GELU()` is the erf form = our `GELU`; LN eps 1e-5.

## The context pairing (`aligned`)

INTACT trains its predictor on (frame t, action taken FROM t) pairs
(`train.py: predict_adjacent_latents`), like LeWM. Its Direct rollout
(`jepa.get_action` / `rollout_one_step`) starts the action history as
[a_{−1}] beside [z_0] and replaces the newest entry with the new action, so
from step 1 on the OLDER context positions pair z_j with a_{j−1}.
`aligned = False` reproduces that (their published numbers come from it);
`aligned = True` uses the training pairing. Step 0 — all that runs at one
block per replan — is identical in both.

Converted weights: `tools/lewm/intact_reference.py` (`ours.actor.*` beside the
LeWM tensors); gate `tests/experimental/lewm/ref/test_intact_direct.mojo`.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn import Sequential, Linear, LayerNorm, GELU
from .ref_model import TORCH_LN_EPS
from .ref_load import load_ref
from .ref_rollout import LeWMRefRollout, REF_EMB, REF_ACT, REF_CTX


comptime ACTOR_IN = 3 * REF_EMB
comptime ACTOR_H = 1024

comptime IntentActorRef = Sequential[
    Linear[ACTOR_IN, ACTOR_H], LayerNorm[ACTOR_H, DT, TORCH_LN_EPS], GELU[ACTOR_H],
    Linear[ACTOR_H, ACTOR_H], LayerNorm[ACTOR_H, DT, TORCH_LN_EPS], GELU[ACTOR_H],
    Linear[ACTOR_H, ACTOR_H], LayerNorm[ACTOR_H, DT, TORCH_LN_EPS], GELU[ACTOR_H],
    Linear[ACTOR_H, 2 * REF_ACT],
]
"""`module.IntentActionActor` without the action-history slot (P8)."""


struct IntactDirect[target: StaticString, HORIZON: Int]:
    """The actor and a one-row predictor: Direct plans for one (start, goal)."""

    var actor: IntentActorRef
    var roll: LeWMRefRollout[Self.target, 1, Self.HORIZON]
    var ctx: Optional[DeviceContext]

    def __init__(out self, dump_dir: String, ctx: Optional[DeviceContext]) raises:
        self.ctx = ctx
        self.actor = IntentActorRef.make[Self.target, Kaiming](ctx)
        var n = load_ref[Self.target](self.actor, dump_dir, String("actor."), ctx)
        if n == 0:
            raise Error("IntactDirect: no `ours.actor.*` in " + dump_dir
                        + " (convert with tools/lewm/intact_reference.py)")
        self.roll = LeWMRefRollout[Self.target, 1, Self.HORIZON](dump_dir, ctx)

    def actor_out(
        mut self, z: List[Scalar[DT]], m: List[Scalar[DT]]
    ) raises -> List[Scalar[DT]]:
        """(mean[10], log σ[10], unclamped) for one latent and one intent."""
        var x = Tensor.alloc(ACTOR_IN)
        for d in range(REF_EMB):
            x.data[d] = z[d]
            x.data[REF_EMB + d] = m[d]
            x.data[2 * REF_EMB + d] = z[d] * m[d]
        comptime if Self.target == "gpu":
            x.upload(self.ctx.value())
        var y = Tensor.alloc(2 * REF_ACT)
        self.actor.forward[Self.target, 1](TensorRefs[1](x), y, self.ctx)
        comptime if Self.target == "gpu":
            self.ctx.value().synchronize()
            y.download(self.ctx.value())
        var out = List[Scalar[DT]](capacity=2 * REF_ACT)
        for i in range(2 * REF_ACT):
            out.append(y.data[i])
        return out^

    def plan(
        mut self,
        z0: List[Scalar[DT]],
        zg: List[Scalar[DT]],
        a_hist: List[Scalar[DT]],
        aligned: Bool,
    ) raises -> List[Scalar[DT]]:
        """The Direct plan, (HORIZON, ACT) in the training normalisation.
        `a_hist` = a_{−1}, read only by the unaligned (INTACT) context."""
        comptime D = REF_EMB
        var embs = List[List[Scalar[DT]]]()
        embs.append(z0.copy())
        var acts = List[List[Scalar[DT]]]()
        if not aligned:
            acts.append(a_hist.copy())
        var plan = List[Scalar[DT]](capacity=Self.HORIZON * REF_ACT)
        for k in range(Self.HORIZON):
            ref cur = embs[len(embs) - 1]
            var m = List[Scalar[DT]](capacity=D)
            for d in range(D):
                m.append(zg[d] - cur[d])
            var o = self.actor_out(cur, m)
            var a = List[Scalar[DT]](capacity=REF_ACT)
            for i in range(REF_ACT):
                a.append(o[i])
                plan.append(o[i])
            if k == Self.HORIZON - 1:
                break  # the last block's successor is never read
            # the context's actions: aligned = a_j beside z_j; INTACT = the
            # history with its newest entry replaced by a_k
            var ctx_a = List[List[Scalar[DT]]]()
            if aligned:
                var L = min(REF_CTX, len(embs))
                for j in range(len(embs) - L, len(embs) - 1):
                    ctx_a.append(acts[j].copy())
                ctx_a.append(a.copy())
            else:
                var L = min(REF_CTX, min(len(embs), len(acts)))
                for j in range(len(acts) - L, len(acts) - 1):
                    ctx_a.append(acts[j].copy())
                ctx_a.append(a.copy())
            var L = len(ctx_a)
            var blocks = List[Scalar[DT]](length=Self.HORIZON * REF_ACT, fill=Scalar[DT](0))
            for j in range(L):
                for i in range(REF_ACT):
                    blocks[j * REF_ACT + i] = ctx_a[j][i]
            var ae = self.roll.embed_actions(blocks)
            var x = List[Scalar[DT]](length=REF_CTX * D, fill=Scalar[DT](0))
            var c = List[Scalar[DT]](length=REF_CTX * D, fill=Scalar[DT](0))
            for j in range(L):
                ref e = embs[len(embs) - L + j]
                for d in range(D):
                    x[j * D + d] = e[d]
                    c[j * D + d] = ae[j * D + d]
            var nxt = self.roll.predict_ctx(x, c, L)
            embs.append(nxt^)
            acts.append(a^)
        return plan^

# +--------------------------------------------------------------------------+ #
# | SmolVLA — one supervised training step over the action heads and expert
# +--------------------------------------------------------------------------+ #
"""Forward to a loss and back to every trainable weight, in one call.

This is the `train_expert_only` + `train_state_proj = False` regime: the
sixteen VLM layers and the SigLIP tower are frozen AND upstream of nothing
that is trained, so the prefix is a constant and the whole gradient lives
below the KV cache. `SmolVLADenoise.backward` forms dL/d(cached prefix K/V)
and drops it; restoring `train_state_proj = True` is what picks it up.

    x_t ──> action_in ──┐
                        ├─> token_concat ─> time_mlp_in ─> SiLU ─> time_mlp_out
    t ──> sinusoidal ───┘                                              │
                                                                       v
                              expert (16 layers, against the KV cache) │
                                                                       v
                                              action_out ──> v_t ──> MSE(u_t)

⚠ **ONE denoising step, at ONE sampled `t`.** Inference runs ten Euler steps;
training runs none. Unrolling the sampler here would be a different objective
(and sixteen times the memory) — see §10.2 of the plan for why prefill + one
step IS the reference's joint training pass.

⚠ **`t` is per SAMPLE.** The time embedding is `[B, EW]` and `token_concat`
gets `B_STRIDE = EW`. Passing a single vector, as inference does, would give
every element of the batch the same timestep, which trains a narrower
distribution than the sampler draws and would never show up as an error.

⚠ **Gradients ACCUMULATE** into every `Param.grd`, per the `nn` convention.
The caller zeroes before the step. Two `run` calls without a zero between them
give a two-batch gradient, which is either exactly what you wanted or a silent
factor of two.

⚠ **`denoise` must be `RECORD = True`** and must be the same instance the
forward used, because the tape it walks is that instance's. `backward` asserts
the first; nothing can assert the second.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.silu import SiLU

from .text import (
    SMOLLM_DIM, SMOLLM_KV_W, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, SMOLLM_LAYERS,
    SMOLLM_HEADS, SMOLLM_THETA,
)
from .heads import SMOLVLA_EXPERT_W, SMOLVLA_ACTION_DIM
from .expert import SmolVLAExpert, EXPERT_FF
from .kv_cache import SmolVLAKVCache
from .fused import SmolVLADenoise
from .flow import token_concat, token_split_a
from .embed import sinusoidal_time_embedding
from .flow_loss import flow_mse, mean_err


struct SmolVLATrainStep[
    CHUNK: Int,
    ADIM_REAL: Int,
    ADIM: Int = SMOLVLA_ACTION_DIM,
    EW: Int = SMOLVLA_EXPERT_W,
    B: Int = 1,
    LAYERS: Int = SMOLLM_LAYERS,
    # ⚠ The expert's geometry, parameterised rather than named from the
    # canonical aliases. Fixing them at the checkpoint's widths would make the
    # only testable fixture the real one — 50M parameters, ~700 MFLOP per
    # forward — and a finite-difference gate needs hundreds of forwards. The
    # defaults ARE the checkpoint, so a caller that wants the real model still
    # writes nothing.
    EFF: Int = EXPERT_FF,
    W: Int = SMOLLM_DIM,
    HEADS: Int = SMOLLM_HEADS,
    N_KV: Int = SMOLLM_KV_HEADS,
    HD: Int = SMOLLM_HEAD_DIM,
    THETA: Float64 = SMOLLM_THETA,
    KVW: Int = SMOLLM_KV_W,
    E_R: Int = 0,
](Movable):
    """`ADIM` is the padded action width (32); `ADIM_REAL` is the robot's.
    `E_R` is the expert's LoRA rank (`layer_weights.mojo`), 0 for none."""

    comptime XN: Int = Self.B * Self.CHUNK * Self.ADIM
    comptime AN: Int = Self.B * Self.CHUNK * Self.EW
    comptime CN: Int = Self.B * Self.CHUNK * 2 * Self.EW
    comptime TN: Int = Self.B * Self.EW
    comptime Act = SiLU[Self.CHUNK * Self.EW]

    comptime AEMB = 0
    comptime TEMB = 1
    comptime CAT = 2
    comptime MID = 3
    comptime SIL = 4
    comptime OUT = 5
    comptime SUF = 6
    comptime V = 7
    comptime ERR = 8
    comptime GV = 9
    comptime GSUF = 10
    comptime GOUT = 11
    comptime GSIL = 12
    comptime GMID = 13
    comptime GCAT = 14
    comptime GAEMB = 15
    comptime GXT = 16
    comptime GCK = 17     # dL/d(cached prefix K) — stage 7's path to the VLM
    comptime GCV = 18
    comptime N_SLOTS = 19

    var act: Self.Act
    var pool: TensorPack[Self.N_SLOTS]
    var profile: Bool
    """⚠ OFF by default. On, `run` SYNCHRONISES twice — after the forward and
    after the backward — so `ns_fwd` / `ns_bwd` are wall times of those phases
    rather than of their enqueues (`_a_per_call_sweep_is_an_upper_bound_on_a_
    step`: host timers across a run of pure enqueues measure the enqueues).
    The two drains cost a few hundred microseconds a step, which is why this is
    a flag and not the default."""
    var ns_fwd: Int
    var ns_bwd: Int

    def __init__(out self):
        comptime assert Self.ADIM_REAL <= Self.ADIM, (
            "SmolVLATrainStep: ADIM_REAL cannot exceed the padded ADIM"
        )
        self.act = Self.Act()
        self.pool = TensorPack[Self.N_SLOTS]()
        self.profile = False
        self.ns_fwd = 0
        self.ns_bwd = 0

    def __init__(out self, *, deinit move: Self):
        self.act = move.act^
        self.pool = move.pool^
        self.profile = move.profile
        self.ns_fwd = move.ns_fwd
        self.ns_bwd = move.ns_bwd

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        s.act = Self.Act.make[target, Deterministic](ctx)
        return s^

    def set_times[
        target: StaticString
    ](
        mut self, ref times: List[Float64],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Build the `[B, EW]` sinusoidal embedding for this step's timesteps.

        ⚠ Recomputed per step and uploaded, unlike the sampler's table: the
        sampler's ten timesteps are a comptime schedule, these are drawn. One
        upload per training step against a whole forward and backward is not
        where the time goes.
        """
        if len(times) != Self.B:
            raise Error(
                "SmolVLATrainStep.set_times: expected " + String(Self.B)
                + " timesteps, one per batch element, got "
                + String(len(times))
            )
        self.pool[Self.TEMB].ensure(Self.TN)
        for b in range(Self.B):
            var te = sinusoidal_time_embedding[Self.EW](times[b])
            for i in range(Self.EW):
                self.pool[Self.TEMB].data[b * Self.EW + i] = te[i]
        comptime if target != "cpu":
            self.pool[Self.TEMB].upload_resident(ctx.value())

    def run[
        target: StaticString, P: Int
    ](
        mut self,
        mut expert: SmolVLAExpert[
            Self.LAYERS, Self.EW, Self.EFF, Self.W, Self.KVW, 2, Self.E_R
        ],
        mut cache: SmolVLAKVCache[
            Self.LAYERS, P, Self.CHUNK, Self.N_KV, Self.HD, Self.B
        ],
        # ⚠ Spelled from Self's parameters, never re-derived. Mojo unifies
        # comptime parameters by EXPRESSION, so `5 * (960 // 15)` and `320`
        # are the same number and different types.
        mut denoise: SmolVLADenoise[
            P, Self.CHUNK, Self.B, Self.LAYERS, Self.EW, Self.EFF, Self.W,
            Self.HEADS, Self.N_KV, Self.HD, Self.THETA, 2, Self.KVW, True,
            Self.E_R,
        ],
        mut action_in: Linear[Self.ADIM, Self.EW],
        mut time_mlp_in: Linear[2 * Self.EW, Self.EW],
        mut time_mlp_out: Linear[Self.EW, Self.EW],
        mut action_out: Linear[Self.EW, Self.ADIM],
        mut x_t: Tensor,
        mut u_t: Tensor,
        mut valid: Tensor,
        n_terms: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Float64:
        """One step. Returns the loss; call `set_times` first.

        `valid` is `[B, CHUNK]` marking the timesteps inside their episode.
        `n_terms` is the averaging denominator: this call's count of 1s when
        used alone, the whole accumulation GROUP's when several calls sum
        their gradients before one optimizer step. A chunk sampled near an episode's end
        has its tail CLAMPED to the last real action by the dataset, and
        training on that teaches the model that episodes end by holding
        still.

        ⚠ Returning the loss SYNCHRONISES — `mean_err` brings the per-element
        errors back to the host. That is one drain per training step, which is
        what you already pay to log a loss.
        """
        comptime TOK = Self.B * Self.CHUNK
        var t_fwd = 0
        if self.profile:
            # ⚠ Drain first, so the forward's clock does not start while the
            # caller's prefix is still executing.
            comptime if target != "cpu":
                ctx.value().synchronize()
            t_fwd = Int(perf_counter_ns())

        # ── forward: embed_suffix ────────────────────────────────────────
        action_in.forward[target, TOK](
            TensorRefs[1](x_t), self.pool[Self.AEMB], ctx
        )
        token_concat[
            target, Self.B, Self.CHUNK, Self.EW, Self.EW, Self.EW
        ](
            self.pool[Self.AEMB], self.pool[Self.TEMB], self.pool[Self.CAT],
            ctx,
        )
        time_mlp_in.forward[target, TOK](
            TensorRefs[1](self.pool[Self.CAT]), self.pool[Self.MID], ctx
        )
        self.act.forward[target, Self.B](
            TensorRefs[1](self.pool[Self.MID]), self.pool[Self.SIL], ctx
        )
        time_mlp_out.forward[target, TOK](
            TensorRefs[1](self.pool[Self.SIL]), self.pool[Self.OUT], ctx
        )

        # ── forward: the expert, recording its tape ──────────────────────
        denoise.step[target](
            expert, cache, self.pool[Self.OUT], self.pool[Self.SUF], ctx
        )
        action_out.forward[target, TOK](
            TensorRefs[1](self.pool[Self.SUF]), self.pool[Self.V], ctx
        )

        # ── the loss ─────────────────────────────────────────────────────
        flow_mse[
            target, Self.B, Self.CHUNK, Self.ADIM, Self.ADIM_REAL
        ](
            self.pool[Self.V], u_t, valid, self.pool[Self.GV],
            self.pool[Self.ERR], n_terms, ctx,
        )
        var t_bwd = 0
        if self.profile:
            comptime if target != "cpu":
                ctx.value().synchronize()
            t_bwd = Int(perf_counter_ns())
            self.ns_fwd += t_bwd - t_fwd

        # ── backward ─────────────────────────────────────────────────────
        action_out.vjp[target, TOK](
            TensorRefs[1](self.pool[Self.SUF]), self.pool[Self.GV],
            TensorRefs[1](self.pool[Self.GSUF]), ctx,
        )
        # ⚠ GCK/GCV receive dL/d(the cached prefix K/V) — the gradient into
        # the frozen VLM and so into `state_proj`. This driver is the
        # `train_state_proj = False` regime and does not consume them; they
        # are kept as named slots rather than dropped so the omission is
        # visible here instead of inferred from what is missing.
        denoise.backward[target](
            expert, cache, self.pool[Self.GSUF], self.pool[Self.GOUT],
            self.pool[Self.GCK], self.pool[Self.GCV], ctx,
        )
        time_mlp_out.vjp[target, TOK](
            TensorRefs[1](self.pool[Self.SIL]), self.pool[Self.GOUT],
            TensorRefs[1](self.pool[Self.GSIL]), ctx,
        )
        self.act.vjp[target, Self.B](
            TensorRefs[1](self.pool[Self.MID]), self.pool[Self.GSIL],
            TensorRefs[1](self.pool[Self.GMID]), ctx,
        )
        time_mlp_in.vjp[target, TOK](
            TensorRefs[1](self.pool[Self.CAT]), self.pool[Self.GMID],
            TensorRefs[1](self.pool[Self.GCAT]), ctx,
        )
        # ⚠ Only the action half of the concat carries a gradient anywhere.
        # The time half's would be the sum over tokens, and the sinusoidal
        # embedding has no parameter for it to reach.
        token_split_a[target, Self.B, Self.CHUNK, Self.EW, Self.EW](
            self.pool[Self.GCAT], self.pool[Self.GAEMB], ctx
        )
        # `x_t` is DATA, so GXT is written and dropped — but `action_in.vjp`
        # needs a destination, and forming it is what accumulates action_in's
        # own weight gradient.
        action_in.vjp[target, TOK](
            TensorRefs[1](x_t), self.pool[Self.GAEMB],
            TensorRefs[1](self.pool[Self.GXT]), ctx,
        )
        if self.profile:
            comptime if target != "cpu":
                ctx.value().synchronize()
            self.ns_bwd += Int(perf_counter_ns()) - t_bwd

        return mean_err[
            target, Self.B, Self.CHUNK, Self.ADIM, Self.ADIM_REAL
        ](self.pool[Self.ERR], n_terms, ctx)

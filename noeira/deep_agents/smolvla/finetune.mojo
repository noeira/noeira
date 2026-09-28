# +--------------------------------------------------------------------------+ #
# | SmolVLA — what gets trained, and the optimizer walk over exactly that set
# +--------------------------------------------------------------------------+ #
"""The trainable set, written down ONCE.

`train_expert_only = True` makes the trainable parameters five components, and
`train_state_proj` decides whether there is a sixth:

    the 16 expert layers + the expert's final norm
    action_in       [ADIM -> EW]
    time_mlp_in     [2*EW -> EW]
    time_mlp_out    [EW   -> EW]
    action_out      [EW   -> ADIM]
    state_proj      [32   -> 960]        only when TRAIN_STATE_PROJ

Everything else — the SigLIP tower, the sixteen VLM layers, the connector, the
token embedding, `state_proj` — is frozen, and under this regime is also
upstream of nothing that is trained, so no gradient is even formed for it.

## Why these two functions exist rather than five call sites

A training loop has to do three things to each component: zero its gradient,
accumulate into it, apply the update. Miss one component from the ZERO list
and its gradient accumulates across every step of the run — a growing,
plausible number that eventually dominates. Miss one from the UPDATE list and
it silently never trains at all. Neither raises, neither shows up in a loss
curve, and both would be found months later by a model that is worse than it
should be for reasons nobody can point at.

So the set is enumerated exactly twice, here, adjacent, in the same order.
`test_finetune_overfit.mojo` then asserts that after a run EVERY parameter
group has actually moved, which is the check that catches an omission from
either list.

⚠ **`TRAIN_STATE_PROJ` is a COMPTIME parameter on both walks, not a runtime
flag and not a sixth call site.** The shipped config trains `state_proj`, and
it is the one trainable parameter that sits UPSTREAM of the frozen tower — its
gradient arrives only after a full backward through all sixteen VLM layers
(`SmolVLAPrefill.backward`). Turning it on therefore changes what a training
step must compute, not just which parameters it updates, so the flag being
visible in the type is worth more than the convenience of a bool.

⚠ **Two modes, and the caller picks with `adopt_trainables`.**

Un-adopted, `adam_step_trainables` walks the five components parameter by
parameter — one GPU kernel each, and no way to clip their JOINT gradient norm.

`adopt_trainables` packs all five into ONE `ParamArena`, which buys both: the
update becomes a single grouped kernel, and `clip_trainables` can clip the
global norm the reference asks for at 10. ⚠ It cannot be five calls to
`Adam.adopt` — that one says "call ONCE" and means it, resetting the arena
each time, so five calls leave only the last component adopted while
`adopted` reads True. `adopt_multi` exists for exactly this.

⚠ **Clipping each component to 10 independently is NOT clipping their joint
norm to 10** — with five components the total can pass through at up to
sqrt(5)x the limit. That is why the arena is what unlocks the clip rather
than being merely faster.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.param import ParamVersionBump, walk_params
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.checkpoint import (
    CheckpointScalars, save_params_multi, load_params_multi,
)
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.linear import Linear

from .expert import SmolVLAExpert
from .grad_ops import suffix_tail


def state_proj_backward[
    target: StaticString, B: Int, P: Int, W: Int, SDIM: Int
](
    mut state_proj: Linear[SDIM, W],
    mut state: Tensor,
    mut grad_x: Tensor,
    mut g_tok: Tensor,
    mut grad_state: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`state_proj`'s weight gradient, from the prefix embeddings' gradient.

    The state is the LAST prefix token — `smolvla_ar` lays out image, then
    language, then state — and it is written UNSCALED, unlike the image and
    language segments which carry a sqrt(W) factor. So its slice of `grad_x`
    is dL/d(state_proj's output) directly, with nothing to undo.

    ⚠ The slice is taken with `suffix_tail`, the same routine that separates
    the expert's K/V from the VLM's in the denoise backward. Same batch-major
    rule, one implementation: `[B, P*W]` with the last `W` of each row.

    `grad_state` is written and discarded — the robot's pose is data — but
    `Linear.vjp` needs a destination, and forming it is what accumulates
    `state_proj.weight.grd`.
    """
    suffix_tail[target, B, P * W, (P - 1) * W, W](grad_x, g_tok, ctx)
    state_proj.vjp[target, B](
        TensorRefs[1](state), g_tok, TensorRefs[1](grad_state), ctx
    )


def adopt_trainables[
    target: StaticString,
    LAYERS: Int, EW: Int, EFF: Int, W: Int, KVW: Int, ADIM: Int,
    SDIM: Int = 32, VW: Int = 960, TRAIN_STATE_PROJ: Bool = False,
](
    mut opt: Adam,
    mut expert: SmolVLAExpert[LAYERS, EW, EFF, W, KVW, 2],
    mut action_in: Linear[ADIM, EW],
    mut time_mlp_in: Linear[2 * EW, EW],
    mut time_mlp_out: Linear[EW, EW],
    mut action_out: Linear[EW, ADIM],
    mut state_proj: Linear[SDIM, VW],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Pack the trainable set into one arena. GPU only; a no-op on CPU.

    ⚠ Call ONCE, after the weights are loaded and before the first step —
    adopting REBINDS every Param's buffers to arena slices, so anything that
    wrote to them beforehand is preserved (the arena copies values in) and
    anything holding a stale handle is not.

    ⚠ The SAME five components in the SAME order as every other walk here.
    """
    opt.adopt_multi[target](
        ctx, expert, action_in, time_mlp_in, time_mlp_out, action_out
    )
    _ = state_proj
    comptime if TRAIN_STATE_PROJ:
        raise Error(
            "adopt_trainables: TRAIN_STATE_PROJ is not wired into the arena"
            " yet — state_proj would train un-adopted while the other five"
            " are adopted, which is two optimizers on one model. Use the"
            " per-parameter path for that regime."
        )


def clip_trainables[
    target: StaticString
](
    mut opt: Adam, max_norm: Scalar[DT], ctx: Optional[DeviceContext] = None
) raises -> Scalar[DT]:
    """Global grad-norm clip over the adopted trainable set; the pre-clip norm.

    `configuration_smolvla.py` sets `optimizer_grad_clip_norm = 10`. Needs
    `adopt_trainables` first — there is no arena to clip otherwise, and
    clipping the five components one at a time is a different operation.
    """
    comptime if target == "cpu":
        return Scalar[DT](0)
    return opt.arena_clip(max_norm, ctx.value())


def zero_trainable_grads[
    target: StaticString,
    LAYERS: Int, EW: Int, EFF: Int, W: Int, KVW: Int, ADIM: Int,
    SDIM: Int = 32, VW: Int = 960, TRAIN_STATE_PROJ: Bool = False,
](
    mut opt: Adam,
    mut expert: SmolVLAExpert[LAYERS, EW, EFF, W, KVW, 2],
    mut action_in: Linear[ADIM, EW],
    mut time_mlp_in: Linear[2 * EW, EW],
    mut time_mlp_out: Linear[EW, EW],
    mut action_out: Linear[EW, ADIM],
    mut state_proj: Linear[SDIM, VW],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Zero every trainable gradient. Call ONCE per step, before the forward.

    ⚠ Takes the optimizer because whether there IS an arena is the
    optimizer's state, and zeroing an adopted set component-by-component
    would work while quietly being five fills instead of one.

    ⚠ `Linear.vjp` ACCUMULATES (`grad_w += ...`), which is the `nn` convention
    and is what makes gradient accumulation across micro-batches possible. It
    also means a forgotten zero is a silent running sum.
    """
    # ⚠ Adopted, the whole arena zeroes in ONE fill; the per-component walk
    # would zero the same memory five times and, worse, read as if the two
    # paths were interchangeable when only one of them can be clipped.
    comptime if target == "gpu":
        if opt.arena.adopted:
            opt.arena.zero_grad(ctx.value())
            return
    expert.zero_grad[target](ctx)
    action_in.zero_grad[target](ctx)
    time_mlp_in.zero_grad[target](ctx)
    time_mlp_out.zero_grad[target](ctx)
    action_out.zero_grad[target](ctx)
    comptime if TRAIN_STATE_PROJ:
        state_proj.zero_grad[target](ctx)


def adam_step_trainables[
    target: StaticString,
    LAYERS: Int, EW: Int, EFF: Int, W: Int, KVW: Int, ADIM: Int,
    SDIM: Int = 32, VW: Int = 960, TRAIN_STATE_PROJ: Bool = False,
](
    mut opt: Adam,
    mut expert: SmolVLAExpert[LAYERS, EW, EFF, W, KVW, 2],
    mut action_in: Linear[ADIM, EW],
    mut time_mlp_in: Linear[2 * EW, EW],
    mut time_mlp_out: Linear[EW, EW],
    mut action_out: Linear[EW, ADIM],
    mut state_proj: Linear[SDIM, VW],
    ctx: Optional[DeviceContext] = None,
) raises:
    """One Adam update over the same five components `zero_trainable_grads`
    covers.

    ⚠ `begin_step()` is called ONCE and then five walks follow. `Adam.step`
    bundles the two, which is right for a single model and wrong here: calling
    it five times would advance the step counter five times per optimizer
    step, so the bias corrections would run ahead of the moments and the early
    steps would take the wrong size.
    """
    comptime if target == "gpu":
        if opt.arena.adopted:
            # ONE grouped kernel over the whole trainable set.
            opt.arena_step(ctx.value())
            var b = ParamVersionBump()
            walk_params[target](expert, b, ctx, String("expert"))
            walk_params[target](action_in, b, ctx, String("action_in"))
            walk_params[target](time_mlp_in, b, ctx, String("time_mlp_in"))
            walk_params[target](time_mlp_out, b, ctx, String("time_mlp_out")
            )
            walk_params[target](action_out, b, ctx, String("action_out"))
            return
    opt.begin_step()
    walk_params[target](expert, opt, ctx, String("expert"))
    walk_params[target](action_in, opt, ctx, String("action_in"))
    walk_params[target](time_mlp_in, opt, ctx, String("time_mlp_in"))
    walk_params[target](time_mlp_out, opt, ctx, String("time_mlp_out"))
    walk_params[target](action_out, opt, ctx, String("action_out"))
    comptime if TRAIN_STATE_PROJ:
        walk_params[target](state_proj, opt, ctx, String("state_proj"))

    # ⚠ The version bump `Adam.step` does after its walk, which is NOT
    # cosmetic: leaves that cache a derived form of a weight (the bf16 cast,
    # split-K's padded copy) gate that cache on the version, and a bump that
    # never happens leaves the forward reading pre-update weights forever.
    # That exact defect has been shipped here before.
    var bump = ParamVersionBump()
    walk_params[target](expert, bump, ctx, String("expert"))
    walk_params[target](action_in, bump, ctx, String("action_in"))
    walk_params[target](time_mlp_in, bump, ctx, String("time_mlp_in"))
    walk_params[target](time_mlp_out, bump, ctx, String("time_mlp_out"))
    walk_params[target](action_out, bump, ctx, String("action_out"))
    comptime if TRAIN_STATE_PROJ:
        walk_params[target](state_proj, bump, ctx, String("state_proj"))


def save_trainables[
    target: StaticString,
    LAYERS: Int, EW: Int, EFF: Int, W: Int, KVW: Int, ADIM: Int,
    SDIM: Int = 32, VW: Int = 960, TRAIN_STATE_PROJ: Bool = False,
](
    path: String,
    mut expert: SmolVLAExpert[LAYERS, EW, EFF, W, KVW, 2],
    mut action_in: Linear[ADIM, EW],
    mut time_mlp_in: Linear[2 * EW, EW],
    mut time_mlp_out: Linear[EW, EW],
    mut action_out: Linear[EW, ADIM],
    mut state_proj: Linear[SDIM, VW],
    save_moments: Bool = True,
    ctx: Optional[DeviceContext] = None,
    scalars: CheckpointScalars = CheckpointScalars(),
) raises:
    """Write the trainable set to one v3 checkpoint, `scalars` as its `K`
    sections — the caller's `opt.put_step_state(sc, "opt")`, without which
    the moments below resume under `t = 0` (see `Adam.put_step_state`).

    ⚠ **ONLY the trainable set** — the SigLIP tower, the sixteen VLM layers,
    the connector and the token embedding are frozen and are already on disk
    as `lerobot/smolvla_base`. Saving them again would triple the file to no
    purpose and, worse, make a checkpoint that could silently disagree with
    the base it was fine-tuned from. The reload path loads the base FIRST and
    then this on top, so the two can never drift apart.

    ⚠ `save_moments=True` writes Adam's per-parameter m/v alongside, which is
    what makes a resume EXACT rather than a restart with a cold optimizer —
    and a cold optimizer is precisely the thing whose first step damages a
    pretrained model (see the runner's warmup note). It roughly triples the
    file: ~98 M parameters is 393 MB of weights and 1.2 GB with moments.

    ⚠ The SAME components, in the SAME order, as `zero_trainable_grads` and
    `adam_step_trainables`. A fourth list is a fourth chance for one of them
    to drift; `load_trainables` below reads them back in this order and the
    v3 format VALIDATES each section's dotted name against the walk, so a
    disagreement raises instead of loading a shifted model.
    """
    comptime if TRAIN_STATE_PROJ:
        save_params_multi[target](
            path, ctx, save_moments, expert, action_in, time_mlp_in,
            time_mlp_out, action_out, state_proj, scalars=scalars,
        )
    else:
        save_params_multi[target](
            path, ctx, save_moments, expert, action_in, time_mlp_in,
            time_mlp_out, action_out, scalars=scalars,
        )


def load_trainables[
    target: StaticString,
    LAYERS: Int, EW: Int, EFF: Int, W: Int, KVW: Int, ADIM: Int,
    SDIM: Int = 32, VW: Int = 960, TRAIN_STATE_PROJ: Bool = False,
](
    path: String,
    mut expert: SmolVLAExpert[LAYERS, EW, EFF, W, KVW, 2],
    mut action_in: Linear[ADIM, EW],
    mut time_mlp_in: Linear[2 * EW, EW],
    mut time_mlp_out: Linear[EW, EW],
    mut action_out: Linear[EW, ADIM],
    mut state_proj: Linear[SDIM, VW],
    ctx: Optional[DeviceContext] = None,
) raises -> CheckpointScalars:
    """Read back what `save_trainables` wrote, over an already-loaded base.
    Returns the file's `K` scalars — hand them to `opt.take_step_state` on a
    resume; a deployment can ignore them.

    ⚠ Load the BASE checkpoint first. This file holds only the trainable set;
    applying it to a freshly initialised policy leaves the vision tower and
    the VLM at their initialiser, which produces finite actions from a random
    prefix and no error anywhere.
    """
    comptime if TRAIN_STATE_PROJ:
        return load_params_multi[target](
            path, ctx, expert, action_in, time_mlp_in, time_mlp_out,
            action_out, state_proj,
        )
    else:
        return load_params_multi[target](
            path, ctx, expert, action_in, time_mlp_in, time_mlp_out,
            action_out,
        )

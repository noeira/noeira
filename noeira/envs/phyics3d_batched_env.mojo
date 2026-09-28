"""GPU-batched physics3d environment on the PER-FIELD tensor path.

`Phyics3dBatchedEnv[MODEL_DEF, CONFIG, N_ENVS]` is the fields-path
counterpart of `BatchedGpuEnv[Phyics3dEnv[...]]`: it implements the
`BatchedEnv` driver ABI directly as a STATEFUL struct (it owns
`Data` / `Model` / `RK4Integrator` plus the driver IO
buffers), so the off-policy GPU drivers (`run_offpolicy_train_batched`,
i.e. `SACAgent.train[...]` etc.) train against fields-path physics with
zero driver changes.

Model: fully on the fields path — the model is built offset-free via
`MODEL_DEF.init_fields` (spec-direct, offset-free),
and the three former model-slab consumers now read `Model` directly:
reset FK -> `forward_kinematics`, `compute_cfrc_ext` (reads
`mf.bodies`), and the reward hook's curriculum params (fed `mf.curriculum`
as a `[1, MODEL_CURRICULUM_SIZE]` view with offset 0 — the CONFIG hook is
generic over MODEL_SIZE + curriculum_offset, so it is UNCHANGED).

G5: the state slab is GONE. Every GPU hook (`MODEL_DEF.reset_env_gpu` /
`apply_actions_kernel_gpu` / `extract_obs_gpu` and the CONFIG
`pre_step_gpu` / `init_qpos_gpu` / `custom_extract_obs_gpu` /
`compute_reward_and_done_gpu`) takes the per-field tensors it needs, and
`compute_cfrc_ext` / `compute_cvel` read/write the Data
cfrc_ext/cvel tensors directly. Hook state (step_count / prev_x) lives in
`d.meta` alongside num_contacts (detection writes ONLY the num_contacts
slot). The hook arithmetic is unchanged — only the addressing moved from
slab+offset to field tensors.

Scope (mirrors `Phyics3dEnv`): contacts + joint limits via the
per-stage RK4 PGS/Newton solve — hopper/walker-class locomotion in scope;
fluid-force models (Swimmer) run via the integrators' passive seam
(Stage A). Mesh-collision models are not in scope.

"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.target_storage import require_ctx

from noeira.deep_agents.training.batched_env import BatchedEnv

from noeira.physics3d.model.model_def import ModelDefLike
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import (
    Data, Model, SpecFields, Dims, DimsLike, AsStatic, DynamicsScratch,
)
from noeira.physics3d.dynamics.osc_control import (
    osc_control_step, osc_reset_batch,
)
from noeira.physics3d.dynamics.osc_pose_gpu import (
    OSC_ACTION_DIM, OSC_IDX_SINGULAR, OSC_REF_WORDS, OSC_STATE_WORDS,
    OSC_WORK_WORDS,
)
from noeira.physics3d.integrator.rk4 import RK4Integrator
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from noeira.physics3d.collision.broadphase_sap import detect_contacts_auto
from noeira.physics3d.joint_types import JNT_FREE
from noeira.physics3d.gpu import compute_cfrc_ext, compute_cvel
from noeira.physics3d.gpu.constants import (
    MAX_GPU_MESHES,
    MESH_ARENA_FLOATS_PER_TRI,
    MODEL_MESH_META_SIZE,
    MAX_GPU_HFIELDS,
    MODEL_HFIELD_META_SIZE,
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    JLIM_SIZE,
    POSE_META_SIZE,
    TPB,
    MODEL_BODY_SIZE,
    BODY_IDX_MOCAP,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_TENDON_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    METADATA_SIZE,
    META_IDX_STEP_COUNT,
    META_IDX_PHI_PREV,
    META_IDX_EPISODE_FLAGS,
    META_IDX_TASK_PARAM_0,
    META_IDX_TASK_PARAM_6,
    META_IDX_NUM_CONTACTS,
    MODEL_CURRICULUM_SIZE,
)

from .phyics3d_env_config import Phyics3dEnvConfig
from noeira.physics3d.model.model_dims import ModelDims


# ──────────────────────────────────────────────────────────────────────
# Sync kernels (transitional slab<->fields bridges; die at P6)
# ──────────────────────────────────────────────────────────────────────


def _rng_counter_bump_kernel(
    counter: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """Bump the device-resident env-reset RNG counter by 1 (capture-safe
    reset randomness; mirrors the legacy `increment_env_rng_kernel`)."""
    if Int(thread_idx.x) == 0:
        counter[0] = counter[0] + UInt64(1)


# ──────────────────────────────────────────────────────────────────────
# The batched fields env
# ──────────────────────────────────────────────────────────────────────


struct Phyics3dBatchedEnv[
    MODEL_DEF: ModelDefLike,
    CONFIG: Phyics3dEnvConfig,
    N_ENVS: Int,
    TERMINATE_ON_UNHEALTHY: Bool = False,
    SOLVER: StaticString = "newton",
    PARALLEL_GPU: Bool = True,
    CRBA_TREEWALK: Bool = True,
](BatchedEnv):
    """GPU-batched MuJoCo env, physics on the per-field tensor path,
    driver IO via the `BatchedEnv` ABI. See module docstring.

    The physics defaults are the LEGACY PRODUCTION bundle
    (CONFIG.physics_substep_gpu = RK4 + Newton at STEP_THREADS=NV with
    RK4_PARALLEL_* on): SOLVER="newton", PARALLEL_GPU=True (cooperative
    _mt kernels, bit-exact vs serial), CRBA_TREEWALK=True (the
    production mass-matrix algorithm — tolerance-equal to dense at
    ~1e-8/eval, exactly as in legacy). Gates that need a bit-exact
    baseline against a serial legacy reference pin these params
    explicitly."""

    comptime ENV_TARGET: StaticString = "gpu"
    comptime OBS_DIM: Int = Self.MODEL_DEF.OBS_DIM
    comptime ACT_DIM: Int = Self.MODEL_DEF.ACTION_DIM

    comptime NQ: Int = Self.MODEL_DEF.NQ

    # ⚠ ONE PROVIDER. `nmesh_verts` stays 0 — the batched path has never run
    # mesh collision, and `ModelDims` defaults it to 0, so this is the same
    # model it always built.
    #
    # ⚠⚠ `nhfield_data` DOES NOT. It defaulted to 0 here, and `Data.hfield_data`
    # is `_at_least_one(BATCH * nhfield_data)` — so a config declaring a
    # 201x201 terrain got a ONE-ELEMENT buffer on the batched path while its
    # single-env twin got 40401. Nothing caught it because no batched model had
    # a heightfield until escape: the grid stayed one element wide and the
    # terrain collided as a flat plane, silently, exactly as
    # `DMQuadrupedEscapeConfig.NHFIELD_DATA` warns for the single-env path.
    #
    # ⚠⚠ AND THE SAME BUG, ONE FIELD OVER: `nmesh_verts` was left at its
    # default 0 too, so NO BATCHED MODEL HAS EVER CARRIED A COLLIDABLE MESH.
    # It surfaced on SO-ARM101 (26 179 hull vertices) as a hard raise from
    # `fields_build`, not silently — that assert was added for exactly this —
    # but every mesh-collidable model was unreachable on this path until now.
    # Threaded from the config for the same reason `nhfield_data` is: the
    # comptime parser cannot read an STL, so the count is hand-supplied.
    comptime MD = ModelDims[
        Self.MODEL_DEF,
        nmesh_verts = Self.CONFIG.NMESH_VERTS,
        nhfield_data = Self.CONFIG.NHFIELD_DATA,
        nmesh_tri = Self.CONFIG.NMESH_TRI,
    ]
    comptime NV: Int = Self.MODEL_DEF.NV
    comptime NBODY: Int = Self.MODEL_DEF.NBODY
    comptime NJOINT: Int = Self.MODEL_DEF.NJOINT
    comptime MC: Int = Self.MODEL_DEF.MAX_CONTACTS
    comptime NGEOM: Int = Self.MODEL_DEF.NGEOM
    comptime NSITE: Int = Self.MODEL_DEF.NSITE
    comptime BLOCKS: Int = (Self.N_ENVS + TPB - 1) // TPB

    # ⚠ FLOORED AT ONE SITE. `Data.site_xpos` is `[BATCH, NSITE*3]` with no
    # zero-extent guard, and FIVE ported models have NSITE == 0 (pendulum,
    # cartpole, cheetah, walker — measured). Binding that as a kernel operand
    # would hand them a zero-extent tensor, which SEGFAULTS
    # (feedback_zero_extent_tensor_operand_crash). The hook ABI therefore
    # carries `SITE_DIM = max(NSITE, 1) * 3`, and for NSITE == 0 the operand is
    # a one-site DUMMY buffer that no hook can legally index — a model with no
    # sites has no site sensor to read.
    #
    # Floored HERE rather than in `Data`: the alloc there is deliberately
    # un-floored so `site_xpos_acc` can shadow it exactly (see fields/data.mojo),
    # and this is the only consumer that binds it unconditionally.
    comptime SITE_DIM: Int = (
        Self.NSITE if Self.NSITE > 0 else 1
    ) * 3
    comptime L_SITE_HOOK = Layout.row_major(Self.N_ENVS, Self.SITE_DIM)
    # `Model.sites` is ALREADY floored at the allocation
    # (`_at_least_one` in fields/model.mojo), so only the LAYOUT needs
    # flooring — no dummy buffer, unlike site_xpos above.
    comptime NSITE_F: Int = Self.NSITE if Self.NSITE > 0 else 1
    comptime L_SITES_HOOK = Layout.row_major(
        Self.NSITE_F, MODEL_SITE_SIZE
    )
    # `Model.geoms` is `_at_least_one`'d too, so again only the layout.
    comptime NGEOM_F: Int = Self.NGEOM if Self.NGEOM > 0 else 1
    comptime L_GEOMS_HOOK = Layout.row_major(
        Self.NGEOM_F, MODEL_GEOM_SIZE
    )
    # `Model.mesh_tris` is `_at_least_one`'d too — layout only. ⚠ ZERO UNLESS
    # THE CONFIG SETS `NMESH_TRI`, and zero means a MESH geom is invisible to
    # `ray_model` here — a ray goes straight through it. `ModelDims` did not
    # forward this at all until 2026-08-26, which went unnoticed because the
    # only ray consumer was `quadruped escape`, whose terrain is a heightfield
    # and whose robot is primitives.
    comptime NMESH_TRI_F: Int = (
        Self.MD.NMESH_TRI if Self.MD.NMESH_TRI > 0 else 1
    )
    comptime L_MESH_META_HOOK = Layout.row_major(
        MAX_GPU_MESHES * MODEL_MESH_META_SIZE
    )
    comptime L_MESH_TRIS_HOOK = Layout.row_major(
        Self.NMESH_TRI_F * MESH_ARENA_FLOATS_PER_TRI
    )
    # `Model.tendons` is `_at_least_one`'d too — layout only.
    comptime NTENDON_F: Int = (
        Self.MODEL_DEF.MAX_TENDON if Self.MODEL_DEF.MAX_TENDON > 0 else 1
    )
    comptime L_TENDONS_HOOK = Layout.row_major(
        Self.NTENDON_F, MODEL_TENDON_SIZE
    )

    # Fields path (the actual physics state)
    var d: Data[DT, Self.MD, Self.N_ENVS]
    var mf: Model[DT, Self.MD]
    # Actuation records (phase 1a.2/1a.3) — the operands
    # `apply_actions_kernel_gpu` reads where it used to read comptime
    # literals. Uploaded once at construction, like `mf`.
    var sf: SpecFields[DT, Self.MD]
    # Both integrators are held; the step comptime-dispatches on
    # CONFIG.INTEGRATOR (HalfCheetah/Pusher/MetaWorld = Euler+Newton, the
    # other 9 envs = RK4+Newton). Only the SELECTED one is `prepare_gpu`'d, so
    # the unused one allocates NO device memory.
    comptime IntegRK4 = RK4Integrator[
        DT, Self.MD, Self.MODEL_DEF.CONE_TYPE, Self.N_ENVS,
        SOLVER = Self.SOLVER, PARALLEL_GPU = Self.PARALLEL_GPU,
        CRBA_TREEWALK = Self.CRBA_TREEWALK,
        MAX_CONDIM = Self.MODEL_DEF.MAX_CONDIM,
        NOSLIP_ITER = Self.MODEL_DEF.NOSLIP_ITER,
    ]
    # ⚠⚠ MAX_CONDIM AND NOSLIP_ITER MUST COME FROM THE MODEL, NOT THE DEFAULT.
    # Both were previously left unpassed, so every batched env silently ran
    # `MAX_CONDIM=3` and `NOSLIP_ITER=0` regardless of its MJCF. That has been
    # invisible only because EVERY GPU-ported env so far is condim 3 with no
    # noslip — the moment one is not, the failure is silent and physical:
    #
    #   condim 6 at MAX_CONDIM=3 -> the pyramidal edge list is sized
    #     2*(3-1)=4 rows per contact instead of 2*(6-1)=10, so the TORSIONAL
    #     and ROLLING rows are never built (`model_def_from_xml` warns about
    #     exactly this), and the GPU quietly solves a different contact model
    #     than the CPU;
    #   noslip_iter 4 at 0 -> `mj_solNoSlip` never runs, dropping a
    #     friction-only post-pass that is first-order on dog.
    #
    # Passing the model's own values is a NO-OP for every currently-gated env
    # (all are condim 3 / noslip 0) and correct for the ones that are not.
    comptime IntegEuler = EulerIntegrator[
        DT, Self.MD, Self.MODEL_DEF.CONE_TYPE, Self.N_ENVS,
        SOLVER = Self.SOLVER, PARALLEL_GPU = Self.PARALLEL_GPU,
        CRBA_TREEWALK = Self.CRBA_TREEWALK,
        RNE_POST = Self.CONFIG.RNE_POST,
        MAX_CONDIM = Self.MODEL_DEF.MAX_CONDIM,
        NOSLIP_ITER = Self.MODEL_DEF.NOSLIP_ITER,
    ]
    var integ_rk4: Self.IntegRK4
    var integ_euler: Self.IntegEuler

    # Driver IO (env-owns-buffers per the BatchedEnv ABI)
    var _obs: DeviceBuffer[DT]
    var _action: DeviceBuffer[DT]
    var _reward: DeviceBuffer[DT]
    var _done: DeviceBuffer[DT]
    var _terminated: DeviceBuffer[DT]
    var _env_rng_counter: DeviceBuffer[DType.uint64]
    # Bound as `site_xpos` only when NSITE == 0; see SITE_DIM above.
    var _site_dummy: DeviceBuffer[DT]
    # E3 — per-env ACTUATOR ACTIVATION (MuJoCo `d->act`), the batched
    # twin of `Phyics3dEnv.act`. An actuator with a `dyntype` feeds its
    # ACTIVATION to the gain where a plain one feeds `ctrl`, and the
    # activation is a first-order lag of ctrl — so without this the
    # force is computed from the wrong quantity entirely. quadruped and
    # dog are the models that have one. Floored at 1 like the CPU side.
    comptime NA_F: Int = Self.MODEL_DEF.NA_F
    comptime L_HF_META_HOOK = Layout.row_major(
        MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE
    )
    comptime L_HF_DATA_HOOK = Layout.row_major(
        Self.N_ENVS * Self.NHFIELD_DATA_F
    )
    """⚠ FLAT, NOT `[N_ENVS, NHFIELD_DATA]`. `ray_model` takes the grid as a
    1-D tensor plus an `env * hf_stride` base offset, and the CPU path builds
    exactly that view. One layout then serves BOTH hooks — the terrain writer
    and the ray reader — instead of two views of one buffer that must be kept
    in step."""
    comptime NHFIELD_DATA_F = Self.CONFIG.NHFIELD_DATA if (
        Self.CONFIG.NHFIELD_DATA > 0
    ) else 1
    """⚠ `_at_least_one`. Every kernel binds `hfield_data` whether the model
    has a heightfield or not, and a zero-extent layout is not a buffer."""

    comptime L_ACT_HOOK = Layout.row_major(Self.N_ENVS, Self.NA_F)
    var _act: DeviceBuffer[DT]

    # ── OSC_POSE, when CONFIG.HAS_OSC_CONTROLLER ──────────────────────────
    #
    # ⚠⚠ THE SCRATCH IS A LIST OF 0 OR 1, NOT A FOLDED BATCH PARAMETER. A
    # struct field cannot be conditionally absent, and a full-batch
    # `DynamicsScratch` allocates three NV*NV host slabs — ~50 MB at 1024 lanes
    # and nv = 37 — which every env that does NOT use OSC would pay.
    #
    # The obvious fix, `BATCH = N_ENVS if HAS_OSC else 1`, does not compile:
    # inside the `comptime if` the two are equal but the parameter is still an
    # unfolded conditional at parse time, and `osc_control_step` then sees a
    # `Data[.., N_ENVS]` beside a `DynamicsScratch[.., OSC_LANES]` and refuses
    # ("types parameters include unfolded expression"). An empty `List` of the
    # N_ENVS-typed scratch costs nothing, keeps ONE type, and `[0]` inside the
    # branch is a `mut` reference of exactly the type the call wants.
    comptime OSC_NACT: Int = Self.MODEL_DEF.NACT_F
    var _osc_scratch: List[DynamicsScratch[DT, Self.MD, Self.N_ENVS]]
    var _osc_state: TensorImpl[DT]
    var _osc_work: TensorImpl[DT]
    var _osc_refs: TensorImpl[DT]
    var _osc_ctrl: TensorImpl[DT]
    var _osc_action: TensorImpl[DT]
    var _osc_ready: Bool
    """False until `set_osc_refs`. See `Phyics3dEnvConfig.HAS_OSC_CONTROLLER`:
    the record is a property of the composed scene and a comptime config cannot
    read it, so `step_batch` refuses rather than driving an all-zero record —
    which is a controller commanding joint 0 with gain 0, i.e. an arm that
    falls over slowly and a training curve that looks merely bad."""
    # 1.0 for a lane the last reset touched, 0.0 otherwise. Written by BOTH
    # reset kernels because `selective_reset_kernel` CLEARS `dones[i]` as it
    # resets, so afterwards nothing else records which lanes moved — and
    # `_find_non_contacting_height_batch` must not raise a lane that is
    # mid-episode.
    var _reset_mask: DeviceBuffer[DT]

    def __init__(out self, ctx: DeviceContext) raises:
        # ⚠ A CONFIG without GPU hooks inherits `compute_reward_and_done_gpu`'s
        # INERT DEFAULT and trains against a flat-zero reward curve — it compiles,
        # it runs, it reports episodes, and it learns nothing. Every dm_control
        # task config was in exactly that state for months (gap G10). Refuse the
        # instantiation instead. See Phyics3dEnvConfig.HAS_GPU_HOOKS.
        # ⚠ RNE_POST lives on the EULER integrator only — RK4 would need the
        # hook inside its base stage and no in-scope model wants both. Mirrors
        # the same assert in `Phyics3dEnv`; without it, a config asking for
        # RNE_POST under RK4 would silently get `cacc`/`cfrc_int` of zero and
        # its acceleration-stage sensors would read 0.
        comptime assert (
            (not Self.CONFIG.RNE_POST)
            or Self.CONFIG.INTEGRATOR == "euler"
        ), (
            "Phyics3dBatchedEnv: CONFIG.RNE_POST is wired into the Euler"
            " integrator only, but this CONFIG selects a different integrator."
            " The acceleration-stage sensors would read zero."
        )
        comptime assert Self.CONFIG.HAS_GPU_HOOKS, (
            "Phyics3dBatchedEnv: this CONFIG does not implement the GPU hooks"
            " (HAS_GPU_HOOKS is False), so its reward would be a constant 0 and"
            " its observation the model default. Implement"
            " compute_reward_and_done_gpu (+ custom_extract_obs_gpu if the model"
            " default is not the task's observation), then set HAS_GPU_HOOKS ="
            " True. See docs/DM_CONTROL_GPU_TRAINING_G10.md."
        )
        # Offset-free fields-native model build: init_fields runs
        # the spec-direct fields build and uploads every record
        # tensor (bodies/joints/meta/curriculum/…) — the reset FK, cfrc_ext,
        # and reward-curriculum hooks now read those directly.
        # The device buffers come FIRST. Every raising call in a constructor
        # carries an unwind path over the fields built before it; with these
        # nine after Model / Data / SpecFields that was nine copies of "free
        # sixty tensors" (docs/COMPILE_TIME_PROFILING.md §3.6).
        self._obs = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.OBS_DIM)
        self._action = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.ACT_DIM
        )
        self._reward = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._done = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._terminated = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._env_rng_counter = ctx.enqueue_create_buffer[DType.uint64](1)
        self._site_dummy = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.SITE_DIM
        )
        self._act = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.NA_F)
        self._reset_mask = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self.mf = type_of(self.mf)()
        Self.MODEL_DEF.init_fields[DT](ctx, self.mf)
        self.sf = type_of(self.sf)()
        Self.MODEL_DEF.init_spec_fields[DT](ctx, self.sf)
        # Fluid forces (density/viscosity) are handled by the fields
        # integrators' passive seam (Stage A: compute_fluid_forces), so
        # no guard — Swimmer runs on this facade.

        self.d = type_of(self.d)()
        self.d.upload_all(ctx)

        # ── OSC_POSE staging ──────────────────────────────────────────────
        # Allocated at OSC_LANES, which is 1 unless the controller is on.
        self._osc_scratch = List[
            DynamicsScratch[DT, Self.MD, Self.N_ENVS]
        ]()
        # ⚠ ONE ELEMENT, NOT ZERO, WHEN THE CONTROLLER IS OFF — a `TensorImpl`
        # of length zero is not a buffer, and these are allocated (never
        # uploaded) either way. The scratch above is the one that is skipped.
        var _osc_n = Self.N_ENVS if Self.CONFIG.HAS_OSC_CONTROLLER else 1
        self._osc_state = TensorImpl[DT].alloc(_osc_n * OSC_STATE_WORDS)
        self._osc_work = TensorImpl[DT].alloc(_osc_n * OSC_WORK_WORDS)
        self._osc_refs = TensorImpl[DT].alloc(OSC_REF_WORDS)
        self._osc_ctrl = TensorImpl[DT].alloc(_osc_n * Self.OSC_NACT)
        self._osc_action = TensorImpl[DT].alloc(_osc_n * OSC_ACTION_DIM)
        self._osc_ready = False
        comptime if Self.CONFIG.HAS_OSC_CONTROLLER:
            # ⚠ THE ACTION WIDTH IS PART OF THE CONTRACT, CHECKED HERE. A model
            # def that left `action_dim_override` at `nact` would hand the
            # controller nine numbers and read the gripper word out of the
            # middle of the arm's deltas — an arm that moves plausibly and
            # never closes its hand.
            comptime assert Self.ACT_DIM == OSC_ACTION_DIM, (
                "HAS_OSC_CONTROLLER needs ACTION_DIM == 7 (six pose deltas and"
                " the gripper); set `action_dim_override=7` on the model def."
            )
            self._osc_scratch.append(
                DynamicsScratch[DT, Self.MD, Self.N_ENVS]()
            )
            self._osc_scratch[0].upload_all(ctx)
            self._osc_state.upload(ctx)
            self._osc_work.upload(ctx)
            self._osc_refs.upload(ctx)
            self._osc_ctrl.upload(ctx)
            self._osc_action.upload(ctx)
        # Construct both (host scratch); prepare only the selected integrator so
        # the unused one allocates no device buffers.
        self.integ_rk4 = Self.IntegRK4()
        self.integ_euler = Self.IntegEuler()
        comptime if Self.CONFIG.INTEGRATOR == "euler":
            self.integ_euler.prepare_gpu(ctx)
        else:
            self.integ_rk4.prepare_gpu(ctx)

        ctx.enqueue_memset(self._obs, 0)
        ctx.enqueue_memset(self._action, 0)
        ctx.enqueue_memset(self._reward, 0)
        ctx.enqueue_memset(self._done, 0)
        ctx.enqueue_memset(self._terminated, 0)
        # Always allocated (a field cannot be conditionally absent); only
        # BOUND when NSITE == 0. N_ENVS * SITE_DIM = N_ENVS * 3 floats.
        self._reset_mask.enqueue_fill(Scalar[DT](0))
        # `mj_resetData` zeroes `act`; so does `Phyics3dEnv._reset_state`.
        var _h_act0 = ctx.enqueue_create_host_buffer[DT](
            Self.N_ENVS * Self.NA_F
        )
        ctx.synchronize()
        for _i in range(Self.N_ENVS * Self.NA_F):
            _h_act0[_i] = Scalar[DT](0)
        ctx.enqueue_copy(self._act, _h_act0)
        ctx.synchronize()

        # ⚠ A model with a mocap body and CONFIG.USES_MOCAP == False would run
        # perfectly and train on a target frozen at its XML pose — an easier
        # task, silently. Read the BUILT model and refuse instead.
        comptime if not Self.CONFIG.USES_MOCAP:
            for _b in range(Self.NBODY):
                if (
                    self.mf.bodies.data[_b * MODEL_BODY_SIZE + BODY_IDX_MOCAP]
                    != 0
                ):
                    raise Error(
                        String(
                            "Phyics3dBatchedEnv: body ", _b, " is a MOCAP body"
                            " but CONFIG.USES_MOCAP is False, so its pose"
                            " would never be synced and the target would sit"
                            " at its XML position for every episode. Set"
                            " USES_MOCAP = True on the config (blocker H in"
                            " docs/DM_CONTROL_GPU_TRAINING_G10.md)."
                        )
                    )
        self._env_rng_counter.enqueue_fill(UInt64(42))

    # ── kinematics ────────────────────────────────────────────────────


    @always_inline
    def _site_xpos_operand(
        mut self,
    ) -> LayoutTensor[DT, Self.L_SITE_HOOK, MutAnyOrigin]:
        """The `site_xpos` kernel operand, floored to one site.

        For NSITE > 0 this IS `d.site_xpos`. For NSITE == 0 it is the dummy —
        `d.site_xpos` is a zero-length allocation there and is never even
        uploaded (`Data.upload_all` guards on `NSITE > 0`), so binding it would
        be a zero-extent operand at best and a null device pointer at worst.
        See the SITE_DIM note above."""
        comptime if Self.NSITE > 0:
            return rebind[LayoutTensor[DT, Self.L_SITE_HOOK, MutAnyOrigin]](
                self.d.site_xpos.lt["gpu", type_of(self.d).L_SITE]()
            )
        else:
            # rebind: the buffer-constructed view carries
            # `origin_of(self._site_dummy)`, the kernel ABI wants
            # MutAnyOrigin. Constructed FROM THE BUFFER, never from
            # `unsafe_ptr()` — that spelling silently miscompiles on GPU.
            return rebind[LayoutTensor[DT, Self.L_SITE_HOOK, MutAnyOrigin]](
                LayoutTensor[DT, Self.L_SITE_HOOK](self._site_dummy)
            )

    @always_inline
    def _site_xpos_acc_operand(
        mut self,
    ) -> LayoutTensor[DT, Self.L_SITE_HOOK, MutAnyOrigin]:
        """`site_xpos` AS IT STOOD WHEN `cacc`/`cfrc_int` WERE WRITTEN.

        Sized and floored exactly like `_site_xpos_operand` — `Data` allocates
        `site_xpos_acc` to shadow `site_xpos` including its lack of a
        zero-extent guard, so the same NSITE == 0 dummy applies. It reuses the
        SAME dummy buffer: for a model with no sites, no hook can legally index
        either, so two distinct dummies would only cost memory.

        ⚠ This is not interchangeable with `_site_xpos_operand`. An
        acceleration-stage sensor that reads the LIVE site pose mixes
        integration stages — dog's accelerometer read 1.484 against
        dm_control's -6.386 that way (defect 19), with `cacc` itself exact to
        4.5e-10. See fields/data.mojo.
        """
        comptime if Self.NSITE > 0:
            return rebind[LayoutTensor[DT, Self.L_SITE_HOOK, MutAnyOrigin]](
                self.d.site_xpos_acc.lt["gpu", type_of(self.d).L_SITE]()
            )
        else:
            return rebind[LayoutTensor[DT, Self.L_SITE_HOOK, MutAnyOrigin]](
                LayoutTensor[DT, Self.L_SITE_HOOK](self._site_dummy)
            )

    @always_inline
    def _act_operand(
        mut self,
    ) -> LayoutTensor[DT, Self.L_ACT_HOOK, MutAnyOrigin]:
        """The per-env actuator activation, as a hook operand.

        `_act` is already floored at `NA_F >= 1` (see the field), so unlike
        `site_xpos` there is no dummy to pick between — this is only here to
        keep the `rebind` off the two call sites."""
        return rebind[LayoutTensor[DT, Self.L_ACT_HOOK, MutAnyOrigin]](
            LayoutTensor[DT, Self.L_ACT_HOOK](self._act)
        )

    def _find_non_contacting_height_batch(
        mut self, mut c: DeviceContext
    ) raises:
        """Raise each lane's free root in 1 cm steps until nothing touches.

        The batched twin of `Phyics3dEnv._find_non_contacting_height`
        (`quadruped._find_non_contacting_height`, suite/quadruped.py:397).
        Lanes settle INDEPENDENTLY — each draws its own random orientation, so
        a sprawled one needs a lower clearance than a rolled one — which is why
        the loop carries a per-lane `done` flag rather than stopping the whole
        batch at the first clear lane.

        ⚠ WITHOUT THIS the GPU lanes spawn at z = 0, i.e. embedded in the
        floor. That is not a crash: the solver pushes the robot out over the
        first few steps and training proceeds from a state the reference never
        visits. Same failure class as blocker H's frozen mocap target.

        COST. One FK + broadphase over the batch per centimetre, with a host
        sync per iteration to test the flags. quadruped settles in tens of
        iterations and its episodes are 1000 steps with no early termination,
        so every lane truncates together and this runs about once per 1000
        steps. It is comptime-gated on `CONFIG.RESET_FIND_HEIGHT`, so no other
        model pays a launch for it.
        """
        # The free root's z, from the joint records rather than assumed at
        # qpos[2] — a model may declare its free joint second. Host-side read
        # of `Model.joints`, which is uploaded once and never mutated.
        var zadr = -1
        for j in range(Self.NJOINT):
            var jt = Int(
                self.mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE]
            )
            if jt == JNT_FREE:
                zadr = (
                    Int(
                        self.mf.joints.data[
                            j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR
                        ]
                    )
                    + 2
                )
                break
        if zadr < 0:
            return

        var done = c.enqueue_create_buffer[DT](Self.N_ENVS)

        @always_inline
        def seed_kernel(
            mask: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            done_t: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
        ):
            # A lane the reset did NOT touch starts already "done", so the
            # search never moves a mid-episode robot's root.
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return
            done_t[env] = Scalar[DT](
                0
            ) if mask[env] != Scalar[DT](0) else Scalar[DT](1)

        @always_inline
        def raise_kernel(
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            done_t: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            z_adr_arg: Int64,
            # ⚠ The HEIGHT, not the attempt index. Computing
            # `0.01 * Float64(attempt)` inside the kernel emits an
            # i64 -> double conversion, and Metal rejects a module
            # containing `double` outright — the error names
            # `air.convert.f.f64.s.i64`, which is exactly this.
            z_val: Scalar[DT],
        ):
            # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
            # a fixed-width `Int64` and re-binds the original name here.
            var z_adr = Int(z_adr_arg)
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return
            if done_t[env] != Scalar[DT](0):
                return
            qpos[env, z_adr] = z_val

        @always_inline
        def settle_kernel(
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
            ],
            done_t: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return
            if meta[env, META_IDX_NUM_CONTACTS] == Scalar[DT](0):
                done_t[env] = Scalar[DT](1)

        var done_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](done)
        var mask_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](
            self._reset_mask
        )
        c.enqueue_function[seed_kernel](
            mask_t, done_t, grid_dim=(Self.BLOCKS,), block_dim=(TPB,)
        )
        var host = List[Scalar[DT]](length=Self.N_ENVS, fill=Scalar[DT](0))

        # Bounded like the reference, which RAISES on exhaustion. We cannot
        # raise per lane here, so an exhausted lane keeps the last height tried
        # and its first step reports the penetration — a model whose legs
        # cannot clear the floor in 100 m is misbuilt in a way a reset failure
        # would not explain either.
        for attempt in range(10000):
            c.enqueue_function[raise_kernel](
                self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
                done_t,
                Int64(zadr),
                Scalar[DT](0.01 * Float64(attempt)),
                grid_dim=(Self.BLOCKS,),
                block_dim=(TPB,),
            )
            self._run_fields_fk(c)
            detect_contacts_auto["gpu", DT, BATCH=Self.N_ENVS](self.d, self.mf, c)
            c.enqueue_function[settle_kernel](
                self.d.meta.lt["gpu", type_of(self.d).L_META](),
                done_t,
                grid_dim=(Self.BLOCKS,),
                block_dim=(TPB,),
            )
            c.enqueue_copy(host.unsafe_ptr(), done)
            c.synchronize()
            var all_done = True
            for i in range(Self.N_ENVS):
                if host[i] == Scalar[DT](0):
                    all_done = False
                    break
            if all_done:
                return

    def set_osc_refs(
        mut self, refs: List[Float64], ctx: DeviceContext
    ) raises:
        """Give the controller its model record — see `HAS_OSC_CONTROLLER`.

        `refs` is what `dynamics.osc_pose_gpu.build_osc_refs` returns from the
        parsed scene: the seven arm joints' velocity indices, qpos addresses,
        joint indices, torque limits and actuators, the grip site and its body,
        the two finger actuators with their ctrl ranges, and the six gains.

        ⚠ IT IS SHARED ACROSS LANES, deliberately. Every lane of a batched env
        is the same model; a per-lane record would be a per-lane ROBOT, which
        is not what a batch is.
        """
        comptime if not Self.CONFIG.HAS_OSC_CONTROLLER:
            raise Error(
                "set_osc_refs: this env's config has HAS_OSC_CONTROLLER False,"
                " so nothing would read the record. Set it on the config."
            )
        if len(refs) != OSC_REF_WORDS:
            raise Error(
                "set_osc_refs: the record is " + String(len(refs))
                + " words and OSC_REF_WORDS is " + String(OSC_REF_WORDS)
                + " — it must come from `build_osc_refs`, not be assembled by"
                " hand."
            )
        for k in range(OSC_REF_WORDS):
            self._osc_refs.data[k] = Scalar[DT](refs[k])
        self._osc_refs.upload(ctx)
        self._osc_ready = True

    def osc_singular_lanes(mut self, c: DeviceContext) raises -> Int:
        """How many lanes reported a singular operational-space inertia.

        ⚠⚠ A DRIVER MUST READ THIS. `osc_run_gpu` leaves a flagged lane's
        torques at ZERO and carries on — a limp arm, not an error, because a
        kernel cannot raise and a batch has no single answer to raise about.
        `OscPose.run` raises for exactly this condition on one lane. A training
        run that never reads it reports a curve for a batch in which some lanes
        were not being controlled at all, which looks like a hard task.

        It costs a device-to-host copy and a synchronize, so call it per
        control step at most — not per substep.
        """
        comptime if not Self.CONFIG.HAS_OSC_CONTROLLER:
            return 0
        self._osc_state.download(c)
        var n = 0
        for e in range(Self.N_ENVS):
            if self._osc_state.data[
                e * OSC_STATE_WORDS + OSC_IDX_SINGULAR
            ] != Scalar[DT](0):
                n += 1
        return n

    def _osc_anchor(mut self, c: DeviceContext) raises:
        """`OscPose.reset` for every lane — after a reset writes the state.

        ⚠ NOT ONLY AFTER A FULL RESET. `q0` and the held goal are per EPISODE,
        so a SELECTIVE reset that restarts four lanes of a thousand must
        re-anchor those four. This re-anchors all of them, which is correct for
        the four and a no-op for the rest ONLY because it reads the current
        pose: a lane mid-episode is re-anchored on where it is now, which
        moves its nullspace target. So it is called from the full reset alone,
        and `reset_done_batch` re-anchors too — the alternative is a per-lane
        mask through the kernel, which is the right shape once a driver
        actually resets subsets under OSC.
        """
        osc_reset_batch["gpu", DT, Self.MD, Self.N_ENVS](
            self.d, self.mf, self._osc_scratch[0], self._osc_refs,
            self._osc_state, self._osc_work, c,
        )

    def _osc_substep(mut self, c: DeviceContext, policy_step: Bool) raises:
        """One control step for every lane, then the ordinary actuation path.

        ⚠ THE ACTION IS COPIED INTO THE CONTROLLER'S OWN BUFFER rather than
        binding `self._action`. The driver's action buffer carries
        `origin_of(self._action)` and handing that to a `mut self` method is an
        exclusivity violation — the same trap `_apply_actions_custom` records
        one method down.
        """
        c.enqueue_copy(self._osc_action.dev.value(), self._action)
        osc_control_step["gpu", DT, Self.MD, Self.N_ENVS](
            self.d, self.mf, self._osc_scratch[0], self._osc_refs,
            self._osc_state, self._osc_work, self._osc_ctrl, self._osc_action,
            Self.OSC_NACT, policy_step, c,
        )
        Self.MODEL_DEF.apply_actions_kernel_gpu[
            DT, Self.N_ENVS, Self.OSC_NACT, NORMALIZED=False
        ](
            c,
            self.d.qfrc.lt["gpu", type_of(self.d).L_NV](),
            self._osc_ctrl.lt[
                "gpu", Layout.row_major(Self.N_ENVS, Self.OSC_NACT)
            ](),
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            rebind[
                LayoutTensor[
                    DT,
                    Layout.row_major(Self.N_ENVS, Self.MODEL_DEF.NA_F),
                    MutAnyOrigin,
                ]
            ](
                LayoutTensor[
                    DT, Layout.row_major(Self.N_ENVS, Self.MODEL_DEF.NA_F)
                ](self._act)
            ),
            self.sf.actuators.lt[
                "gpu",
                Layout.row_major(
                    Self.MODEL_DEF.NACT_F * MODEL_ACTUATOR_SIZE
                ),
            ](),
            # ⚠ `MODEL_DEF.NTEN_F`, NOT `Self.NTENDON_F`. The two are the same
            # number and NOT the same comptime expression, and
            # `apply_actions_kernel_gpu` declares its parameter with the former
            # — the layouts then fail to unify ("unfolded expression at parser
            # time"). The model-default call site above spells it the same way.
            self.sf.act_tendons.lt[
                "gpu",
                Layout.row_major(
                    Self.MODEL_DEF.NTEN_F * MODEL_ACT_TENDON_SIZE
                ),
            ](),
            self.sf.joint_limits.lt[
                "gpu",
                Layout.row_major(Self.MODEL_DEF.NJOINT * JLIM_SIZE),
            ](),
            self.d.dof_actdamp.lt["gpu", type_of(self.d).L_NV](),
            self.d.actdamp_act.lt[
                "gpu",
                Layout.row_major(Self.N_ENVS, Self.MODEL_DEF.NACT_F),
            ](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
        )

    def _apply_actions_custom(mut self, c: DeviceContext) raises:
        """Launch `CONFIG.custom_apply_actions_gpu` over the batch.

        The batched twin of `Phyics3dEnv`'s `custom_applied` branch, and it
        REPLACES the model default including its `qfrc` zeroing — the hook
        zeroes for itself. Called at the top of every substep, which is where
        the model default is called, so a state-dependent force law is right
        here even though the CPU hook's once-per-control-step cadence would
        not be.
        """

        @always_inline
        def act_kernel(
            qfrc: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            actions: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.ACT_DIM), MutAnyOrigin
            ],
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            qvel: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            act: LayoutTensor[DT, Self.L_ACT_HOOK, MutAnyOrigin],
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
            ],
            joints: LayoutTensor[
                DT,
                Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE),
                MutAnyOrigin,
            ],
            tendons: LayoutTensor[DT, Self.L_TENDONS_HOOK, MutAnyOrigin],
            acts: LayoutTensor[
                DT,
                Layout.row_major(
                    Self.MODEL_DEF.NACT_F * MODEL_ACTUATOR_SIZE
                ),
                MutAnyOrigin,
            ],
            act_tendons: LayoutTensor[
                DT,
                Layout.row_major(Self.NTENDON_F * MODEL_ACT_TENDON_SIZE),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return
            Self.CONFIG.custom_apply_actions_gpu[
                DT, Self.N_ENVS, Self.NQ, Self.NV, Self.NJOINT,
                Self.NTENDON_F, Self.ACT_DIM, Self.NA_F,
                Self.MODEL_DEF.NACT_F,
            ](
                qfrc, actions, qpos, qvel, act, meta, joints, tendons,
                acts, act_tendons, env,
            )

        # ⚠ The action view is built HERE rather than passed in: `actions_t`
        # at the call site carries `origin_of(self._action)`, and handing that
        # to a `mut self` method is an exclusivity violation ("aliasing values
        # passed mutably to 'self' and to 'actions_t'"). The model-default
        # path does not hit this because it is a static method on MODEL_DEF.
        c.enqueue_function[act_kernel](
            self.d.qfrc.lt["gpu", type_of(self.d).L_NV](),
            rebind[
                LayoutTensor[
                    DT,
                    Layout.row_major(Self.N_ENVS, Self.ACT_DIM),
                    MutAnyOrigin,
                ]
            ](
                LayoutTensor[
                    DT, Layout.row_major(Self.N_ENVS, Self.ACT_DIM)
                ](self._action)
            ),
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self._act_operand(),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            self.mf.joints.lt["gpu", type_of(self.mf).L_JOINT](),
            self.mf.tendons.lt["gpu", Self.L_TENDONS_HOOK](),
            self.sf.actuators.lt[
                "gpu",
                Layout.row_major(Self.MODEL_DEF.NACT_F * MODEL_ACTUATOR_SIZE),
            ](),
            self.sf.act_tendons.lt[
                "gpu",
                Layout.row_major(Self.NTENDON_F * MODEL_ACT_TENDON_SIZE),
            ](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )

    def _sync_mocap_batch(mut self, c: DeviceContext) raises:
        """Push `mocap_pos`/`mocap_quat` into the mocap bodies' world pose.

        The batched twin of `Phyics3dEnv._sync_mocap_to_fields`, and it must be
        called at the SAME points relative to FK: immediately BEFORE it. That
        ordering is not arbitrary — the fields FK SKIPS mocap bodies, so
        writing the pose first leaves it in place for the weld/equality solve
        to track, and writing it after would be redundant at best.

        ⚠ Without this, a config's per-episode mocap target simply never
        reaches the body: `d.mocap_pos` gets written by the reset hook and
        nothing reads it, so the target sits at its XML pose for every episode
        of every lane. That is a SILENTLY EASIER TASK, not a crash — blocker H.

        Comptime-gated on `CONFIG.USES_MOCAP` so the ~11 non-mocap envs pay no
        launch; `__init__` raises if a model contradicts that flag.
        """

        @always_inline
        def mocap_kernel(
            bodies: LayoutTensor[
                DT,
                Layout.row_major(Self.NBODY, MODEL_BODY_SIZE),
                MutAnyOrigin,
            ],
            mocap_pos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NBODY * 3), MutAnyOrigin
            ],
            mocap_quat: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NBODY * 4), MutAnyOrigin
            ],
            xpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NBODY * 3), MutAnyOrigin
            ],
            xipos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NBODY * 3), MutAnyOrigin
            ],
            xquat: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NBODY * 4), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return
            for b in range(Self.NBODY):
                if bodies[b, BODY_IDX_MOCAP] == Scalar[DT](0):
                    continue
                # xipos too, not just xpos: a mocap body's inertial frame is
                # its body frame, and reward hooks read xipos.
                for k in range(3):
                    var pv = mocap_pos[env, b * 3 + k]
                    xpos[env, b * 3 + k] = pv
                    xipos[env, b * 3 + k] = pv
                for k in range(4):
                    xquat[env, b * 4 + k] = mocap_quat[env, b * 4 + k]

        c.enqueue_function[mocap_kernel](
            self.mf.bodies.lt["gpu", type_of(self.mf).L_BODY](),
            self.d.mocap_pos.lt["gpu", type_of(self.d).L_B3](),
            self.d.mocap_quat.lt["gpu", type_of(self.d).L_B4](),
            self.d.xpos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xipos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xquat.lt["gpu", type_of(self.d).L_B4](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )

    def _run_fields_fk(mut self, c: DeviceContext) raises:
        """Fields FK over the whole batch (mf -> Data xpos/xquat/xipos
        [+ site_xpos]). Replaces the legacy slab `forward_kinematics_gpu` in
        the reset paths so reset no longer reads the model slab."""
        forward_kinematics["gpu", DT, BATCH=Self.N_ENVS](self.d, self.mf, c)

    def _run_fields_vel(mut self, c: DeviceContext) raises:
        """Body world velocities (xvel/xangvel) over the batch, from the
        current qvel. Companion to `_run_fields_fk` — the integrators compute
        these mid-step, so hooks reading them after integration need a
        refresh. Mirrors `Phyics3dEnv._fields_vel`."""
        compute_body_velocities["gpu", DT, BATCH=Self.N_ENVS](self.d, self.mf, c)

    # ── hook kernels (legacy Phyics3dEnv GPU code, verbatim) ──────────

    def _extract_obs_only(mut self, c: DeviceContext) raises:
        """Obs from the field tensors: CONFIG custom extraction else
        MODEL_DEF default qpos[skip:]+qvel."""

        @always_inline
        def obs_kernel(
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            qvel: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            xpos: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            xquat: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 4),
                MutAnyOrigin,
            ],
            xvel: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            bodies: LayoutTensor[
                DT,
                Layout.row_major(Self.NBODY, MODEL_BODY_SIZE),
                MutAnyOrigin,
            ],
            site_xpos: LayoutTensor[
                DT, Self.L_SITE_HOOK, MutAnyOrigin
            ],
            contacts: LayoutTensor[
                DT, Data[DT, Self.MD, Self.N_ENVS].L_CONTACTS, MutAnyOrigin
            ],
            sites: LayoutTensor[DT, Self.L_SITES_HOOK, MutAnyOrigin],
            geoms: LayoutTensor[DT, Self.L_GEOMS_HOOK, MutAnyOrigin],
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
            ],
            obs: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.OBS_DIM),
                MutAnyOrigin,
            ],
            xipos: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            xangvel: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            cvel: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 6),
                MutAnyOrigin,
            ],
            cacc: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 6),
                MutAnyOrigin,
            ],
            cfrc_int: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 6),
                MutAnyOrigin,
            ],
            subtree_com: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            site_xpos_acc: LayoutTensor[
                DT, Self.L_SITE_HOOK, MutAnyOrigin
            ],
            xquat_acc: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 4),
                MutAnyOrigin,
            ],
            act: LayoutTensor[DT, Self.L_ACT_HOOK, MutAnyOrigin],
            mesh_meta: LayoutTensor[
                DT, Self.L_MESH_META_HOOK, MutAnyOrigin
            ],
            mesh_tris: LayoutTensor[
                DT, Self.L_MESH_TRIS_HOOK, MutAnyOrigin
            ],
            hfield_meta: LayoutTensor[DT, Self.L_HF_META_HOOK, MutAnyOrigin],
            hfield_data: LayoutTensor[DT, Self.L_HF_DATA_HOOK, MutAnyOrigin],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return
            # ⚠ THE RAY-CAPABLE HOOK, WHICH DEFAULTS TO FORWARDING to
            # `custom_extract_obs_gpu` — see `Phyics3dEnvConfig`. Calling the
            # narrow one here too would give a config that overrode the ray
            # hook two chances to write the observation. Same shape as
            # `Phyics3dEnv._get_obs` on the single-env path.
            if not Self.CONFIG.custom_extract_obs_ray_gpu[
                DT, Self.N_ENVS, Self.NQ, Self.NV, Self.NBODY,
                Self.OBS_DIM, Self.SITE_DIM, Self.MC, Self.NSITE_F,
                Self.NGEOM_F, Self.NA_F, Self.NMESH_TRI_F,
                Self.NHFIELD_DATA_F,
            ](
                qpos, qvel, xpos, xquat, xvel, bodies, site_xpos,
                contacts, sites, geoms, meta, obs,
                xipos, xangvel, cvel, cacc, cfrc_int, subtree_com,
                site_xpos_acc, xquat_acc, act,
                mesh_meta, mesh_tris, hfield_meta, hfield_data, env,
            ):
                Self.MODEL_DEF.extract_obs_gpu[
                    DT, Self.N_ENVS, Self.OBS_DIM
                ](qpos, qvel, obs, env)

        var obs_t = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.OBS_DIM)
        ](self._obs)
        c.enqueue_function[obs_kernel](
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.xpos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xquat.lt["gpu", type_of(self.d).L_B4](),
            self.d.xvel.lt["gpu", type_of(self.d).L_B3](),
            self.mf.bodies.lt["gpu", type_of(self.mf).L_BODY](),
            self._site_xpos_operand(),
            self.d.contacts.lt["gpu", type_of(self.d).L_CONTACTS](),
            self.mf.sites.lt["gpu", Self.L_SITES_HOOK](),
            self.mf.geoms.lt["gpu", Self.L_GEOMS_HOOK](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            obs_t,
            self.d.xipos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xangvel.lt["gpu", type_of(self.d).L_B3](),
            self.d.cvel.lt["gpu", type_of(self.d).L_B6](),
            self.d.cacc.lt["gpu", type_of(self.d).L_B6](),
            self.d.cfrc_int.lt["gpu", type_of(self.d).L_B6](),
            self.d.subtree_com.lt["gpu", type_of(self.d).L_B3](),
            self._site_xpos_acc_operand(),
            self.d.xquat_acc.lt["gpu", type_of(self.d).L_B4](),
            self._act_operand(),
            self.mf.mesh_meta.lt["gpu", Self.L_MESH_META_HOOK](),
            self.mf.mesh_tris.lt["gpu", Self.L_MESH_TRIS_HOOK](),
            self.mf.hfield_meta.lt["gpu", Self.L_HF_META_HOOK](),
            self.d.hfield_data.lt["gpu", Self.L_HF_DATA_HOOK](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )

    def _extract_obs_rewards_dones(mut self, c: DeviceContext) raises:
        """Step-count bump + obs + CONFIG reward/termination from the field
        tensors (hook arithmetic unchanged from the slab era)."""

        @always_inline
        def extract_kernel(
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            qvel: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            xpos: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            xipos: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            xquat: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 4),
                MutAnyOrigin,
            ],
            xvel: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            bodies: LayoutTensor[
                DT,
                Layout.row_major(Self.NBODY, MODEL_BODY_SIZE),
                MutAnyOrigin,
            ],
            site_xpos: LayoutTensor[
                DT, Self.L_SITE_HOOK, MutAnyOrigin
            ],
            contacts: LayoutTensor[
                DT, Data[DT, Self.MD, Self.N_ENVS].L_CONTACTS, MutAnyOrigin
            ],
            sites: LayoutTensor[DT, Self.L_SITES_HOOK, MutAnyOrigin],
            geoms: LayoutTensor[DT, Self.L_GEOMS_HOOK, MutAnyOrigin],
            cfrc_ext: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 6),
                MutAnyOrigin,
            ],
            cvel: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 6),
                MutAnyOrigin,
            ],
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
            ],
            curriculum: LayoutTensor[
                DT, Layout.row_major(1, MODEL_CURRICULUM_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.ACT_DIM),
                MutAnyOrigin,
            ],
            rewards: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            dones: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            terminated_out: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            obs: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.OBS_DIM),
                MutAnyOrigin,
            ],
            # ── E2: the acceleration-stage set ────────────────────────────
            # ⚠ OPERAND BUDGET. This kernel now binds 27 operands against a
            # MEASURED Metal cliff of 28 (29 = JIT abort, not a slowdown).
            # Anything further has to displace something, not append — the
            # obvious lever is that `cfrc_ext` is read by three Gym configs
            # and nothing in the suite.
            xangvel: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            cacc: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 6),
                MutAnyOrigin,
            ],
            cfrc_int: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 6),
                MutAnyOrigin,
            ],
            subtree_com: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            site_xpos_acc: LayoutTensor[
                DT, Self.L_SITE_HOOK, MutAnyOrigin
            ],
            xquat_acc: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 4),
                MutAnyOrigin,
            ],
            act: LayoutTensor[DT, Self.L_ACT_HOOK, MutAnyOrigin],
            mesh_meta: LayoutTensor[
                DT, Self.L_MESH_META_HOOK, MutAnyOrigin
            ],
            mesh_tris: LayoutTensor[
                DT, Self.L_MESH_TRIS_HOOK, MutAnyOrigin
            ],
            hfield_meta: LayoutTensor[DT, Self.L_HF_META_HOOK, MutAnyOrigin],
            hfield_data: LayoutTensor[DT, Self.L_HF_DATA_HOOK, MutAnyOrigin],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return

            var step_count = Int(
                rebind[Scalar[DT]](meta[env, META_IDX_STEP_COUNT])
            )
            step_count += 1
            meta[env, META_IDX_STEP_COUNT] = Scalar[DT](step_count)

            # ⚠⚠ THE RAY-CAPABLE HOOK, AND THIS IS THE CALL SITE THAT
            # MATTERS. There are TWO obs dispatches in this file — `obs_kernel`
            # runs at reset, and THIS one runs every step. Wiring only the
            # first one silently gave escape the MODEL DEFAULT observation
            # (raw qpos from index 0) on every step while its reward hook,
            # dispatched four lines below, ran correctly — a config that looks
            # half-connected because it is. Both dispatch through the ray hook
            # now; see `Phyics3dEnvConfig` for why it defaults to forwarding.
            if not Self.CONFIG.custom_extract_obs_ray_gpu[
                DT, Self.N_ENVS, Self.NQ, Self.NV, Self.NBODY,
                Self.OBS_DIM, Self.SITE_DIM, Self.MC, Self.NSITE_F,
                Self.NGEOM_F, Self.NA_F, Self.NMESH_TRI_F,
                Self.NHFIELD_DATA_F,
            ](
                qpos, qvel, xpos, xquat, xvel, bodies, site_xpos,
                contacts, sites, geoms, meta, obs,
                xipos, xangvel, cvel, cacc, cfrc_int, subtree_com,
                site_xpos_acc, xquat_acc, act,
                mesh_meta, mesh_tris, hfield_meta, hfield_data, env,
            ):
                Self.MODEL_DEF.extract_obs_gpu[
                    DT, Self.N_ENVS, Self.OBS_DIM
                ](qpos, qvel, obs, env)

            var result = Self.CONFIG.compute_reward_and_done_gpu[
                DT, Self.N_ENVS, Self.NQ, Self.NV, Self.NBODY,
                Self.ACT_DIM, Self.SITE_DIM, Self.MC, Self.NSITE_F,
                Self.NGEOM_F, Self.NA_F,
            ](
                qpos,
                qvel,
                xpos,
                xipos,
                xquat,
                xvel,
                bodies,
                site_xpos,
                contacts,
                sites,
                geoms,
                cfrc_ext,
                cvel,
                meta,
                curriculum,
                actions,
                xangvel,
                cacc,
                cfrc_int,
                subtree_com,
                site_xpos_acc,
                xquat_acc,
                act,
                env,
                step_count,
                Self.CONFIG.FRAME_SKIP,
                Scalar[DT](Self.CONFIG.get_timestep()),
            )
            rewards[env] = result[0]

            var is_terminated = result[1]
            comptime if not Self.TERMINATE_ON_UNHEALTHY:
                is_terminated = False
            var truncated = step_count >= Self.CONFIG.MAX_STEPS

            if is_terminated or truncated:
                dones[env] = Scalar[DT](1.0)
            else:
                dones[env] = Scalar[DT](0.0)
            terminated_out[env] = Scalar[DT](
                1.0
            ) if is_terminated else Scalar[DT](0.0)

        var rewards_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](
            self._reward
        )
        var dones_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](
            self._done
        )
        var term_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](
            self._terminated
        )
        var obs_t = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.OBS_DIM)
        ](self._obs)
        var actions_t = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.ACT_DIM)
        ](self._action)
        c.enqueue_function[extract_kernel](
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.xpos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xipos.lt["gpu", type_of(self.d).L_B3](),
            self.d.xquat.lt["gpu", type_of(self.d).L_B4](),
            self.d.xvel.lt["gpu", type_of(self.d).L_B3](),
            self.mf.bodies.lt["gpu", type_of(self.mf).L_BODY](),
            self._site_xpos_operand(),
            self.d.contacts.lt["gpu", type_of(self.d).L_CONTACTS](),
            self.mf.sites.lt["gpu", Self.L_SITES_HOOK](),
            self.mf.geoms.lt["gpu", Self.L_GEOMS_HOOK](),
            self.d.cfrc_ext.lt["gpu", type_of(self.d).L_B6](),
            self.d.cvel.lt["gpu", type_of(self.d).L_B6](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            self.mf.curriculum.lt[
                "gpu", Layout.row_major(1, MODEL_CURRICULUM_SIZE)
            ](),
            actions_t,
            rewards_t,
            dones_t,
            term_t,
            obs_t,
            self.d.xangvel.lt["gpu", type_of(self.d).L_B3](),
            self.d.cacc.lt["gpu", type_of(self.d).L_B6](),
            self.d.cfrc_int.lt["gpu", type_of(self.d).L_B6](),
            self.d.subtree_com.lt["gpu", type_of(self.d).L_B3](),
            self._site_xpos_acc_operand(),
            self.d.xquat_acc.lt["gpu", type_of(self.d).L_B4](),
            self._act_operand(),
            self.mf.mesh_meta.lt["gpu", Self.L_MESH_META_HOOK](),
            self.mf.mesh_tris.lt["gpu", Self.L_MESH_TRIS_HOOK](),
            self.mf.hfield_meta.lt["gpu", Self.L_HF_META_HOOK](),
            self.d.hfield_data.lt["gpu", Self.L_HF_DATA_HOOK](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )

    # ── BatchedEnv ABI ────────────────────────────────────────────────

    def reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "Phyics3dBatchedEnv: reset_batch BATCH must match"
            " N_ENVS"
        )
        var c = require_ctx["Phyics3dBatchedEnv.reset_batch"](ctx)

        # Reset every lane on the field tensors (joint noise + CONFIG qpos +
        # hook metadata), then FK for the reset observation.
        @always_inline
        def reset_kernel(
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            qvel: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            qacc: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            qfrc: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            # ⚠ `mj_resetData` ZEROES `qacc_warmstart`, so a lane that starts a
            # new episode must too — otherwise its first primal solve prices
            # the PREVIOUS episode's acceleration. The cost comparison would
            # usually discard it, but "usually" is not the reference's
            # algorithm.
            qacc_ws: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
            ],
            joints: LayoutTensor[
                DT,
                Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE),
                MutAnyOrigin,
            ],
            mocap_pos: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            mocap_quat: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 4),
                MutAnyOrigin,
            ],
            reset_mask: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            bodies: LayoutTensor[
                DT,
                Layout.row_major(Self.NBODY, MODEL_BODY_SIZE),
                MutAnyOrigin,
            ],
            geoms: LayoutTensor[DT, Self.L_GEOMS_HOOK, MutAnyOrigin],
            act: LayoutTensor[DT, Self.L_ACT_HOOK, MutAnyOrigin],
            hfield_meta: LayoutTensor[DT, Self.L_HF_META_HOOK, MutAnyOrigin],
            hfield_data: LayoutTensor[DT, Self.L_HF_DATA_HOOK, MutAnyOrigin],
            seed_arg: Int64,
            qpos0: LayoutTensor[
                DT, Layout.row_major(Self.MODEL_DEF.NQ_F), MutAnyOrigin
            ],
            pose_meta: LayoutTensor[
                DT, Layout.row_major(POSE_META_SIZE), MutAnyOrigin
            ],
        ):
            # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
            # a fixed-width `Int64` and re-binds the original name here.
            var seed = Int(seed_arg)
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= Self.N_ENVS:
                return
            Self._reset_env_lane(
                qpos, qvel, qacc, qfrc, qacc_ws, meta, joints, mocap_pos,
                mocap_quat, bodies, geoms, act, hfield_meta, hfield_data,
                qpos0, pose_meta, i, seed,
            )
            reset_mask[i] = Scalar[DT](1)

        c.enqueue_function[reset_kernel](
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.qacc.lt["gpu", type_of(self.d).L_NV](),
            self.d.qfrc.lt["gpu", type_of(self.d).L_NV](),
            self.d.qacc_warmstart.lt["gpu", type_of(self.d).L_NV](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            self.mf.joints.lt["gpu", type_of(self.mf).L_JOINT](),
            self.d.mocap_pos.lt["gpu", type_of(self.d).L_B3](),
            self.d.mocap_quat.lt["gpu", type_of(self.d).L_B4](),
            LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](self._reset_mask),
            self.mf.bodies.lt["gpu", type_of(self.mf).L_BODY](),
            self.mf.geoms.lt["gpu", Self.L_GEOMS_HOOK](),
            self._act_operand(),
            self.mf.hfield_meta.lt["gpu", Self.L_HF_META_HOOK](),
            self.d.hfield_data.lt["gpu", Self.L_HF_DATA_HOOK](),
            Int64(rng_seed),
            self.sf.qpos0.lt[
                "gpu", Layout.row_major(Self.MODEL_DEF.NQ_F)
            ](),
            self.sf.pose_meta.lt[
                "gpu", Layout.row_major(POSE_META_SIZE)
            ](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )
        # After the orientation draw, before the reset observation — the same
        # order `Phyics3dEnv._reset_state` uses.
        comptime if Self.CONFIG.RESET_FIND_HEIGHT:
            self._find_non_contacting_height_batch(c)
        comptime if Self.CONFIG.USES_MOCAP:
            self._sync_mocap_batch(c)
        self._run_fields_fk(c)
        # ⚠ THE CONTROLLER IS ANCHORED HERE, AFTER FK AND BEFORE THE FIRST
        # STEP. `q0` and the held goal are per EPISODE and are read off the
        # CURRENT pose, so anchoring before the reset writes `qpos` would pull
        # every arm toward the previous episode's configuration — a plausible
        # drift with no error anywhere.
        comptime if Self.CONFIG.HAS_OSC_CONTROLLER:
            if self._osc_ready:
                self._osc_anchor(c)
        self._extract_obs_only(c)

    def step_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        # Trait-conforming entry; delegates to the instrumentable impl.
        self._step_impl[BATCH, False](ctx, rng_seed)

    def _step_impl[
        BATCH: Int, DEBUG: Bool = False
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "Phyics3dBatchedEnv: step_batch BATCH must match N_ENVS"
        )
        _ = rng_seed
        var c = require_ctx["Phyics3dBatchedEnv.step_batch"](ctx)

        # 1) CONFIG pre-step hook (save prev_x etc. into d.meta).
        @always_inline
        def pre_step_kernel(
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            qvel: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= Self.N_ENVS:
                return
            # ⚠ THE WIDE HOOK, WHICH DEFAULTS TO FORWARDING to `pre_step_gpu`
            # — see `Phyics3dEnvConfig`. Calling the narrow one here too would
            # give a config that overrode it TWO invocations per step, and a
            # hook whose job is to stash the previous value is silently wrong
            # when called twice. Same shape as the two obs hooks below.
            Self.CONFIG.pre_step_full_gpu[DT, Self.N_ENVS, Self.NQ, Self.NV](
                qpos, qvel, meta, env
            )

        c.enqueue_function[pre_step_kernel](
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 1 pre_step ok")

        var actions_t = LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.ACT_DIM)
        ](self._action)

        # 2a') Record the action for configs that put the PREVIOUS one in
        # their observation. ⚠ HERE, not in the reward hook: this path runs
        # obs BEFORE reward while `Phyics3dEnv` runs reward before obs, so a
        # write from the reward hook lands one step apart on the two devices.
        # See `Phyics3dEnvConfig.RECORD_PREV_ACTION`.
        comptime if Self.CONFIG.RECORD_PREV_ACTION:

            @always_inline
            def record_action_kernel(
                actions: LayoutTensor[
                    DT,
                    Layout.row_major(Self.N_ENVS, Self.ACT_DIM),
                    MutAnyOrigin,
                ],
                meta: LayoutTensor[
                    DT,
                    Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                    MutAnyOrigin,
                ],
            ):
                var env = Int(block_dim.x * block_idx.x + thread_idx.x)
                if env >= Self.N_ENVS:
                    return
                for j in range(Self.ACT_DIM):
                    meta[env, META_IDX_TASK_PARAM_0 + j] = meta[
                        env, META_IDX_TASK_PARAM_6 + j
                    ]
                    meta[env, META_IDX_TASK_PARAM_6 + j] = actions[env, j]

            c.enqueue_function[record_action_kernel](
                actions_t,
                self.d.meta.lt["gpu", type_of(self.d).L_META](),
                grid_dim=(Self.BLOCKS,),
                block_dim=(TPB,),
            )

        # 2b) Mocap-controlled models: push the updated target into the
        #     body pose BEFORE the step so the weld solve tracks it.
        #     Mirrors `Phyics3dEnv.step`, which syncs at this exact point.
        comptime if Self.CONFIG.USES_MOCAP:
            self._sync_mocap_batch(c)

        # 3) Physics: fields integrator (RK4 or Euler per CONFIG.INTEGRATOR)
        #    with per-substep contact/limit solving, with actuation re-applied
        #    at the top of EVERY substep.
        #
        # ⚠ THE ACTUATOR CALL MOVED INSIDE THIS LOOP (2026-08-07). It used to
        # run ONCE per control step while `Phyics3dEnv.step` calls its CPU twin
        # once per SUBSTEP. For a plain `<motor>` the two are identical — its
        # force is `gear * coef * kp * ctrl`, constant across the step, and the
        # kernel zeroes `qfrc` before writing, so re-applying is idempotent.
        # That is why every model gated to date was unaffected, and it is the
        # evidence that this move is a no-op for them.
        #
        # It is NOT identical for anything that reads `qpos`/`qvel` or carries
        # state: a position servo, a fixed-tendon spring, and a `dyntype`
        # activation all change every substep. Those three were refused by
        # comptime asserts in `apply_actions_kernel_gpu` precisely because of
        # this cadence; the asserts are gone now that the cadence matches.
        # quadruped needs all three at once (`<general biastype="affine"
        # dyntype="filter">` on tendon transmissions).
        comptime if Self.CONFIG.HAS_OSC_CONTROLLER:
            if not self._osc_ready:
                raise Error(
                    "this env is configured for OSC_POSE but `set_osc_refs` was"
                    " never called, so the controller's model record is all"
                    " zeros: joint 0 for every arm joint, gain 0, no grip site."
                    " That does not fail — it commands nothing and the arm"
                    " sags. See Phyics3dEnvConfig.HAS_OSC_CONTROLLER."
                )
        for _substep in range(Self.CONFIG.FRAME_SKIP):
            # A config whose transmission is randomized PER EPISODE cannot be
            # driven from the comptime actuator tables, which are baked from
            # the XML — see `CONFIG.HAS_CUSTOM_ACTUATION_GPU`. The choice is
            # comptime because it is a choice between kernel launches, not a
            # per-lane `if` like the CPU path's.
            comptime if Self.CONFIG.HAS_OSC_CONTROLLER:
                # ⚠⚠ `policy_step` IS THE FIRST SUBSTEP ONLY. robosuite's
                # `Robot.control(action, policy_step)` gates `set_goal` on it
                # while running the torque law every substep, and the two
                # cadences are not interchangeable: `set_goal` adds the
                # action's delta to the CURRENT end-effector pose, so calling
                # it every substep applies a 5 cm command twenty-five times.
                # `tests/libero/test_osc_control_batched.mojo` measures what
                # that costs (0.387 N·m against a 1.7e-14 agreement) precisely
                # so this line cannot be changed silently.
                self._osc_substep(c, _substep == 0)
            elif Self.CONFIG.HAS_CUSTOM_ACTUATION_GPU:
                self._apply_actions_custom(c)
            else:
                Self.MODEL_DEF.apply_actions_kernel_gpu[
                    DT,
                    Self.N_ENVS,
                    Self.ACT_DIM,
                    NORMALIZED = Self.CONFIG.NORMALIZED_ACTIONS,
                ](
                    c,
                    self.d.qfrc.lt["gpu", type_of(self.d).L_NV](),
                    rebind[
                        LayoutTensor[
                            DT,
                            Layout.row_major(Self.N_ENVS, Self.ACT_DIM),
                            MutAnyOrigin,
                        ]
                    ](actions_t),
                    self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
                    self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
                    rebind[
                        LayoutTensor[
                            DT,
                            Layout.row_major(
                                Self.N_ENVS, Self.MODEL_DEF.NA_F
                            ),
                            MutAnyOrigin,
                        ]
                    ](
                        LayoutTensor[
                            DT,
                            Layout.row_major(
                                Self.N_ENVS, Self.MODEL_DEF.NA_F
                            ),
                        ](self._act)
                    ),
                    self.sf.actuators.lt[
                        "gpu",
                        Layout.row_major(
                            Self.MODEL_DEF.NACT_F * MODEL_ACTUATOR_SIZE
                        ),
                    ](),
                    self.sf.act_tendons.lt[
                        "gpu",
                        Layout.row_major(
                            Self.MODEL_DEF.NTEN_F * MODEL_ACT_TENDON_SIZE
                        ),
                    ](),
                    # `jnt_actfrcrange` — the per-JOINT force clamp
                    # `mj_fwdActuation` applies after every actuator has
                    # contributed. Passed so this target and the CPU one
                    # cannot compute different forces from the same action.
                    self.sf.joint_limits.lt[
                        "gpu",
                        Layout.row_major(Self.MODEL_DEF.NJOINT * JLIM_SIZE),
                    ](),
                    # This step's actuator damping diagonal + the flag. The
                    # batched env builds Euler/RK4, neither of which reads
                    # them; they are filled anyway so a future implicit
                    # caller cannot silently get the model-time value.
                    self.d.dof_actdamp.lt["gpu", type_of(self.d).L_NV](),
                    self.d.actdamp_act.lt[
                        "gpu",
                        Layout.row_major(
                            Self.N_ENVS, Self.MODEL_DEF.NACT_F
                        ),
                    ](),
                    self.d.meta.lt["gpu", type_of(self.d).L_META](),
                )
            comptime if Self.CONFIG.INTEGRATOR == "euler":
                self.integ_euler.step["gpu"](self.d, self.mf, ctx)
            else:
                self.integ_rk4.step["gpu"](self.d, self.mf, ctx)
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 4 physics step ok")

        # 3b) Put the FK products and body velocities in sync with the
        #     INTEGRATED qpos/qvel before anything derived is computed from
        #     them. Both integrators run FK -> body velocities -> subtree_com
        #     at the START of each substep, so after the frame-skip loop
        #     xpos/xquat/xipos/site_xpos/xvel describe the state BEFORE the
        #     last substep. That is raw `mj_step` — right for the Gym-derived
        #     envs and wrong for dm_control, whose tasks read mjData in sync
        #     with qpos/qvel. See Phyics3dEnvConfig.SYNC_FK_AFTER_STEP.
        #
        # ⚠ This was missing on the batched path entirely until 2026-08-06 —
        #     `Phyics3dEnv` (single-env, CPU) honoured the flag and this file
        #     did not, so a suite config ported to the GPU hooks would have
        #     produced rewards one control step stale WITH ITS CPU GATE STILL
        #     PASSING. See docs/DM_CONTROL_GPU_TRAINING_G10.md §4.
        #
        # Comptime-gated, so the Gym envs pay nothing. Both calls are
        # deterministic and RNG-free, so they stay inside a USE_ENV_CUDA_GRAPH
        # capture safely.
        comptime if Self.CONFIG.SYNC_FK_AFTER_STEP:
            comptime if Self.CONFIG.USES_MOCAP:
                self._sync_mocap_batch(c)
            self._run_fields_fk(c)
            self._run_fields_vel(c)
            comptime if DEBUG:
                c.synchronize()
                print("[step_batch] 4b sync_fk_after_step ok")

        # 4) Derived quantities the reward hooks may read (cfrc_ext / cvel),
        #    straight on the field tensors.
        compute_cfrc_ext[DT, Self.N_ENVS](
            c,
            AsStatic[Self.MD](),
            self.d.xipos.lt["gpu", type_of(self.d).L_B3](),
            self.d.contacts.lt["gpu", type_of(self.d).L_CONTACTS](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            self.d.cfrc_ext.lt["gpu", type_of(self.d).L_B6](),
            self.mf.bodies.lt["gpu", type_of(self.mf).L_BODY](),
        )
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 5b compute_cfrc_ext ok")
        # ⚠⚠ `d.cvel` HAS TWO PRODUCERS WRITING DIFFERENT QUANTITIES, and this
        # call is the one that must NOT run last for an RNE_POST model.
        #
        #   rne_post          -> MuJoCo's `d->cvel`: the spatial velocity
        #                        referenced at `subtree_com[rootid]`, written
        #                        DURING the substep. This is what
        #                        `mju_transformSpatial` in the accelerometer
        #                        transports, and what `Phyics3dEnv` leaves in
        #                        the field (it never calls the helper below).
        #   gpu/cvel_gpu      -> a per-body CoM velocity referenced at
        #                        `xipos[b]`, written AFTER the substep loop.
        #
        # The two differ by the reference point, so overwriting the first with
        # the second hands the acceleration-stage sensors a different physical
        # quantity. Measured on quadruped's torso at step 0: worst |cvel diff|
        # vs the CPU path was 45.6, against 3.5e-7 on the FK snapshots — i.e.
        # not rounding, a different vector. It reached the observation as
        # accelerometer[0] = 0.102 where the CPU reads 0.027, and rebuilding
        # the sensor in float64 from the GPU's own downloaded fields
        # reproduced 0.102 to 5.6e-9, which is how the sensor was cleared and
        # the input blamed.
        #
        # Gated rather than deleted: no config reads the helper's form today
        # (grep says `cvel` reaches only quadruped's and dog's hooks, both
        # acceleration-stage), but every Gym-derived model was built while it
        # ran, so leaving it in place for them keeps this change a provable
        # no-op there.
        comptime if not Self.CONFIG.RNE_POST:
            compute_cvel[DT, Self.N_ENVS](
                c,
                AsStatic[Self.MD](),
                self.d.xpos.lt["gpu", type_of(self.d).L_B3](),
                self.d.xvel.lt["gpu", type_of(self.d).L_B3](),
                self.d.xangvel.lt["gpu", type_of(self.d).L_B3](),
                self.d.xipos.lt["gpu", type_of(self.d).L_B3](),
                self.d.cvel.lt["gpu", type_of(self.d).L_B6](),
            )
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 5c compute_cvel ok")

        # 6) Obs + reward + done/terminated (CONFIG hooks).
        self._extract_obs_rewards_dones(c)
        comptime if DEBUG:
            c.synchronize()
            print("[step_batch] 6 extract_obs_rewards_dones ok")

    def selective_reset_batch[
        BATCH: Int
    ](mut self, ctx: Optional[DeviceContext], rng_seed: UInt64,) raises:
        comptime assert BATCH == Self.N_ENVS, (
            "Phyics3dBatchedEnv: selective_reset_batch BATCH must"
            " match N_ENVS"
        )
        _ = rng_seed  # device counter drives reset randomness (capture-safe)
        var c = require_ctx[
            "Phyics3dBatchedEnv.selective_reset_batch"
        ](ctx)
        var cnt_t = LayoutTensor[DType.uint64, Layout.row_major(1)](
            self._env_rng_counter
        )
        c.enqueue_function[_rng_counter_bump_kernel](
            cnt_t, grid_dim=(1,), block_dim=(1,)
        )

        @always_inline
        def selective_reset_kernel(
            qpos: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
            ],
            qvel: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            qacc: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            qfrc: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            # ⚠ `mj_resetData` ZEROES `qacc_warmstart`, so a lane that starts a
            # new episode must too — otherwise its first primal solve prices
            # the PREVIOUS episode's acceleration. The cost comparison would
            # usually discard it, but "usually" is not the reference's
            # algorithm.
            qacc_ws: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
            ],
            meta: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, METADATA_SIZE),
                MutAnyOrigin,
            ],
            dones: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            joints: LayoutTensor[
                DT,
                Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE),
                MutAnyOrigin,
            ],
            mocap_pos: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 3),
                MutAnyOrigin,
            ],
            mocap_quat: LayoutTensor[
                DT,
                Layout.row_major(Self.N_ENVS, Self.NBODY * 4),
                MutAnyOrigin,
            ],
            counter: LayoutTensor[
                DType.uint64, Layout.row_major(1), MutAnyOrigin
            ],
            reset_mask: LayoutTensor[
                DT, Layout.row_major(Self.N_ENVS), MutAnyOrigin
            ],
            bodies: LayoutTensor[
                DT,
                Layout.row_major(Self.NBODY, MODEL_BODY_SIZE),
                MutAnyOrigin,
            ],
            geoms: LayoutTensor[DT, Self.L_GEOMS_HOOK, MutAnyOrigin],
            act: LayoutTensor[DT, Self.L_ACT_HOOK, MutAnyOrigin],
            hfield_meta: LayoutTensor[DT, Self.L_HF_META_HOOK, MutAnyOrigin],
            hfield_data: LayoutTensor[DT, Self.L_HF_DATA_HOOK, MutAnyOrigin],
            qpos0: LayoutTensor[
                DT, Layout.row_major(Self.MODEL_DEF.NQ_F), MutAnyOrigin
            ],
            pose_meta: LayoutTensor[
                DT, Layout.row_major(POSE_META_SIZE), MutAnyOrigin
            ],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= Self.N_ENVS:
                return
            reset_mask[i] = Scalar[DT](0)
            if dones[i] > Scalar[DT](0.5):
                Self._reset_env_lane(
                    qpos,
                    qvel,
                    qacc,
                    qfrc,
                    qacc_ws,
                    meta,
                    joints,
                    mocap_pos,
                    mocap_quat,
                    bodies,
                    geoms,
                    act,
                    hfield_meta,
                    hfield_data,
                    qpos0,
                    pose_meta,
                    i,
                    Int(rebind[Scalar[DType.uint64]](counter[0])),
                )
                dones[i] = Scalar[DT](0.0)
                reset_mask[i] = Scalar[DT](1)

        var dones_t = LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](
            self._done
        )
        c.enqueue_function[selective_reset_kernel](
            self.d.qpos.lt["gpu", type_of(self.d).L_QPOS](),
            self.d.qvel.lt["gpu", type_of(self.d).L_NV](),
            self.d.qacc.lt["gpu", type_of(self.d).L_NV](),
            self.d.qfrc.lt["gpu", type_of(self.d).L_NV](),
            self.d.qacc_warmstart.lt["gpu", type_of(self.d).L_NV](),
            self.d.meta.lt["gpu", type_of(self.d).L_META](),
            dones_t,
            self.mf.joints.lt["gpu", type_of(self.mf).L_JOINT](),
            self.d.mocap_pos.lt["gpu", type_of(self.d).L_B3](),
            self.d.mocap_quat.lt["gpu", type_of(self.d).L_B4](),
            cnt_t,
            LayoutTensor[DT, Layout.row_major(Self.N_ENVS)](self._reset_mask),
            self.mf.bodies.lt["gpu", type_of(self.mf).L_BODY](),
            self.mf.geoms.lt["gpu", Self.L_GEOMS_HOOK](),
            self._act_operand(),
            self.mf.hfield_meta.lt["gpu", Self.L_HF_META_HOOK](),
            self.d.hfield_data.lt["gpu", Self.L_HF_DATA_HOOK](),
            self.sf.qpos0.lt[
                "gpu", Layout.row_major(Self.MODEL_DEF.NQ_F)
            ](),
            self.sf.pose_meta.lt[
                "gpu", Layout.row_major(POSE_META_SIZE)
            ](),
            grid_dim=(Self.BLOCKS,),
            block_dim=(TPB,),
        )
        # Only the lanes this call reset get raised; the mask is what keeps a
        # mid-episode lane's root where the physics left it.
        comptime if Self.CONFIG.RESET_FIND_HEIGHT:
            self._find_non_contacting_height_batch(c)
        # FK for the batch (idempotent for the live lanes, whose state is
        # unchanged since the last step), then refresh obs so reset lanes
        # start their episode from the reset observation.
        comptime if Self.CONFIG.USES_MOCAP:
            self._sync_mocap_batch(c)
        self._run_fields_fk(c)
        self._extract_obs_only(c)

    @always_inline
    @staticmethod
    def _reset_env_lane(
        qpos: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NQ), MutAnyOrigin
        ],
        qvel: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
        ],
        qacc: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
        ],
        qfrc: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
        ],
        qacc_ws: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NV), MutAnyOrigin
        ],
        meta: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, METADATA_SIZE), MutAnyOrigin
        ],
        joints: LayoutTensor[
            DT, Layout.row_major(Self.NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
        ],
        mocap_pos: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NBODY * 3), MutAnyOrigin
        ],
        mocap_quat: LayoutTensor[
            DT, Layout.row_major(Self.N_ENVS, Self.NBODY * 4), MutAnyOrigin
        ],
        bodies: LayoutTensor[
            DT, Layout.row_major(Self.NBODY, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        geoms: LayoutTensor[DT, Self.L_GEOMS_HOOK, MutAnyOrigin],
        act: LayoutTensor[DT, Self.L_ACT_HOOK, MutAnyOrigin],
        hfield_meta: LayoutTensor[DT, Self.L_HF_META_HOOK, MutAnyOrigin],
        hfield_data: LayoutTensor[DT, Self.L_HF_DATA_HOOK, MutAnyOrigin],
        qpos0: LayoutTensor[
            DT, Layout.row_major(Self.MODEL_DEF.NQ_F), MutAnyOrigin
        ],
        pose_meta: LayoutTensor[
            DT, Layout.row_major(POSE_META_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """One env lane's reset on the field tensors (arithmetic verbatim
        from the slab era: joint reset noise + CONFIG qpos + hook metadata).

        ⚠ `bodies` / `geoms` are the MODEL records, and the hook must not
        expect FK products alongside them: this runs BEFORE `_run_fields_fk`,
        so `Data.xpos` still holds the PREVIOUS episode's pose. `Phyics3dEnv`
        has the same ordering (`_reset_state` runs `_fields_fk()` after
        `custom_reset_cpu`), which is exactly how ball_in_cup's rejection
        sampler ended up testing against a stale cup position."""
        # `mj_resetData`'s zero for the carried acceleration. Written HERE
        # rather than inside `reset_env_gpu`, which is a MODEL_DEF method with
        # its own callers and no business knowing about the solver's carry.
        for _wi in range(Self.NV):
            qacc_ws[env, _wi] = Scalar[DT](0)
        var RESET_NOISE = Scalar[DT](Self.CONFIG.get_reset_noise())
        Self.MODEL_DEF.reset_env_gpu[DT, Self.N_ENVS](
            qpos, qvel, qacc, qfrc, qpos0, pose_meta, env, RESET_NOISE, seed
        )
        Self.CONFIG.init_qpos_gpu[
            DT, Self.N_ENVS, Self.NQ, Self.NJOINT, Self.NV, Self.NBODY,
            Self.NGEOM_F,
        ](
            qpos, qvel, joints, mocap_pos, mocap_quat, bodies, geoms,
            meta, env, seed,
        )

        # ⚠ THE TERRAIN, BEFORE THE HEIGHT SEARCH. `raise_kernel` /
        # `settle_kernel` lift this lane until nothing touches, so the surface
        # it will stand on has to exist first. Default is a NO-OP, so a model
        # whose grid came from a file keeps the one `init_hfield_data` wrote
        # at construction.
        Self.CONFIG.init_hfield_gpu[DT, Self.N_ENVS, Self.NHFIELD_DATA_F](
            hfield_meta, hfield_data, env, seed
        )

        # ⚠⚠ `mj_resetData` ZEROES `act`, AND THIS PATH DID NOT — a real bug,
        # not just a dog gap. `_act` was zeroed ONCE at construction and never
        # again, so on GPU an actuator activation SURVIVED the episode
        # boundary: a lane that finished with a loaded filter started its next
        # episode already actuated. `Phyics3dEnv._reset_state` has always
        # zeroed it ("MuJoCo's mj_resetData zeroes `act` along with
        # qpos/qvel"), so CPU and GPU disagreed about what a reset IS.
        #
        # ⚠ IT AFFECTS EVERY MODEL WITH A `dyntype`, INCLUDING QUADRUPED,
        # which has been gated and trained in this state. The GPU-vs-CPU gates
        # cannot see it: they inject a shared qpos/qvel and compare a window
        # that never crosses a reset, and the reset tests only check
        # height/ncon. Dog is simply the model that made someone look.
        for a in range(Self.NA_F):
            act[env, a] = Scalar[DT](0)
        # Then the episode's draw, for the models whose `initialize_episode`
        # makes one (default: no-op, leaving MuJoCo's zero).
        Self.CONFIG.init_act_gpu[DT, Self.N_ENVS, Self.NA_F](act, env, seed)

        meta[env, META_IDX_STEP_COUNT] = Scalar[DT](0.0)
        # The reward's per-episode state (`constants.META_IDX_PHI_PREV`): a
        # potential delta must not span the episode boundary.
        meta[env, META_IDX_PHI_PREV] = Scalar[DT](0.0)
        meta[env, META_IDX_EPISODE_FLAGS] = Scalar[DT](0.0)
        Self.CONFIG.pre_step_full_gpu[DT, Self.N_ENVS, Self.NQ, Self.NV](
            qpos, qvel, meta, env
        )

    # ── pointer accessors ─────────────────────────────────────────────

    def obs_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._obs.unsafe_ptr())

    def action_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._action.unsafe_ptr())

    def reward_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._reward.unsafe_ptr())

    def done_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._done.unsafe_ptr())

    def terminated_ptr(self) -> Pointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._terminated.unsafe_ptr())

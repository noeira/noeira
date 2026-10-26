"""Every LIBERO family on the batched GPU env, driven by OSC_POSE — ONE config.

    from noeira.envs.libero.models.libero_kitchen_scene3_xml import (
        LiberoKitchenScene3OscEnv,
    )
    var env = LiberoKitchenScene3OscEnv[N](ctx, seed)
    env.set_osc_refs(build_osc_refs(...), ctx)     # from the parsed scene
    env.reset_batch[N](ctx, seed)
    env.step_batch[N](ctx, seed)

## ⚠⚠ ONE CONFIG, PARAMETERISED ON THE FAMILY'S PLACEMENT TABLE

`LiberoOscConfig[P]` is what `LiberoGoalOscConfig` was, with its table as a
parameter. Every hook it carries is a call into code every family shares, over
`P`:

    init_qpos_gpu            placement/table.reset_task_slots (jinit=, placements)
    pre_step_full_gpu        task_hooks.repark_inactive_slots
    custom_extract_obs_*     task_hooks.write_task_obs(_host)
    compute_reward_and_done  gpu_eval.eval_tape_gpu -> reward AND GOAL_HELD

and nothing else in it differs between the 23 LIBERO families: all of them run
robosuite's Panda under OSC_POSE at `control_freq=20` on a 2 ms timestep, for a
600-step horizon, under Euler. A per-family copy of this file would be 23 copies
of one rule, and `_a_rule_written_inline_twice_drifts` is this tree's most
frequent defect. What IS per family is the model def and its contact budget —
`libero_scenes/<family>_xml.mojo` (generated) and the three hand-written
`libero_{goal,object,spatial}_xml.mojo`.

⚠ THE THREE CONSTANTS BELOW ARE RESTATED FROM EVERY `.family` AND EVERY
GENERATED DIMS FILE, NOT READ — a config is a comptime type, a `.family` is a
runtime file. `tests/libero/test_libero_task_hooks.mojo` reads all 23 families
and their dims and fails if one disagrees, so a family that ever runs at another
`control_freq` is refused by a gate rather than silently stepped at 20 Hz.

What is per LANE is data in `meta`, written by the driver per episode: the tape
(`tape.encode_goal`), the mask (`active.active_mask`), the init words
(`active.init_region_words`), the joint words (`placement/check.
joint_init_words`); and, once, the region table in `curriculum`
(`gpu_eval.region_table_words`).

## ⚠ FRAME_SKIP IS 25 AND IT IS NOT A TUNING KNOB

`control_freq=20` and a 2 ms timestep make one policy step 1/20 / 0.002 = 25
substeps — robosuite's own arithmetic (`MujocoEnv.__init__`:
`self.control_timestep = 1. / control_freq`). The controller's `set_goal` fires
on substep 0 of those 25 and the torque law runs on all of them;
`dynamics/osc_control.mojo` says why that gate matters.

## ⚠ `NORMALIZED_ACTIONS` IS FALSE, AND THAT IS THE OPPOSITE OF THE SO-101

There the action is [-1, 1] per joint mapped onto each actuator's `ctrlrange`,
because the policy commands joint POSITIONS. Here it commands end-effector
DELTAS in metres and radians, which `osc_set_goal_gpu` scales by
`output_max_pos` / `output_max_ori` itself (0.05 m, 0.5 rad — robosuite's
`osc_pose.json`). Normalising on top of that would scale the deltas twice.
"""


from noeira.physics3d.fields import Data, DimsLike
from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE, METADATA_SIZE, MODEL_BODY_SIZE, MODEL_CURRICULUM_SIZE,
    MODEL_GEOM_SIZE, MODEL_JOINT_SIZE, MODEL_SITE_SIZE, META_IDX_GOAL_HELD,
)
from noeira.tasks.task_hooks import (
    repark_inactive_slots, write_task_obs, write_task_obs_host,
)
from noeira.tasks.placement.table import PlacementTable, reset_task_slots
from noeira.tasks.gpu_eval import eval_tape_gpu
from noeira.envs.phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


comptime LIBERO_CONTROL_FREQ: Int = 20
"""`control_freq=` in every LIBERO `.family`. Restated, not read — see the header."""

comptime LIBERO_TIMESTEP: Float64 = 0.002
"""Every composed LIBERO scene's `<option timestep>` (robosuite's `base.xml`).

⚠⚠ A SCALAR `comptime` LITERAL, NOT A GENERATED DIMS FIELD. The batched env
calls `get_timestep` INSIDE the reward kernel, and a field of a comptime STRUCT
INSTANCE read at runtime is materialised as a lazily-initialised global: the
NVIDIA build died in ptxas with "Unresolved extern function
'KGEN_CompilerRT_GetOrCreateGlobal'". The generated dims are still the
authority — the hooks test compares all 23 against this."""

comptime LIBERO_FRAME_SKIP: Int = 25
"""`1 / LIBERO_CONTROL_FREQ / LIBERO_TIMESTEP`. See the header."""

comptime LIBERO_HORIZON: Int = 600
"""`horizon=` in every `.family`, and `cfg.eval.max_steps` in
`lifelong/metric.py`. Restated, not read."""


struct LiberoOscConfig[P: PlacementTable](Phyics3dEnvConfig):
    """One LIBERO family's batched control path, over its table `P`."""

    comptime FRAME_SKIP: Int = LIBERO_FRAME_SKIP
    comptime MAX_STEPS: Int = LIBERO_HORIZON

    # ⚠ EULER, AS robosuite RUNS THIS MODEL. `base.xml` sets no
    # `<option integrator>`, so MuJoCo steps it under the default; RK4 would be
    # four times the cost of a pipeline the demonstrations were never recorded
    # under. The same reasoning as `So101TabletopConfig`, which measured it.
    comptime INTEGRATOR: StaticString = "euler"

    comptime INTEGRATOR_WS_EXTRA: Int = 0
    """0 for Euler and RK4; only ImplicitFast wants workspace beyond them."""

    comptime HAS_GPU_HOOKS: Bool = True
    comptime HAS_OSC_CONTROLLER: Bool = True
    comptime NORMALIZED_ACTIONS: Bool = False

    # ⚠ THE SCENE IS MESH-HEAVY AND THE COLLIDERS ARE NOT. LIBERO's objects
    # collide as 859 BOXES with 14 meshes; the `.msh` visuals carry
    # `contype=0`. The mesh vertex budget still has to cover the colliding
    # ones, and `parse_model_runtime` doubles until it fits — this is the
    # comptime figure the batched path needs up front.
    comptime NMESH_VERTS: Int = 65536

    # `mj_step`'s own order: the FK products after a step describe the state
    # BEFORE the last substep. LIBERO's `_check_success` reads object poses in
    # sync with qpos, the way dm_control does, so the flag is on.
    comptime SYNC_FK_AFTER_STEP: Bool = True

    # === the task hooks — `tasks/task_hooks.mojo`, on this family's table ===
    @always_inline
    @staticmethod
    def pre_step_full_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
    ):
        """Pin every INACTIVE free slot at its park pose — `task_hooks.
        repark_inactive_slots`. Inert on a family whose every task activates
        every slot (`libero_goal`); a union family (`libero_object`, 7 of 11
        active per task) needs exactly this, and an inactive slot is what the
        observation's mask words describe."""
        repark_inactive_slots[Self.P, DTYPE, BATCH_SIZE, NQ, NV](
            qpos, qvel, meta, env
        )

    @always_inline
    @staticmethod
    def custom_extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NV_F: Int,
        NBODY_F: Int,
        OBS_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ_F],
        qvel: Tile[DTYPE, BATCH_SIZE, NV_F],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY_F * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        bodies: Tile[DTYPE, NBODY_F, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY_F * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY_F * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY_F * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
    ) -> Bool:
        """`task_hooks.write_task_obs`: `qpos`, `qvel`, one active word per
        free slot, then the nine goal words — measured from `robot_grip_site`,
        with an `In`/`On` target at ITS OWN region's site.

        ⚠ EACH FAMILY'S `<FAMILY>_OBS_DIM` IS THAT WIDTH, defined beside its
        model def as `NQ + NV + P.N_FREE + TASK_GOAL_WORDS`, so the number the
        env allocates and the layout written here are one expression."""
        write_task_obs[
            Self.P, DTYPE, BATCH_SIZE, NQ_F, NV_F, NBODY_F,
            SITE_DIM, OBS_DIM,
        ](qpos, qvel, xpos, site_xpos, meta, obs, env)
        _ = xquat
        _ = xvel
        _ = bodies
        _ = contacts
        _ = sites
        _ = geoms
        _ = xipos
        _ = xangvel
        _ = cvel
        _ = cacc
        _ = cfrc_int
        _ = subtree_com
        _ = site_xpos_acc
        _ = xquat_acc
        _ = act
        return True

    @staticmethod
    def custom_extract_obs_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """The single-env twin — `task_hooks.write_task_obs_host`. A batched
        run writes the checkpoint a single-env eval loads, so the two must agree
        word for word; `tests/libero/test_libero_task_hooks.mojo` demands it."""
        write_task_obs_host[Self.P, DTYPE, D](d, obs)
        _ = m_bodies
        _ = m_joints
        _ = m_geoms
        _ = m_sites
        _ = act
        return True

    @always_inline
    @staticmethod
    def compute_reward_and_done_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        ACTION_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        cfrc_ext: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        curriculum: Tile[DTYPE, 1, MODEL_CURRICULUM_SIZE],
        actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        env: Int,
        step_count: Int,
        frame_skip: Int,
        timestep: Scalar[DTYPE],
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """LIBERO'S REWARD IS ITS GOAL: sparse, +1 when the tape holds.

        ⚠⚠ NO SHAPING, DELIBERATELY. `lifelong/metric.py` scores an episode by
        `_check_success()` alone and every published LIBERO number is that. A
        shaped reward is a research choice about one experiment; putting it in
        the config that also defines the benchmark would make two runs
        incomparable while looking identical. `So101TabletopConfig` shapes
        because its family is ours to design — this one is not.

        ⚠ THE WIDE OVERLOAD OF `eval_tape_gpu`, with this hook's own qpos,
        sites, bodies and contacts. Most LIBERO goals are CONTACT
        predicates (`On`, `In`); the narrow overload compiles those
        branches out and would read every one of them as False — a reward that
        is zero for a reason no log would show.

        ⚠ THE TAPE COMES FROM `meta`, WRITTEN PER EPISODE BY THE DRIVER. Until
        one writes it, `eval_tape_gpu` reads an all-zero tape and returns False,
        so the reward is 0 — the same contract the SO-101 family runs under, and
        the reason `tasks/gpu_eval.mojo` exists.
        """
        var holds = eval_tape_gpu[
            DTYPE, BATCH_SIZE, NBODY, SITE_DIM, NQ, NSITE_F, MC_F
        ](
            meta, curriculum, xpos, xquat, site_xpos, qpos, sites, bodies,
            contacts, env,
        )
        # ⚠⚠ THE SUCCESS BIT GOES TO `META_IDX_GOAL_HELD` TOO. Here the reward
        # IS that bit, so a driver could read either — but the drivers that
        # count success (`task_batched_gpu`, `task_eval_frozen`) read the
        # word, because on a SHAPED family the reward is not the bit. Writing
        # it on every family keeps one way to ask.
        meta[env, META_IDX_GOAL_HELD] = (
            Scalar[DTYPE](1) if holds else Scalar[DTYPE](0)
        )
        # ⚠ THE `Bool` IS AN ASK AND IS DISCARDED UNLESS THE ENV WAS BUILT WITH
        # `TERMINATE_ON_UNHEALTHY=True` (`phyics3d_batched_env.mojo`). A driver
        # reading success out of `_done` without that flag reads a constant
        # zero; read `_reward` instead, which needs no flag.
        _ = qvel
        _ = xipos
        _ = xvel
        _ = geoms
        _ = cfrc_ext
        _ = cvel
        _ = actions
        _ = xangvel
        _ = cacc
        _ = cfrc_int
        _ = subtree_com
        _ = site_xpos_acc
        _ = xquat_acc
        _ = act
        _ = step_count
        _ = frame_skip
        _ = timestep
        return (Scalar[DTYPE](1) if holds else Scalar[DTYPE](0), holds)

    @staticmethod
    def compute_reward_and_done_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        """ZERO REWARD, NEVER TERMINATED — and that is the honest state of it.

        ⚠⚠ THIS IS NOT A PLACEHOLDER THAT SOMEONE FORGOT. LIBERO's reward is
        its GOAL, which lives in the task layer (`tasks/predicates.mojo` bound
        per task, `tasks/tape.mojo` for the device, `tasks/gpu_eval.mojo` for
        the per-lane bit) and is selected by the `.task` file the episode is
        running — not by the config, which is one comptime type for every
        task of a family. Wiring it here would be a second copy of the tape.

        A driver that trains on this config would see a flat zero curve, which
        is why the header says so in as many words. `examples/libero/libero_eval.mojo`
        is the loop that evaluates the real goal, on the CPU, today.
        """
        _ = len(m_bodies) + len(m_joints) + len(m_geoms) + len(m_sites)
        _ = len(actions) + step_count + frame_skip
        _ = prev_x
        return (Scalar[DTYPE](0), False)

    @always_inline
    @staticmethod
    def init_qpos_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ_F: Int,
        NJOINT_F: Int,
        NV_F: Int,
        NBODY_M: Int,
        NGEOM_F: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ_F],
        qvel: Tile[DTYPE, BATCH_SIZE, NV_F],
        joints: Tile[DTYPE, NJOINT_F, MODEL_JOINT_SIZE],
        mocap_pos: Tile[DTYPE, BATCH_SIZE, NBODY_M * 3],
        mocap_quat: Tile[DTYPE, BATCH_SIZE, NBODY_M * 4],
        bodies: Tile[DTYPE, NBODY_M, MODEL_BODY_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        env: Int,
        seed: Int,
    ):
        """THE BASE POSE, THE JOINT DRAWS AND FREE-SLOT PLACEMENT AT RESET.

        ⚠ THE SAME KERNEL AS `so101_tabletop`, reading the family's GENERATED
        table (`placement/<family>.mojo`), and gated against the host sampler
        on every LIBERO task by `tests/tasks/test_device_placement.mojo`. It
        places what the lane's init words in `meta` say; a driver that writes
        none leaves every free slot where `_reset_env_lane` put it.

        ⚠ WORDS, NOT A TASK. The driver writes them per lane with
        `active.init_region_words` and `placement.check.joint_init_words`
        after `require_device_placement[P]`.

        ⚠ `jinit=` FIRST, THEN PLACEMENTS (`reset_task_slots`): a prop drawn
        into a drawer region stands in the drawer the draw just opened."""
        reset_task_slots[Self.P, DTYPE, BATCH_SIZE, NQ_F, NV_F](
            qpos, qvel, meta, env, seed
        )
        _ = joints
        _ = mocap_pos
        _ = mocap_quat
        _ = bodies
        _ = geoms

    @staticmethod
    def get_timestep() -> Float64:
        """`LIBERO_TIMESTEP` — a scalar literal, for the reason its docstring
        gives (ptxas's `KGEN_CompilerRT_GetOrCreateGlobal`). The generated dims
        of all 23 families are compared against it by the hooks test."""
        return LIBERO_TIMESTEP



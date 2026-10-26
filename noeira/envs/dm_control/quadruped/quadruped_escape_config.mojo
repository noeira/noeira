"""`dm_control` `quadruped escape` — the 49th and last suite task.

`suite/quadruped.py::Escape`. Everything `Move` observes, plus where the origin
is and what twenty rangefinders see; the reward is uprightness times how far
the robot has escaped a bowl-shaped heightfield.

⚠⚠ THIS IS THE ONLY SUITE TASK WHOSE TERRAIN IS STATE. `<hfield nrow ncol>`
with no `file` is a grid of zeros that `initialize_episode` overwrites on every
reset — see `custom_reset_cpu`. Every other model in the suite is fixed once
built.

WHAT ESCAPE ADDS TO `_common_observations`, in order:

  · `origin` (3) — the world origin expressed in the TORSO frame, i.e.
    `-torso_pos . torso_xmat`. ⚠ NOT the torso's position: it is the vector
    the robot would have to travel to get back, in its own frame, so it
    rotates with the robot.
  · `rangefinder` (20) — `where(reading == -1, 1.0, tanh(reading))`.
    ⚠⚠ A MISS BECOMES 1.0, NOT A NEGATIVE. `-1` is the sentinel `mj_ray`
    returns for no intersection and dm_control replaces it BEFORE the `tanh`,
    so a miss reads as MAXIMUM range — the same value a very distant hit
    saturates to. Feeding -1 through `tanh` instead gives -0.76 on exactly the
    states where the robot has escaped, which is where the reward is highest.

THE REWARD is `_upright_reward(deviation_angle=20) * escape_reward`.
⚠ `deviation_angle=20` here and `0` in `Move` — the bound is
`cos(20 deg) = 0.9397` rather than 1, so the torso may lean 20 degrees before
the term starts falling. Copying `Move`'s upright term would make the task
harder in a way no gate on the walk model can see.
`escape_reward` is `tolerance(|site_xpos['workspace']|, bounds=(30, inf),
margin=30, value_at_margin=0, linear)` — 0 at the centre of the bowl, 1 once
the workspace site is `hfield_size[0]` metres out.
"""

from std.math import sqrt, tanh, inf, cos, pi
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom

from noeira.physics3d.fields import Data, Model
from noeira.physics3d.fields.dims import DimsLike
from noeira.physics3d.ray.model import ray_model
from noeira.physics3d.kinematics.site_frame import (
    site_world_quat_list,
    site_world_quat,
)
from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from noeira.physics3d.sensors.rangefinder import rangefinder_site
from noeira.physics3d.kinematics.xmat import (
    xmat_elem,
    xmat_elem_gpu,
    XMAT_ZZ,
)
from noeira.physics3d.gpu.constants import (
    MODEL_HFIELD_META_SIZE,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
    HFIELD_META_IDX_NCOL,
    HFIELD_META_IDX_SIZE_X,
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.gpu.constants import (
    MAX_GPU_HFIELDS,
    MAX_GPU_MESHES,
    MESH_ARENA_FLOATS_PER_TRI,
    MODEL_MESH_META_SIZE,
    MODEL_BODY_SIZE,
    MODEL_GEOM_SIZE,
    METADATA_SIZE,
    CONTACT_SIZE,
)
from ..gpu_reset import reset_seed
from ..dtype_math import sqrt_dt, cos_dt, tanh_dt, inf_dt
from ..rewards import tolerance, SIGMOID_LINEAR
from .quadruped_config import (
    _upright_reward_at,
    _common_obs_cpu,
    _common_obs_gpu,
    _random_root_orientation,
    QUADRUPED_FRAME_SKIP,
    QUADRUPED_MAX_STEPS,
)
from .quadruped_xml import (
    TORSO_BODY_IDX,
    TORSO_SITE_IDX,
    TOE_SITE_0,
)
from .quadruped_dims import DM_QUADRUPED_ESCAPE_DIMS as qep
from ...phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.tiles import Tile


# ── Model indices, resolved by NAME against the asset ─────────────────────
# ⚠ Pinned in `test_quadruped_escape_vs_dm_control` by looking each one up in
# MuJoCo rather than trusting these; a task fragment that inserted a site
# would otherwise shift every rangefinder silently.
comptime ESCAPE_WORKSPACE_SITE: Int = 2
comptime ESCAPE_RF_SITE_0: Int = 3
comptime ESCAPE_N_RF: Int = 20
comptime ESCAPE_TERRAIN_GEOM: Int = 1

# `hfield_size[0]`, the terrain's x radius. Both the reward's bound and its
# margin.
comptime ESCAPE_TERRAIN_RADIUS: Float64 = 30.0

# `_upright_reward(deviation_angle=20)` — cos(20 deg).
comptime ESCAPE_UPRIGHT_DEVIATION: Float64 = 0.9396926207859084

# `_HEIGHTFIELD_ID`, `_TERRAIN_SMOOTHNESS`, `_TERRAIN_BUMP_SCALE`.
comptime ESCAPE_HFIELD_ID: Int = 0
comptime ESCAPE_TERRAIN_SMOOTHNESS: Float64 = 0.15
comptime ESCAPE_TERRAIN_BUMP_SCALE: Float64 = 2.0

# ⚠ A SALT, so the terrain stream is independent of the ORIENTATION stream.
# `custom_reset_cpu`/`init_qpos_gpu` already draw from `reset_seed(env, seed)`;
# reusing that key unsalted would tie which way the quadruped faces to which
# bumps it faces, and the reference draws the two independently.
comptime ESCAPE_TERRAIN_PHILOX_SALT: UInt64 = 0x9E3779B97F4A7C15


def _escape_bump[
    DTYPE: DType
](
    key: UInt64, k: Int, lo: Scalar[DTYPE], span: Scalar[DTYPE]
) -> Scalar[DTYPE]:
    """Bump `k` of the terrain grid: `uniform(smoothness, 1)`.

    Counter-addressed rather than sequential — `offset=k` names the draw, so
    any lane can evaluate any bump with no state. Only lane 0 of the returned
    quad is used; taking `[k % 4]` would be a runtime index into a SIMD to save
    three quarters of the Philox work, and this runs once per episode.
    """
    var rng = PhiloxRandom(seed=key, offset=UInt64(k))
    var v = rng.step_uniform()
    return lo + Scalar[DTYPE](v[0]) * span


struct DMQuadrupedEscapeConfig(Phyics3dEnvConfig):
    """`Escape` — bowl terrain, twenty rangefinders, distance-from-origin."""

    comptime FRAME_SKIP: Int = QUADRUPED_FRAME_SKIP
    comptime MAX_STEPS: Int = QUADRUPED_MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR: StaticString = "euler"
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime RNE_POST: Bool = True
    comptime RESET_FIND_HEIGHT: Bool = True
    # ⚠ THE THREE GPU HOOKS ARE ALL PRESENT, and all three are required —
    # claiming `True` with any one missing runs the batched path with a zero
    # observation for the last 23 dims, or a flat terrain, or no reward.
    #   · `init_hfield_gpu`          — the per-lane bowl, at reset
    #   · `custom_extract_obs_ray_gpu` — 78 common + origin + 20 rangefinders
    #   · `compute_reward_and_done_gpu` — upright(20 deg) x escape
    comptime HAS_GPU_HOOKS: Bool = True

    # ⚠ THE HEIGHTFIELD GRID IS 201x201 AND MUST BE ALLOCATED EXACTLY.
    # `<hfield nrow ncol>` with NO `file` is a grid of zeros that
    # `custom_reset_full_cpu` overwrites every reset — the only model in the
    # suite whose terrain is STATE rather than an asset. Left at the default 0
    # the grid is one element wide and the terrain collides as a flat plane.
    comptime NHFIELD_DATA: Int = 201 * 201

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(qep.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        # `Escape.initialize_episode` randomizes the ORIENTATION only.
        return 0.0

    @staticmethod
    def custom_extract_obs_ray_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        mut mf: Model[DTYPE, D],
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) raises -> Bool:
        """`_common_observations` + `origin` + `rangefinder`."""
        if not _common_obs_cpu[DTYPE, TORSO_SITE_IDX, TOE_SITE_0](
            d, mf.bodies.data, mf.joints.data, mf.geoms.data, mf.sites.data,
            act, obs,
        ):
            return False

        # ── origin, in the torso frame ────────────────────────────────────
        # `-torso_pos.dot(torso_frame)`; numpy's `v.dot(M)` is `M^T v`, so
        # this is the torso rotation's INVERSE applied to `-torso_pos`.
        var bq = QuatGeneric[DTYPE](
            d.xquat.data[TORSO_BODY_IDX * 4 + 3],
            d.xquat.data[TORSO_BODY_IDX * 4 + 0],
            d.xquat.data[TORSO_BODY_IDX * 4 + 1],
            d.xquat.data[TORSO_BODY_IDX * 4 + 2],
        )
        var tp = Vec3Generic[DTYPE](
            d.xpos.data[TORSO_BODY_IDX * 3 + 0],
            d.xpos.data[TORSO_BODY_IDX * 3 + 1],
            d.xpos.data[TORSO_BODY_IDX * 3 + 2],
        )
        var org = bq.rotate_vec_inverse(
            Vec3Generic[DTYPE](-tp.x, -tp.y, -tp.z)
        )
        obs.append(org.x)
        obs.append(org.y)
        obs.append(org.z)

        # ── the twenty rangefinders ───────────────────────────────────────
        # ⚠ `comptime if` RATHER THAN A CONSTRAINT ON THE HOOK. The trait
        # declares this over an unconstrained `DTYPE` and `ray_model` needs
        # floating-point evidence; putting `where DTYPE.is_floating_point()`
        # on the trait member would impose it on all fifty implementers.
        comptime if DTYPE.is_floating_point():
            for i in range(ESCAPE_N_RF):
                var raw = rangefinder_site[DTYPE, D, 1](
                    d, mf, ESCAPE_RF_SITE_0 + i
                )
                # `np.where(rf == -1.0, 1.0, np.tanh(rf))` — A MISS IS 1.0.
                obs.append(
                    Scalar[DTYPE](1.0) if raw == -1.0
                    else Scalar[DTYPE](tanh(raw))
                )
            return True
        else:
            # An integer-typed physics model is not a thing this engine
            # builds; returning False would silently fall back to the default
            # observation, which is the WRONG LENGTH.
            return False

    @staticmethod
    def custom_reset_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`orientation = randn(4); orientation /= norm(orientation)`.

        The same draw `Move.initialize_episode` makes; the height is found
        afterwards by `RESET_FIND_HEIGHT`. Shared verbatim with walk/run
        rather than re-derived — see `DMQuadrupedConfig.custom_reset_cpu`.
        """
        _random_root_orientation[DTYPE, D](d)

    @staticmethod
    def custom_reset_full_cpu[DTYPE: DType, D: DimsLike](
        mut d: Data[DTYPE, D, 1],
        mut mf: Model[DTYPE, D],
    ) raises:
        """Rewrite the heightfield — `Escape.initialize_episode`.

        ⚠⚠ THIS RUNS BEFORE `_find_non_contacting_height`, and the order is
        load-bearing: that routine raises the robot in 1 cm steps until nothing
        touches, so it must see the terrain it will stand on. Generating the
        terrain afterwards would spawn the quadruped inside a hill.

            radius     = clip(hypot(col, row), .04, 1)   over [-1, 1]^2
            bowl       = .5 - cos(2*pi*radius)/2
            bumps      = uniform(0.15, 1, (bump_res, bump_res))
            terrain    = bowl * zoom(bumps, res/bump_res)

        `bump_res = int(2*hfield_size[0] / 2) = 30` and `res = 201`.

        ⚠⚠ ONE LABELLED DEVIATION: THE UPSAMPLING KERNEL. dm_control calls
        `scipy.ndimage.zoom(..., order=3)` — a cubic B-spline with a recursive
        prefilter and scipy's own boundary handling. This uses BILINEAR
        interpolation on the same grid at the same sample positions. The bowl,
        the bump distribution, the grid, the product and the `[0, 1]`
        normalisation MuJoCo applies on load are all identical; what differs is
        the shape of the interpolation between bump centres, so a given seed
        produces a different-but-equivalent terrain rather than a different
        KIND of terrain.
        ⚠ The consequence, stated rather than buried: a per-episode terrain
        cannot be compared against dm_control's, so
        `test_quadruped_escape_vs_dm_control` writes the SAME grid into both
        engines and gates everything downstream of it. Nothing here gates the
        generator itself beyond its shape.
        """
        var base = ESCAPE_HFIELD_ID * MODEL_HFIELD_META_SIZE
        var adr = Int(mf.hfield_meta.data[base + HFIELD_META_IDX_ADR])
        var nrow = Int(mf.hfield_meta.data[base + HFIELD_META_IDX_NROW])
        var ncol = Int(mf.hfield_meta.data[base + HFIELD_META_IDX_NCOL])
        if nrow != ncol:
            raise Error(
                "escape: the heightfield must be square, got "
                + String(nrow) + "x" + String(ncol)
            )
        var res = nrow
        var sx = Float64(mf.hfield_meta.data[base + HFIELD_META_IDX_SIZE_X])
        var bump_res = Int((2.0 * sx) / ESCAPE_TERRAIN_BUMP_SCALE)
        if bump_res < 2:
            bump_res = 2

        # The bumps, drawn once per reset.
        var bumps = List[Float64](capacity=bump_res * bump_res)
        for _ in range(bump_res * bump_res):
            bumps.append(
                ESCAPE_TERRAIN_SMOOTHNESS
                + random_float64() * (1.0 - ESCAPE_TERRAIN_SMOOTHNESS)
            )

        # `np.ogrid[-1:1:res*1j]` — res samples INCLUSIVE of both ends.
        var step = 2.0 / Float64(res - 1)
        for r in range(res):
            var ry = -1.0 + Float64(r) * step
            for c in range(res):
                var cx = -1.0 + Float64(c) * step
                var rad = sqrt(cx * cx + ry * ry)
                # `np.clip(radius, .04, 1)` — the floor keeps the bowl's
                # centre flat instead of cusped, the ceiling stops the cosine
                # turning back up outside the unit disc.
                if rad < 0.04:
                    rad = 0.04
                if rad > 1.0:
                    rad = 1.0
                var bowl = 0.5 - cos(2.0 * pi * rad) / 2.0

                # Bilinear sample of `bumps` at the same position `zoom` uses
                # with `grid_mode=False`: input index = out * (n_in-1)/(n_out-1).
                var fy = Float64(r) * Float64(bump_res - 1) / Float64(res - 1)
                var fx = Float64(c) * Float64(bump_res - 1) / Float64(res - 1)
                var y0 = Int(fy)
                var x0 = Int(fx)
                var y1 = y0 + 1 if y0 + 1 < bump_res else y0
                var x1 = x0 + 1 if x0 + 1 < bump_res else x0
                var ty = fy - Float64(y0)
                var tx = fx - Float64(x0)
                var b00 = bumps[y0 * bump_res + x0]
                var b01 = bumps[y0 * bump_res + x1]
                var b10 = bumps[y1 * bump_res + x0]
                var b11 = bumps[y1 * bump_res + x1]
                var smooth = (
                    b00 * (1.0 - tx) * (1.0 - ty)
                    + b01 * tx * (1.0 - ty)
                    + b10 * (1.0 - tx) * ty
                    + b11 * tx * ty
                )
                # ⚠⚠ ROUNDED TO float32, BECAUSE `mjModel.hfield_data` IS
                # `float*`. MuJoCo computes the terrain in double and stores
                # it as float, and every ray it casts reads those floats.
                # Keeping our copy in double makes the two engines intersect
                # DIFFERENT SURFACES — measured at 1.1e-07 on the rangefinder
                # parity gate before this cast, from an elevation difference
                # of ~5e-08 m amplified by shallow incidence. Same rule as
                # `load_mesh_hull`'s hull vertices.
                d.hfield_data.data[adr + r * res + c] = Scalar[DTYPE](
                    Float64(bowl * smooth).cast[DType.float32]()
                )

    @always_inline
    @staticmethod
    def custom_extract_obs_ray_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        OBS_DIM: Int,
        SITE_DIM: Int,
        MC_F: Int,
        NSITE_F: Int,
        NGEOM_F: Int,
        NA_F: Int,
        NMESH_TRI_F: Int,
        NHFIELD_DATA: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, NV],
        xpos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xquat: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        xvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
        site_xpos: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        contacts: Tile[DTYPE, BATCH_SIZE, MC_F * CONTACT_SIZE],
        sites: Tile[DTYPE, NSITE_F, MODEL_SITE_SIZE],
        geoms: Tile[DTYPE, NGEOM_F, MODEL_GEOM_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
        xipos: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        xangvel: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        cvel: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cacc: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        cfrc_int: Tile[DTYPE, BATCH_SIZE, NBODY * 6],
        subtree_com: Tile[DTYPE, BATCH_SIZE, NBODY * 3],
        site_xpos_acc: Tile[DTYPE, BATCH_SIZE, SITE_DIM],
        xquat_acc: Tile[DTYPE, BATCH_SIZE, NBODY * 4],
        act: Tile[DTYPE, BATCH_SIZE, NA_F],
        mesh_meta: Tile[DTYPE, MAX_GPU_MESHES * MODEL_MESH_META_SIZE],
        mesh_tris: Tile[DTYPE, NMESH_TRI_F * MESH_ARENA_FLOATS_PER_TRI],
        hfield_meta: Tile[DTYPE, MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE],
        hfield_data: Tile[DTYPE, BATCH_SIZE * NHFIELD_DATA],
        env: Int,
    ) -> Bool:
        """`custom_extract_obs_ray_cpu` against the batched field tensors.

        Block for block the same three pieces — `_common_observations`, the
        origin in the torso frame, twenty rangefinders — reading the same
        indices. The CPU twin is the one to read for WHY each block is what it
        is; this one only says what differs.

        ⚠ `comptime if` RATHER THAN A CONSTRAINT ON THE HOOK, exactly as on the
        CPU side: the trait declares this over an unconstrained `DTYPE` and
        `ray_model` needs floating-point evidence, and putting
        `where DTYPE.is_floating_point()` on the trait member would impose it
        on all fifty implementers.

        ⚠ THE COST IS PER LANE. Twenty rays over eighteen geoms is 360 geom
        queries per lane per step on top of the physics — the reason
        `HAS_GPU_HOOKS` is worth having at all is that those 360 are
        embarrassingly parallel across the batch, not that any one is cheap.
        """
        # ── `_common_observations` (78) ───────────────────────────────────
        # ⚠ `COMMON_DIM` PASSED EXPLICITLY. The default is `OBS_DIM`, and
        # escape's is 23 wider; without this the size assert inside would
        # reject the very block it is meant to protect.
        if not _common_obs_gpu[
            DTYPE, BATCH_SIZE, NQ, NV, NBODY, OBS_DIM, SITE_DIM, NSITE_F,
            NA_F, TORSO_SITE_IDX, TOE_SITE_0,
            OBS_DIM - 3 - ESCAPE_N_RF,
        ](
            qpos, qvel, xipos, xquat, xvel, xangvel, bodies, site_xpos,
            sites, cvel, cacc, cfrc_int, subtree_com, site_xpos_acc,
            xquat_acc, act, obs, env,
        ):
            return False

        var k = OBS_DIM - 3 - ESCAPE_N_RF

        # ── origin, in the torso frame (3) ────────────────────────────────
        var bq = QuatGeneric[DTYPE](
            rebind[Scalar[DTYPE]](xquat[env, TORSO_BODY_IDX * 4 + 3]),
            rebind[Scalar[DTYPE]](xquat[env, TORSO_BODY_IDX * 4 + 0]),
            rebind[Scalar[DTYPE]](xquat[env, TORSO_BODY_IDX * 4 + 1]),
            rebind[Scalar[DTYPE]](xquat[env, TORSO_BODY_IDX * 4 + 2]),
        )
        var org = bq.rotate_vec_inverse(
            Vec3Generic[DTYPE](
                -rebind[Scalar[DTYPE]](xpos[env, TORSO_BODY_IDX * 3 + 0]),
                -rebind[Scalar[DTYPE]](xpos[env, TORSO_BODY_IDX * 3 + 1]),
                -rebind[Scalar[DTYPE]](xpos[env, TORSO_BODY_IDX * 3 + 2]),
            )
        )
        obs[env, k] = org.x
        obs[env, k + 1] = org.y
        obs[env, k + 2] = org.z
        k += 3

        # ── the twenty rangefinders (20) ──────────────────────────────────
        # ⚠ THE POSITIVE BRANCH, NOT AN EARLY RETURN. `comptime if not
        # is_floating_point(): return False` does NOT carry the evidence
        # `ray_model` needs into the code after it — the compiler rejects the
        # call with "lacking evidence to prove correctness". Wrapping the
        # users of that evidence is what supplies it, and it is the shape the
        # CPU twin already uses.
        comptime if DTYPE.is_floating_point():
            for i in range(ESCAPE_N_RF):
                var site = ESCAPE_RF_SITE_0 + i
                var sbody = Int(
                    rebind[Scalar[DTYPE]](sites[site, SITE_IDX_BODY])
                )
                var origin = Vec3Generic[DTYPE](
                    rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 0]),
                    rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 1]),
                    rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 2]),
                )
                var q4 = site_world_quat[DTYPE](env, site, sites, xquat)
                var sq = QuatGeneric[DTYPE](q4[3], q4[0], q4[1], q4[2])
                # ⚠ +Z. A rangefinder fires along the site's own +Z; a CAMERA
                # looks down its -Z. See `sensors/rangefinder.mojo`.
                var rvec = sq.rotate_vec(Vec3Generic[DTYPE](0, 0, 1))

                var hit = ray_model[DTYPE](
                    geoms, NGEOM_F, bodies, xpos, xquat, env,
                    mesh_meta, mesh_tris, hfield_meta, hfield_data,
                    NHFIELD_DATA, origin, rvec, sbody,
                )
                # `np.where(rf == -1.0, 1.0, np.tanh(rf))` — A MISS IS 1.0.
                obs[env, k] = (
                    Scalar[DTYPE](1.0) if hit.t == Scalar[DTYPE](-1.0)
                    else tanh_dt[DTYPE](hit.t)
                )
                k += 1

            return True
        else:
            # An integer-typed physics model is not a thing this engine
            # builds; returning False falls back to a DIFFERENT
            # observation of the wrong length, so say so rather than
            # leave the last 23 dims at whatever the buffer last held.
            return False

    @staticmethod
    def init_hfield_gpu[
        DTYPE: DType, BATCH_SIZE: Int, NHFIELD_DATA: Int
    ](
        hfield_meta: Tile[DTYPE, MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE],
        hfield_data: Tile[DTYPE, BATCH_SIZE * NHFIELD_DATA],
        env: Int,
        seed: Int,
    ):
        """`custom_reset_full_cpu`'s terrain, per lane, on device.

        Same bowl, same bump distribution, same bilinear upsample — see the
        CPU docstring for the reference formula and for the one labelled
        deviation (scipy's cubic `zoom` vs bilinear).

        TWO GPU-ONLY DEVIATIONS, both stated rather than buried:

        ⚠ THE ARITHMETIC IS `DTYPE`, NOT `Float64`. Float64 is banned on this
        engine's device path, so the bowl, the radius and the bilinear blend
        are computed in float32 where the CPU computes them in double and
        rounds once at the end. The stored grid therefore differs from the CPU
        one in the last few ulps for the SAME seed. This costs nothing that is
        gated: the terrain is redrawn every episode from a per-lane stream that
        does not match the CPU's anyway (below), and
        `test_quadruped_escape_vs_dm_control` injects a SHARED grid rather than
        comparing two generated ones.

        ⚠ THE BUMP STREAM IS PER-LANE PHILOX, ADDRESSED BY BUMP INDEX. The CPU
        draws `bump_res^2` values in order from host `random_float64`; here bump
        `k` is `Random(seed=key, offset=k).step_uniform()[0]`, so a lane can
        evaluate any bump without holding the grid. That matters twice over:
        a 900-entry per-thread `Array` would be 3.6 KB of stack per lane,
        AND indexing one by a runtime value is the Metal miscompute class this
        engine has now hit four times. No array, no index, no exposure.

        The four corners are refetched only when `x0` moves — `res / bump_res`
        is about 7, so this is ~4 draws per 7 columns rather than per column.
        """
        comptime if not DTYPE.is_floating_point():
            return

        var base = ESCAPE_HFIELD_ID * MODEL_HFIELD_META_SIZE
        var adr = Int(rebind[Scalar[DTYPE]](hfield_meta[base + HFIELD_META_IDX_ADR]))
        var nrow = Int(rebind[Scalar[DTYPE]](hfield_meta[base + HFIELD_META_IDX_NROW]))
        var ncol = Int(rebind[Scalar[DTYPE]](hfield_meta[base + HFIELD_META_IDX_NCOL]))
        # ⚠ NO `raise` HERE — a kernel cannot. The CPU hook rejects a
        # non-square grid; this one leaves the terrain untouched, which is the
        # flat plane the model already had rather than a half-written grid.
        if nrow != ncol or nrow < 2:
            return
        var res = nrow
        if res * res > NHFIELD_DATA:
            return

        var sx = rebind[Scalar[DTYPE]](
            hfield_meta[base + HFIELD_META_IDX_SIZE_X]
        )
        var bump_res = Int(
            (Scalar[DTYPE](2.0) * sx)
            / Scalar[DTYPE](ESCAPE_TERRAIN_BUMP_SCALE)
        )
        if bump_res < 2:
            bump_res = 2

        var key = reset_seed(env, seed) ^ ESCAPE_TERRAIN_PHILOX_SALT
        var smooth_lo = Scalar[DTYPE](ESCAPE_TERRAIN_SMOOTHNESS)
        var smooth_span = Scalar[DTYPE](1.0 - ESCAPE_TERRAIN_SMOOTHNESS)

        var step = Scalar[DTYPE](2.0) / Scalar[DTYPE](res - 1)
        var scale = Scalar[DTYPE](bump_res - 1) / Scalar[DTYPE](res - 1)
        var two_pi = Scalar[DTYPE](2.0 * pi)

        for r in range(res):
            var ry = Scalar[DTYPE](-1.0) + Scalar[DTYPE](r) * step
            var fy = Scalar[DTYPE](r) * scale
            var y0 = Int(fy)
            if y0 > bump_res - 1:
                y0 = bump_res - 1
            var y1 = y0 + 1 if y0 + 1 < bump_res else y0
            var ty = fy - Scalar[DTYPE](y0)

            # Corner cache, valid while `x0` holds. Plain scalars — see the
            # docstring on why this is deliberately not an array.
            var cached_x0 = -1
            var b00 = Scalar[DTYPE](0)
            var b01 = Scalar[DTYPE](0)
            var b10 = Scalar[DTYPE](0)
            var b11 = Scalar[DTYPE](0)

            for c in range(res):
                var cx = Scalar[DTYPE](-1.0) + Scalar[DTYPE](c) * step
                var rad = sqrt_dt[DTYPE](cx * cx + ry * ry)
                if rad < Scalar[DTYPE](0.04):
                    rad = Scalar[DTYPE](0.04)
                if rad > Scalar[DTYPE](1.0):
                    rad = Scalar[DTYPE](1.0)
                var bowl = (
                    Scalar[DTYPE](0.5)
                    - cos_dt[DTYPE](two_pi * rad) / Scalar[DTYPE](2.0)
                )

                var fx = Scalar[DTYPE](c) * scale
                var x0 = Int(fx)
                if x0 > bump_res - 1:
                    x0 = bump_res - 1
                var x1 = x0 + 1 if x0 + 1 < bump_res else x0
                var tx = fx - Scalar[DTYPE](x0)

                if x0 != cached_x0:
                    cached_x0 = x0
                    b00 = _escape_bump[DTYPE](
                        key, y0 * bump_res + x0, smooth_lo, smooth_span
                    )
                    b01 = _escape_bump[DTYPE](
                        key, y0 * bump_res + x1, smooth_lo, smooth_span
                    )
                    b10 = _escape_bump[DTYPE](
                        key, y1 * bump_res + x0, smooth_lo, smooth_span
                    )
                    b11 = _escape_bump[DTYPE](
                        key, y1 * bump_res + x1, smooth_lo, smooth_span
                    )

                var one = Scalar[DTYPE](1.0)
                var smooth = (
                    b00 * (one - tx) * (one - ty)
                    + b01 * tx * (one - ty)
                    + b10 * (one - tx) * ty
                    + b11 * tx * ty
                )
                hfield_data[
                    env * NHFIELD_DATA + adr + r * res + c
                ] = bowl * smooth

    @staticmethod
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
        """`_upright_reward(deviation_angle=20) * escape_reward`, per lane.

        The CPU twin block for block; see it for what each term means. The two
        deviations from `Move`'s GPU reward are the ones the CPU docstring
        already names — the upright bound is `cos(20 deg)` rather than 1, and
        the second term is distance from the origin rather than forward speed.

        ⚠ `origin_distance()` IS THE FULL 3-VECTOR NORM of the workspace
        site's world position, not a horizontal one. On a bowl the robot
        climbs as it escapes, so dropping z would under-report the distance
        exactly where the reward is meant to be rising.
        """
        var zz = xmat_elem_gpu[DTYPE](xquat, env, TORSO_BODY_IDX, XMAT_ZZ)
        var dev = Scalar[DTYPE](ESCAPE_UPRIGHT_DEVIATION)
        var upright = tolerance[SIGMOID_LINEAR, 0.0, DTYPE](
            zz, dev, inf_dt[DTYPE](), Scalar[DTYPE](1.0) + dev
        )

        var wx = rebind[Scalar[DTYPE]](
            site_xpos[env, ESCAPE_WORKSPACE_SITE * 3 + 0]
        )
        var wy = rebind[Scalar[DTYPE]](
            site_xpos[env, ESCAPE_WORKSPACE_SITE * 3 + 1]
        )
        var wz = rebind[Scalar[DTYPE]](
            site_xpos[env, ESCAPE_WORKSPACE_SITE * 3 + 2]
        )
        var dist = sqrt_dt[DTYPE](wx * wx + wy * wy + wz * wz)

        var escape = tolerance[SIGMOID_LINEAR, 0.0, DTYPE](
            dist,
            Scalar[DTYPE](ESCAPE_TERRAIN_RADIUS),
            inf_dt[DTYPE](),
            Scalar[DTYPE](ESCAPE_TERRAIN_RADIUS),
        )
        return (upright * escape, False)

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
        """`_upright_reward(deviation_angle=20) * escape_reward`."""
        var zz = xmat_elem(d, TORSO_BODY_IDX, XMAT_ZZ)
        var upright = _upright_reward_at(zz, ESCAPE_UPRIGHT_DEVIATION)

        # `origin_distance()` = |site_xpos['workspace']|, the FULL 3-vector
        # norm and not a horizontal one.
        var wx = Float64(d.site_xpos.data[ESCAPE_WORKSPACE_SITE * 3 + 0])
        var wy = Float64(d.site_xpos.data[ESCAPE_WORKSPACE_SITE * 3 + 1])
        var wz = Float64(d.site_xpos.data[ESCAPE_WORKSPACE_SITE * 3 + 2])
        var dist = sqrt(wx * wx + wy * wy + wz * wz)

        var escape = tolerance[SIGMOID_LINEAR, 0.0](
            dist,
            ESCAPE_TERRAIN_RADIUS,
            inf[DType.float64](),
            ESCAPE_TERRAIN_RADIUS,
        )
        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](upright * escape), False)


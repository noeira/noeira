"""Per-record tensor container for static model config (migration P1).

`Model` is the end-state replacement for the flat GPU model slab: each
record FAMILY becomes one owned packed tensor (`bodies [NBODY, 25]`,
`joints [NJOINT, 26]`, `geoms [NGEOM, 30]`, …) rather than one giant buffer.
Column indices are the existing `BODY_IDX_*` / `JOINT_IDX_*` / … constants —
after sunset they move here and become the single, local column definition.

Records stay PACKED (not one tensor per column) by design: the P0 audit +
the measured Metal operand cliff (28 ok / 29 = JIT abort) make per-column
splits of body/joint/geom records blow the kernel operand budget on the wide
kernels. Kernels bind one record tensor per entity kind and read columns —
same addressing as today's record-strided slab, minus the cross-family
offset math.

Build: `load_from_model` fills the record tensors DIRECTLY from the CPU
`Model` (`i * MODEL_<KIND>_SIZE + <KIND>_IDX_*`), no flat slab. The
transitional `load_from_slab` / `store_to_slab` bridges + the cross-family
`model_*_offset` tables were deleted at the P6 sunset.

Dtype note: all-`DTYPE` (int columns like `BODY_IDX_PARENT` are float-encoded).
Honest `int32` record tensors are a later cleanup.
"""

from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl

from .dims import DimsLike
from .data import Data

from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_TREE_SIZE,
    MODEL_META_IDX_NTREE,
    MODEL_CURRICULUM_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_EQ_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    MODEL_SENSOR_SIZE,
    MODEL_PAIR_SIZE,
    ACTDAMP_TRN_SIZE,
    MODEL_MESH_META_SIZE,
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
    MODEL_CAM_SIZE,
    MAX_GPU_CAMERAS,
    MODEL_GEOM_RGBA_SIZE,
    MODEL_MESH_POLY_SIZE,
    MAX_GPU_MESHES,
    MESH_ARENA_FLOATS_PER_TRI,
    mesh_max_poly,
    mesh_max_polyvert,
    mesh_max_edge,
    MODEL_META_IDX_CCD_TOLERANCE,
    MODEL_META_IDX_CCD_ITERATIONS,
    MODEL_META_IDX_SOLVER_ITERATIONS,
    MODEL_META_IDX_SOLVER_TOLERANCE,
    MODEL_META_IDX_LS_ITERATIONS,
    MODEL_META_IDX_LS_TOLERANCE,
    MODEL_META_IDX_NOSLIP_ITERATIONS,
    MJ_CCD_TOLERANCE,
    MJ_CCD_ITERATIONS,
    MJ_SOLVER_ITERATIONS,
    MJ_SOLVER_TOLERANCE,
    MJ_LS_ITERATIONS,
    MJ_LS_TOLERANCE,
)
from noeira.core.tiles import RowMajor

@always_inline
def _at_least_one(n: Int) -> Int:
    # Zero-entity record tensors still get a 1-element buffer so `.lt["gpu"]`
    # can always bind them as kernel operands (the layouts are zero-sized and
    # never indexed; an empty Optional[DeviceBuffer] would abort at bind).
    return n if n > 0 else 1


struct Model[
    DTYPE: DType,
    D: DimsLike,
](Movable):
    """Static model config as one packed tensor per record family (13
    tensors). See module docstring."""

    # ⚠ THE POSITIONAL HAZARD THIS STRUCT DOCUMENTED IS GONE.
    #
    # The parameter list used to carry a standing warning: every entry was an
    # `Int`, so inserting one mid-list silently shifted every positional
    # instantiation — `NMESH_VERTS` would take `NPAIR`'s value and mesh
    # collision would switch itself off across the tree with nothing to
    # compile-error on, and the comment recorded that "that exact failure has
    # happened here before". New dimensions had to go on the END, forever.
    #
    # With one `D` there are no positions to shift: a dimension is named, and
    # the 272 call sites that used to spell 7, 9 or 10 `Int`s in a fixed order
    # now name none. New dimensions can go anywhere in `DimsLike`.
    comptime NV = Self.D.NV
    comptime NBODY = Self.D.NBODY
    comptime NJOINT = Self.D.NJOINT
    comptime NGEOM = Self.D.NGEOM
    comptime NEQUALITY = Self.D.NEQUALITY
    comptime NTENDON = Self.D.NTENDON
    comptime NSITE = Self.D.NSITE
    comptime NEXCLUDE = Self.D.NEXCLUDE
    comptime NMESH_VERTS = Self.D.NMESH_VERTS
    comptime NPAIR = Self.D.NPAIR


    # Record view layouts ([N_ENTITY, RECORD_SIZE] row-major; tails 1-D).
    comptime L_BODY = RowMajor[Self.NBODY, MODEL_BODY_SIZE]
    comptime L_JOINT = RowMajor[Self.NJOINT, MODEL_JOINT_SIZE]
    comptime L_META = RowMajor[MODEL_META_SIZE]
    comptime L_CURRICULUM = RowMajor[MODEL_CURRICULUM_SIZE]
    comptime L_GEOM = RowMajor[Self.NGEOM, MODEL_GEOM_SIZE]
    comptime L_EQ = RowMajor[Self.NEQUALITY, MODEL_EQ_SIZE]
    comptime L_TENDON = RowMajor[Self.NTENDON, MODEL_TENDON_SIZE]
    comptime L_SITE = RowMajor[Self.NSITE, MODEL_SITE_SIZE]
    comptime L_BODY_INVW = RowMajor[Self.NBODY, 2]
    comptime L_DOF_INVW = RowMajor[Self.NV]
    # ⚠ `[NV, 3]` IS EXACT, NOT A CAP. A model cannot have more trees than
    # dofs, so this is a comptime bound that needs no `CAP_*` parameter — which
    # matters because `fields/dims.mojo` treats caps as load-bearing (a cap
    # answers "can I stack-allocate", not "does this exist"). The live row
    # count is `meta[MODEL_META_IDX_NTREE]`.
    comptime L_TREE = RowMajor[Self.NV, MODEL_TREE_SIZE]
    comptime L_EXCLUDE = RowMajor[Self.NEXCLUDE, 2]
    comptime L_PAIR = RowMajor[Self.NPAIR, MODEL_PAIR_SIZE]
    comptime L_MESH_META = RowMajor[MAX_GPU_MESHES, MODEL_MESH_META_SIZE]
    # ⚠ FLAT, NOT `[N, SIZE]`. Every consumer of these two indexes them as
    # `base + FIELD`, and a 2-D `LayoutTensor` given ONE index returns a ROW,
    # not an element — the mismatch that cost a debugging session on
    # `hfield_data` in `feb1b6c7`. Spelling the layout flat makes the wrong
    # access a compile error instead of a silent row read.
    comptime L_CAM = RowMajor[MAX_GPU_CAMERAS * MODEL_CAM_SIZE]
    comptime L_GEOM_RGBA = RowMajor[Self.NGEOM * MODEL_GEOM_RGBA_SIZE]
    comptime L_MESH_VERT = RowMajor[Self.NMESH_VERTS, 3]
    # Mesh POLYGON topology for the native multi-contact path. The capacities
    # are Euler's formula on NMESH_VERTS, so they need no type parameter of
    # their own — see `mesh_max_poly` / `mesh_max_polyvert`.
    comptime NMESH_POLY = mesh_max_poly(Self.NMESH_VERTS)
    comptime NMESH_POLYVERT = mesh_max_polyvert(Self.NMESH_VERTS)
    comptime L_MESH_POLY = RowMajor[Self.NMESH_POLY, MODEL_MESH_POLY_SIZE]
    comptime L_MESH_POLYVERT = RowMajor[Self.NMESH_POLYVERT]
    comptime L_MESH_POLYMAP = RowMajor[Self.NMESH_POLYVERT]
    comptime L_MESH_VERT_POLYMAP = RowMajor[Self.NMESH_VERTS, 2]
    comptime NMESH_EDGE = mesh_max_edge(Self.NMESH_VERTS)
    comptime L_MESH_VERT_EDGEADR = RowMajor[Self.NMESH_VERTS]
    comptime L_MESH_EDGE = RowMajor[Self.NMESH_EDGE]

    var bodies: TensorImpl[Self.DTYPE]  # [NBODY, MODEL_BODY_SIZE]
    var joints: TensorImpl[Self.DTYPE]  # [NJOINT, MODEL_JOINT_SIZE]
    var meta: TensorImpl[Self.DTYPE]  # [MODEL_META_SIZE]
    var curriculum: TensorImpl[Self.DTYPE]  # [MODEL_CURRICULUM_SIZE]
    var geoms: TensorImpl[Self.DTYPE]  # [NGEOM, MODEL_GEOM_SIZE]
    var equality: TensorImpl[Self.DTYPE]  # [NEQUALITY, MODEL_EQ_SIZE]
    var tendons: TensorImpl[Self.DTYPE]  # [NTENDON, MODEL_TENDON_SIZE]
    var sites: TensorImpl[Self.DTYPE]  # [NSITE, MODEL_SITE_SIZE]
    var sensors: TensorImpl[Self.DTYPE]  # [NSENSOR, MODEL_SENSOR_SIZE]
    var body_invweight0: TensorImpl[Self.DTYPE]  # [NBODY, 2]
    var dof_invweight0: TensorImpl[Self.DTYPE]  # [NV]
    var dof_M0: TensorImpl[Self.DTYPE]  # [NV]
    """Diagonal of the mass matrix at qpos0, armature included — MuJoCo's
    `dof_M0`.

    ⚠ NOT `1/dof_invweight0`. That is the diagonal of M INVERSE, and the two
    are reciprocals only for a diagonal M. MuJoCo keeps both for a reason and
    reads THIS one when it turns an actuator's `dampratio` into a `kv`
    (`engine_setconst.c:1025`).

    Filled by `compute_invweight0`, which already forms M at qpos0 and must
    read the diagonal BEFORE `ldl_factor` overwrites it in place."""
    var dof_parentid: TensorImpl[Self.DTYPE]  # [NV]
    """MuJoCo's `dof_parentid`: the dof each dof hangs from, -1 for a root.

    Within a body the dofs chain (dof `d`'s parent is `d-1` when both belong
    to the same body, across joints too); the first dof of a body hangs from
    the LAST dof of the nearest ancestor body that has one. Built by
    `fields_build` beside `trees` and gated against MuJoCo's own table
    (`test_dof_parentid_vs_mujoco`). It is what makes the tree-ordered LDL
    (`dynamics/ldl.mojo`, `_ldl_factor_tree_env`) possible: L keeps M's own
    sparsity — row `k` is nonzero at `k`'s ancestors only — so every solve is
    O(nC) rather than O(nv^2). ⚠ `-1` everywhere means "no table" (a `Model`
    built without the parser), and the dispatchers fall back to the dense
    factorisation on `MODEL_META_IDX_NTREE == 0`, exactly as `_dof_block`
    does for `trees`."""
    var trees: TensorImpl[Self.DTYPE]  # [NV, MODEL_TREE_SIZE]
    """The kinematic trees as `(dof_adr, dof_num, kind)` — M's diagonal blocks.

    ⚠ THE ROW COUNT IS `meta[MODEL_META_IDX_NTREE]`, NOT `NV`. The allocation
    is `[NV, 3]` because a model cannot have more trees than dofs; the rows past
    `ntree` are zero, and a zero row reads as "a tree at dof 0 of length 0" —
    legal-looking and wrong. See `MODEL_TREE_SIZE` for why a tree is a block of
    the mass matrix and why `kind` is CLASSIFIED rather than read off the joint
    type.

    Filled by `build_model_fields_from_flat` from `BODY_IDX_ROOTID`, which it
    has just written. It is model-time topology: nothing here depends on
    `qpos`.

    ⚠ BUILT BEFORE `compute_invweight0`, which calls `ldl_factor` during model
    build (`dynamics/invweight.mojo:236`) and will read this table once the
    factorisation is block-aware."""

    var dof_actdamp: TensorImpl[Self.DTYPE]  # [NV]
    """Per-dof actuator damping, `sum over actuators of kv * trn^2`.

    ⚠ THIS IS NOT `dof_damping`. `<joint damping>` is passive and MuJoCo's
    Euler already folds it into the velocity update; THIS is the `-kv*vel`
    term of a `<position>`/`<velocity>` servo, which Euler integrates
    EXPLICITLY and `implicitfast` folds into the mass matrix
    (`mjd_actuator_vel`). The distinction is the whole reason spot flies: its
    `dof_damping` is 0, so the passive path has nothing to stabilise and the
    only damping in the model is this.

    ⚠ THE DIAGONAL ONLY. MuJoCo forms the full `J^T diag(kv) J`, which for a
    JOINT transmission (one dof, `trn = gear*coef`) is exactly this diagonal
    and for a multi-dof transmission (tendon, site, ball) also has
    off-diagonal terms. `build_actuator_damping` WARNS when it meets one
    rather than dropping them quietly.

    Filled alongside the `dampratio` conversion, which is the point where the
    final `kv` of every actuator is known."""
    var actdamp_trn: TensorImpl[Self.DTYPE]  # [NACT * ACTDAMP_TRN_SIZE]
    """The OFF-DIAGONAL half of `mjd_actuator_vel`, per actuator.

    `dof_actdamp` above is the DIAGONAL of `J^T diag(kv) J`; MuJoCo adds the
    whole outer product `moment^T * biasprm[2] * moment` to `d->qDeriv`
    (`engine_derivative.c:1213`). For a JOINT transmission those are the same
    thing — one dof, one entry — and for a multi-dof one they are not.

    ⚠⚠ IT IS WORTH A FIELD BECAUSE IT IS A WHOLE SCENE. Seven models here have
    a multi-dof `kv` transmission and all seven are a tendon; forcing both
    engines to Euler — which never touches `qDeriv` — takes
    `hello_robot_stretch` from **4.406e-05 to 1.823e-10** while 49 of the
    other 50 `implicitfast` scenes do not move.

    ⚠ IN `Model`, NOT IN `Data`, because it is a model-time constant: the
    `(dof, gear*coef)` pairs of a joint or FIXED-tendon transmission do not
    depend on `qpos`. What IS state-dependent is whether the actuator is
    `forcerange`-saturated, and that rides on `Data.actdamp_act` — the same
    split `dof_actdamp` already uses. See `ACTDAMP_TRN_SIZE`."""
    var excludes: TensorImpl[Self.DTYPE]  # [NEXCLUDE, 2]
    var pairs: TensorImpl[Self.DTYPE]  # [NPAIR, MODEL_PAIR_SIZE]
    var mesh_meta: TensorImpl[Self.DTYPE]  # [MAX_GPU_MESHES, 4]
    var hfield_data0: TensorImpl[Self.DTYPE]  # [NHFIELD_DATA]
    """The heightfield grids AS PARSED — the reset value, not the live one.

    ⚠⚠ THE LIVE GRID IS `Data.hfield_data`, ONE PER ENVIRONMENT. This is the
    same relationship `SpecFields.qpos0` has to `Data.qpos`: the model carries
    the pose a reset restores, the state carries what is actually being
    simulated. `quadruped escape` rewrites its terrain every episode and the
    lanes of a batch reset at different times, so a shared live grid would
    hand every environment whichever terrain reset last.

    ⚠ `init_hfield_data` is what copies this into every lane, and a `Data`
    that never had it called has a grid of ZEROS — i.e. a flat terrain, which
    collides and rays perfectly happily and is simply the wrong shape.
    """

    var hfield_meta: TensorImpl[Self.DTYPE]  # [MAX_GPU_HFIELDS, 7]
    """`mjModel.hfield_*` minus the grid — adr, nrow, ncol and the four sizes.

    ⚠⚠ THE GRID ITSELF LIVES IN `Data.hfield_data`, NOT HERE, and that is the
    one place this engine deliberately disagrees with MuJoCo's split.
    `mjModel.hfield_data` is model data because MuJoCo simulates one world;
    `quadruped escape` rewrites the terrain on EVERY EPISODE, and in a batch
    the lanes reset at different times, so a shared grid would give every
    environment whichever terrain reset last. The metadata is genuinely
    per-asset and stays.
    """
    var cameras: TensorImpl[Self.DTYPE]  # [MAX_GPU_CAMERAS, MODEL_CAM_SIZE]
    """`mjModel.cam_*` as a tensor, for the batched ray tracer.

    ⚠⚠ THE POSE HERE IS LOCAL TO `CAM_IDX_BODY`, exactly as `mjModel.cam_pos`
    is. Composing it is `raytrace/camera.camera_world_frame`, which is
    `mj_camlight` — the batched twin the `kinematics/camera_frame.mojo`
    docstring said this would need. Reading these three fields as a world pose
    is right for every camera in `<worldbody>` and wrong for every camera on a
    moving body, which is the one failure mode this table exists to make
    spellable.

    ⚠ CAPPED AT `MAX_GPU_CAMERAS` AND ANNOUNCED, NOT TRUNCATED — see
    `fields_build`. Rows past the model's camera count have
    `CAM_IDX_ACTIVE == 0`.
    """
    var geom_rgba: TensorImpl[Self.DTYPE]  # [NGEOM, 4]
    """Each geom's visual colour, material already resolved.

    ⚠ THE PARSER RESOLVED THE MATERIAL, NOT THIS FILE.
    `_resolve_geom_materials` writes the material's colour into the geom's own
    `rgba` where MuJoCo says it should win (XMLreference: only where the geom's
    rgba still equals the default `0.5 0.5 0.5 1`), so this is a straight copy
    of `GeomData.rgba_*`.

    ⚠ THE SDL RENDERER DOES NOT USE THIS RULE. `model_def_from_xml` re-derives
    the colour at draw time as "the material always wins if there is one",
    which is a SECOND copy of a rule that has already drifted. The two paths
    can therefore disagree on a geom that carries both an explicit `rgba` and a
    `material`. This one follows the parser, i.e. MuJoCo; reconciling the
    renderer is a separate change and is not made here.

    ⚠ NOT IN `MODEL_GEOM_SIZE`. See `MODEL_GEOM_RGBA_SIZE`.
    """
    var mesh_verts: TensorImpl[Self.DTYPE]  # [NMESH_VERTS, 3]
    var mesh_tris: TensorImpl[Self.DTYPE]  # [NMESH_TRI * 3, 9] — an ARENA
    """The meshes' ORIGINAL triangles, nine floats each, principal frame.

    ⚠ A DIFFERENT SURFACE FROM `mesh_verts`, which is the convex HULL. The
    hull is what collision wants and what MuJoCo collides too; it cannot
    answer a RAY, because a ray aimed into a bracket's cutout must find the
    hole and the hull has none. `mj_rayMesh` walks `mesh_face` for exactly
    that reason. Rounded to float32 on the way in, like the hull, because
    `mjModel.mesh_vert` is `float*` and a double copy puts our surface a few
    hundred picometres from the one the reference intersects.

    Sized by `nmesh_tri`, which is 0 unless something asked for it, so a model
    nobody rays pays nothing.

    ⚠⚠ IT IS AN ARENA OF TWO RECORD KINDS, NOT AN ARRAY OF TRIANGLES. Nine
    floats per record; the triangles occupy records `[0, ntri)` and the
    per-mesh BVHs the records after them, addressed by
    `MESH_META_IDX_BVHADR`/`BVHNUM`. `MESH_ARENA_FLOATS_PER_TRI` is 27 rather
    than 9 for that reason — a one-triangle-per-leaf tree over n triangles has
    exactly 2n-1 nodes, so the arena is a FIXED multiple of the triangle
    budget and `nmesh_tri` still means triangles. See `gpu/constants.mojo` for
    the node layout and for why this is one tensor rather than two.
    """
    var mesh_polys: TensorImpl[Self.DTYPE]  # [NMESH_POLY, 5]
    var mesh_polyvert: TensorImpl[Self.DTYPE]  # [NMESH_POLYVERT]
    var mesh_polymap: TensorImpl[Self.DTYPE]  # [NMESH_POLYVERT]
    var mesh_vert_polymap: TensorImpl[Self.DTYPE]  # [NMESH_VERTS, 2] adr,num
    # Hull vertex adjacency for `mjc_PlaneConvex`'s graph path — see
    # `collision/convex_hull.mojo::build_hull_edge_graph`. `mesh_vert_edgeadr`
    # is indexed by GLOBAL hull vertex (parallel to `mesh_verts`) and points
    # into `mesh_edges`, whose per-vertex runs are -1 terminated.
    var mesh_vert_edgeadr: TensorImpl[Self.DTYPE]  # [NMESH_VERTS]
    var mesh_edges: TensorImpl[Self.DTYPE]  # [NMESH_EDGE_SLOTS(NMESH_VERTS)]

    # The provider as a VALUE (3a) — see the same field on `Data` for why a
    # dispatcher cannot synthesize one and why storing it costs the static
    # leg nothing.
    var dims: Self.D

    def __init__(out self) raises:
        """Dimensions from the comptime provider; raises on a dynamic one.
        See `DimsLike.comptime_value`."""
        self = Self(Self.D.comptime_value())

    def __init__(out self, dims: Self.D) raises:
        """Dimensions passed in, and ALLOCATED FROM (3b).

        ⚠ Every size below reads `dims`, never a comptime member. Those
        members still exist and still size the GPU layouts, but they are
        `DIM_POISON` on a dynamic provider, so an `alloc` that read one
        would ask for a NEGATIVE length. See the twin on `Data`."""
        self.dims = dims
        self.bodies = TensorImpl[Self.DTYPE].alloc(
            dims.get_nbody() * MODEL_BODY_SIZE
        )
        self.joints = TensorImpl[Self.DTYPE].alloc(
            dims.get_njoint() * MODEL_JOINT_SIZE
        )
        self.meta = TensorImpl[Self.DTYPE].alloc(MODEL_META_SIZE)
        # ⚠ MUJOCO'S CCD DEFAULTS, SEEDED HERE BECAUSE A ZERO IS A LEGAL-LOOKING
        # VALUE FOR BOTH. `alloc` does not promise zeroed memory and a zero
        # tolerance / zero iteration count would silently mean "iterate to the
        # array cap on every pair" and "never iterate" respectively — neither of
        # which resembles the reference. Every hand-built Model (the GPU env
        # specs, test fixtures) gets MuJoCo's behaviour without knowing this
        # slot exists; `fields_build` overwrites both from `<option>`.
        self.meta.data[MODEL_META_IDX_CCD_TOLERANCE] = Scalar[Self.DTYPE](
            MJ_CCD_TOLERANCE
        )
        self.meta.data[MODEL_META_IDX_CCD_ITERATIONS] = Scalar[Self.DTYPE](
            MJ_CCD_ITERATIONS
        )
        # ⚠⚠ `alloc` DOES NOT ZERO, AND `iterations` IS A LOOP BOUND. A hand-
        # built `Data` — the board, the GPU env specs, every fixture that skips
        # the parser — would otherwise solve an arbitrary number of times, or
        # zero. Seeded with MuJoCo's defaults so a `Model` built without a
        # parser behaves like the reference. Same reasoning as
        # `MODEL_META_IDX_NOSLIP_ITERATIONS` below.
        self.meta.data[
            MODEL_META_IDX_SOLVER_ITERATIONS
        ] = Scalar[Self.DTYPE](MJ_SOLVER_ITERATIONS)
        self.meta.data[
            MODEL_META_IDX_SOLVER_TOLERANCE
        ] = Scalar[Self.DTYPE](MJ_SOLVER_TOLERANCE)
        self.meta.data[MODEL_META_IDX_LS_ITERATIONS] = Scalar[Self.DTYPE](
            MJ_LS_ITERATIONS
        )
        self.meta.data[MODEL_META_IDX_LS_TOLERANCE] = Scalar[Self.DTYPE](
            MJ_LS_TOLERANCE
        )
        # ⚠ SEEDED FOR THE SAME REASON, AND THE SAFE VALUE HERE IS 0. `alloc`
        # does not zero, and this slot is a LOOP BOUND — `mj_solNoSlip` runs
        # `opt.noslip_iterations` friction-only sweeps — so uninitialized
        # memory would mean an arbitrary number of them on any hand-built
        # Model whose caller enabled the pass. 0 is MuJoCo's default and means
        # "no pass at all"; `fields_build` overwrites it from `<option>`.
        #
        # ⚠ THE COMPTIME `NOSLIP_ITER` CANNOT BE READ FROM HERE — `Model` is
        # parameterized on `DimsLike`, not on the model def — which is exactly
        # why the count travels in meta rather than only as a parameter.
        self.meta.data[MODEL_META_IDX_NOSLIP_ITERATIONS] = Scalar[Self.DTYPE](
            0
        )
        self.curriculum = TensorImpl[Self.DTYPE].alloc(MODEL_CURRICULUM_SIZE)
        self.geoms = TensorImpl[Self.DTYPE].alloc(_at_least_one(dims.get_ngeom() * MODEL_GEOM_SIZE))
        self.equality = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_nequality() * MODEL_EQ_SIZE)
        )
        self.tendons = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_ntendon() * MODEL_TENDON_SIZE)
        )
        self.sites = TensorImpl[Self.DTYPE].alloc(_at_least_one(dims.get_nsite() * MODEL_SITE_SIZE))
        self.sensors = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_nsensor() * MODEL_SENSOR_SIZE)
        )
        self.body_invweight0 = TensorImpl[Self.DTYPE].alloc(dims.get_nbody() * 2)
        self.dof_invweight0 = TensorImpl[Self.DTYPE].alloc(dims.get_nv())
        self.dof_M0 = TensorImpl[Self.DTYPE].alloc(dims.get_nv())
        # `_at_least_one`: an nv=0 model still needs a bindable operand.
        self.trees = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_nv()) * MODEL_TREE_SIZE
        )
        # ⚠ ZEROED, AND THE ZERO IS THE SAFE VALUE. `alloc` does not promise
        # zeroed memory, and an unbuilt table read as garbage `(adr, num)` is
        # an out-of-range block, not an empty one. A hand-built `Model` that
        # skips the parser gets `ntree = 0` and the whole-nv fallback that
        # `MODEL_META_IDX_NTREE`'s default implies.
        for i in range(_at_least_one(dims.get_nv()) * MODEL_TREE_SIZE):
            self.trees.data[i] = Scalar[Self.DTYPE](0)
        self.dof_parentid = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_nv())
        )
        for i in range(_at_least_one(dims.get_nv())):
            self.dof_parentid.data[i] = Scalar[Self.DTYPE](-1)
        self.dof_actdamp = TensorImpl[Self.DTYPE].alloc(dims.get_nv())
        self.actdamp_trn = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_nact() * ACTDAMP_TRN_SIZE)
        )
        self.excludes = TensorImpl[Self.DTYPE].alloc(_at_least_one(dims.get_nexclude() * 2))
        self.pairs = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_npair() * MODEL_PAIR_SIZE)
        )
        self.mesh_meta = TensorImpl[Self.DTYPE].alloc(
            MAX_GPU_MESHES * MODEL_MESH_META_SIZE
        )
        # HEIGHTFIELDS. The META table is capped (448 bytes) and the grid is
        # not — see `MAX_GPU_HFIELDS`. A model with no hfield allocates one
        # float, like every other `_at_least_one` here.
        self.hfield_data0 = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_nhfield_data())
        )
        self.hfield_meta = TensorImpl[Self.DTYPE].alloc(
            MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE
        )
        # CAMERAS. A capped table like `hfield_meta`, and for the same
        # reason — 1.5 KB at float64, so every model pays for it and no model
        # needs a dimension for it. Zeroed, so an unfilled row reads
        # `CAM_IDX_ACTIVE == 0`.
        self.cameras = TensorImpl[Self.DTYPE].alloc(
            MAX_GPU_CAMERAS * MODEL_CAM_SIZE
        )
        for _c in range(MAX_GPU_CAMERAS * MODEL_CAM_SIZE):
            self.cameras.data[_c] = Scalar[Self.DTYPE](0)
        self.geom_rgba = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_ngeom() * MODEL_GEOM_RGBA_SIZE)
        )

        self.mesh_verts = TensorImpl[Self.DTYPE].alloc(_at_least_one(dims.get_nmesh_verts() * 3))
        self.mesh_tris = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_nmesh_tri() * MESH_ARENA_FLOATS_PER_TRI)
        )
        self.mesh_polys = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(mesh_max_poly(dims.get_nmesh_verts()) * MODEL_MESH_POLY_SIZE)
        )
        self.mesh_polyvert = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(mesh_max_polyvert(dims.get_nmesh_verts()))
        )
        self.mesh_polymap = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(mesh_max_polyvert(dims.get_nmesh_verts()))
        )
        self.mesh_vert_polymap = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_nmesh_verts() * 2)
        )
        self.mesh_vert_edgeadr = TensorImpl[Self.DTYPE].alloc(
            _at_least_one(dims.get_nmesh_verts())
        )
        # ⚠⚠ -1 MEANS "NO GRAPH", AND ZERO WOULD NOT. Zero is a valid offset
        # into `mesh_edges`, so a vertex whose adjacency was never built would
        # silently borrow the FIRST mesh's neighbour list and collide against
        # unrelated vertices. That is not hypothetical: it is what
        # `test_plane_mesh_fields` does — it appends a synthetic tetrahedron
        # straight into the packed tensors, past everything `fields_build`
        # wrote — and it turned 1 contact into 3. `_plane_mesh_contacts` reads
        # -1 as MuJoCo reads `mesh_graphadr < 0` and takes the exhaustive
        # branch instead.
        for v in range(_at_least_one(dims.get_nmesh_verts())):
            self.mesh_vert_edgeadr.data[v] = Scalar[Self.DTYPE](-1)
        self.mesh_edges = TensorImpl[Self.DTYPE].alloc(mesh_max_edge(dims.get_nmesh_verts()))
        # Pre-filled with the terminator, not zero: a neighbour walk reads
        # until it sees -1, and zero is a VALID vertex index. The real guard
        # against an unbuilt graph is the -1 in `mesh_vert_edgeadr` above; this
        # is belt and braces for a run that is entered anyway.
        for k in range(mesh_max_edge(dims.get_nmesh_verts())):
            self.mesh_edges.data[k] = Scalar[Self.DTYPE](-1)

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Host -> device for every record tensor (static config: called once
        after setup, like the model slab upload today)."""
        self.bodies.upload(ctx)
        self.joints.upload(ctx)
        self.meta.upload(ctx)
        self.curriculum.upload(ctx)
        self.geoms.upload(ctx)
        self.equality.upload(ctx)
        self.tendons.upload(ctx)
        self.sites.upload(ctx)
        self.sensors.upload(ctx)
        self.body_invweight0.upload(ctx)
        self.dof_invweight0.upload(ctx)
        self.dof_M0.upload(ctx)
        self.trees.upload(ctx)
        self.dof_parentid.upload(ctx)
        self.dof_actdamp.upload(ctx)
        self.actdamp_trn.upload(ctx)
        self.excludes.upload(ctx)
        self.pairs.upload(ctx)
        self.mesh_meta.upload(ctx)
        self.hfield_data0.upload(ctx)
        self.hfield_meta.upload(ctx)
        self.cameras.upload(ctx)
        self.geom_rgba.upload(ctx)
        self.mesh_verts.upload(ctx)
        self.mesh_tris.upload(ctx)
        self.mesh_polys.upload(ctx)
        self.mesh_polyvert.upload(ctx)
        self.mesh_polymap.upload(ctx)
        self.mesh_vert_polymap.upload(ctx)
        self.mesh_vert_edgeadr.upload(ctx)
        self.mesh_edges.upload(ctx)

    # `load_from_model` (CPU `Model` -> record fill) was deleted at the G4
    # fields sunset — the spec-direct build is
    # `fields_build.build_model_fields_from_flat`.


def init_hfield_data[
    DTYPE: DType, D: DimsLike, BATCH: Int
](mut d: Data[DTYPE, D, BATCH], m: Model[DTYPE, D]):
    """Copy `Model.hfield_data0` into every lane of `Data.hfield_data`.

    The heightfield's `qpos0 -> qpos`. Called once after both exist; a task
    that rewrites its terrain per episode (`quadruped escape`) overwrites the
    lane it owns afterwards.

    ⚠ NOT CALLING IT LEAVES A FLAT TERRAIN, silently. A grid of zeros is a
    perfectly valid heightfield — it collides, it rays, it just is not the
    surface the model declared. There is no error to raise, which is why this
    is a named step rather than something a caller might reasonably forget.
    """
    var n = d.dims.get_nhfield_data()
    if n < 1:
        return
    for e in range(BATCH):
        for k in range(n):
            d.hfield_data.data[e * n + k] = m.hfield_data0.data[k]

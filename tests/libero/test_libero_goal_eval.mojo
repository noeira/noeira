"""LIBERO-Goal's TEN goals, evaluated on the composed scene — L3's gate.

    pixi run mojo run -I . tests/libero/test_libero_goal_eval.mojo              # host + kernel loop (f64)
    pixi run -e apple mojo run -I . tests/libero/test_libero_goal_eval.mojo     # + the device leg (f32)

## WHAT THIS ASSERTS

1. **every one of the ten `.task` files binds** against the composed
   `libero_goal` scene with the L3 language: `Joint` through the joint
   table, `On(obj, obj)` to `OP_ON_BODY`, box regions with their contact
   slot; `require_tier_a` and `require_gpu_regions` accept all ten and
   `encode_goal` fits each on the twelve-word tape.
2. **the three readers agree on every state**: `eval.eval_goal` (host,
   `BoundGoal`), `tape.eval_tape` (host, twelve floats) and
   `gpu_eval.eval_tape_gpu` (the KERNEL loop, instantiated on CPU tensors at
   float64) — and `tape_distance_gpu` is zero exactly when the goal holds.
3. **every goal SHAPE flips**: the states below are built so that each of
   `On(obj, zone)`, `On(obj, obj)`, `On(obj, fixture site)`, `In(obj, site)`
   and `Joint` holds in at least one state and fails in at least one. Three
   readers agreeing on all-False is what three `return False`s achieve.
4. with an accelerator, **the device leg**: the same ten tapes in ten
   lanes, evaluated by `eval_tape_gpu` INSIDE a kernel at float32, agree
   with the host on every state (`N states x 10 lanes, 0 mismatches`).

⚠ THE STATES ARE SIMULATED, NOT CONSTRUCTED. `On(obj, obj)` and
`On(obj, fixture site)` need a CONTACT (LIBERO's `check_ontop` and
`SiteObjectState.check_ontop`), which only the solver can produce; so every
"holds" state below drops the prop from a few centimetres and settles it
under gravity with the real elliptic-cone Newton step, arm held at
`base_qpos`. A state that fails to settle where it should is a test
failure, not a skipped check — the flip counts assert it.

⚠ THE WINE RACK IS REPORTED, NOT ASSERTED. Its site is tilted 62 degrees
and LIBERO's `under` multiplies by `site_xmat` where a frame change would
use the transpose (`eval.mojo` header); where the bottle has to rest for
that line to hold is a question about the benchmark's own geometry, and
the two rack tasks are the same SHAPE as the stove and cabinet-top ones,
which are asserted.
"""

from std.os.path import exists
from std.sys import has_accelerator
from max.gpu import thread_idx, block_idx, block_dim
from layout import TileTensor
from noeira.core.tiles import RowMajor
from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_NUM_CONTACTS,
    MODEL_CURRICULUM_SIZE, MODEL_SITE_SIZE, MODEL_BODY_SIZE,
    CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B,
)
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.studio.stepping import StudioIntegEll
from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family, FamilySpec,
)
from noeira.tasks.family import scene_path
from noeira.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, joint_qpos_addresses, joint_id,
    slot_body_id, BoundGoal, OP_ON_BODY, OP_JOINT, OP_IN, OP_ON,
)
from noeira.tasks.eval import (
    eval_goal, HostState, region_sites, region_rects, region_half_heights,
    region_box_flags, region_contact_bodies,
)
from noeira.tasks.tape import encode_goal, eval_tape, TAPE_WORDS
from noeira.tasks.gpu_eval import (
    eval_tape_gpu, tape_distance_gpu, region_table_words, require_gpu_regions,
)
from noeira.envs.libero.models.libero_goal_dims import LIBERO_GOAL_DIMS
from noeira.envs.libero.models.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS


comptime FAMILY = "noeira/envs/libero/families/libero_goal.family"
comptime TASK_DIR = "noeira/envs/libero/tasks/libero_goal__"
comptime PACK = "noeira/envs/libero/assets"

comptime DT = DType.float64
comptime F32 = DType.float32
comptime N_TASKS = 10
comptime NB = LIBERO_GOAL_DIMS.NBODY
comptime NS = LIBERO_GOAL_DIMS.NSITE
comptime NQ = LIBERO_GOAL_DIMS.NQ
comptime MC = LIBERO_GOAL_MAX_CONTACTS

comptime SETTLE_STEPS = 400   # 0.8 s at the family's 2 ms timestep
comptime DROP = 0.04          # metres above the resting surface
comptime TABLE_Z = 0.9        # the workspace site's z

comptime L_META = RowMajor[1, METADATA_SIZE]
comptime L_CUR = RowMajor[1, MODEL_CURRICULUM_SIZE]
comptime L_XP = RowMajor[1, NB * 3]
comptime L_XQ = RowMajor[1, NB * 4]
comptime L_SP = RowMajor[1, NS * 3]
comptime L_QP = RowMajor[1, NQ]
comptime L_SITES = RowMajor[NS, MODEL_SITE_SIZE]
comptime L_BODIES = RowMajor[NB, MODEL_BODY_SIZE]
comptime L_CON = RowMajor[1, MC * CONTACT_SIZE]

# the device leg: ten lanes, one task each
comptime L_META32 = RowMajor[N_TASKS, METADATA_SIZE]
comptime L_CUR32 = RowMajor[1, MODEL_CURRICULUM_SIZE]
comptime L_XP32 = RowMajor[N_TASKS, NB * 3]
comptime L_XQ32 = RowMajor[N_TASKS, NB * 4]
comptime L_SP32 = RowMajor[N_TASKS, NS * 3]
comptime L_QP32 = RowMajor[N_TASKS, NQ]
comptime L_CON32 = RowMajor[N_TASKS, MC * CONTACT_SIZE]
comptime L_OUT32 = RowMajor[N_TASKS, 2]


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def _task_stems() -> List[String]:
    var out = List[String]()
    out.append(String("open_the_middle_drawer_of_the_cabinet"))
    out.append(String("open_the_top_drawer_and_put_the_bowl_inside"))
    out.append(String("push_the_plate_to_the_front_of_the_stove"))
    out.append(String("put_the_bowl_on_the_plate"))
    out.append(String("put_the_bowl_on_the_stove"))
    out.append(String("put_the_bowl_on_top_of_the_cabinet"))
    out.append(String("put_the_cream_cheese_in_the_bowl"))
    out.append(String("put_the_wine_bottle_on_the_rack"))
    out.append(String("put_the_wine_bottle_on_top_of_the_cabinet"))
    out.append(String("turn_on_the_stove"))
    return out^


# ── the device kernel: one lane, one tape ──────────────────────────────────


def _lane_kernel(
    meta: TileTensor[F32, L_META32, MutAnyOrigin],
    cur: TileTensor[F32, L_CUR32, MutAnyOrigin],
    xpos: TileTensor[F32, L_XP32, MutAnyOrigin],
    xquat: TileTensor[F32, L_XQ32, MutAnyOrigin],
    sxp: TileTensor[F32, L_SP32, MutAnyOrigin],
    qpos: TileTensor[F32, L_QP32, MutAnyOrigin],
    sites: TileTensor[F32, L_SITES, MutAnyOrigin],
    bodies: TileTensor[F32, L_BODIES, MutAnyOrigin],
    contacts: TileTensor[F32, L_CON32, MutAnyOrigin],
    res: TileTensor[F32, L_OUT32, MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= N_TASKS:
        return
    var h = eval_tape_gpu[F32, N_TASKS, NB, NS * 3, NQ, NS, MC](
        meta, cur, xpos, xquat, sxp, qpos, sites, bodies, contacts, env
    )
    var dd = tape_distance_gpu[F32, N_TASKS, NB, NS * 3, NQ, NS, MC](
        meta, cur, xpos, xquat, sxp, qpos, sites, bodies, contacts, env
    )
    res[env, 0] = Scalar[F32](1) if h else Scalar[F32](0)
    res[env, 1] = dd


# ── scene helpers: the arm held, props placed, the solver settling ─────────


def _hold_arm(mut d: Data[DT, DynDims, 1], base_qpos: List[Float64]):
    """The Panda has motor actuators and no controller here; pin it."""
    for i in range(len(base_qpos)):
        d.qpos.data[i] = Scalar[DT](base_qpos[i])
        d.qvel.data[i] = Scalar[DT](0)


def _place(mut d: Data[DT, DynDims, 1], adr: Int, x: Float64, y: Float64, z: Float64, nv: Int):
    """A free joint at (x, y, z), identity orientation; everything at rest."""
    d.qpos.data[adr] = Scalar[DT](x)
    d.qpos.data[adr + 1] = Scalar[DT](y)
    d.qpos.data[adr + 2] = Scalar[DT](z)
    # ⚠ qpos quaternion is W-FIRST (test_task_reset_steps asserts it)
    d.qpos.data[adr + 3] = Scalar[DT](1)
    d.qpos.data[adr + 4] = Scalar[DT](0)
    d.qpos.data[adr + 5] = Scalar[DT](0)
    d.qpos.data[adr + 6] = Scalar[DT](0)
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)


def _settle(
    mut integ: StudioIntegEll, mut d: Data[DT, DynDims, 1],
    mut m: Model[DT, DynDims], base_qpos: List[Float64], steps: Int,
) raises:
    for _ in range(steps):
        integ.step["cpu"](d, m)
        _hold_arm(d, base_qpos)
    # one more FK so xpos/site_xpos describe the state we read
    forward_kinematics["cpu", DT, DynDims, 1](d, m)


def _body_pos(d: Data[DT, DynDims, 1], b: Int) -> List[Float64]:
    var out = List[Float64]()
    for k in range(3):
        out.append(Float64(d.xpos.data[b * 3 + k]))
    return out^


def _site_pos(d: Data[DT, DynDims, 1], name: String, site_names: List[String]) raises -> List[Float64]:
    var s = -1
    for i in range(len(site_names)):
        if String(site_names[i]) == name:
            s = i
    if s < 0:
        raise Error("no site " + name)
    var out = List[Float64]()
    for k in range(3):
        out.append(Float64(d.site_xpos.data[s * 3 + k]))
    return out^


def _ncon(d: Data[DT, DynDims, 1]) -> Int:
    var n = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    return MC if n > MC else n


def _snapshot(
    d: Data[DT, DynDims, 1], nb: Int, ns: Int, nq: Int,
    site_body: List[Int], site_quat: List[Float64], body_parent: List[Int],
) -> HostState:
    var xp = List[Float64]()
    for i in range(nb * 3):
        xp.append(Float64(d.xpos.data[i]))
    var xq = List[Float64]()
    for i in range(nb * 4):
        xq.append(Float64(d.xquat.data[i]))
    var sp = List[Float64]()
    for i in range(ns * 3):
        sp.append(Float64(d.site_xpos.data[i]))
    var st = HostState(xp^, xq^, sp^)
    for i in range(nq):
        st.qpos.append(Float64(d.qpos.data[i]))
    st.site_body = site_body.copy()
    st.site_quat = site_quat.copy()
    st.body_parent = body_parent.copy()
    var n = _ncon(d)
    st.ncon = n
    for k in range(n):
        var base = k * CONTACT_SIZE
        st.con_a.append(Int(d.contacts.data[base + CONTACT_IDX_BODY_A]))
        st.con_b.append(Int(d.contacts.data[base + CONTACT_IDX_BODY_B]))
    return st^


def _zone_centre(f: FamilySpec, region: String) raises -> List[Float64]:
    """World (x, y) of a region's rect centre (the anchors sit at x=y=0)."""
    var ri = f.region_index(region)
    if ri < 0:
        raise Error("no region " + region)
    ref r = f.regions[ri]
    var out = List[Float64]()
    out.append(0.5 * (r.x_min + r.x_max))
    out.append(0.5 * (r.y_min + r.y_max))
    return out^


def _rest_state(
    mut integ: StudioIntegEll, mut d: Data[DT, DynDims, 1],
    mut m: Model[DT, DynDims], base_qpos: List[Float64], nq: Int, nv: Int,
    adrs: List[Int], rest_xy: List[List[Float64]],
) raises:
    """Every prop at its init rect centre, dropped and settled."""
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    _hold_arm(d, base_qpos)
    for i in range(len(adrs)):
        _place(d, adrs[i], rest_xy[i][0], rest_xy[i][1], TABLE_Z + DROP, nv)
    _settle(integ, d, m, base_qpos, SETTLE_STEPS)


def main() raises:
    print("=== LIBERO-Goal's ten goals on the composed scene — L3 ===")
    if not exists(PACK):
        print("  SKIPPED: no LIBERO pack at", PACK,
              "— run `pixi run assets-pull libero`")
        print("=== SKIPPED (no pack — this is not a pass) ===")
        return
    var ta = Tally()

    var f = load_family(FAMILY)
    var fmd = parse_model_runtime(scene_path(f))
    if len(fmd.body_names) != NB or len(fmd.site_names) != NS:
        raise Error(
            "libero_goal eval: the composed scene has "
            + String(len(fmd.body_names)) + " bodies / "
            + String(len(fmd.site_names)) + " sites but the dims say "
            + String(NB) + " / " + String(NS) + ". Run `pixi run"
            " gen-family-scenes && pixi run gen-dims`."
        )
    var nqs = List[Int]()
    for i in range(len(fmd.joints)):
        nqs.append(fmd.joints[i].nq)
    var jadr = joint_qpos_addresses(nqs)
    var site_body = List[Int]()
    var site_quat = List[Float64]()
    for i in range(len(fmd.sites)):
        site_body.append(fmd.sites[i].body_id)
        site_quat.append(fmd.sites[i].quat_x)
        site_quat.append(fmd.sites[i].quat_y)
        site_quat.append(fmd.sites[i].quat_z)
        site_quat.append(fmd.sites[i].quat_w)
    var body_parent = List[Int]()
    body_parent.append(-1)
    for i in range(len(fmd.bodies)):
        body_parent.append(fmd.bodies[i].parent)

    var rsites = region_sites(f, fmd.site_names)
    var rcontact = region_contact_bodies(f, fmd.body_names)
    var rects = region_rects(f)
    var rh = region_half_heights(f)
    var rbox = region_box_flags(f)
    var r_x0 = List[Float64]()
    var r_y0 = List[Float64]()
    var r_x1 = List[Float64]()
    var r_y1 = List[Float64]()
    var n_box = 0
    for i in range(len(f.regions)):
        r_x0.append(rects[i][0])
        r_y0.append(rects[i][1])
        r_x1.append(rects[i][2])
        r_y1.append(rects[i][3])
        n_box += rbox[i]
    var cw = region_table_words(f, rsites, rcontact)
    print("  family:", f.name, "| regions:", len(f.regions), "| box regions:", n_box)

    # ── 1. every task binds, is Tier A, fits the table and the tape ──────
    print("--- the ten tasks bind ---")
    var stems = _task_stems()
    var goal_texts = List[String]()
    var tapes = List[List[Float64]]()
    var n_on_body = 0
    var n_joint = 0
    var n_box_in = 0
    var n_box_on = 0
    var n_zone_on = 0
    for i in range(len(stems)):
        var t = load_task(TASK_DIR + stems[i] + ".task")
        validate_task_against_family(t, f)
        var g = bind_goal(
            parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
            fmd.joint_names, jadr,
        )
        require_tier_a(g, t.name)
        require_gpu_regions(g, t.name)
        var tp = encode_goal(g)
        for k in range(len(g.terms)):
            ref bt = g.terms[k]
            if bt.op == OP_ON_BODY:
                n_on_body += 1
            elif bt.op == OP_JOINT:
                n_joint += 1
            elif bt.op == OP_IN and rbox[bt.b] != 0:
                n_box_in += 1
            elif bt.op == OP_ON and rbox[bt.b] != 0:
                if rcontact[bt.b] >= 0:
                    n_box_on += 1
                else:
                    n_zone_on += 1
        print("   ", stems[i], "->", t.goal)
        goal_texts.append(t.goal)
        tapes.append(tp^)
    ta.check(len(goal_texts) == N_TASKS, "all ten tasks bind, Tier A, on the tape")
    ta.check(n_on_body == 2, "two On(obj, obj) -> OP_ON_BODY (bowl/plate, cheese/bowl)")
    ta.check(n_joint == 2, "two Joint terms (middle drawer, stove button)")
    ta.check(n_box_in == 1, "one In(obj, box site) (bowl in the top drawer)")
    ta.check(n_box_on == 4, "four On(obj, fixture site) with a contact slot")
    ta.check(n_zone_on == 1, "one On(obj, table zone), no contact")

    # ── 2. the scene ──────────────────────────────────────────────────────
    print("--- building the scene (Newton, elliptic cone, arm held) ---")
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=MC, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=MC, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var integ = StudioIntegEll(dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nb = dims.get_nbody()
    var ns = dims.get_nsite()
    var base_qpos = f.base_qpos.copy()
    print("  nq", nq, " nv", nv, " nbody", nb, " nsite", ns, " mesh verts", verts)
    if nb != NB or ns != NS or nq != NQ:
        raise Error("dims disagree with LIBERO_GOAL_DIMS")

    # the id conventions the evaluator relies on
    var cab = slot_body_id(String("wooden_cabinet_1"), fmd.body_names)
    var top_body = -1
    for b in range(len(fmd.body_names)):
        if String(fmd.body_names[b]) == "wooden_cabinet_1_cabinet_top":
            top_body = b
    ta.check(top_body > cab, "a fixture's child body has a larger id than its root")
    var walk = top_body
    while walk > cab:
        walk = body_parent[walk]
    ta.check(walk == cab, "the parent walk from the drawer reaches the cabinet root")

    # ── 3. the states ─────────────────────────────────────────────────────
    var slots = List[String]()
    slots.append(String("akita_black_bowl_1"))
    slots.append(String("cream_cheese_1"))
    slots.append(String("wine_bottle_1"))
    slots.append(String("plate_1"))
    var adrs = List[Int]()
    var rest_xy = List[List[Float64]]()
    for i in range(len(slots)):
        adrs.append(jadr[joint_id(slots[i] + "_joint0", fmd.joint_names)])
        var stem = String(slots[i][byte = 0 : slots[i].byte_length() - 2])
        rest_xy.append(_zone_centre(f, String("main_table_") + stem + "_region"))
    var bowl_b = slot_body_id(String("akita_black_bowl_1"), fmd.body_names)
    var plate_b = slot_body_id(String("plate_1"), fmd.body_names)
    var cheese_b = slot_body_id(String("cream_cheese_1"), fmd.body_names)
    var a_bowl = adrs[0]
    var a_cheese = adrs[1]
    var a_bottle = adrs[2]
    var a_plate = adrs[3]
    var j_top = jadr[joint_id(String("wooden_cabinet_1_top_level"), fmd.joint_names)]
    var j_mid = jadr[joint_id(String("wooden_cabinet_1_middle_level"), fmd.joint_names)]
    var j_btn = jadr[joint_id(String("flat_stove_1_button"), fmd.joint_names)]

    var state_names = List[String]()
    var states = List[HostState]()

    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    state_names.append(String("rest (every prop settled at its init centre)"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))
    print("  rest: contacts", _ncon(d), " bowl z", _body_pos(d, bowl_b)[2],
          " plate z", _body_pos(d, plate_b)[2])

    # plate pushed to the front of the stove
    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    var zc = _zone_centre(f, String("main_table_stove_front_region_zone"))
    _place(d, a_plate, zc[0], zc[1], TABLE_Z + DROP, nv)
    _settle(integ, d, m, base_qpos, SETTLE_STEPS)
    state_names.append(String("plate in the stove-front zone"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))

    # bowl on the plate
    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    var pp = _body_pos(d, plate_b)
    _place(d, a_bowl, pp[0], pp[1], pp[2] + 0.02 + DROP, nv)
    _settle(integ, d, m, base_qpos, SETTLE_STEPS)
    state_names.append(String("bowl dropped onto the plate"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))
    var bp = _body_pos(d, bowl_b)
    pp = _body_pos(d, plate_b)
    print("  bowl/plate: bowl z", bp[2], " plate z", pp[2], " dxy",
          bp[0] - pp[0], bp[1] - pp[1], " contacts", _ncon(d))

    # bowl on the stove's burner
    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    var cook = _site_pos(d, String("flat_stove_1_cook_region"), fmd.site_names)
    _place(d, a_bowl, cook[0], cook[1], cook[2] + 0.03 + DROP, nv)
    _settle(integ, d, m, base_qpos, SETTLE_STEPS)
    state_names.append(String("bowl dropped onto the burner"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))
    bp = _body_pos(d, bowl_b)
    print("  bowl/stove: bowl z", bp[2], " cook site z", cook[2], " contacts", _ncon(d))

    # bowl on top of the cabinet
    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    var top = _site_pos(d, String("wooden_cabinet_1_top_side"), fmd.site_names)
    _place(d, a_bowl, top[0], top[1], top[2] + 0.03 + DROP, nv)
    _settle(integ, d, m, base_qpos, SETTLE_STEPS)
    state_names.append(String("bowl dropped onto the cabinet top"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))
    bp = _body_pos(d, bowl_b)
    print("  bowl/cabinet: bowl z", bp[2], " top_side z", top[2], " contacts", _ncon(d))

    # wine bottle on top of the cabinet
    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    _place(d, a_bottle, top[0], top[1], top[2] + 0.12 + DROP, nv)
    _settle(integ, d, m, base_qpos, SETTLE_STEPS)
    state_names.append(String("wine bottle dropped onto the cabinet top"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))

    # cream cheese into the bowl
    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    bp = _body_pos(d, bowl_b)
    _place(d, a_cheese, bp[0], bp[1], bp[2] + 0.06 + DROP, nv)
    _settle(integ, d, m, base_qpos, SETTLE_STEPS)
    state_names.append(String("cream cheese dropped into the bowl"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))
    var cp = _body_pos(d, cheese_b)
    bp = _body_pos(d, bowl_b)
    print("  cheese/bowl: cheese z", cp[2], " bowl z", bp[2], " dxy",
          cp[0] - bp[0], cp[1] - bp[1], " contacts", _ncon(d))

    # the middle drawer open
    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    d.qpos.data[j_mid] = Scalar[DT](-0.15)
    _settle(integ, d, m, base_qpos, 10)
    state_names.append(String("middle drawer open (qpos -0.15)"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))
    print("  middle_level qpos after settle:", Float64(d.qpos.data[j_mid]))

    # the top drawer open with the bowl inside it
    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    d.qpos.data[j_top] = Scalar[DT](-0.15)
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    var drawer = _site_pos(d, String("wooden_cabinet_1_top_region"), fmd.site_names)
    _place(d, a_bowl, drawer[0], drawer[1], drawer[2], nv)
    _settle(integ, d, m, base_qpos, 5)
    state_names.append(String("top drawer open, bowl teleported into its box"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))
    bp = _body_pos(d, bowl_b)
    drawer = _site_pos(d, String("wooden_cabinet_1_top_region"), fmd.site_names)
    print("  bowl/drawer: bowl", bp[0], bp[1], bp[2], " drawer site", drawer[0],
          drawer[1], drawer[2], " top_level qpos", Float64(d.qpos.data[j_top]))

    # the stove button turned
    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    d.qpos.data[j_btn] = Scalar[DT](0.6)
    _settle(integ, d, m, base_qpos, 10)
    state_names.append(String("stove button at 0.6 rad"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))
    print("  button qpos after settle:", Float64(d.qpos.data[j_btn]))

    # the wine bottle laid on the rack's site (reported, not asserted)
    _rest_state(integ, d, m, base_qpos, nq, nv, adrs, rest_xy)
    var rack = _site_pos(d, String("wine_rack_1_top_region"), fmd.site_names)
    _place(d, a_bottle, rack[0], rack[1], rack[2] + 0.02 + DROP, nv)
    _settle(integ, d, m, base_qpos, SETTLE_STEPS)
    state_names.append(String("wine bottle dropped onto the rack's site"))
    states.append(_snapshot(d, nb, ns, nq, site_body, site_quat, body_parent))

    # ── 4. three readers, every state, every task ─────────────────────────
    print("--- host goal == host tape == kernel loop, on", len(states), "states ---")
    var meta = TensorImpl[DT].alloc(METADATA_SIZE)
    var cur = TensorImpl[DT].alloc(MODEL_CURRICULUM_SIZE)
    for i in range(MODEL_CURRICULUM_SIZE):
        cur.data[i] = Scalar[DT](cw[i])
    var t_xp = TensorImpl[DT].alloc(NB * 3)
    var t_xq = TensorImpl[DT].alloc(NB * 4)
    var t_sp = TensorImpl[DT].alloc(NS * 3)
    var t_qp = TensorImpl[DT].alloc(NQ)
    var t_con = TensorImpl[DT].alloc(MC * CONTACT_SIZE)
    var t_sites = TensorImpl[DT].alloc(NS * MODEL_SITE_SIZE)
    var t_bodies = TensorImpl[DT].alloc(NB * MODEL_BODY_SIZE)
    for i in range(NS * MODEL_SITE_SIZE):
        t_sites.data[i] = m.sites.data[i]
    for i in range(NB * MODEL_BODY_SIZE):
        t_bodies.data[i] = m.bodies.data[i]

    var holds = List[List[Bool]]()   # [state][task]
    var agree = 0
    var total = 0
    var dist_ok = 0
    for si in range(len(states)):
        ref st = states[si]
        for i in range(NB * 3):
            t_xp.data[i] = Scalar[DT](st.xpos[i])
        for i in range(NB * 4):
            t_xq.data[i] = Scalar[DT](st.xquat[i])
        for i in range(NS * 3):
            t_sp.data[i] = Scalar[DT](st.site_xpos[i])
        for i in range(NQ):
            t_qp.data[i] = Scalar[DT](st.qpos[i])
        for i in range(MC * CONTACT_SIZE):
            t_con.data[i] = Scalar[DT](0)
        for k in range(st.ncon):
            t_con.data[k * CONTACT_SIZE + CONTACT_IDX_BODY_A] = Scalar[DT](st.con_a[k])
            t_con.data[k * CONTACT_SIZE + CONTACT_IDX_BODY_B] = Scalar[DT](st.con_b[k])
        for i in range(METADATA_SIZE):
            meta.data[i] = Scalar[DT](0)
        meta.data[META_IDX_NUM_CONTACTS] = Scalar[DT](st.ncon)

        var row = List[Bool]()
        var line = String("")
        for ti in range(N_TASKS):
            var g = bind_goal(
                parse_goal(goal_texts[ti]), f, fmd.body_names, fmd.site_names,
                fmd.joint_names, jadr,
            )
            var h = eval_goal(g, f, st, rsites, rcontact)
            var tp = eval_tape(
                tapes[ti], 0, st.xpos, st.xquat, st.site_xpos,
                rsites, r_x0, r_y0, r_x1, r_y1, rh, rbox, rcontact,
                st.qpos, st.site_body, st.site_quat, st.body_parent,
                st.ncon, st.con_a, st.con_b,
            )
            for k in range(TAPE_WORDS):
                meta.data[META_IDX_TASK_PARAM_0 + k] = Scalar[DT](tapes[ti][k])
            var kl = eval_tape_gpu[DT, 1, NB, NS * 3, NQ, NS, MC](
                meta.tt["cpu", 1, METADATA_SIZE](), cur.tt["cpu", 1, MODEL_CURRICULUM_SIZE](),
                t_xp.tt["cpu", 1, NB * 3](), t_xq.tt["cpu", 1, NB * 4](),
                t_sp.tt["cpu", 1, NS * 3](), t_qp.tt["cpu", 1, NQ](),
                t_sites.tt["cpu", NS, MODEL_SITE_SIZE](), t_bodies.tt["cpu", NB, MODEL_BODY_SIZE](),
                t_con.tt["cpu", 1, MC * CONTACT_SIZE](), 0,
            )
            var dd = tape_distance_gpu[DT, 1, NB, NS * 3, NQ, NS, MC](
                meta.tt["cpu", 1, METADATA_SIZE](), cur.tt["cpu", 1, MODEL_CURRICULUM_SIZE](),
                t_xp.tt["cpu", 1, NB * 3](), t_xq.tt["cpu", 1, NB * 4](),
                t_sp.tt["cpu", 1, NS * 3](), t_qp.tt["cpu", 1, NQ](),
                t_sites.tt["cpu", NS, MODEL_SITE_SIZE](), t_bodies.tt["cpu", NB, MODEL_BODY_SIZE](),
                t_con.tt["cpu", 1, MC * CONTACT_SIZE](), 0,
            )
            total += 1
            if h == tp and tp == kl:
                agree += 1
            if (dd == 0.0) == kl:
                dist_ok += 1
            row.append(h)
            line += "1" if h else "."
        print("   ", line, " ", state_names[si], "(contacts", st.ncon, ")")
        holds.append(row^)
    print("  ", agree, "of", total, "evaluations agree across the three readers")
    ta.check(agree == total, "host goal == host tape == kernel loop on EVERY (state, task)")
    ta.check(dist_ok == total, "tape_distance_gpu is zero EXACTLY when the goal holds")

    # ── 5. every shape flips ──────────────────────────────────────────────
    print("--- flips per task (column = task index above) ---")
    var must_flip = List[Int]()
    must_flip.append(0)   # Joint (middle drawer)
    must_flip.append(1)   # In(bowl, top drawer box)
    must_flip.append(2)   # On(plate, zone)
    must_flip.append(3)   # On(bowl, plate)  — OP_ON_BODY
    must_flip.append(4)   # On(bowl, cook_region) — box + contact
    must_flip.append(5)   # On(bowl, cabinet top_side) — box + contact
    must_flip.append(6)   # On(cheese, bowl) — OP_ON_BODY
    must_flip.append(8)   # On(bottle, cabinet top_side)
    must_flip.append(9)   # Joint (stove button)
    for k in range(len(must_flip)):
        var ti = must_flip[k]
        var n_true = 0
        for si in range(len(states)):
            if holds[si][ti]:
                n_true += 1
        ta.check(
            n_true > 0 and n_true < len(states),
            stems[ti] + " holds in " + String(n_true) + " of "
            + String(len(states)) + " states (flips)",
        )
    var rack_true = 0
    for si in range(len(states)):
        if holds[si][7]:
            rack_true += 1
    print("  (reported) put_the_wine_bottle_on_the_rack holds in", rack_true,
          "of", len(states), "states")
    var rest_any = False
    for ti in range(N_TASKS):
        if holds[0][ti]:
            rest_any = True
    ta.check(not rest_any, "no goal holds at rest (every prop at its init centre)")

    # ── 6. the device leg ─────────────────────────────────────────────────
    comptime if has_accelerator():
        print("--- device leg: ten lanes, one task each, float32 ---")
        var ctx = DeviceContext()
        var g_meta = TensorImpl[F32].alloc(N_TASKS * METADATA_SIZE)
        var g_cur = TensorImpl[F32].alloc(MODEL_CURRICULUM_SIZE)
        var g_xp = TensorImpl[F32].alloc(N_TASKS * NB * 3)
        var g_xq = TensorImpl[F32].alloc(N_TASKS * NB * 4)
        var g_sp = TensorImpl[F32].alloc(N_TASKS * NS * 3)
        var g_qp = TensorImpl[F32].alloc(N_TASKS * NQ)
        var g_con = TensorImpl[F32].alloc(N_TASKS * MC * CONTACT_SIZE)
        var g_sites = TensorImpl[F32].alloc(NS * MODEL_SITE_SIZE)
        var g_bodies = TensorImpl[F32].alloc(NB * MODEL_BODY_SIZE)
        var g_out = TensorImpl[F32].alloc(N_TASKS * 2)
        for i in range(MODEL_CURRICULUM_SIZE):
            g_cur.data[i] = Scalar[F32](cw[i])
        for i in range(NS * MODEL_SITE_SIZE):
            g_sites.data[i] = Scalar[F32](Float64(m.sites.data[i]))
        for i in range(NB * MODEL_BODY_SIZE):
            g_bodies.data[i] = Scalar[F32](Float64(m.bodies.data[i]))
        g_cur.upload(ctx)
        g_sites.upload(ctx)
        g_bodies.upload(ctx)
        var dev_agree = 0
        var dev_total = 0
        var dev_dist_ok = 0
        for si in range(len(states)):
            ref st = states[si]
            for e in range(N_TASKS):
                for i in range(NB * 3):
                    g_xp.data[e * NB * 3 + i] = Scalar[F32](st.xpos[i])
                for i in range(NB * 4):
                    g_xq.data[e * NB * 4 + i] = Scalar[F32](st.xquat[i])
                for i in range(NS * 3):
                    g_sp.data[e * NS * 3 + i] = Scalar[F32](st.site_xpos[i])
                for i in range(NQ):
                    g_qp.data[e * NQ + i] = Scalar[F32](st.qpos[i])
                for i in range(MC * CONTACT_SIZE):
                    g_con.data[e * MC * CONTACT_SIZE + i] = Scalar[F32](0)
                for k in range(st.ncon):
                    g_con.data[e * MC * CONTACT_SIZE + k * CONTACT_SIZE + CONTACT_IDX_BODY_A] = Scalar[F32](st.con_a[k])
                    g_con.data[e * MC * CONTACT_SIZE + k * CONTACT_SIZE + CONTACT_IDX_BODY_B] = Scalar[F32](st.con_b[k])
                for i in range(METADATA_SIZE):
                    g_meta.data[e * METADATA_SIZE + i] = Scalar[F32](0)
                g_meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS] = Scalar[F32](st.ncon)
                for k in range(TAPE_WORDS):
                    g_meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + k] = Scalar[F32](tapes[e][k])
            for i in range(N_TASKS * 2):
                g_out.data[i] = Scalar[F32](-1)
            g_meta.upload(ctx)
            g_xp.upload(ctx)
            g_xq.upload(ctx)
            g_sp.upload(ctx)
            g_qp.upload(ctx)
            g_con.upload(ctx)
            g_out.upload(ctx)
            ctx.enqueue_function[_lane_kernel](
                g_meta.tt["gpu", N_TASKS, METADATA_SIZE](), g_cur.tt["gpu", 1, MODEL_CURRICULUM_SIZE](),
                g_xp.tt["gpu", N_TASKS, NB * 3](), g_xq.tt["gpu", N_TASKS, NB * 4](),
                g_sp.tt["gpu", N_TASKS, NS * 3](), g_qp.tt["gpu", N_TASKS, NQ](),
                g_sites.tt["gpu", NS, MODEL_SITE_SIZE](), g_bodies.tt["gpu", NB, MODEL_BODY_SIZE](),
                g_con.tt["gpu", N_TASKS, MC * CONTACT_SIZE](), g_out.tt["gpu", N_TASKS, 2](),
                grid_dim=(1,), block_dim=(N_TASKS,),
            )
            g_out.download(ctx)
            ctx.synchronize()
            var line = String("")
            for e in range(N_TASKS):
                var dh = Float64(g_out.data[e * 2]) > 0.5
                var ddist = Float64(g_out.data[e * 2 + 1])
                dev_total += 1
                if dh == holds[si][e]:
                    dev_agree += 1
                if (ddist == 0.0) == dh:
                    dev_dist_ok += 1
                line += "1" if dh else "."
            print("   ", line, " ", state_names[si])
        print("  ", dev_agree, "of", dev_total, "device (float32) evaluations agree with the host")
        ta.check(dev_agree == dev_total, "device leg == host on EVERY (state, lane)")
        ta.check(dev_dist_ok == dev_total, "device distance is zero EXACTLY when the lane's goal holds")
    else:
        print("--- device leg: SKIPPED (no accelerator; run with -e apple / -e nvidia) ---")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "libero_goal eval: " + String(ta.failures) + " of "
            + String(ta.checks) + " failed"
        )
    print("=== PASS ===")

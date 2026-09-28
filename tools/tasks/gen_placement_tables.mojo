"""Every generatable family's device placement table — GENERATED artifacts.

    pixi run gen-placement-tables           # write <root>/placement/*.mojo (noeira/tasks, noeira/envs/libero)
    pixi run gen-placement-tables --check   # CI: fail if one is stale

`placement/table.place_free_slots` is the device twin of
`sampler.sample_placements`, and it reads a family's geometry through a
comptime `PlacementTable` because a kernel cannot read a `.family`. This writes
one per `.family` whose free slots ALL carry `slot_geom=` (the 23 LIBERO
families and `so101_tower`), from the SAME inputs the host sampler uses: the loaded
spec, `reset.free_slot_addresses`, and the region sites' world frames from
forward kinematics on the composed scene.

⚠ EVERY FLOAT IS `Scalar[DTYPE](<String(Float64)>)`. That decimal is measured
to round-trip to the same bits (see `_f`), so the table matches the frames it
was generated from exactly and `check.placement_table_drift` can demand it; and
it is a `DTYPE` constant in the kernel, because Metal has no `double`.

⚠ A FREE SLOT WITHOUT `slot_geom=` CANNOT BE GENERATED. The host would use the
caller's radius for it, which is not in the `.family`, so there is nothing to
generate from — `so101_tabletop`, the one family like that, has a hand-written
table (`family_config.So101TabletopPlacement`) and is SKIPPED here by that
rule, not by name. A family that gains `slot_geom=` on every free slot joins
this generator on the next run, and its hand-written table becomes dead code.

⚠ THE GRIPPER SITE IS PER BASE ROBOT: `robot_grip_site` on the vendored Panda,
`robot_gripperframe` on the SO-101. The first one the composed scene has wins;
a scene with neither is refused, because the goal words need an origin.

⚠ `region_move_joint` IS THE SITE'S BODY CHAIN, not a list. No joint above the
site: -1. Exactly one joint and it is a drawable SLIDE: that joint, with the world
axis measured by FK. Anything else: -2, which `check.require_device_placement`
refuses, because the kernel runs before FK.
"""

from std.os import listdir
from std.sys import argv

from noeira.tasks.spec import load_family, task_root_of, FamilySpec, SLOT_FREE
from noeira.tasks.family import scene_path, park_pos
from noeira.tasks.reset import free_slot_addresses
from noeira.tasks.eval import region_sites
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics

comptime DT = DType.float64
comptime TASK_ROOTS = "noeira/tasks,noeira/envs/libero"
"""Every task root; a family's table is written to `<its root>/placement/`."""
comptime GRIPPER_SITE_NAMES = "robot_grip_site,robot_grasp_center,robot_gripperframe"
"""The end-effector site, per base robot, first match wins: the vendored
Panda's (the one OSC_POSE drives, `examples/libero/libero_eval.mojo`), the
tower follower's PINCH CENTRE (`grasp_center`, the bake's step 4c — where a
held brick sits, 3 cm above the tip) and the stock SO-101's `gripperframe`
(the jaw TIP, the tabletop family). The goal words' origin, and the reach
term's."""


def _generatable(f: FamilySpec) -> Bool:
    """Every FREE slot carries `slot_geom=` — the one thing the table needs
    that the host sampler otherwise takes from its caller."""
    var any_free = False
    for si in range(len(f.slots)):
        if f.slots[si].kind != SLOT_FREE:
            continue
        any_free = True
        if not f.slots[si].has_geom:
            return False
    return any_free


def _families() raises -> List[String]:
    """Every generatable `.family`, sorted — enumerated like `gen_family_scene`.

    ⚠ BY RULE, NOT BY PREFIX. It was `libero*`; `so101_tower` is the first
    non-LIBERO family with a full `slot_geom=` table, and a prefix list would
    have left it hand-written."""
    var out = List[String]()
    for root in String(TASK_ROOTS).split(","):
        var dir = String(root) + "/families"
        for e in listdir(dir):
            var n = String(e)
            if not n.endswith(".family"):
                continue
            var f = load_family(dir + "/" + n)
            if _generatable(f):
                out.append(dir + "/" + n)
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    return out^


def struct_name(family: String) -> String:
    """`libero_kitchen_scene10` -> `LiberoKitchenScene10Placement`."""
    var out = String("")
    var up = True
    for cp in family.codepoint_slices():
        var c = String(cp)
        if c == "_":
            up = True
            continue
        out += c.upper() if up else c
        up = False
    return out + "Placement"


def _f(v: Float64) -> String:
    """`Scalar[DTYPE](<v>)`, with `v` as Mojo prints it.

    ⚠ MEASURED EXACT BOTH WAYS: `String(Float64)` round-trips on 4000 values
    (random bit patterns included) and the compiler turns those literals back
    into the same bits on 3701. So the decimal IS the bit pattern — and unlike
    a `bitcast` it folds to a `DTYPE` constant, which Metal needs (no double).
    A zero is written `0.0`: the sign of a zero rect bound changes no draw."""
    if v == 0.0:
        return "Scalar[DTYPE](0.0)"
    return "Scalar[DTYPE](" + String(v) + ")"


def _b(v: Bool) -> String:
    return "True" if v else "False"


def _method(
    name: String, arg: String, ret: String, values: List[String]
) -> String:
    """A static method returning `values[arg]` as an `if` chain.

    ⚠ AN `if` CHAIN, NOT A TABLE LITERAL, because this body runs inside a GPU
    kernel: a chain of constant returns lowers on every backend, and a
    comptime aggregate indexed at runtime is one more thing a metallib could
    refuse. Collapsed to one `return` when every entry is the same."""
    var s = "    @staticmethod\n    def " + name + "(" + arg + ": Int) -> " + ret + ":\n"
    if ret == "Float64":
        s = (
            "    @staticmethod\n    def " + name + "[DTYPE: DType](" + arg
            + ": Int) -> Scalar[DTYPE]:\n"
        )
    var same = True
    for i in range(1, len(values)):
        if values[i] != values[0]:
            same = False
    if len(values) == 0:
        # a family with no regions still has to satisfy the trait
        var zero = String("Scalar[DTYPE](0.0)") if ret == "Float64" else (
            String("False") if ret == "Bool" else (
                String('String("")') if ret == "String" else String("0")
            )
        )
        return s + "        return " + zero + "\n\n"
    if same:
        return s + "        return " + values[0] + "\n\n"
    for i in range(len(values) - 1):
        s += "        if " + arg + " == " + String(i) + ":\n"
        s += "            return " + values[i] + "\n"
    s += "        return " + values[len(values) - 1] + "\n\n"
    return s


def family_name(path: String) -> String:
    """`<root>/families/<name>.family` -> `<name>`."""
    var parts = path.split("/")
    var base = String(parts[len(parts) - 1])
    return String(base[byte = 0 : base.byte_length() - 7])


def generate(family_path: String) raises -> String:
    var f = load_family(family_path)
    var family = f.name
    var fmd = parse_model_runtime(scene_path(f))
    # ⚠ `fmd.bodies` HAS NO WORLDBODY RECORD: model body id `bi` is
    # `fmd.bodies[bi - 1]` (`fields_build`), while `body_names`, a site's
    # `body_id`, a joint's `body_id` and a body's `parent` all count the
    # worldbody as 0. Checked, because an off-by-one here would walk a
    # NEIGHBOUR's chain and mark the wrong regions as moving.
    if len(fmd.bodies) + 1 != len(fmd.body_names):
        raise Error(
            family + ": " + String(len(fmd.bodies)) + " body records against "
            + String(len(fmd.body_names)) + " names — expected one more name"
            " (the worldbody)"
        )
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    # ⚠ THE HOST'S OWN RESET STATE BEFORE ITS FK: zeros, then `base_qpos`
    # (`test_libero_scenes`). Only a site under a joint could see the
    # difference, and those are marked `moves` and refused.
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    forward_kinematics["cpu", DT, DynDims, 1](d, m)

    var jt = List[Int]()
    var jqn = List[Int]()
    var jvn = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jqn.append(fmd.joints[i].nq)
        jvn.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jqn, jvn)
    var rsites = region_sites(f, fmd.site_names)

    # ── the drawable joints: every hinge/slide of a STATIC slot, scene order ──
    # ⚠ BY NAME PREFIX `<slot>_`, the composer's own spelling (`wooden_cabinet_1
    # _top_level`). The underscore keeps `white_cabinet_1` from claiming
    # `white_cabinet_10`'s joints.
    var jnames = List[String]()
    var jqadr = List[Int]()
    var jdadr = List[Int]()
    var jidx = List[Int]()
    var qa_run = 0
    var da_run = 0
    for i in range(len(fmd.joints)):
        var owned = False
        for si in range(len(f.slots)):
            if f.slots[si].kind == SLOT_FREE:
                continue
            if String(fmd.joint_names[i]).startswith(f.slots[si].name + "_"):
                owned = True
        if owned and (jt[i] == JNT_HINGE or jt[i] == JNT_SLIDE) and jqn[i] == 1:
            jnames.append(String(fmd.joint_names[i]))
            jqadr.append(qa_run)
            jdadr.append(da_run)
            jidx.append(i)
        qa_run += jqn[i]
        da_run += jvn[i]

    var fslot = List[String]()
    var fqadr = List[String]()
    var fdadr = List[String]()
    var fgeom = List[String]()
    var frest = List[String]()
    var frad = List[String]()
    var fbot = List[String]()
    var ftop = List[String]()
    var fpx = List[String]()
    var fpy = List[String]()
    var fpz = List[String]()
    for si in range(len(f.slots)):
        ref s = f.slots[si]
        if s.kind != SLOT_FREE:
            continue
        if not s.has_geom:
            raise Error(
                family + ": free slot '" + s.name + "' has no slot_geom=. The"
                " host sampler would use the CALLER's radius for it, which the"
                " .family does not hold; write this family's table by hand."
            )
        fslot.append(String(si) + "  # " + s.name)
        fqadr.append(String(addrs[si].qadr))
        fdadr.append(String(addrs[si].dadr))
        fgeom.append(_b(True))
        frest.append(_f(-s.bottom_z))
        frad.append(_f(s.h_radius))
        fbot.append(_f(s.bottom_z))
        ftop.append(_f(s.top_z))
        var pp = park_pos(f, si)
        fpx.append(_f(pp[0]))
        fpy.append(_f(pp[1]))
        fpz.append(_f(pp[2]))

    var grip = -1
    var grip_name = String("")
    for cand in String(GRIPPER_SITE_NAMES).split(","):
        if grip >= 0:
            break
        for i in range(len(fmd.site_names)):
            if fmd.site_names[i] == String(cand):
                grip = i
                grip_name = String(cand)
    if grip < 0:
        raise Error(
            family + ": the composed scene has none of the sites '"
            + String(GRIPPER_SITE_NAMES) + "' — the goal words' origin. Is"
            " the base robot one this generator knows?"
        )
    var rsid = List[String]()
    for r in range(len(f.regions)):
        rsid.append(String(rsites[r]))
    var rx_raw = List[Float64]()
    for r in range(len(f.regions)):
        for c in range(3):
            rx_raw.append(Float64(d.site_xpos.data[rsites[r] * 3 + c]))
    var rx = List[String]()
    var ry = List[String]()
    var rz = List[String]()
    var rrect = List[String]()
    var rx0 = List[String]()
    var ry0 = List[String]()
    var rx1 = List[String]()
    var ry1 = List[String]()
    var ranch = List[String]()
    var rcg = List[String]()
    var rctop = List[String]()
    var rmove = List[String]()
    var rax = List[String]()
    var ray = List[String]()
    var raz = List[String]()
    var n_moves = 0
    var n_followed = 0
    for r in range(len(f.regions)):
        ref reg = f.regions[r]
        var sid = rsites[r]
        rx.append(_f(Float64(d.site_xpos.data[sid * 3])))
        ry.append(_f(Float64(d.site_xpos.data[sid * 3 + 1])))
        rz.append(_f(Float64(d.site_xpos.data[sid * 3 + 2])))
        rrect.append(_b(reg.has_rect) + "  # " + reg.name)
        rx0.append(_f(reg.x_min))
        ry0.append(_f(reg.y_min))
        rx1.append(_f(reg.x_max))
        ry1.append(_f(reg.y_max))
        var anchored = reg.contact.byte_length() > 0
        ranch.append(_b(anchored))
        var cg = False
        var ctop = 0.0
        if anchored:
            var ci = f.slot_index(reg.contact)
            if ci >= 0 and f.slots[ci].has_geom:
                cg = True
                ctop = f.slots[ci].top_z
        rcg.append(_b(cg))
        rctop.append(_f(ctop))
        # every joint above the site, by the body chain
        var chain = List[Int]()
        var b = fmd.sites[sid].body_id
        var guard = 0
        while b > 0:
            for k in range(len(fmd.joints)):
                if fmd.joints[k].body_id == b:
                    chain.append(k)
            b = fmd.bodies[b - 1].parent
            guard += 1
            if guard > len(fmd.bodies):
                raise Error(family + ": a cycle in the body parent chain")
        var carried = -1
        var ax = 0.0
        var ay = 0.0
        var az = 0.0
        if len(chain) > 0:
            n_moves += 1
            carried = -2
            if len(chain) == 1 and jt[chain[0]] == JNT_SLIDE:
                for q in range(len(jidx)):
                    if jidx[q] == chain[0]:
                        carried = q
            if carried >= 0:
                # ⚠ THE AXIS IS MEASURED, FK at q = 1 minus FK at q = 0, not
                # read off `<joint axis>` — a slide's world axis is the parent
                # chain's rotation applied to it, and FK already composes that.
                n_followed += 1
                var adr = jqadr[carried]
                d.qpos.data[adr] = Scalar[DT](1)
                forward_kinematics["cpu", DT, DynDims, 1](d, m)
                ax = Float64(d.site_xpos.data[sid * 3]) - rx_raw[r * 3]
                ay = Float64(d.site_xpos.data[sid * 3 + 1]) - rx_raw[r * 3 + 1]
                az = Float64(d.site_xpos.data[sid * 3 + 2]) - rx_raw[r * 3 + 2]
                d.qpos.data[adr] = Scalar[DT](0)
                forward_kinematics["cpu", DT, DynDims, 1](d, m)
        rmove.append(String(carried))
        rax.append(_f(ax))
        ray.append(_f(ay))
        raz.append(_f(az))

    var name = struct_name(family)
    var o = String("")
    o += '"""`' + family + "`'s device placement table — GENERATED, DO NOT EDIT.\n\n"
    o += "Regenerate with:  pixi run gen-placement-tables\n"
    o += "CI checks it with: pixi run gen-placement-tables --check\n\n"
    o += "From `" + family_path + "`,\n"
    o += "`" + scene_path(f) + "` and forward kinematics on it.\n"
    o += String(len(fslot)) + " free slots, " + String(len(f.regions))
    o += " regions (" + String(n_moves) + " moving, " + String(n_followed)
    o += " followed on one slide), " + String(len(jnames)) + " drawable joints.\n"
    o += "See `placement/table.mojo` for what each method means.\n"
    o += '"""\n\n'
    o += "from noeira.tasks.placement.table import PlacementTable\n\n\n"
    o += "struct " + name + "(PlacementTable):\n"
    o += "    comptime N_SLOTS: Int = " + String(len(f.slots)) + "\n"
    o += "    comptime N_FREE: Int = " + String(len(fslot)) + "\n"
    o += "    comptime N_REGIONS: Int = " + String(len(f.regions)) + "\n"
    o += "    comptime NQ: Int = " + String(nq) + "\n"
    o += "    comptime NV: Int = " + String(nv) + "\n"
    o += "    comptime N_JOINTS: Int = " + String(len(jnames)) + "\n"
    o += "    comptime NBODY: Int = " + String(dims.get_nbody()) + "\n"
    o += "    comptime NSITE: Int = " + String(dims.get_nsite()) + "\n"
    o += "    comptime GRIPPER_SITE: Int = " + String(grip) + "  # "
    o += grip_name + "\n"
    o += "    comptime N_BASE_QPOS: Int = " + String(len(f.base_qpos)) + "\n\n"
    var bq = List[String]()
    for i in range(len(f.base_qpos)):
        bq.append(_f(f.base_qpos[i]))
    o += _method("base_qpos", "i", "Float64", bq)
    # only a family that declares it: every other table stays byte-identical
    # and takes the trait's default of 0
    if len(f.base_qpos_jitter) > 0:
        var bj = List[String]()
        for i in range(len(f.base_qpos_jitter)):
            bj.append(_f(f.base_qpos_jitter[i]))
        o += _method("base_qpos_jitter", "i", "Float64", bj)
    o += _method("free_slot", "j", "Int", fslot)
    o += _method("free_qadr", "j", "Int", fqadr)
    o += _method("free_dadr", "j", "Int", fdadr)
    o += _method("free_has_geom", "j", "Bool", fgeom)
    o += _method("free_rest", "j", "Float64", frest)
    o += _method("free_radius", "j", "Float64", frad)
    o += _method("free_park_x", "j", "Float64", fpx)
    o += _method("free_park_y", "j", "Float64", fpy)
    o += _method("free_park_z", "j", "Float64", fpz)
    o += _method("free_bottom_z", "j", "Float64", fbot)
    o += _method("free_top_z", "j", "Float64", ftop)
    o += _method("region_site", "r", "Int", rsid)
    o += _method("region_site_x", "r", "Float64", rx)
    o += _method("region_site_y", "r", "Float64", ry)
    o += _method("region_site_z", "r", "Float64", rz)
    o += _method("region_has_rect", "r", "Bool", rrect)
    o += _method("region_x0", "r", "Float64", rx0)
    o += _method("region_y0", "r", "Float64", ry0)
    o += _method("region_x1", "r", "Float64", rx1)
    o += _method("region_y1", "r", "Float64", ry1)
    o += _method("region_anchored", "r", "Bool", ranch)
    o += _method("region_contact_has_geom", "r", "Bool", rcg)
    o += _method("region_contact_top_z", "r", "Float64", rctop)
    o += _method("region_move_joint", "r", "Int", rmove)
    o += _method("region_move_axis_x", "r", "Float64", rax)
    o += _method("region_move_axis_y", "r", "Float64", ray)
    o += _method("region_move_axis_z", "r", "Float64", raz)
    var jn = List[String]()
    var jq = List[String]()
    var jd = List[String]()
    for k in range(len(jnames)):
        jn.append('String("' + jnames[k] + '")')
        jq.append(String(jqadr[k]))
        jd.append(String(jdadr[k]))
    o += _method("joint_name", "k", "String", jn)
    o += _method("joint_qadr", "k", "Int", jq)
    o += _method("joint_dadr", "k", "Int", jd)
    # one trailing newline, not two
    return String(o[byte = 0 : o.byte_length() - 1])


def main() raises:
    var args = argv()
    var check = False
    for i in range(len(args)):
        if String(args[i]) == "--check":
            check = True
    var fams = _families()
    if len(fams) == 0:
        raise Error("no generatable .family under " + String(TASK_ROOTS))
    var stale = 0
    for i in range(len(fams)):
        var text = generate(fams[i])
        var path = (
            task_root_of(fams[i]) + "/placement/" + family_name(fams[i])
            + ".mojo"
        )
        var old = String("")
        var have = True
        try:
            with open(path, "r") as fh:
                old = fh.read()
        except:
            have = False
        if check:
            if not have or old != text:
                print("  STALE:", path)
                stale += 1
            else:
                print("  up to date:", path)
        else:
            with open(path, "w") as fh:
                fh.write(text)
            print("  wrote", path)
    if check and stale > 0:
        raise Error(
            String(stale) + " placement table(s) stale — run"
            " `pixi run gen-placement-tables`"
        )
    print(len(fams), "placement tables", "checked" if check else "written")

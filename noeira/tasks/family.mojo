"""A `.family` slot table -> a composed MJCF scene — P1c.

    var f = load_family("families/so101_tabletop.family")
    var xml = compose_family(f)          # MJCF text MuJoCo loads unchanged

## ⚠ THIS CALLS THE STUDIO COMPOSER. IT DOES NOT REIMPLEMENT ONE.

`TASK_LAYER_PLAN.md` §7: `tasks/` calls `physics3d/studio`'s composer, does not
know about MJCF text, and `physics3d` never imports `tasks`. Everything below
goes through `SceneDoc` / `Instance` / `scene_from_base` / `to_mjcf`, which is
also what keeps the payoff of §2.1 — **MuJoCo loads the composed document
unchanged, so `mjModel` is the oracle for the whole thing.** A second writer
here would forfeit that on its first line.

The one liberty taken is appending to `SceneDoc.instances` directly rather than
calling `place()`: `place` invents a unique prefix, and a family needs the
prefix to BE the slot name, because §2.1 records that the prefix is the
instance identity and the key for state remap. Same struct, same serialiser,
chosen name.

## ⚠⚠ EVERY SLOT IS INSTANTIATED IN EVERY TASK. THAT IS THE BUDGET.

§3.1: a family declares the scene once and every task carries every slot,
active or not, so `nbody`/`nq`/`nv`/`ngeom` are CONSTANT across the family and
the GPU leg stays one monomorphisation. A task varies which slots are active —
by POSE, at reset — never which exist. If a task needs an object the family did
not declare, that is a new family and a rebuild, and `spec.validate_task_against_family`
refuses it long before here.

## ⚠ WHY THE PARK POSE IS HIGH, LATERAL AND SPREAD

Measured against MuJoCo 3.10.0 on the real SO-ARM101 asset:

    park pose        contacts at t=0        after 1.2 s
    (0, 0, -2)       4, at dist = -2.02     EJECTED to z = +36.7
    (100, 0, 0.5)    0                      LANDED, resting at z = 0.02
    (10,  0, 10)     0                      z = 2.95, never reaches the plane

`TASK_LAYER_PLAN.md` §4.2 specifies the first row. A floor written
`size="0 0 ..."` is an INFINITE plane, so there is no "below" it — only a deep
penetration, and at four contacts per parked slot a 16-contact budget overflows
by the fourth slot. An overflowed budget DROPS contacts silently.

⚠ AND THEY ARE SPREAD, NOT STACKED. `k` slots at one pose interpenetrate each
other, which is `k*(k-1)/2` contact pairs from objects that are supposed to be
absent. `PARK_SPACING` is 25x a 2 cm prop's half-extent.
"""

from std.math import cos, sin
from noeira.physics3d.studio.scene import SceneDoc, Instance, scene_from_base
from noeira.physics3d.parser.expander import compiler_attr
from .spec import FamilySpec, SLOT_FREE, SLOT_STATIC


# Metres between adjacent parked slots. See the header: stacked slots collide
# with each other, which puts contacts into a scene whose parked half is
# supposed to be invisible.
comptime PARK_SPACING: Float64 = 0.5

# The base model's instance prefix. Fixed, because the prefix is the instance
# identity (§2.1) and a numbered one would move when a slot is added.
comptime BASE_PREFIX: String = "robot_"

# Where composed family scenes are generated. Project-root relative, and the
# directory every asset path is rewritten against.
comptime SCENE_DIR: String = "noeira/tasks/scenes"


def _relative_to(path: String, dir: String) -> String:
    """`path`, rewritten relative to `dir`. Both are project-root-relative.

    ⚠⚠ THIS EXISTS BECAUSE MuJoCo RESOLVES `<model file=>` AGAINST THE SCENE
    FILE'S OWN DIRECTORY, not against the process CWD — §10.5 decision 1, and
    the same rule `ModelDefFromXML.asset_base_dir` follows. A `.family`
    declares its assets from the project root because that is what a human can
    read and a manifest can pin; the COMPOSED scene has to spell them relative
    to wherever it is written.

    Getting this wrong is not subtle for long but it is confusing while it
    lasts: composing to `/tmp` produced
    `Failed to open file '/tmp/noeira/envs/robots/assets/so_arm101.xml'` —
    the project-root path glued onto the scene's directory. Caught by P1c's
    own gate on its first run.
    """
    if dir.byte_length() == 0:
        return path
    var pp = path.split("/")
    var dd = dir.split("/")
    # Drop the shared leading directories.
    var common = 0
    while common < len(pp) - 1 and common < len(dd):
        if String(pp[common]) != String(dd[common]):
            break
        common += 1
    var out = String("")
    for _ in range(len(dd) - common):
        out += "../"
    for i in range(common, len(pp)):
        out += String(pp[i])
        if i + 1 < len(pp):
            out += "/"
    return out^


def _whole_tag(xml: String, open_at: String) -> String:
    """The first `<open_at ... />` opening tag, verbatim, or "".

    ⚠ ONE TAG, COPIED WHOLE, NEVER PARSED. `<option>` has ~30 attributes and
    this must not become a second reader of them; the composed scene is
    loaded by MuJoCo and by our parser, and both read the tag themselves.
    """
    var i = xml.find(open_at)
    if i < 0:
        return String("")
    var j = xml.find(">", i)
    if j < 0:
        return String("")
    return String(xml[byte=i : j + 1])


def _tag_attr(xml: String, open_at: String, attr: String) -> String:
    """`attr="..."` from the first `<open_at ...>` tag, or ""."""
    var tag = _whole_tag(xml, open_at)
    var needle = String(" ") + attr + '="'
    var k = tag.find(needle)
    if k < 0:
        return String("")
    var start = k + needle.byte_length()
    var end = tag.find('"', start)
    if end < 0:
        return String("")
    return String(tag[byte=start:end])


def _asset_key(path: String) -> String:
    """`props/box_small.xml` -> `box_small`. The `<asset><model name=>` key.

    ⚠ KEYED ON THE FILE, NOT ON THE SLOT. Two slots sharing one asset —
    `cube_a` and `cube_b` both `props/cube.xml` — must produce ONE asset entry
    and TWO instances; that is the case `<attach prefix=>` exists for, and it
    is how §3.6's LIBERO-Object shape (declare the union, park the unused)
    stays affordable in the asset table.
    """
    var base = path
    var cut = path.rfind("/")
    if cut >= 0:
        base = String(path[byte = cut + 1 : path.byte_length()])
    var dot = base.rfind(".")
    if dot <= 0:
        return base^
    return String(base[byte=0:dot])


def park_pos(f: FamilySpec, slot_index: Int) -> List[Float64]:
    """Where slot `slot_index` sits when parked. `[x, y, z]`.

    Exposed because the RESET path (P2) has to write exactly this pose for an
    inactive slot, and a second copy of the arithmetic is how the scene and the
    reset drift apart — the shape `_a_rule_written_inline_twice_drifts` names.
    """
    var out = List[Float64]()
    out.append(f.park_x + Float64(slot_index) * PARK_SPACING)
    out.append(f.park_y)
    out.append(f.park_z)
    return out^


def compose_family(f: FamilySpec, scene_dir: String) raises -> String:
    """The family's slot table as MJCF. MuJoCo loads the result unchanged.

    `scene_dir` is the directory the result will be WRITTEN to, project-root
    relative — every asset path is rewritten relative to it, because MuJoCo
    resolves `<model file=>` against the scene file's own directory. Pass ""
    only when the scene is written at the project root.

    ⚠ SLOT ORDER IS THE OBSERVATION LAYOUT and the `qpos` layout, so the
    instances are emitted in declaration order and nothing here sorts them.
    """
    if len(f.slots) == 0:
        raise Error(
            "tasks: family '" + f.name + "' declares no slots. A family with"
            " no slots is a scene, and `physics3d/studio` already composes"
            " those — the family layer exists for the slot TABLE."
        )

    # ⚠ `scene_from_base("")` — THE FLOOR AND THE LIGHT, AND NOTHING ELSE.
    # Passing `f.base` here would work, but it routes the robot through
    # `place()`, which invents a prefix by NUMBERING (`so_arm1011_`). That
    # number depends on what else is in the scene, so the robot's identity
    # would shift when a slot is added — and §2.1 records that the prefix IS
    # the instance identity and the key for state remap. P3 addresses lanes by
    # prefix. So the base is attached explicitly, under a fixed name.
    #
    # Calling it with "" rather than writing the floor by hand is what keeps
    # this file free of MJCF text (§7).
    var d = scene_from_base(String(""), floor=f.floor)
    if f.sunlight >= 0.0:
        # `sunlight=` (see `FamilySpec.sunlight`): the floor light's diffuse
        var sun_old = String('<light pos="0 0 3" dir="0 0 -1" directional="true"/>')
        if d.base_world.find(sun_old) < 0:
            raise Error(
                "tasks: family '" + f.name + "' sets sunlight= but the composed"
                " floor has no default light to set (floor=0?)"
            )
        var g = String(f.sunlight)
        d.base_world = d.base_world.replace(
            sun_old,
            String('<light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="')
            + g + " " + g + " " + g + '"/>',
        )

    # ⚠⚠ THE HOST MUST RESTATE THE BASE'S ANGLE UNIT, AND OMITTING IT FROZE
    # THE ARM. MuJoCo's default for `<compiler angle>` is **degree**, so a
    # composed scene that declares no `<compiler>` reads every attached
    # `angle="radian"` asset in degrees. `so101_tabletop.xml` was written that
    # way and our parser gave `robot_shoulder_pan` a range of +-0.0335 rad
    # where MuJoCo 3.10 gives +-1.9199 — a factor of 57.3, i.e. the arm could
    # move +-1.9 DEGREES. Nothing caught it: `nq`, `nv`, `ngeom` and every
    # contact count were right, the scene rendered, the props settled, and no
    # gate had ever asked whether the arm could REACH anything.
    #
    # ⚠ AND THE EXPANDER'S GUARD CANNOT CATCH IT EITHER, BY CONSTRUCTION. It
    # refuses a sub-model whose angle differs from the host's, but only when
    # BOTH are present — an ABSENT host angle reads as "no opinion" there and
    # as "degree" in MuJoCo. Stating it here closes the case for composed
    # families; the general fix belongs in the guard.
    #
    # ⚠ READ FROM THE BASE, NOT HARDCODED. `radian` is right for this tree's
    # robots and wrong for an asset that ships in degrees; the composer's job
    # is to agree with what it attaches, which the expander then enforces for
    # every other slot.
    var base_xml_text: String
    with open(f.base, "r") as bf:
        base_xml_text = bf.read()
    # ⚠ THE COMPILER SETTINGS ARE GLOBAL ONCE SPLICED. `<attach>` compiles
    # the child under the PARENT's compiler, so whatever the base asset
    # relies on has to be restated here. `angle` always was; with
    # `inherit_option=1` L2 also carries `inertiagrouprange` and
    # `autolimits`, which robosuite's base.xml sets and LIBERO's objects
    # depend on (a visual `.msh` with `density=` would otherwise enter the
    # inertia).
    var header = String("")
    var ctag = String("  <compiler")
    var any_attr = False
    var base_angle = compiler_attr(base_xml_text, "angle")
    if base_angle.byte_length() > 0:
        ctag += ' angle="' + base_angle + '"'
        any_attr = True
    if f.inherit_option:
        # ⚠ GATED WITH THE OPTION, NOT ALWAYS: `so_arm101.xml` carries
        # `autolimits="true"` too, and copying it unconditionally rewrote
        # `scenes/so101_tabletop.xml` for no physical change (it is
        # MuJoCo's default). A family that predates L2 stays byte-identical.
        var igr = _tag_attr(base_xml_text, "<compiler", "inertiagrouprange")
        if igr.byte_length() > 0:
            ctag += ' inertiagrouprange="' + igr + '"'
            any_attr = True
        var al = _tag_attr(base_xml_text, "<compiler", "autolimits")
        if al.byte_length() > 0:
            ctag += ' autolimits="' + al + '"'
            any_attr = True
    if any_attr:
        header += ctag + "/>"
    if f.inherit_option:
        # ⚠ COPIED WHOLE, NEVER PARSED — see `_whole_tag`. MuJoCo ignores an
        # attached child's <option>, so a base authored with one runs under
        # the composed scene's defaults unless it is restated here.
        var opt = _whole_tag(base_xml_text, "<option")
        if opt.byte_length() == 0:
            raise Error(
                "tasks: family '" + f.name + "' sets inherit_option=1 but its"
                " base '" + f.base + "' has no <option> tag to inherit"
            )
        if header.byte_length() > 0:
            header += "\n"
        header += "  " + opt
    if len(f.headlight) == 3:
        # `headlight=` (see `FamilySpec.headlight`): the fill light a white
        # part seen edge-on needs. Grey levels, written as MuJoCo's rgb.
        var a = String(f.headlight[0])
        var dd = String(f.headlight[1])
        var sp = String(f.headlight[2])
        if header.byte_length() > 0:
            header += "\n"
        header += (
            '  <visual><headlight ambient="' + a + " " + a + " " + a
            + '" diffuse="' + dd + " " + dd + " " + dd + '" specular="' + sp
            + " " + sp + " " + sp + '"/></visual>'
        )
    d.base_xml = header

    var base_key = _asset_key(f.base)
    d.add_asset(base_key, _relative_to(f.base, scene_dir))
    d.instances.append(
        Instance(base_key, BASE_PREFIX, f.base_x, f.base_y, f.base_z)
    )

    for i in range(len(f.slots)):
        ref s = f.slots[i]
        var key = _asset_key(s.asset)

        # ⚠ `SceneDoc.add_asset` SILENTLY RETURNS on a duplicate NAME, which
        # is right for re-adding the same asset and wrong for two different
        # files whose stems collide (`a/cube.xml` and `b/cube.xml`). That
        # would attach the FIRST file twice and the second never — a slot
        # holding the wrong object, with nothing said. Checked here because
        # `add_asset` cannot tell the two cases apart.
        var seen = -1
        for j in range(len(d.asset_names)):
            if d.asset_names[j] == key:
                seen = j
        var rel = _relative_to(s.asset, scene_dir)
        if seen >= 0 and d.asset_files[seen] != rel:
            raise Error(
                "tasks: family '" + f.name + "' has two assets whose file"
                " stems collide on '" + key + "': '" + d.asset_files[seen]
                + "' and '" + s.asset + "'. Rename one file — the stem is the"
                " `<asset><model name=>` key and MJCF has no way to hold both."
            )
        d.add_asset(key, rel)

        # ⚠ THE PREFIX IS THE SLOT NAME. §2.1: the prefix IS the instance
        # identity and the key for state remap across a rebuild. `place()`
        # would invent `cube1_`; a family needs `brick_`, because that is what
        # a `.task` writes in `active=` and `goal=`.
        #
        # ⚠⚠ A STATIC SLOT IS COMPOSED WHERE IT LIVES; ONLY A FREE ONE IS
        # PARKED. Parking is BY POSE, and a static slot has no joint — so
        # composing a fixture at the park pose welds it there forever, and the
        # region attached to its site goes with it. `spec.SlotSpec` records
        # what that looked like the first time.
        var px = s.px
        var py = s.py
        var pz = s.pz
        if not s.has_pose:
            var p = park_pos(f, i)
            px = p[0]
            py = p[1]
            pz = p[2]
        var inst = Instance(key, s.name + "_", px, py, pz)
        if s.has_pose and s.yaw != 0.0:
            # A static slot's yaw, as the frame's quat about +z. `Instance`
            # already carries a unit quaternion; only the z-rotation is set,
            # which is all `slot=` can express.
            inst.qw = cos(0.5 * s.yaw)
            inst.qx = 0.0
            inst.qy = 0.0
            inst.qz = sin(0.5 * s.yaw)
        d.instances.append(inst^)

    return d.to_mjcf(f.name)


def scene_path(f: FamilySpec) -> String:
    """Where a family's composed scene is generated. ONE spelling of it.

    ⚠ GENERATED, NOT AUTHORED — the same status as
    `envs/robots/assets/so101_park_k*.xml`. It is what `gen_model_dims.py`
    reads and what a `ModelDefFromXML` points at, so it must be a stable path
    and not a temp file.
    """
    return scene_dir(f) + "/" + f.name + ".xml"


def scene_dir(f: FamilySpec) -> String:
    """`<root>/scenes` — `SCENE_DIR` for the task layer's own families."""
    return f.root + "/scenes"


def task_path(f: FamilySpec, task: String) -> String:
    """`<root>/tasks/<task>.task` — a task lives beside its family."""
    return f.root + "/tasks/" + task + ".task"


def write_family_scene(f: FamilySpec) raises -> String:
    """Compose and write to `scene_path(f)`; returns that path."""
    var out = scene_path(f)
    var xml = compose_family(f, scene_dir(f))
    with open(out, "w") as fh:
        fh.write(xml)
    return out^

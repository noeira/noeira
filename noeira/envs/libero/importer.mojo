"""BDDL -> `.family` + `.task`, AND A REFUSAL FOR EVERYTHING ELSE. P5.

    var f = translate_family(p)     # slots + regions
    var t = translate_task(p, f)    # language + goal + init + active

`bddl.mojo` reads LIBERO's format; this maps it onto ours. The interesting
half is what it will NOT map.

## ⚠⚠ IT REFUSES RATHER THAN APPROXIMATING, AND THAT IS THE WHOLE DESIGN

A goal this cannot express is a goal that must not be emitted. `spec.mojo`
already fixed this rule for our own files — *"a silently dropped `goal=` is a
task that always succeeds"* — and an importer is where the temptation to
approximate is strongest: `On(bowl, plate)` is ALMOST `On(bowl, plate_region)`,
and a policy trained against the almost is a policy evaluated on the wrong
benchmark.

`classify_goal` returns WHICH capability is missing, so the survey can count
gaps by kind instead of reporting one opaque failure total.

## THE GAPS, MEASURED ON ALL 130 FILES — AND WHAT L3 CLOSED

| gap | what LIBERO writes | status |
|---|---|---|
| `GAP_OBJECT_TARGET` | `In(bowl_1, plate_1)` — In whose container is an OBJECT | still refused: LIBERO's `ObjectState.check_contain` calls `object.in_box`, which no object class defines; the corpus never uses it. `On(bowl_1, plate_1)` is NOT this gap since L3: it binds to `OP_ON_BODY` = `check_ontop` |
| `GAP_FIXTURE_REGION` | `On(bowl_1, cabinet_1_top_region)` | CLOSED (L3): the region's rect is the `<site>`'s `size` in the fixture's asset, read here into a `:box:` region with the fixture as contact partner |
| `GAP_ARTICULATION` | `Open(cabinet_1_middle_region)`, `Turnon(stove_1)` | CLOSED (L3): `Joint(<fixture>_<joint>, <cmp>, <thr>)` from `categories.kv`'s per-class thresholds, `Or` over the joints for Open/Turnon (ANY), `And` for Close/Turnoff (ALL) — `ObjectState.is_open/is_close` quoted; refused only when the class has no threshold row |
| `GAP_UNKNOWN_PRED` | — | a predicate outside the measured seven |

## WHICH JOINTS `Open(x)` READS — quoted, because it is not obvious

`Open(<site region>)` → `SiteObjectState.is_open` → the joints LIBERO
attached to that site when it built `object_sites_dict`: it walks every
body of the fixture in document order, keeps `part.findall("./joint")` (the
body's DIRECT joints) for every part whose DESCENDANT sites include the
name, and the last match wins — i.e. the site's innermost enclosing body.
`Open(<fixture>)` → `ObjectState.is_open` → `MujocoXMLObject.joints`, every
joint in the asset. `site_joints` / `asset_joints` below are those two
rules on the asset text.

## TABLE REGIONS ARE TWO THINGS IN LIBERO, AND TWO REGIONS HERE

A `:regions` entry with `:ranges` is a PLACEMENT region (the init sampler,
`reference_pos = workspace_offset`) AND, when a goal names it, a
`TargetZone` SITE the problem class appends to the workspace body at its
own z convention (`zone_z=` in the table). The placement region keeps its
composed name (`main_table_stove_front_region`, plain `:site:` on the
workspace anchor — the sampler drops props at the anchor's z); the zone is
a `:box:` region named `<composed>_zone` on the arena's `zone_plane` site,
and only for regions a `:goal` names. `On(plate_1, main_table_stove_front_
region)` therefore translates to `On(plate_1, main_table_stove_front_region_zone)`.
"""

from .bddl import BddlProblem, BddlAtom, BddlRegion
from std.math import sqrt

from noeira.core.kv import split_on
from .categories import (
    LiberoTable, LiberoCategory, LiberoProblem, Threshold, JointRange, cmp_name,
)
from noeira.tasks.spec import (
    FamilySpec, TaskSpec, SlotSpec, RegionSpec, InitSpec, JointInitSpec,
    order_inits,
    SLOT_FREE, SLOT_STATIC, SCHEMA_VERSION,
)
from noeira.tasks.predicates import parse_goal
from noeira.tasks.tape import MAX_TAPE_TERMS


# `TargetZone.zone_height` — the half-z of every table target zone's site.
comptime TARGET_ZONE_HALF_HEIGHT: Float64 = 0.007
comptime ZONE_SUFFIX: String = "_zone"
comptime ZONE_SITE: String = "zone_plane"


comptime GAP_NONE: Int = 0
comptime GAP_OBJECT_TARGET: Int = 1
comptime GAP_FIXTURE_REGION: Int = 2
comptime GAP_ARTICULATION: Int = 3
comptime GAP_UNKNOWN_PRED: Int = 4
comptime GAP_ARITY: Int = 5
comptime GAP_TAPE_TERMS: Int = 6


def gap_name(kind: Int) -> String:
    if kind == GAP_NONE:
        return String("expressible")
    if kind == GAP_OBJECT_TARGET:
        return String(
            "In whose container is an OBJECT (LIBERO's check_contain calls"
            " object.in_box, which no object class defines)"
        )
    if kind == GAP_FIXTURE_REGION:
        return String(
            "On/In onto a region whose target is neither a fixture, an object"
            " nor the workspace"
        )
    if kind == GAP_ARTICULATION:
        return String(
            "Open/Close/Turnon/Turnoff on something that is neither a"
            " fixture nor a fixture's site region"
        )
    if kind == GAP_UNKNOWN_PRED:
        return String("a predicate outside the measured seven")
    if kind == GAP_TAPE_TERMS:
        return String(
            "a goal with more terms than the device tape holds (three, in"
            " twelve `meta` words)"
        )
    return String("a predicate with an argument count we do not expect")


struct GoalGap(Copyable, ImplicitlyCopyable, Movable):
    """Why a goal does not translate, and which term caused it."""

    var kind: Int
    var term: String

    def __init__(out self, kind: Int, var term: String):
        self.kind = kind
        self.term = term^


def _region_target(p: BddlProblem, composed: String) -> String:
    try:
        var ri = p.region_index(composed)
        if ri < 0:
            return String("")
        return String(p.regions[ri].target)
    except:
        return String("")


def classify_goal(p: BddlProblem) raises -> GoalGap:
    """`GAP_NONE` if every goal term maps, else the FIRST gap and its term.

    ⚠ THE FIRST, NOT ALL OF THEM. A goal is a conjunction and one
    untranslatable term makes the whole task untranslatable, so reporting
    further gaps in the same file would inflate the survey's counts past the
    number of TASKS blocked.

    ⚠ SYNTACTIC. What the corpus SAYS can be classified from the file; whether
    the table has a threshold row for a class, or the asset the site,
    surfaces from `translate_task`, which raises. The survey counts both.
    """
    for i in range(len(p.goal)):
        ref g = p.goal[i]
        if (
            g.pred == "Open" or g.pred == "Close"
            or g.pred == "Turnon" or g.pred == "Turnoff"
        ):
            if len(g.args) != 1:
                return GoalGap(GAP_ARITY, g.show())
            var x = String(g.args[0])
            if p.is_fixture(x) or p.is_object(x):
                continue
            var tgt = _region_target(p, x)
            if tgt.byte_length() > 0 and (p.is_fixture(tgt) or p.is_object(tgt)):
                continue
            return GoalGap(GAP_ARTICULATION, g.show())
        if g.pred != "On" and g.pred != "In":
            return GoalGap(GAP_UNKNOWN_PRED, g.show())
        if len(g.args) != 2:
            return GoalGap(GAP_ARITY, g.show())
        var target = String(g.args[1])
        if p.is_object(target) or p.is_fixture(target):
            if g.pred == "In":
                return GoalGap(GAP_OBJECT_TARGET, g.show())
            continue
        var ri = p.region_index(target)
        if ri < 0:
            return GoalGap(GAP_UNKNOWN_PRED, g.show())
        if not p.regions[ri].has_ranges:
            var t2 = String(p.regions[ri].target)
            if not (p.is_fixture(t2) or p.is_object(t2)):
                return GoalGap(GAP_FIXTURE_REGION, g.show())
    # ⚠⚠ THE DEVICE TAPE'S CAPACITY IS A LIMIT ON WHAT CAN BE EXPRESSED, so
    # it is classified here and not left to surface as a translation error.
    # `n` goal atoms left-fold into `n` leaves and `n - 1` `And`s; the tape
    # holds `MAX_TAPE_TERMS`. This is a LOWER bound — an articulation atom
    # expands to one Joint term per joint, which needs the asset — so
    # `translate_task` refuses the rest. Measured on the corpus: 110 files
    # have one goal atom, 19 have two (three terms, exactly the tape), and
    # ONE has three (`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`: two
    # `On`s and a `Turnon`, five terms).
    if 2 * len(p.goal) - 1 > MAX_TAPE_TERMS:
        var terms = String(2 * len(p.goal) - 1)
        return GoalGap(
            GAP_TAPE_TERMS,
            String(len(p.goal)) + " goal atoms -> " + terms + " tape terms",
        )
    return GoalGap(GAP_NONE, String(""))


def goal_zone_regions(p: BddlProblem) raises -> List[Int]:
    """Indices of the RANGED regions a `:goal` names — the ones that get a
    `_zone` box region (header: table regions are two things)."""
    var out = List[Int]()
    for i in range(len(p.goal)):
        ref g = p.goal[i]
        if (g.pred != "On" and g.pred != "In") or len(g.args) != 2:
            continue
        var ri = p.region_index(String(g.args[1]))
        if ri >= 0 and p.regions[ri].has_ranges:
            var seen = False
            for k in range(len(out)):
                if out[k] == ri:
                    seen = True
            if not seen:
                out.append(ri)
    return out^


def translate_family(p: BddlProblem) raises -> FamilySpec:
    """Fixtures -> static slots, objects -> free slots, ranged regions ->
    regions.

    ⚠⚠ THE ASSET PATHS ARE THE CATEGORY NAME, NOT A FILE THAT EXISTS. LIBERO
    resolves `akita_black_bowl` through a Python object registry onto
    `assets/stable_scanned_objects/...`; the `.bddl` names only the category.
    Emitting a path here would be inventing one. So the slot's asset is the
    CATEGORY, and a `.family` written from this does not compose until someone
    maps categories to files — which is a deliberate, visible TODO rather than
    a path that looks right and is not.

    ⚠ A REGION WITH NO `:ranges` IS DROPPED, and the goal that names it is
    refused by `classify_goal` — the two must agree, or a task would reference
    a region the family does not declare and `validate_task_against_family`
    would reject it with a confusing message instead of the real reason.
    """
    var f = FamilySpec()
    f.schema_version = SCHEMA_VERSION
    f.name = String(p.problem)
    f.base = String("TODO:") + p.domain
    for i in range(len(p.fixtures)):
        var s = SlotSpec(
            String(p.fixtures[i].name), SLOT_STATIC,
            String("TODO:") + p.fixtures[i].category,
        )
        # ⚠ A STATIC SLOT REQUIRES A POSE and the `.bddl` gives it as an INIT
        # RANGE, not a point — `(On wooden_cabinet_1 main_table_cabinet_region)`
        # with a rectangle. The centre of that rectangle is the honest reading
        # and it is what LIBERO samples around; a fixture whose init region is
        # missing keeps (0,0,0) and the caller is told by `family_todo_count`.
        s.has_pose = True
        s.px = 0.0
        s.py = 0.0
        s.pz = 0.0
        for k in range(len(p.init)):
            ref a = p.init[k]
            if a.pred == "On" and len(a.args) == 2 and a.args[0] == s.name:
                var ri = p.region_index(String(a.args[1]))
                if ri >= 0 and p.regions[ri].has_ranges:
                    ref r = p.regions[ri]
                    s.px = 0.5 * (r.x0 + r.x1)
                    s.py = 0.5 * (r.y0 + r.y1)
                    s.pz = 0.0
        f.slots.append(s^)
    for i in range(len(p.objects)):
        var s2 = SlotSpec(
            String(p.objects[i].name), SLOT_FREE,
            String("TODO:") + p.objects[i].category,
        )
        f.slots.append(s2^)
    for i in range(len(p.regions)):
        ref r = p.regions[i]
        if not r.has_ranges:
            continue
        var rs = RegionSpec(r.composed_name(), r.composed_name())
        rs.has_rect = True
        rs.x_min = r.x0
        rs.y_min = r.y0
        rs.x_max = r.x1
        rs.y_max = r.y1
        f.regions.append(rs^)
    return f^


# ── L3: the asset text, read for what the .bddl does not carry ─────────────


def _tag_attr_or(tag: String, name: String, dflt: String) -> String:
    var v = _tag_attr(tag, name)
    return v if v.byte_length() > 0 else dflt


def _floats(text: String) -> List[Float64]:
    var out = List[Float64]()
    var toks = text.split(" ")
    for k in range(len(toks)):
        var t = String(String(toks[k]).strip())
        if t.byte_length() > 0:
            try:
                out.append(Float64(t))
            except:
                pass
    return out^


def has_site(xml: String, site: String) -> Bool:
    """Does the asset declare `<site name="<site>">`? LIBERO registers a
    fixture's site region only when some body of the fixture carries the
    site; a `.bddl` may name one the asset lacks (it names regions for
    fixtures it does not even declare), and then nothing is registered."""
    var pos = 0
    while True:
        var i = xml.find("<site", pos)
        if i < 0:
            return False
        var j = xml.find(">", i)
        if j < 0:
            return False
        if _tag_attr(String(xml[byte=i:j]), String("name")) == site:
            return True
        pos = j + 1


def site_box(xml: String, site: String) raises -> List[Float64]:
    """`(hx, hy, hz)` of `<site type="box" name="<site>" size=...>` — the
    region a LIBERO goal means by that site. RAISES if absent or not a box:
    `SiteObject.in_box` / `under` index `size[:2]` and `size[2]`, which only
    a box has."""
    var pos = 0
    while True:
        var i = xml.find("<site", pos)
        if i < 0:
            break
        var j = xml.find(">", i)
        if j < 0:
            break
        var tag = String(xml[byte=i:j])
        if _tag_attr(tag, String("name")) == site:
            var ty = _tag_attr_or(tag, String("type"), String("sphere"))
            var sz = _floats(_tag_attr(tag, String("size")))
            if ty != "box" or len(sz) != 3:
                raise Error(
                    "libero: site '" + site + "' is type '" + ty + "' with "
                    + String(len(sz)) + " size numbers; LIBERO's in_box/under"
                    " read a box's three half-sizes"
                )
            return sz^
        pos = j + 1
    raise Error(
        "libero: no <site name=\"" + site + "\"> in the asset — the region"
        " a goal names is that site's box, and there is none"
    )


def _scan_joints(
    xml: String, site: String, mut of_site: List[String], mut all: List[String]
) raises -> Bool:
    """One pass over the tags: `all` gets every joint; `of_site` the direct
    joints of the innermost body enclosing `site` (LIBERO's rule, header).
    Returns whether the site was seen."""
    var stack_joints = List[List[String]]()
    var found = False
    var pos = 0
    while True:
        var i = xml.find("<", pos)
        if i < 0:
            break
        var j = xml.find(">", i)
        if j < 0:
            break
        var tag = String(xml[byte=i:j])
        pos = j + 1
        if tag.startswith("<!--"):
            var e = xml.find("-->", i)
            pos = e + 3 if e >= 0 else pos
            continue
        if tag.startswith("<body"):
            stack_joints.append(List[String]())
        elif tag.startswith("</body"):
            if len(stack_joints) > 0:
                _ = stack_joints.pop()
        elif tag.startswith("<joint"):
            var jn = _tag_attr(tag, String("name"))
            if jn.byte_length() > 0:
                all.append(jn)
                if len(stack_joints) > 0:
                    stack_joints[len(stack_joints) - 1].append(jn)
        elif tag.startswith("<site"):
            if site.byte_length() > 0 and _tag_attr(tag, String("name")) == site:
                found = True
                of_site = List[String]()
                if len(stack_joints) > 0:
                    for k in range(len(stack_joints[len(stack_joints) - 1])):
                        of_site.append(stack_joints[len(stack_joints) - 1][k])
    return found


def asset_placement_geom(
    xml: String, asset_path: String
) raises -> Tuple[Float64, Float64, Float64]:
    """`(bottom_z, top_z, h_radius)` from the asset's own robosuite sites.

    Every robosuite `MujocoXMLObject` declares three sites, and all 93 of
    LIBERO's object assets have them:

        bottom_site             z  -> `bottom_offset[-1]`
        top_site                z  -> `top_offset[-1]`
        horizontal_radius_site  sqrt(x^2 + y^2) -> `horizontal_radius`

    ⚠⚠ THESE ARE THE NUMBERS THE SAMPLER NEEDS AND THEY ARE NOT INTERCHANGEABLE
    WITH EACH OTHER, LET ALONE WITH ONE CONSTANT.
    `SiteRegionRandomSampler.sample` places an object's ORIGIN at
    `site_z - bottom_offset[-1]` and rejects a draw within
    `other.horizontal_radius + horizontal_radius` — a height and a distance.
    `sample_placements` used one caller-supplied `radius` for both.

    ⚠ `bottom_site` IS DELIBERATELY BELOW THE COLLISION GEOMETRY.
    `akita_black_bowl` puts it at z = -0.06 while its collision boxes reach
    -0.012, so LIBERO starts the bowl 4.8 cm clear of the surface and lets it
    fall — which is the same protocol §6k measured in the recorded states
    (a 2.6-7.2 cm settle over five zero-action steps). Reading the collision
    extent instead would place it flush and change every episode's first
    moments.

    ⚠ RAISES ON A MISSING SITE rather than defaulting. A default here is a
    made-up resting height, which is exactly what this function exists to
    remove.
    """
    var have = List[Bool](length=3, fill=False)
    var bottom_z = 0.0
    var top_z = 0.0
    var h_radius = 0.0
    var pos = 0
    while True:
        var i = xml.find("<site", pos)
        if i < 0:
            break
        var j = xml.find(">", i)
        if j < 0:
            break
        var tag = String(xml[byte=i:j])
        pos = j + 1
        var nm = _tag_attr(tag, String("name"))
        if nm != "bottom_site" and nm != "top_site" and nm != "horizontal_radius_site":
            continue
        var pv = _tag_attr(tag, String("pos"))
        var n = split_on(pv, String(" "))
        var xyz = List[Float64]()
        for k in range(len(n)):
            var t = String(String(n[k]).strip())
            if t.byte_length() > 0:
                xyz.append(Float64(t))
        if len(xyz) != 3:
            raise Error(
                "libero: <site name=\"" + nm + "\"> in " + asset_path
                + " has pos='" + pv + "', which is not three numbers"
            )
        if nm == "bottom_site":
            bottom_z = xyz[2]
            have[0] = True
        elif nm == "top_site":
            top_z = xyz[2]
            have[1] = True
        else:
            # ⚠ `sqrt`, NOT `** 0.5`. The exponent form goes through exp/log
            # and returned 0.049999999998 for hypot(0.03, 0.04) — 2.2e-12 off,
            # and this number is written into a checked-in `.family` file where
            # it becomes the exact value every later comparison uses.
            h_radius = sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1])
            have[2] = True
    if not (have[0] and have[1] and have[2]):
        raise Error(
            "libero: " + asset_path + " declares "
            + ("bottom_site " if not have[0] else "")
            + ("top_site " if not have[1] else "")
            + ("horizontal_radius_site " if not have[2] else "")
            + "nowhere. Every robosuite MujocoXMLObject has all three and the"
            " sampler needs them: the resting height, the stacking height and"
            " the rejection radius. Refused rather than defaulted — a default"
            " is a made-up resting height."
        )
    return (bottom_z, top_z, h_radius)


def site_joints(xml: String, site: String) raises -> List[String]:
    """The joints `Open(<site region>)` reads — see the module header."""
    var of_site = List[String]()
    var all = List[String]()
    if not _scan_joints(xml, site, of_site, all):
        raise Error("libero: no <site name=\"" + site + "\"> in the asset")
    return of_site^


def asset_joints(xml: String) raises -> List[String]:
    """Every joint in the asset — `MujocoXMLObject.joints`, what
    `Open(<fixture>)` / `Turnon(<fixture>)` read."""
    var of_site = List[String]()
    var all = List[String]()
    _ = _scan_joints(xml, String(""), of_site, all)
    return all^


def _articulation_joints(
    p: BddlProblem, table: LiberoTable, pred: String, x: String,
    pack_dir: String,
) raises -> Tuple[String, String, List[String]]:
    """`(fixture, category, joints)` for an articulation term's argument.

    ⚠ ONE RESOLUTION, TWO CALLERS. `_articulation_terms` builds a GOAL out of
    these joints and `translate_task` builds an INIT draw out of the same ones;
    written twice they would drift on exactly the case that is hard — a term
    naming a REGION (`Open(wooden_cabinet_1_top_region)`), where the joints are
    the ones on that region's site's body and NOT the asset's whole set.
    """
    var fixture = x
    var site = String("")
    if not (p.is_fixture(x) or p.is_object(x)):
        var ri = p.region_index(x)
        if ri < 0:
            raise Error("libero: " + pred + "(" + x + ") names nothing declared")
        fixture = String(p.regions[ri].target)
        site = String(p.regions[ri].name)
    var cat_name = String("")
    for i in range(len(p.fixtures)):
        if p.fixtures[i].name == fixture:
            cat_name = String(p.fixtures[i].category)
    for i in range(len(p.objects)):
        if p.objects[i].name == fixture:
            cat_name = String(p.objects[i].category)
    var cat = table.category(cat_name)
    var xml: String
    with open(pack_dir + "/" + cat.asset, "r") as fh:
        xml = fh.read()
    var joints: List[String]
    if site.byte_length() > 0:
        joints = site_joints(xml, site)
    else:
        joints = asset_joints(xml)
    if len(joints) == 0:
        raise Error(
            "libero: " + pred + "(" + x + ") reads no joint — the site's body"
            " (or the asset) declares none, so LIBERO's is_open would loop"
            " over nothing and return "
            + ("False" if pred == "Open" or pred == "Turnon" else "True")
        )
    return (fixture^, cat_name^, joints^)


def _articulation_terms(
    p: BddlProblem, f: FamilySpec, table: LiberoTable, pred: String,
    x: String, pack_dir: String,
) raises -> String:
    """`Open/Close/Turnon/Turnoff(x)` as Joint terms over the right joints,
    Or-joined for ANY (Open, Turnon), And-joined for ALL (Close, Turnoff)."""
    var res = _articulation_joints(p, table, pred, x, pack_dir)
    var fixture = String(res[0])
    var cat = table.category(String(res[1]))
    var joints = res[2].copy()
    var thr: Threshold
    if pred == "Open":
        thr = cat.open
    elif pred == "Close":
        thr = cat.close
    elif pred == "Turnon":
        thr = cat.on
    else:
        thr = cat.off
    if not thr.present():
        raise Error(
            "libero: " + pred + "(" + x + ") — category '" + cat.name + "' has"
            " no " + pred.lower() + "= threshold in categories.kv; LIBERO's"
            " class defines the comparison and it must be quoted there, not"
            " guessed from the joint range"
        )
    var any_ = pred == "Open" or pred == "Turnon"
    var out = String("")
    for k in range(len(joints)):
        var term = (
            String("Joint(") + fixture + "_" + joints[k] + ", " + cmp_name(thr.op)
            + ", " + String(thr.thr) + ")"
        )
        if k == 0:
            out = term^
        else:
            out = (String("Or(") if any_ else String("And(")) + out + ", " + term + ")"
    return out^


struct RegionAlias(Copyable, ImplicitlyCopyable, Movable):
    """One `.bddl` composed region name -> the family's name for the SAME
    rectangle.

    ## ⚠⚠ WHY A REGION'S NAME IS NOT ITS IDENTITY ACROSS A SUITE

    A `.bddl` region name is a ROLE in that one file. `libero_object`'s ten
    files declare the same eight floor rectangles under names that MOVE
    between them: `target_object_region` is
    `(-0.145 -0.265 -0.095 -0.215)` in five files and
    `(0.025 -0.125 0.075 -0.075)` in the other five, with
    `other_object_region_0` holding whichever one it does not. Nothing about
    the scene changes — the target object simply starts in the other spot.

    A union family (one slot table and one region table for a suite, see
    `gen_libero_family`) must therefore key a region by its GEOMETRY and
    let each file resolve its own names through this map. Keying by name
    would silently place the props in the wrong halves of the floor for five
    of the ten tasks, with every downstream number still agreeing.
    """

    var bddl: String
    """The composed `<target>_<name>` as the `.bddl` writes it."""
    var family: String
    """The name the family gives that same rectangle."""

    def __init__(out self, var bddl: String, var family: String):
        self.bddl = bddl^
        self.family = family^


def alias_region(name: String, region_alias: List[RegionAlias]) -> String:
    """`name` under the family's spelling. Identity when unlisted — a slot
    name (`On(bowl, plate_1)`) or a region whose name the suite agrees on."""
    for i in range(len(region_alias)):
        if region_alias[i].bddl == name:
            return String(region_alias[i].family)
    return String(name)


def translate_task(p: BddlProblem, f: FamilySpec) raises -> TaskSpec:
    """The pre-L3 signature: no table, so no articulation and no asset
    access. Kept for `translate_family`'s survey shape; RAISES on an
    articulation goal."""
    var t = LiberoTable()
    return translate_task(p, f, t, String(""))


def translate_task(
    p: BddlProblem, f: FamilySpec, table: LiberoTable, pack_dir: String,
) raises -> TaskSpec:
    """One file against its own family: no region aliasing needed."""
    return translate_task(p, f, table, pack_dir, List[RegionAlias]())


def translate_task(
    p: BddlProblem, f: FamilySpec, table: LiberoTable, pack_dir: String,
    region_alias: List[RegionAlias],
) raises -> TaskSpec:
    """The `.task`. RAISES with the gap's name if the goal does not map.

    `region_alias` maps this file's region names onto a UNION family's —
    empty when the family was resolved from this file alone (see
    `RegionAlias`). ⚠ `alias` is a reserved word."""
    var gap = classify_goal(p)
    if gap.kind != GAP_NONE:
        raise Error(
            "libero: task '" + p.problem + "' (" + p.language + ") cannot be"
            " translated — " + gap_name(gap.kind) + ". The blocking term is "
            + gap.term + ". Refused rather than approximated: a goal that is"
            " ALMOST the benchmark's is a policy evaluated on the wrong task."
        )
    var t = TaskSpec()
    t.schema_version = SCHEMA_VERSION
    t.name = String(p.problem)
    t.family = String(f.name)
    t.language = String(p.language)

    var goal = String("")
    for i in range(len(p.goal)):
        ref g = p.goal[i]
        var term: String
        if (
            g.pred == "Open" or g.pred == "Close"
            or g.pred == "Turnon" or g.pred == "Turnoff"
        ):
            if pack_dir.byte_length() == 0:
                raise Error(
                    "libero: " + g.show() + " needs the category table and"
                    " the asset pack — call translate_task(p, f, table,"
                    " pack_dir)"
                )
            term = _articulation_terms(
                p, f, table, String(g.pred), String(g.args[0]), pack_dir
            )
        else:
            var ri = p.region_index(String(g.args[1]))
            var target = alias_region(String(g.args[1]), region_alias)
            if ri >= 0 and p.regions[ri].has_ranges:
                # a table zone — the `_zone` box region (header)
                target = (
                    alias_region(p.regions[ri].composed_name(), region_alias)
                    + ZONE_SUFFIX
                )
            term = String(g.pred) + "(" + g.args[0] + ", " + target + ")"
            if f.region_index(target) < 0 and f.slot_index(target) < 0:
                raise Error(
                    "libero: " + g.show() + " names '" + target + "', which"
                    " the family declares neither as a region nor as a slot"
                )
        goal = term^ if i == 0 else (String("And(") + goal + ", " + term + ")")
    # ⚠ THE DEVICE TAPE HOLDS THREE TERMS. A goal past that would parse and
    # bind and then be refused by `encode_goal` at the driver; refusing it
    # here keeps "written" in the survey honest about what can RUN.
    var parsed = parse_goal(goal)
    if len(parsed.terms) > MAX_TAPE_TERMS:
        raise Error(
            "libero: task '" + p.problem + "' (" + p.language + ") needs "
            + String(len(parsed.terms)) + " goal terms; the device tape holds "
            + String(MAX_TAPE_TERMS) + " (twelve `meta` words). Goal: " + goal
        )
    t.goal = goal^

    # ⚠ ACTIVE = EVERY SLOT THIS FILE DECLARES, not `:obj_of_interest`.
    # LIBERO's field names what the INSTRUCTION is about; our `active=` names
    # what the SCENE contains, and a prop that is present but not mentioned is
    # still on the table. Using obj_of_interest would park the distractors and
    # quietly make every task easier than the benchmark's.
    #
    # ⚠⚠ AND IT IS THE FILE'S `:objects`, NOT THE FAMILY'S SLOT TABLE. For a
    # suite whose ten files declare the same props, the two are the same list
    # and this reads identically. For a UNION family they are not:
    # `libero_object` pools ELEVEN props and each of its ten files puts SEVEN
    # of them on the floor. Taking the family's table would activate all
    # eleven — four props with no `init=`, which
    # `validate_task_against_family` refuses, and if it ever stopped refusing
    # it would be four objects stacked at the family's park pose.
    #
    # The arena is the exception: it is slot 0 of every family and no `.bddl`
    # names it (the `:fixtures` entry is `main_table` / `floor`, which
    # `resolve_family` folds INTO the arena asset).
    for i in range(len(f.slots)):
        var nm = String(f.slots[i].name)
        if nm == ARENA_SLOT or p.is_object(nm) or p.is_fixture(nm):
            t.active.append(nm^)

    # ── `:init` -> `init=`, and the extent comes from the FAMILY ──────────
    #
    # ⚠⚠ THE OLD TEST WAS `p.regions[ri].has_ranges`, AND IT ASKED THE WRONG
    # FILE. A `.bddl` gives `:ranges` only for a region on the TABLE; a region
    # on a fixture is `(cook_region (:target flat_stove_1))` with no extent at
    # all, because the extent is the `<site>` inside the fixture's asset XML —
    # which `resolve_family` has already read into the family's own `region=`
    # line as a box rect. So the bddl's silence meant "no init here" and the
    # placement was dropped, silently, for every object LIBERO starts on a
    # stove, a cabinet top or inside a drawer.
    #
    # It surfaced one step later as `gen_libero_family` refusing the task —
    # "free slot X is active but has no init=" — which named the symptom and
    # not this line. Six of `libero_spatial`'s ten tasks, and it is why the
    # suite sat at 4.
    #
    # ⚠ `In` IS ACCEPTED BESIDE `On`. `(In akita_black_bowl_1
    # wooden_cabinet_1_top_region)` is a bowl INSIDE a drawer, and the
    # placement is the same question — where in that region does it start. The
    # difference between resting on a surface and sitting in a box is the
    # drawer's own joint value, which is a separate `:init` term (`Open`) and a
    # separate feature; a task needing it is refused by name below rather than
    # placed into a shut drawer.
    for k in range(len(p.init)):
        ref a = p.init[k]
        if len(a.args) != 2:
            continue
        if a.pred != "On" and a.pred != "In":
            continue
        var slot = String(a.args[0])
        var target = String(a.args[1])
        var ri = p.region_index(target)
        if ri < 0:
            # ⚠ NOT A REGION — `(On bowl cookies_1)`, a STACK on another free
            # object. The target is the other slot's name, which is exactly what
            # the goal language already does with `On(bowl, plate_1)`.
            var tsi = f.slot_index(target)
            var is_free_t = tsi >= 0 and f.slots[tsi].kind == SLOT_FREE
            var is_free_s = False
            for s2 in range(len(f.slots)):
                if f.slots[s2].name == slot and f.slots[s2].kind == SLOT_FREE:
                    is_free_s = True
            if is_free_t and is_free_s:
                # ⚠ A STACK IS ALWAYS `On` — `ObjectBasedSampler` is a third
                # class and adds the reference's `top_offset` unconditionally
                # (`spec.STACK_Z_OFFSET`). `(In obj other_obj)` appears once in
                # the whole corpus and names a REGION, not a prop.
                t.inits.append(InitSpec(slot, String(target), False))
            continue
        var fam_name = alias_region(p.regions[ri].composed_name(), region_alias)
        var fri = f.region_index(fam_name)
        if fri < 0:
            continue
        # ⚠ THE FAMILY MUST CARRY A RECT. A region with none is a bare site,
        # and every draw would land on the same point — an "init" that
        # randomises nothing, which is the degeneracy `init=` exists to avoid.
        if not f.regions[fri].has_rect:
            continue
        # only FREE slots take an init; a fixture's placement is its pose
        var is_free = False
        for s in range(len(f.slots)):
            if f.slots[s].name == slot and f.slots[s].kind == SLOT_FREE:
                is_free = True
        if not is_free:
            continue
        t.inits.append(InitSpec(slot, fam_name, a.pred == "In"))
    # ── `(Open X)` / `(Turnon X)` in `:init` -> `jinit=` ──────────────────
    #
    # ⚠⚠ THIS IS A DRAW, NOT A JUMP TO THE THRESHOLD.
    # `bddl_base_domain._reset_internal` builds an `OpenCloseSampler` from the
    # class's `default_open_ranges` and calls `np.random.uniform` on it every
    # reset, so a drawer starts SOMEWHERE in [-0.16, -0.14] and not at -0.14.
    # `categories.kv` carries the range beside the threshold for exactly this;
    # the two are different questions and only range -> threshold is derivable.
    #
    # ⚠ DROPPING THESE WAS NOT HARMLESS. `(In akita_black_bowl_1
    # wooden_cabinet_1_top_region)` places a bowl INSIDE the top drawer, and
    # `(Open wooden_cabinet_1_top_region)` is what makes that drawer open
    # enough to hold it. Emitting the placement without the opening puts the
    # bowl inside a SHUT cabinet — a scene MuJoCo resolves by ejecting it, one
    # step in, far from the cause. 28 of the corpus' 130 tasks carry one.
    for k in range(len(p.init)):
        ref a = p.init[k]
        if len(a.args) != 1:
            continue
        var ap = String(a.pred)
        if ap != "Open" and ap != "Close" and ap != "Turnon" and ap != "Turnoff":
            continue
        if pack_dir.byte_length() == 0:
            raise Error(
                "libero: " + a.show() + " in `:init` needs the category table"
                " and the asset pack — call translate_task(p, f, table,"
                " pack_dir)"
            )
        var res = _articulation_joints(p, table, ap, String(a.args[0]), pack_dir)
        var jfixture = String(res[0])
        var jcat = table.category(String(res[1]))
        var jjoints = res[2].copy()
        var rng: JointRange
        if ap == "Open":
            rng = jcat.open_range
        elif ap == "Close":
            rng = jcat.close_range
        elif ap == "Turnon":
            rng = jcat.on_range
        else:
            rng = jcat.off_range
        if not rng.present():
            raise Error(
                "libero: " + a.show() + " in `:init` — category '" + jcat.name
                + "' has no " + ap.lower() + "_range= in categories.kv."
                " LIBERO's class defines `default_" + ap.lower()
                + "_ranges` and it must be QUOTED there; the threshold does"
                " not determine it (open=lt:-0.14 says nothing about -0.16)."
            )
        # ⚠ ONE JOINT, ASSERTED. `set_joint(qpos)` gives every joint of the
        # object THE SAME draw, and a `.task` carries one `jinit=` per joint —
        # so several joints would be drawn INDEPENDENTLY here and the scene
        # would differ from LIBERO's in a way nothing downstream could see.
        # Every articulation init in the corpus resolves to exactly one; this
        # refuses rather than diverging if that ever stops being true.
        if len(jjoints) != 1:
            raise Error(
                "libero: " + a.show() + " in `:init` resolves to "
                + String(len(jjoints)) + " joints. LIBERO's `set_joint` gives"
                " them ONE shared draw; a `jinit=` per joint would draw each"
                " independently. Refused — see the note above this check."
            )
        t.joint_inits.append(
            JointInitSpec(jfixture + "_" + jjoints[0], rng.lo, rng.hi)
        )

    # ⚠ THE ORDER IS `spec.order_inits`, NOT A SECOND COPY OF IT. It is family
    # slot order with stacks moved after their references, and
    # `validate_task_against_family` checks the written file against the same
    # function — so the importer and the validator cannot drift.
    t.inits = order_inits(t, f)
    return t^


def family_todo_count(f: FamilySpec) -> Int:
    """How many asset paths are still `TODO:` placeholders.

    ⚠ EXISTS SO A CALLER CANNOT MISS THEM. A `.family` written straight out of
    a `.bddl` names CATEGORIES, not files; composing it would fail at the
    first `<attach>`. Counting them makes that a number a tool can print
    rather than a surprise at compose time."""
    var n = 0
    if f.base.startswith("TODO:"):
        n += 1
    for i in range(len(f.slots)):
        if f.slots[i].asset.startswith("TODO:"):
            n += 1
    return n


# ── L1: the resolved family — real assets, real fixture poses ──────────────


def _tag_attr(tag: String, name: String) -> String:
    """`name="..."` inside one opening tag, or "" if absent."""
    var needle = String(" ") + name + '="'
    var i = tag.find(needle)
    if i < 0:
        return String("")
    var start = i + needle.byte_length()
    var end = tag.find('"', start)
    if end < 0:
        return String("")
    return String(tag[byte=start:end])


def site_pos(xml: String, site: String) raises -> List[Float64]:
    """The `pos` of `<site ... name="<site>" .../>` in an asset XML.

    ⚠ A TEXT SCAN, NOT A PARSE, on purpose: the three robosuite bookkeeping
    sites (`bottom_site`, `top_site`, `horizontal_radius_site`) sit on the
    wrapper body of every LIBERO object and this is the only thing the
    importer reads out of an asset. `MujocoXMLObject.bottom_offset` reads the
    same attribute the same way (`string_to_array(site.get("pos"))`).
    RAISES if the site is absent — an object without one cannot be placed by
    LIBERO either.
    """
    var pos = 0
    while True:
        var i = xml.find("<site", pos)
        if i < 0:
            break
        var j = xml.find(">", i)
        if j < 0:
            break
        var tag = String(xml[byte=i:j])
        if _tag_attr(tag, String("name")) == site:
            var ps = _tag_attr(tag, String("pos"))
            if ps.byte_length() == 0:
                ps = String("0 0 0")
            var out = List[Float64]()
            var toks = ps.split(" ")
            for k in range(len(toks)):
                var t = String(String(toks[k]).strip())
                if t.byte_length() > 0:
                    out.append(Float64(t))
            if len(out) != 3:
                raise Error(
                    "libero: site '" + site + "' has a pos with "
                    + String(len(out)) + " numbers"
                )
            return out^
        pos = j + 1
    raise Error(
        "libero: no <site name=\"" + site + "\"> in the asset — LIBERO"
        " places every object by this site and so must we"
    )


# ⚠ A FIXTURE YAW RANGE NARROWER THAN THIS IS TAKEN AT ITS MIDPOINT. Measured
# on the corpus: 475 of 501 yaw blocks are a point; the only RANGED fixture is
# `libero_spatial`'s wooden cabinet, [2.6614, 2.7242] rad — a 3.6 degree band.
# LIBERO itself re-draws that yaw at EVERY reset, frozen init or not (a
# fixture's pose is `model.body_pos`, which `sim.get_state()` does not carry),
# so the band is reset NOISE in the benchmark, not part of a task's identity.
# A static slot has one pose; the midpoint is the mean of what LIBERO draws
# and the deviation is at most 1.8 degrees. Wider than this and it is a
# distribution the family cannot express, so it RAISES.
comptime FIXTURE_YAW_TOLERANCE: Float64 = 0.1


# ⚠ THE ROBOT AND THE ARENA ARE L2's HALF OF THE FAMILY. With `robot_xml`
# empty (`robot_dir`) the family's base is the raw arena scene (L1's shape, kept for the
# survey and the tests); with it set, the base is the vendored Panda placed
# at the problem's `base_pos`, the styled arena from `arena_dir` is the
# first static slot at the origin, the composer's own floor is off, and the
# Panda's `<option>` is inherited. Regions then anchor on the arena's
# `workspace` site, which `gen_libero_arenas.py` places at LIBERO's
# `workspace_offset` — the point every region rect is measured from.
comptime LIBERO_ARENA_DIR: String = "noeira/envs/libero/arenas"
# ⚠ IN ROBOT MODE THE OBJECTS ARE THE GENERATED COPIES, NOT THE PACK'S.
# `tools/libero/gen_libero_objects.py` writes one attachable XML per category:
# the pack's files are byte-identical to upstream, and upstream never loads
# them directly — robosuite's `merge_assets` drops a repeated (tag, name)
# asset declaration, which `flat_stove.xml` relies on and which MuJoCo's
# `<attach>` refuses. Without the robot (L1's survey shape) the pack path is
# still emitted, so the survey needs no generated files.
comptime LIBERO_OBJECT_DIR: String = "noeira/envs/libero/objects"
comptime LIBERO_ROBOT_DIR: String = "noeira/envs/robots/assets"
comptime ARENA_SLOT: String = "arena"


# LIBERO `MountedPanda.init_qpos` / `OnTheGroundPanda.init_qpos` (identical),
# `envs/robots/*_panda.py`, plus `PandaGripper.init_qpos` (0.020833,
# -0.020833). The family's `base_qpos=`; the oracle and the env read it.
comptime PANDA_INIT_QPOS: String = (
    "0,-0.161037389,0,-2.44459747,0,2.2267522,0.7853981633974483,"
    "0.020833,-0.020833"
)


def robot_path(prob: LiberoProblem, robot_dir: String) -> String:
    """`panda_robosuite.xml` for `robot=mounted`, `panda_robosuite_nomount.xml`
    for `robot=on_the_ground` — both written by
    `tools/robots/vendor_panda_robosuite.py`."""
    if prob.robot == "on_the_ground":
        return robot_dir + "/panda_robosuite_nomount.xml"
    return robot_dir + "/panda_robosuite.xml"


def arena_path(prob: LiberoProblem, arena_dir: String) -> String:
    """`<arena_dir>/<problem, lowercased>_arena.xml` — ONE spelling, shared
    with `tools/libero/gen_libero_arenas.py`."""
    return arena_dir + "/" + prob.name.lower() + "_arena.xml"


def resolve_family(
    p: BddlProblem, table: LiberoTable, pack_dir: String,
) raises -> FamilySpec:
    var narrowed = 0
    return resolve_family(p, table, pack_dir, narrowed, String(""), String(""))


def resolve_family(
    p: BddlProblem, table: LiberoTable, pack_dir: String, mut narrowed: Int,
) raises -> FamilySpec:
    return resolve_family(p, table, pack_dir, narrowed, String(""), String(""))


def resolve_family(
    p: BddlProblem, table: LiberoTable, pack_dir: String, mut narrowed: Int,
    robot_dir: String, arena_dir: String,
) raises -> FamilySpec:
    """`translate_family` with the registry: every slot names a FILE under
    `pack_dir`, every fixture carries the pose LIBERO computes for it, and
    the workspace (`main_table - table`) is the base scene rather than a slot.

    ## The fixture pose, quoted from LIBERO

    `bddl_base_domain._add_placement_initializer` places a fixture with a
    `MultiRegionRandomSampler(z_offset=self.z_offset, rotation=yaw_rotation,
    rotation_axis="z", reference_pos=self.workspace_offset)`, and
    `base_region_sampler.sample` then does

        object_x = sample_x + reference_pos[0]
        object_y = sample_y + reference_pos[1]
        object_z = z_offset + reference_pos[2] - bottom_offset[2]

    A static slot cannot sample, so x,y are the rect's CENTRE — the mean of
    what LIBERO draws — and the yaw is the range's MIDPOINT when the range
    is narrower than `FIXTURE_YAW_TOLERANCE` (see it for why), counted in
    `narrowed`, and REFUSED when wider.

    ⚠ THE REGION SITE IS AN OBLIGATION ON L2. Table regions are emitted
    anchored on `robot_<workspace>_top`, i.e. a site named `<workspace>_top`
    in the base scene at `workspace_offset`; robosuite's `TableArena`
    re-poses the arena's `table_top` site there at load time, and the styled
    base scene L2 generates must do the same. Until then the family composes
    but its table regions bind to nothing — `family_todo_count` does not
    count that, `validate_task_against_family` does.
    """
    var prob = table.problem(p.problem)
    var f = FamilySpec()
    f.schema_version = SCHEMA_VERSION
    f.name = String(p.problem)
    f.horizon = 600
    f.control_freq = 20
    var with_robot = robot_dir.byte_length() > 0
    var anchor = String("robot_") + prob.workspace + "_top"
    if with_robot:
        f.base = robot_path(prob, robot_dir)
        f.base_x = prob.base_x
        f.base_y = prob.base_y
        f.base_z = prob.base_z
        f.floor = False
        f.inherit_option = True
        var q = PANDA_INIT_QPOS.split(",")
        for k in range(len(q)):
            f.base_qpos.append(Float64(String(q[k])))
        f.slots.append(
            SlotSpec(
                String(ARENA_SLOT), SLOT_STATIC, arena_path(prob, arena_dir),
                0.0, 0.0, 0.0,
            )
        )
        anchor = String(ARENA_SLOT) + "_workspace"
    else:
        f.base = pack_dir + "/" + prob.scene

    for i in range(len(p.fixtures)):
        var cat = table.category(String(p.fixtures[i].category))
        if cat.is_workspace():
            if cat.name != prob.workspace:
                raise Error(
                    "libero: task '" + p.problem + "' declares workspace"
                    " fixture '" + p.fixtures[i].name + " - " + cat.name
                    + "' but its problem class works on '" + prob.workspace
                    + "'"
                )
            continue
        var asset = pack_dir + "/" + cat.asset
        if with_robot:
            asset = String(LIBERO_OBJECT_DIR) + "/" + cat.name + ".xml"
        # the init line that places it, and its region
        var ri = -1
        for k in range(len(p.init)):
            ref a = p.init[k]
            if a.pred == "On" and len(a.args) == 2 and a.args[0] == p.fixtures[i].name:
                ri = p.region_index(String(a.args[1]))
        if ri < 0 or not p.regions[ri].has_ranges:
            raise Error(
                "libero: fixture '" + p.fixtures[i].name + "' in task '"
                + p.problem + "' has no `(On <fixture> <ranged region>)`"
                " init, so it has no pose"
            )
        ref r = p.regions[ri]
        var yaw = 0.0
        if r.has_yaw:
            var band = r.yaw_hi - r.yaw_lo
            if band < 0.0:
                band = -band
            if band > FIXTURE_YAW_TOLERANCE:
                raise Error(
                    "libero: fixture '" + p.fixtures[i].name + "' in task '"
                    + p.problem + "' samples its yaw in ["
                    + String(r.yaw_lo) + ", " + String(r.yaw_hi) + "], a "
                    + String(band) + " rad band; a static slot has ONE pose"
                    " and only a band under " + String(FIXTURE_YAW_TOLERANCE)
                    + " rad is taken at its midpoint. Not approximated."
                )
            if band > 0.0:
                narrowed += 1
            yaw = 0.5 * (r.yaw_lo + r.yaw_hi)
        # ⚠ THE BOOKKEEPING SITE IS READ FROM THE PACK'S FILE, never from the
        # generated copy: robosuite reads `bottom_site` off the wrapper body
        # and then discards the wrapper, and the generated copy does the same.
        var xml: String
        with open(pack_dir + "/" + cat.asset, "r") as fh:
            xml = fh.read()
        var bottom = site_pos(xml, String("bottom_site"))
        var s = SlotSpec(
            String(p.fixtures[i].name), SLOT_STATIC, asset,
            0.5 * (r.x0 + r.x1) + prob.off_x,
            0.5 * (r.y0 + r.y1) + prob.off_y,
            prob.z_offset + prob.off_z - bottom[2],
            yaw,
        )
        # ⚠⚠ A STATIC FIXTURE CARRIES `slot_geom=` TOO, AND ITS `top_z` IS
        # LOAD-BEARING. `SiteRegionRandomSampler.sample` places an object on a
        # fixture's site at `base_offset[2] + site_z - bottom_offset[-1]`,
        # where `base_offset` is the fixture's own pose PLUS
        # `ref_obj.top_offset[-1]` — the fixture's `top_site`. Every LIBERO
        # fixture declares it at 0.045, and leaving it out placed a bowl 4.5 cm
        # into the stove and into the cabinet's roof. MEASURED against
        # LIBERO's frozen states: 1.0100 / 1.2315 with it, 0.9650 / 1.1865
        # without (`tools/libero/libero_init_z.py`).
        var gf = asset_placement_geom(xml, String(cat.asset))
        s.set_geom(gf[0], gf[1], gf[2])
        f.slots.append(s^)

    for i in range(len(p.objects)):
        var cat2 = table.category(String(p.objects[i].category))
        if cat2.is_workspace():
            raise Error(
                "libero: '" + p.objects[i].name + "' is declared as an OBJECT"
                " but its category '" + cat2.name + "' is a workspace"
            )
        var asset2 = pack_dir + "/" + cat2.asset
        if with_robot:
            # ⚠ `_free`: the generated copy WITH robosuite's free joint. The
            # pack's XML has none — robosuite injects it at load for
            # `:objects` and not for `:fixtures` — so a free slot on the
            # plain copy would be welded to the world and never move.
            asset2 = String(LIBERO_OBJECT_DIR) + "/" + cat2.name + "_free.xml"
        var slot2 = SlotSpec(String(p.objects[i].name), SLOT_FREE, asset2)
        # ⚠⚠ THE PLACEMENT GEOMETRY, READ FROM THE PACK'S OWN ASSET, NOT THE
        # GENERATED `_free` COPY. `asset2` points at the copy when
        # `with_robot`, and the copy adds a free joint and nothing else — but
        # the three sites live in the pack XML either way and reading the one
        # we KNOW has them is what makes this unconditional. See
        # `SlotSpec.has_geom` for what the alternative cost.
        var geom_xml: String
        with open(pack_dir + "/" + cat2.asset, "r") as gh:
            geom_xml = gh.read()
        var g3 = asset_placement_geom(geom_xml, String(cat2.asset))
        slot2.set_geom(g3[0], g3[1], g3[2])
        f.slots.append(slot2^)

    # ── regions: GOAL regions first (the device table is 16 deep and a
    # goal must index it; `init=` regions are sampled on the host) ────────
    var zone_anchor = String("robot_") + prob.workspace + "_" + ZONE_SITE
    if with_robot:
        zone_anchor = String(ARENA_SLOT) + "_" + ZONE_SITE
    if not prob.has_zone_z:
        raise Error(
            "libero: problem '" + prob.name + "' has no zone_z= in the table;"
            " the table target zones' height is a per-class convention that"
            " must be quoted there"
        )
    # fixture / object sites named as regions → `:box:` with a contact slot
    for i in range(len(p.regions)):
        ref r3 = p.regions[i]
        if r3.has_ranges:
            continue
        var tgt = String(r3.target)
        var cat_name = String("")
        for k in range(len(p.fixtures)):
            if p.fixtures[k].name == tgt:
                cat_name = String(p.fixtures[k].category)
        for k in range(len(p.objects)):
            if p.objects[k].name == tgt:
                cat_name = String(p.objects[k].category)
        if cat_name.byte_length() == 0:
            # ⚠ SKIPPED, AS LIBERO SKIPS IT. All ten `libero_goal` files
            # declare `bowl_drainer_1_*` regions for a fixture none of them
            # declares (assessment §1.1); `_load_sites_in_arena` finds no
            # body carrying the site and registers nothing, and no goal
            # names them. A goal that DID would be refused by
            # `classify_goal` as GAP_FIXTURE_REGION.
            continue
        var cat3 = table.category(cat_name)
        var xml3: String
        with open(pack_dir + "/" + cat3.asset, "r") as fh:
            xml3 = fh.read()
        if not has_site(xml3, String(r3.name)):
            # the asset carries no such site: LIBERO registers nothing, and
            # a goal naming it is refused at `translate_task` ("declares
            # neither as a region nor as a slot")
            continue
        var hs = site_box(xml3, String(r3.name))
        var rb = RegionSpec(
            r3.composed_name(), r3.composed_name(),
            -hs[0], -hs[1], hs[0], hs[1], hs[2],
        )
        rb.is_box = True
        rb.contact = tgt
        f.regions.append(rb^)
    # table zones a goal names → `<composed>_zone` box on the zone plane
    var zones = goal_zone_regions(p)
    for k in range(len(zones)):
        ref rz = p.regions[zones[k]]
        var rq = RegionSpec(
            rz.composed_name() + ZONE_SUFFIX, zone_anchor,
            rz.x0, rz.y0, rz.x1, rz.y1, TARGET_ZONE_HALF_HEIGHT,
        )
        rq.is_box = True
        f.regions.append(rq^)
    # placement regions, plain, on the workspace anchor (the sampler's z)
    for i in range(len(p.regions)):
        ref r2 = p.regions[i]
        if not r2.has_ranges:
            continue
        var rs = RegionSpec(r2.composed_name(), anchor)
        rs.has_rect = True
        rs.x_min = r2.x0
        rs.y_min = r2.y0
        rs.x_max = r2.x1
        rs.y_max = r2.y1
        f.regions.append(rs^)
    return f^

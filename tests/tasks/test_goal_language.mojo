"""The goal language — parse, bind, and the Tier A rule. P2a's gate.

⚠ THE BINDING HALF RUNS AGAINST THE REAL COMPOSED FAMILY, not a fixture. A
goal's whole job is to name things in a scene, so a gate that binds against
invented names would pass while every real task failed to resolve. This binds
`In(brick, table_top)` against `noeira/tasks/scenes/so101_tabletop.xml` and
checks it lands on the body `<attach prefix="brick_">` actually produced.

Run: pixi run mojo run -I . tests/tasks/test_goal_language.mojo
"""

from noeira.tasks.spec import load_family
from noeira.tasks.family import scene_path
from noeira.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, slot_body_id,
    joint_qpos_addresses, OP_JOINT, OP_ON_BODY, OP_ON, CMP_GE,
    op_is_composite, op_name,
    OP_IN, OP_AND, OP_NEAR, OP_GRASPED, MAX_GOAL_TERMS,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime


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

    def refuses(mut self, text: String, what: String):
        self.checks += 1
        var raised = False
        try:
            _ = parse_goal(text)
        except e:
            raised = True
        if raised:
            print("  ok: refused —", what)
        else:
            self.failures += 1
            print("  FAIL: ACCEPTED —", what)


def main() raises:
    print("=== the goal language — P2a ===")
    var ta = Tally()

    # ── 1. parse + round-trip ─────────────────────────────────────────────
    print("--- parse ---")
    var g1 = parse_goal(String("In(brick, table_top)"))
    ta.check(len(g1.terms) == 1, "a leaf goal is one term")
    ta.check(g1.terms[0].op == OP_IN, "In parsed")
    ta.check(g1.describe() == "In(brick, table_top)", "round-trips as written")

    var g2 = parse_goal(
        String("And(In(brick, table_top), Near(cube_a, cube_b, 0.05))")
    )
    ta.check(len(g2.terms) == 3, "And + two leaves is three terms")
    ta.check(g2.terms[g2.root()].op == OP_AND, "the ROOT is the composite")
    ta.check(
        g2.describe() == "And(In(brick, table_top), Near(cube_a, cube_b, 0.05))",
        "a nested goal round-trips",
    )
    ta.check(g2.terms[1].param == 0.05, "Near's distance survives")

    # ⚠⚠ THE POST-ORDER INVARIANT IS WHAT MAKES STACKLESS EVALUATION LEGAL.
    # A GPU kernel sweeps this array FORWARDS and reads each child's already
    # -computed result; if a parent could reference a later term, that sweep
    # would read an uninitialised slot and the goal would evaluate to garbage
    # ONLY on device. Asserted here, where it is cheap.
    var ordered = True
    for i in range(len(g2.terms)):
        if op_is_composite(g2.terms[i].op):
            if g2.terms[i].kid0 >= i:
                ordered = False
            if g2.terms[i].kid1 >= i:
                ordered = False
    ta.check(ordered, "every child index is LESS than its parent's")

    var g3 = parse_goal(String("Not(Above(brick, cube_a, 0.05))"))
    ta.check(g3.describe() == "Not(Above(brick, cube_a, 0.05))", "Not is unary")

    # ── 2. refusals ───────────────────────────────────────────────────────
    print("--- refusals ---")
    ta.refuses(String("Floats(brick, table_top)"), "an unknown predicate")
    ta.refuses(String("In(brick"), "an unclosed paren")
    ta.refuses(String("In(brick, table_top) On(cube_a, table_top)"),
               "TWO goals with no operator between them")
    ta.refuses(String("In()"), "no arguments")
    ta.refuses(String("Near(a, b)"), "Near with a missing distance")
    ta.refuses(String(""), "an empty goal")
    # ⚠ THE TERM CAP IS A DEVICE-SIDE ONE — P3's tape is comptime-sized, so a
    # goal past it must be refused rather than truncated.
    var deep = String("In(brick, table_top)")
    for _ in range(MAX_GOAL_TERMS):
        deep = String("And(") + deep + ", In(brick, table_top))"
    ta.refuses(deep, "a goal past MAX_GOAL_TERMS")

    # ── 3. binding against the REAL composed family ───────────────────────
    print("--- bind against the composed scene ---")
    var f = load_family("noeira/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))
    print("    scene has", len(fmd.body_names), "body names,",
          len(fmd.site_names), "sites")

    var brick_id = slot_body_id(String("brick"), fmd.body_names)
    print("    slot 'brick' -> body", brick_id, "=",
          fmd.body_names[brick_id])
    ta.check(
        String(fmd.body_names[brick_id]).startswith("brick_"),
        "a slot resolves to the body its PREFIX produced",
    )
    # ⚠ body_names[0] is the WORLDBODY. A binder that forgot that would
    # resolve one body early and still return an id — the silent kind.
    ta.check(fmd.body_names[0] == "world", "body 0 is the worldbody")

    # ⚠ A COMPOSED BODY'S OWN NAME BINDS TOO, AFTER THE PREFIX. A base model's
    # free body is in no slot (the G1 room's crate, `robot_box`), so a goal
    # names it as the scene spells it. The prefix must still win, or a name
    # that resolved before would move to another body.
    var whole = String(fmd.body_names[brick_id])
    ta.check(
        slot_body_id(whole, fmd.body_names) == brick_id,
        "a composed body's own name ('" + whole + "') resolves to that body",
    )
    var mixed: List[String] = ["world", "a_x", "a", "b"]
    ta.check(slot_body_id(String("a"), mixed) == 1, "the prefix wins over a body spelled like the slot")
    ta.check(slot_body_id(String("b"), mixed) == 3, "a name that is no prefix binds to the body spelled so")
    var unknown_raised = False
    try:
        _ = slot_body_id(String("c"), mixed)
    except e:
        unknown_raised = True
    ta.check(unknown_raised, "a name that is neither a prefix nor a body still raises")
    var part_raised = False
    try:
        _ = slot_body_id(String("a_"), mixed)
    except e:
        part_raised = True
    ta.check(part_raised, "a part of a body's name ('a_') is not that body")

    var b1 = bind_goal(g1, f, fmd.body_names, fmd.site_names)
    ta.check(len(b1.terms) == 1, "bound goal has the same shape")
    ta.check(b1.terms[0].a == brick_id, "In's first arg is the brick's body")
    ta.check(b1.terms[0].b == 0, "In's second arg is the REGION index")
    ta.check(b1.is_tier_a(), "In is Tier A")
    require_tier_a(b1, String("t"))
    print("  ok: require_tier_a accepts it")
    ta.checks += 1

    var b2 = bind_goal(g2, f, fmd.body_names, fmd.site_names)
    ta.check(
        b2.terms[b2.root()].a == 0 and b2.terms[b2.root()].b == 1,
        "And's args are TERM indices, not body ids",
    )

    # ── 4. Tier B binds, and is refused where it matters ──────────────────
    print("--- Tier B ---")
    var gb = parse_goal(String("Grasped(brick)"))
    var bb = bind_goal(gb, f, fmd.body_names, fmd.site_names)
    ta.check(not bb.is_tier_a(), "Grasped is NOT Tier A")
    var raised = False
    try:
        require_tier_a(bb, String("pick"))
    except e:
        raised = True
    ta.check(raised, "require_tier_a REFUSES a Tier B goal (§5.1)")

    # ── 5. binding refuses what does not exist ────────────────────────────
    print("--- binding refusals ---")
    var bad = 0
    try:
        _ = bind_goal(parse_goal(String("In(hammer, table_top)")), f,
                      fmd.body_names, fmd.site_names)
    except e:
        bad += 1
    try:
        _ = bind_goal(parse_goal(String("In(brick, nowhere)")), f,
                      fmd.body_names, fmd.site_names)
    except e:
        bad += 1
    ta.check(bad == 2,
             "a goal naming an unknown SLOT or REGION is refused at bind")

    # ⚠ AtRegion resolves a SITE, not a slot — the gripper belongs to no slot.
    var gr = parse_goal(String("AtRegion(robot_gripperframe, table_top)"))
    var br = bind_goal(gr, f, fmd.body_names, fmd.site_names)
    ta.check(br.terms[0].a >= 0 and br.terms[0].a < len(fmd.site_names),
             "AtRegion binds its first arg to a SITE id")

    # ── 4. the L3 language: Joint, On(obj, obj), contacts ──────────────────
    print("--- L3: Joint / On(obj, obj) / Touching ---")
    var gj = parse_goal(String("Joint(robot_shoulder_pan, lt, -0.14)"))
    ta.check(
        gj.terms[0].op == OP_JOINT and gj.terms[0].arg1 == "lt"
        and gj.terms[0].param == -0.14,
        "Joint(j, cmp, thr) parses with the comparison as a NAME and the"
        " threshold as the number",
    )
    var raised_j = False
    try:
        _ = bind_goal(gj, f, fmd.body_names, fmd.site_names)
    except e:
        raised_j = True
    ta.check(raised_j, "Joint is REFUSED by the pre-L3 bind (no joint table)")
    var nqs = List[Int]()
    for i in range(len(fmd.joints)):
        nqs.append(fmd.joints[i].nq)
    var jadr = joint_qpos_addresses(nqs)
    var jname = String(fmd.joint_names[0])
    var bj = bind_goal(
        parse_goal(String("Joint(") + jname + ", ge, 0.5)"), f,
        fmd.body_names, fmd.site_names, fmd.joint_names, jadr,
    )
    ta.check(
        bj.terms[0].op == OP_JOINT and bj.terms[0].a == 0
        and bj.terms[0].b == CMP_GE and bj.terms[0].param == 0.5,
        "Joint binds to (qpos address, CMP code, threshold) — joint 0 is at"
        " qpos 0",
    )
    ta.check(bj.is_tier_a(), "Joint is Tier A")
    var bo = bind_goal(parse_goal(String("On(brick, cube_a)")), f,
                       fmd.body_names, fmd.site_names)
    ta.check(
        bo.terms[0].op == OP_ON_BODY
        and bo.terms[0].a == slot_body_id(String("brick"), fmd.body_names)
        and bo.terms[0].b == slot_body_id(String("cube_a"), fmd.body_names),
        "On(obj, SLOT) binds to OP_ON_BODY with both root bodies",
    )
    var bo2 = bind_goal(parse_goal(String("On(brick, table_top)")), f,
                        fmd.body_names, fmd.site_names)
    ta.check(bo2.terms[0].op == OP_ON, "On(obj, REGION) still binds to OP_ON")
    var raised_in = False
    try:
        _ = bind_goal(parse_goal(String("In(brick, cube_a)")), f,
                      fmd.body_names, fmd.site_names)
    except e:
        raised_in = True
    ta.check(raised_in, "In(obj, SLOT) is refused (LIBERO has no object in_box)")
    var bt = bind_goal(parse_goal(String("Touching(brick, cube_a)")), f,
                       fmd.body_names, fmd.site_names)
    ta.check(bt.is_tier_a(), "Touching is Tier A since L3 (the reward kernel"
             " has the lane's contacts)")
    ta.refuses(String("Joint(j, lt)"), "Joint with a missing threshold")
    ta.refuses(String("Joint(j, 0.5)"), "Joint with a missing comparison")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "goal language: " + String(ta.failures) + " of "
            + String(ta.checks) + " check(s) failed"
        )
    print("=== PASS ===")

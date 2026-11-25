"""The goal language — `goal=In(brick, box_inside)` — P2.

A `.task`'s `goal=` is TEXT in `spec.mojo` and a small NUMERIC PROGRAM here.
Two steps, and the split is the design:

    parse_goal("In(brick, box_inside)")   -> a tree of terms holding NAMES
    bind_goal(goal, family, model)        -> the same tree holding INDICES

## ⚠⚠ THE NUMERIC PROGRAM IS THE POINT, AND IT IS FOR THE GPU

`TASK_LAYER_PLAN.md` §5.1: LIBERO calls `_check_success()` from Python every
step; at 1024 lanes a per-step host round-trip would dominate the step, so
**the goal predicate is part of the reward and runs where the reward runs.** A
`BoundGoal` is a fixed-capacity array of `(op, a, b, param)` — exactly the
shape that becomes P3's device tape, read by a switch in the reward kernel.
Nothing here allocates, nothing here holds a string.

That is also why binding is a separate pass: the strings are resolved ONCE, on
the host, at task-load time. A kernel never sees a name.

## ⚠ TIER A vs TIER B — AND WHAT L3 MOVED

* **Tier A** reads per-lane `Data` the reward kernel is already handed:
  `xpos` / `xquat` / `site_xpos` / `qpos`, the model's `sites` and `bodies`
  tables, and — since L3 — the lane's own CONTACT LIST (`contacts` +
  `meta[META_IDX_NUM_CONTACTS]`, exactly what `sensors/touch.mojo` reads).
  In, On, Near, Above, Upright, AtRegion, Joint, Touching, and `On(obj, obj)`.
* **Tier B** is what has NO defined semantics yet: Grasped. It parses and
  binds and is refused by `require_tier_a` and by the tape.

⚠ THE OLD SPLIT WAS "Tier B needs the contact array, which is CPU-only".
That premise was wrong by the time L3 looked: `Phyics3dBatchedEnv` passes
`contacts` to `compute_reward_and_done_gpu` and the touch sensor evaluates
per lane from it. The LIBERO port needed `On(a, b)` = `b.check_ontop(a)` =
z order + CONTACT + xy < 0.03, so the contact test moved to Tier A rather
than the benchmark's most common predicate being approximated.

## THE L3 OPS, AND THE LIBERO SEMANTICS THEY CARRY

* `Joint(joint, cmp, thr)` — `qpos[joint] <cmp> thr`, cmp in lt/le/gt/ge.
  LIBERO's `Open/Close/Turnon/Turnoff` are this with a per-CLASS comparison
  (`categories.kv`); `Open` on a multi-joint object is ANY joint (`Or`),
  `Close` is ALL (`And`), quoted from `ObjectState.is_open/is_close`.
* `On(obj, obj)` — bound to `OP_ON_BODY`: `check_ontop` verbatim, with the
  ARGUMENT INVERSION LIBERO's `On.__call__` performs (`arg2.check_ontop(arg1)`)
  resolved here, once — see `eval.pred_ontop`.
* `On(obj, box region)` / `In(obj, box region)` — a region declared
  `:box:` (`spec.RegionSpec.is_box`) evaluates as LIBERO's `SiteObject.under`
  / `in_box`, in the site's frame, with an optional CONTACT partner slot.

## ⚠ WHAT THIS FILE DOES NOT DO

It does not evaluate. Evaluation reads `Data` and belongs with the env wiring;
keeping the LANGUAGE separate is what lets the parser be gated without a
physics step, and what stops a second half-parser appearing in a kernel.
"""

from noeira.core.kv import split_on
from .spec import FamilySpec


# ── ops ────────────────────────────────────────────────────────────────────
# ⚠ THE VALUES ARE THE WIRE FORMAT. A `BoundGoal` becomes a device tape in P3,
# so these numbers end up in GPU memory. Append; do not renumber.
comptime OP_IN: Int = 0
comptime OP_ON: Int = 1
comptime OP_NEAR: Int = 2
# ⚠ `Above(a, b, margin)` — THREE ARGUMENTS, AND THE MARGIN IS WHY. Comparing
# body ORIGINS alone made `Above(brick, table)` TRUE AT RESET: a brick resting
# on a table's top face is already above the table's origin, so the "lift"
# task was satisfied before the arm moved. Found by the viewer's `--check`,
# not by a test — every gate had only ever asked whether `Above` fired, never
# whether the TASK it expressed was non-trivial.
comptime OP_ABOVE: Int = 3
comptime OP_UPRIGHT: Int = 4
comptime OP_OPEN: Int = 5
comptime OP_AT_REGION: Int = 6
# Contact readers. ⚠ TOUCHING IS TIER A SINCE L3 (see the header); GRASPED
# stays Tier B — it has no defined semantics and is refused everywhere.
comptime OP_TOUCHING: Int = 7
comptime OP_GRASPED: Int = 8
# Composition.
comptime OP_AND: Int = 9
comptime OP_OR: Int = 10
comptime OP_NOT: Int = 11
# L3 — appended, never renumbered (the values are in `meta` on device).
# `Joint(joint, cmp, thr)`: a = qpos address, b = CMP_* code, param = thr.
comptime OP_JOINT: Int = 12
# `On(obj, obj)` after binding: a = subject root body, b = target root body.
# ⚠ THE PARSER NEVER EMITS THIS — `On` parses to OP_ON and `bind_goal`
# picks OP_ON_BODY when the second name is a SLOT and not a region.
comptime OP_ON_BODY: Int = 13

comptime OP_COUNT: Int = 14

# ── comparison codes for `Joint` ───────────────────────────────────────────
# ⚠ ALSO THE WIRE FORMAT: `b` of an OP_JOINT term. `libero_categories.kv`
# spells them lt/le/gt/ge and imports these, so the table and the language
# cannot disagree on a code.
comptime CMP_NONE: Int = -1
comptime CMP_LT: Int = 0
comptime CMP_LE: Int = 1
comptime CMP_GT: Int = 2
comptime CMP_GE: Int = 3


def cmp_from_name(s: String) raises -> Int:
    if s == "lt":
        return CMP_LT
    if s == "le":
        return CMP_LE
    if s == "gt":
        return CMP_GT
    if s == "ge":
        return CMP_GE
    raise Error(
        "tasks: unknown comparison '" + s + "'. Known: lt, le, gt, ge."
    )


def cmp_name(op: Int) -> String:
    if op == CMP_LT:
        return String("lt")
    if op == CMP_LE:
        return String("le")
    if op == CMP_GT:
        return String("gt")
    if op == CMP_GE:
        return String("ge")
    return String("none")

# ⚠ A CAP, AND IT IS A DEVICE-SIDE ONE. P3's tape is `[N_TASKS, MAX_TERMS, 4]`
# and must be comptime-sized, so a goal that needs more terms than this cannot
# be expressed rather than silently truncated. Sixteen is far past anything
# LIBERO expresses (its goals are one or two predicates under an And).
comptime MAX_GOAL_TERMS: Int = 16


def op_from_name(s: String) raises -> Int:
    if s == "In":
        return OP_IN
    if s == "On":
        return OP_ON
    if s == "Near":
        return OP_NEAR
    if s == "Above":
        return OP_ABOVE
    if s == "Upright":
        return OP_UPRIGHT
    if s == "Open":
        return OP_OPEN
    if s == "AtRegion":
        return OP_AT_REGION
    if s == "Touching":
        return OP_TOUCHING
    if s == "Grasped":
        return OP_GRASPED
    if s == "And":
        return OP_AND
    if s == "Or":
        return OP_OR
    if s == "Not":
        return OP_NOT
    if s == "Joint":
        return OP_JOINT
    raise Error(
        "tasks: unknown predicate '" + s + "'. Known: In, On, Near, Above,"
        " Upright, Open, AtRegion, Joint, Touching, Grasped, And, Or, Not."
    )


def op_name(op: Int) -> String:
    if op == OP_IN:
        return String("In")
    if op == OP_ON:
        return String("On")
    if op == OP_NEAR:
        return String("Near")
    if op == OP_ABOVE:
        return String("Above")
    if op == OP_UPRIGHT:
        return String("Upright")
    if op == OP_OPEN:
        return String("Open")
    if op == OP_AT_REGION:
        return String("AtRegion")
    if op == OP_TOUCHING:
        return String("Touching")
    if op == OP_GRASPED:
        return String("Grasped")
    if op == OP_AND:
        return String("And")
    if op == OP_OR:
        return String("Or")
    if op == OP_JOINT:
        return String("Joint")
    if op == OP_ON_BODY:
        return String("On")
    return String("Not")


def op_arity(op: Int) -> Int:
    """How many arguments the predicate takes, INCLUDING its numeric one."""
    if op == OP_NEAR or op == OP_ABOVE or op == OP_JOINT:
        return 3
    if op == OP_GRASPED or op == OP_NOT:
        return 1
    return 2


def op_is_tier_a(op: Int) -> Bool:
    """Can this run on device, reading only per-lane `Data`?

    ⚠ SINCE L3 EVERYTHING BUT GRASPED. The contact list is per lane in the
    reward kernel (header), so Touching and `On(obj, obj)` read it there.
    """
    return op != OP_GRASPED


def op_reads_contacts(op: Int) -> Bool:
    """Does the op read the lane's contact list? (Box regions may too — that
    is a REGION property, `RegionSpec.contact`, not an op property)."""
    return op == OP_TOUCHING or op == OP_GRASPED or op == OP_ON_BODY


def op_is_composite(op: Int) -> Bool:
    return op == OP_AND or op == OP_OR or op == OP_NOT


def op_takes_region(op: Int) -> Bool:
    """Is the SECOND argument a region name rather than a slot name?"""
    return op == OP_IN or op == OP_ON or op == OP_AT_REGION


def op_takes_number(op: Int) -> Bool:
    """Does the LAST argument parse as a number rather than a name?

    ⚠ `Near(a, b, d)` has THREE, the last numeric; `Upright(obj, tol)` and
    `Open(joint, frac)` have two, the last numeric. Getting this table wrong
    reads a distance as a body name and raises with a confusing message.
    """
    return (
        op == OP_NEAR or op == OP_UPRIGHT or op == OP_OPEN or op == OP_ABOVE
        or op == OP_JOINT
    )


# ── the parsed, still-symbolic tree ────────────────────────────────────────


struct GoalTerm(Copyable, ImplicitlyCopyable, Movable):
    """One node. Leaves carry NAMES; composites carry child term indices.

    ⚠ POST-ORDER: a child's index is always LESS than its parent's, so the
    root is the last term and an evaluator can sweep the array forwards with
    no stack. That is what makes this shape usable inside a GPU kernel.
    """

    var op: Int
    var arg0: String
    var arg1: String
    var param: Float64
    var kid0: Int
    var kid1: Int

    def __init__(out self, op: Int):
        self.op = op
        self.arg0 = String("")
        self.arg1 = String("")
        self.param = 0.0
        self.kid0 = -1
        self.kid1 = -1


struct Goal(Movable & Deinitable):
    var terms: List[GoalTerm]

    def __init__(out self):
        self.terms = List[GoalTerm]()

    def __init__(out self, *, deinit move: Self):
        self.terms = move.terms^

    def root(self) -> Int:
        return len(self.terms) - 1

    def describe(self) raises -> String:
        return self._describe(self.root())

    def _describe(self, i: Int) raises -> String:
        ref t = self.terms[i]
        var s = op_name(t.op) + "("
        if op_is_composite(t.op):
            s += self._describe(t.kid0)
            if t.kid1 >= 0:
                s += ", " + self._describe(t.kid1)
        else:
            s += t.arg0
            if t.arg1.byte_length() > 0:
                s += ", " + t.arg1
            if op_takes_number(t.op):
                s += ", " + String(t.param)
        return s + ")"


struct _Parser(Movable & Deinitable):
    var src: String
    var pos: Int
    var terms: List[GoalTerm]

    def __init__(out self, src: String):
        self.src = src
        self.pos = 0
        self.terms = List[GoalTerm]()

    def __init__(out self, *, deinit move: Self):
        self.src = move.src^
        self.pos = move.pos
        self.terms = move.terms^

    def _at(self) -> String:
        if self.pos >= self.src.byte_length():
            return String("")
        return String(self.src[byte = self.pos : self.pos + 1])

    def _skip_ws(mut self):
        while self.pos < self.src.byte_length():
            var c = self._at()
            if c != " " and c != "\t" and c != "\n" and c != "\r":
                break
            self.pos += 1

    def _fail(self, why: String) raises:
        raise Error(
            "tasks: bad goal at offset " + String(self.pos) + ": " + why
            + " — in '" + self.src + "'"
        )

    def _ident(mut self) raises -> String:
        """A bare token: a name, a number, or a predicate head."""
        self._skip_ws()
        var start = self.pos
        while self.pos < self.src.byte_length():
            var c = self._at()
            if (
                c == "(" or c == ")" or c == "," or c == " " or c == "\t"
                or c == "\n" or c == "\r"
            ):
                break
            self.pos += 1
        if self.pos == start:
            self._fail(String("expected a name"))
        return String(self.src[byte=start : self.pos])

    def _expect(mut self, c: String) raises:
        self._skip_ws()
        if self._at() != c:
            self._fail(String("expected '") + c + "'")
        self.pos += 1

    def parse_expr(mut self) raises -> Int:
        """Parse one predicate; append its term; return the term's index."""
        if len(self.terms) >= MAX_GOAL_TERMS:
            raise Error(
                "tasks: goal has more than " + String(MAX_GOAL_TERMS)
                + " terms. That cap is a DEVICE-SIDE one — P3's tape is"
                " comptime-sized — so a longer goal cannot be expressed"
                " rather than silently truncated."
            )
        var head = self._ident()
        var op = op_from_name(head)
        self._expect(String("("))

        var t = GoalTerm(op)
        if op_is_composite(op):
            # ⚠ CHILDREN ARE PARSED FIRST, so their indices are lower than the
            # parent's and the array stays post-order. See GoalTerm.
            t.kid0 = self.parse_expr()
            if op != OP_NOT:
                self._expect(String(","))
                t.kid1 = self.parse_expr()
        else:
            var n = op_arity(op)
            t.arg0 = self._ident()
            if n >= 2:
                self._expect(String(","))
                if op_takes_number(op) and n == 2:
                    t.param = Float64(String(self._ident().strip()))
                else:
                    t.arg1 = self._ident()
            if n == 3:
                self._expect(String(","))
                t.param = Float64(String(self._ident().strip()))
        self._expect(String(")"))
        self.terms.append(t^)
        return len(self.terms) - 1


def parse_goal(text: String) raises -> Goal:
    """`In(brick, box_inside)` -> a post-order term tree.

    ⚠ TRAILING TEXT IS AN ERROR, not ignored. `In(a,b) On(c,d)` is two goals
    with no operator between them, and silently keeping the first would make a
    task succeed on half of what it says.
    """
    var p = _Parser(text)
    _ = p.parse_expr()
    p._skip_ws()
    # ⚠ RAISED INLINE, NOT VIA `p._fail`. Calling a method on `p` here and
    # then moving `p.terms` out below makes the compiler see a value that is
    # partially moved on one path and whole on the other — "field 'p.terms'
    # destroyed out of the middle of a value". Cheaper to spell the error.
    var leftover = p.pos != p.src.byte_length()
    var g = Goal()
    # ⚠ `.copy()`, NOT `^`. `_Parser` has a destructor, so moving a field out
    # leaves a partially-moved value the compiler refuses to destroy. The list
    # is capped at MAX_GOAL_TERMS (16) and this runs once at load time, so the
    # copy is free in every sense that matters.
    g.terms = p.terms.copy()
    if leftover:
        raise Error(
            "tasks: trailing text after the goal in '" + text + "'. Two"
            " predicates with no operator between them is not a goal —"
            " combine them with And(...) or Or(...)."
        )
    return g^


# ── the bound, numeric program — what a kernel would read ──────────────────


struct BoundTerm(Copyable, ImplicitlyCopyable, Movable):
    """`(op, a, b, param)`. No strings, no allocation, fixed width.

    `a` / `b` mean different things per op, and the table is the ABI:

        In / On / AtRegion   a = body id,   b = region index
        Near / Above         a = body id,   b = body id
        Upright              a = body id,   b = -1        param = tolerance
        Open                 a = joint id,  b = -1        param = fraction
        Touching             a = body id,   b = body id   (slot ROOTS — the
                             evaluator walks `body_parent` so any body of
                             either slot counts, as robosuite's
                             `check_contact` over a model's contact_geoms)
        On (OP_ON_BODY)      a = body id,   b = body id   (roots, as above)
        Joint                a = qpos adr,  b = CMP_*     param = threshold
        Grasped              a = body id,   b = -1        (Tier B)
        And / Or             a = term idx,  b = term idx
        Not                  a = term idx,  b = -1
    """

    var op: Int
    var a: Int
    var b: Int
    var param: Float64

    def __init__(out self, op: Int, a: Int, b: Int, param: Float64):
        self.op = op
        self.a = a
        self.b = b
        self.param = param


struct BoundGoal(Movable & Deinitable):
    var terms: List[BoundTerm]

    def __init__(out self):
        self.terms = List[BoundTerm]()

    def __init__(out self, *, deinit move: Self):
        self.terms = move.terms^

    def root(self) -> Int:
        return len(self.terms) - 1

    def is_tier_a(self) -> Bool:
        for i in range(len(self.terms)):
            if not op_is_tier_a(self.terms[i].op):
                return False
        return True


def slot_body_id(slot: String, body_names: List[String]) raises -> Int:
    """The body a slot's instance contributes, by `Model` body id.

    ⚠⚠ THE PREFIX IS THE IDENTITY, AND THIS IS WHERE IT CASHES IN. §2.1: the
    `<attach prefix=>` string is the instance identity, so slot `brick` owns
    every body whose name begins `brick_` — in the composed SO-101 tabletop
    that is `brick_cube`, from the body named `cube` inside `cube.xml`.

    ⚠ THE LOWEST MATCHING ID IS THE SLOT'S ROOT. Bodies are emitted in tree
    order, so for a multi-body asset the first match is the root — the one a
    free joint attaches to and the one a goal means when it says "the brick".

    ⚠ `body_names[0]` IS THE WORLDBODY and these ids include it, matching
    `FlatModelDef.body_names`' own convention. Off-by-one here would name the
    wrong object and still resolve, which is the silent kind.

    ⚠ A NAME THAT IS NO PREFIX MAY BE A COMPOSED BODY'S OWN NAME. A base model
    can bring a movable body of its own (the G1 room's crate is `robot_box`,
    a free body of the base beside the robot, in no slot), and a goal about
    it had no way to say so. The prefix is tried FIRST, so every name that
    resolved before resolves to the same body; only a name that used to raise
    can now bind, to the one body spelled exactly that way. It is what `Joint`
    and `AtRegion` already do for joints and sites: name the composed element.
    Such a body is its own root for `Touching`: `eval.body_in_slot` walks up
    from a contact's body and stops at it.
    """
    var want = slot + "_"
    for i in range(len(body_names)):
        if String(body_names[i]).startswith(want):
            return i
    for i in range(len(body_names)):
        if String(body_names[i]) == slot:
            return i
    raise Error(
        "tasks: slot '" + slot + "' has no body in the composed scene — no"
        " body name starts with '" + want + "' and none is named '" + slot
        + "'. Either the slot is not in the family's table, or the scene is"
        " stale (`pixi run gen-family-scenes`)."
    )


def site_id(name: String, site_names: List[String]) raises -> Int:
    for i in range(len(site_names)):
        if String(site_names[i]) == name:
            return i
    raise Error(
        "tasks: no site named '" + name + "' in the composed scene. ⚠ A"
        " region names the COMPOSED site — `<slot>_<site in the asset>` —"
        " because `<attach prefix=>` renames every element it splices."
    )


def joint_qpos_addresses(joint_nq: List[Int]) -> List[Int]:
    """Each joint's first `qpos` index, from the per-joint `nq` in joint
    order — `FlatModelDef.joints[i].nq`. The runtime parser carries no
    address column; the joints are in qpos order, so the prefix sum IS the
    address, the same rule `reset.free_slot_addresses` applies."""
    var out = List[Int]()
    var adr = 0
    for i in range(len(joint_nq)):
        out.append(adr)
        adr += joint_nq[i]
    return out^


def joint_id(name: String, joint_names: List[String]) raises -> Int:
    for i in range(len(joint_names)):
        if String(joint_names[i]) == name:
            return i
    raise Error(
        "tasks: no joint named '" + name + "' in the composed scene. ⚠ A"
        " goal names the COMPOSED joint — `<slot>_<joint in the asset>` —"
        " because `<attach prefix=>` renames every element it splices."
    )


def bind_goal(
    g: Goal,
    f: FamilySpec,
    body_names: List[String],
    site_names: List[String],
) raises -> BoundGoal:
    """The pre-L3 signature: no joint table. A goal with a `Joint` term
    RAISES here — pass `joint_names` and `joint_qpos_adr` (see
    `joint_qpos_addresses`) through the wide overload."""
    var no_names = List[String]()
    var no_adr = List[Int]()
    return bind_goal(g, f, body_names, site_names, no_names, no_adr)


def bind_goal(
    g: Goal,
    f: FamilySpec,
    body_names: List[String],
    site_names: List[String],
    joint_names: List[String],
    joint_qpos_adr: List[Int],
) raises -> BoundGoal:
    """Resolve every name to an index. Runs ONCE, on the host, at load time.

    ⚠ REGIONS BIND TO THEIR INDEX IN THE FAMILY, not to a site id. The region
    carries its own site plus a rectangle, and the evaluator needs both; a
    goal term only has to say WHICH region. Resolving the site here would
    throw away the rectangle and quietly turn `In` into `AtSite`.

    ⚠ `On`'s SECOND NAME IS A REGION OR A SLOT, DECIDED HERE. A region wins;
    a name that is BOTH a region and a slot is refused rather than guessed,
    because the two evaluate differently (`pred_in_rect` / `under` against
    `check_ontop`) and a family that overloads a name has said nothing about
    which it meant. `In(obj, slot)` is refused outright: LIBERO's
    `ObjectState.check_contain` calls `object.in_box`, which no object
    class in its corpus defines — only sites and target zones have one.
    """
    var out = BoundGoal()
    for i in range(len(g.terms)):
        ref t = g.terms[i]
        # ⚠ `-1` IS THE ABI's "no second argument", not a placeholder — see
        # BoundTerm's table. Every branch below sets `a`; `b` keeps this value
        # for the unary ops (Upright, Grasped, Not), and a kernel reads it as
        # "absent".
        var a: Int
        var b = -1
        var op = t.op
        if op_is_composite(t.op):
            a = t.kid0
            b = t.kid1
        elif t.op == OP_OPEN:
            # A joint fraction of its range. Left unresolved: LIBERO's four
            # articulation spellings are absolute thresholds with a per-class
            # direction, which `Joint(joint, cmp, thr)` carries exactly; a
            # range fraction has no user yet and would need the joint range
            # here. Refused loudly rather than half-bound.
            raise Error(
                "tasks: Open(joint, frac) is not bound — use"
                " Joint(<joint>, lt|le|gt|ge, <threshold>), which is what"
                " LIBERO's Open/Close/Turnon/Turnoff translate to."
            )
        elif t.op == OP_JOINT:
            if len(joint_names) == 0:
                raise Error(
                    "tasks: Joint(" + t.arg0 + ", ...) needs the joint table"
                    " — call bind_goal with joint_names and joint_qpos_adr"
                    " (predicates.joint_qpos_addresses)."
                )
            var jid = joint_id(t.arg0, joint_names)
            if jid >= len(joint_qpos_adr):
                raise Error(
                    "tasks: joint_qpos_adr has " + String(len(joint_qpos_adr))
                    + " entries but joint '" + t.arg0 + "' is index "
                    + String(jid)
                )
            a = joint_qpos_adr[jid]
            b = cmp_from_name(t.arg1)
        elif t.op == OP_AT_REGION:
            # ⚠ AT_REGION's FIRST ARGUMENT IS A SITE, NOT A SLOT. It is what
            # asks "is the gripper over the drop zone" — `robot_gripperframe`
            # is a site on the robot and belongs to no slot at all. Binding it
            # through `slot_body_id` like every other op would look for a body
            # named `robot_gripperframe_*` and raise on a goal that is
            # perfectly well formed.
            a = site_id(t.arg0, site_names)
            b = f.region_index(t.arg1)
            if b < 0:
                raise Error(
                    "tasks: AtRegion names region '" + t.arg1 + "', which"
                    " family '" + f.name + "' does not declare"
                )
        else:
            a = slot_body_id(t.arg0, body_names)
            if op_takes_region(t.op):
                var ri = f.region_index(t.arg1)
                var si = f.slot_index(t.arg1)
                if ri >= 0 and si >= 0:
                    raise Error(
                        "tasks: '" + t.arg1 + "' is BOTH a region and a slot"
                        " of family '" + f.name + "'; " + op_name(t.op)
                        + "(" + t.arg0 + ", " + t.arg1 + ") is ambiguous."
                        " Rename one."
                    )
                if ri >= 0:
                    b = ri
                elif si >= 0 and t.op == OP_ON:
                    op = OP_ON_BODY
                    b = slot_body_id(t.arg1, body_names)
                elif si >= 0:
                    raise Error(
                        "tasks: In(" + t.arg0 + ", " + t.arg1 + ") names a"
                        " SLOT as its container. LIBERO defines In only"
                        " against a site or a target zone (`in_box`); an"
                        " object has no interior. Declare a `:box:` region"
                        " on the container's site instead."
                    )
                else:
                    raise Error(
                        "tasks: goal names region '" + t.arg1 + "', which"
                        " family '" + f.name + "' does not declare"
                    )
            elif t.arg1.byte_length() > 0:
                b = slot_body_id(t.arg1, body_names)
        out.terms.append(BoundTerm(op, a, b, t.param))
    return out^


def require_tier_a(g: BoundGoal, task_name: String) raises:
    """⚠ `TASK_LAYER_PLAN.md` §5.1's RULE, made real. P3 calls this.

    Since L3 the only Tier B op is `Grasped`, which has no semantics defined
    anywhere: evaluating it would return a constant, and a constant goal
    trains against a flat reward with every curve looking healthy.
    """
    for i in range(len(g.terms)):
        if not op_is_tier_a(g.terms[i].op):
            raise Error(
                "task '" + task_name + "': goal uses "
                + op_name(g.terms[i].op) + ", which is TIER B — it has no"
                " defined evaluation (neither host nor device). Spell the"
                " grasp as Touching(obj, <finger slot>) or a Joint term."
            )

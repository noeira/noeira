"""The KERNEL loop and the HOST loop agree — P3b's gate.

`tape.eval_tape` reads `List[Float64]`. `gpu_eval.eval_tape_gpu` reads
`LayoutTensor`s, because that is what a reward kernel holds. Two loops, one set
of predicates, and this is where they are pinned against each other.

⚠ RUN ON THE CPU, AT float64, ON PURPOSE. `eval_tape_gpu` is generic over
dtype and over the tensor's origin, so instantiating it here needs no GPU —
which means the LOOP can be gated on any machine, separately from the
device-vs-host DTYPE question. Isolating the two is the point: a disagreement
found here is an indexing bug, and one found in P3c is a rounding one.

⚠⚠ THE GRID MUST MAKE EACH GOAL FLIP, and the gate asserts it. Two loops
agreeing on states where every goal is False is what two `return False`s also
achieve.

⚠ AND THE TAPE IS WRITTEN INTO `meta` AT ITS REAL OFFSET — `env *
METADATA_SIZE + META_IDX_TASK_PARAM_0` — with a SECOND LANE carrying a
DIFFERENT task. A loop that ignored `env` and read lane 0's tape would pass
every single-lane test ever written; this is the smallest arrangement that
catches it, and it is the same defect P3c's negative leg exists for at 1024
lanes.

Run: pixi run mojo run -I . tests/tasks/test_tape_gpu_parity.mojo
"""

from noeira.core.tiles import RowMajor
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, MODEL_CURRICULUM_SIZE,
)
from noeira.tasks.spec import load_family, load_task
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TabletopConfig
from noeira.tasks.predicates import parse_goal, bind_goal
from noeira.tasks.eval import (
    region_sites, region_rects, region_half_heights,
)
from noeira.tasks.tape import encode_goal, eval_tape, TAPE_WORDS
from noeira.tasks.gpu_eval import eval_tape_gpu, region_table_words
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.so101_tabletop_xml import So101TabletopModel


comptime DTYPE = DType.float64
comptime BATCH = 2

# ⚠ COMPTIME, FROM THE FAMILY'S COMPILE UNIT. A `LayoutTensor`'s shape is a
# type parameter, so these cannot come from `len(fmd.body_names)` — and taking
# them from `So101TabletopModel` is the right source anyway: it is the same
# CI-checked `*_dims.mojo` the batched env would be built from. The runtime
# parse is asserted against them below, so a stale scene is a failure here
# rather than an out-of-bounds read.
comptime NB = So101TabletopModel.NBODY
comptime NS = So101TabletopModel.NSITE


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


def main() raises:
    print("=== kernel loop vs host loop — P3b ===")
    var ta = Tally()

    var f = load_family("noeira/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var nb = NB
    var ns = NS
    if len(fmd.body_names) != NB or len(fmd.site_names) != NS:
        raise Error(
            "tape parity: the composed scene has "
            + String(len(fmd.body_names)) + " bodies / "
            + String(len(fmd.site_names)) + " sites but the model def says "
            + String(NB) + " / " + String(NS) + ". Run `pixi run"
            " gen-family-scenes && pixi run gen-dims`."
        )

    # ⚠ TWO LANES, TWO DIFFERENT TASKS. See the header.
    var t0 = load_task("noeira/tasks/tasks/so101_gather_bricks.task")
    var t1 = load_task("noeira/tasks/tasks/so101_lift_brick.task")
    var g0 = bind_goal(parse_goal(t0.goal), f, fmd.body_names, fmd.site_names)
    var g1 = bind_goal(parse_goal(t1.goal), f, fmd.body_names, fmd.site_names)
    var tp0 = encode_goal(g0)
    var tp1 = encode_goal(g1)
    print("  lane 0:", t0.goal)
    print("  lane 1:", t1.goal)

    # host-side flat region table
    var r_site = List[Int]()
    var r_x0 = List[Float64]()
    var r_y0 = List[Float64]()
    var r_x1 = List[Float64]()
    var r_y1 = List[Float64]()
    # ⚠ THE PER-REGION Z BAND, which used to be `eval.IN_HALF_HEIGHT` inside
    # both evaluators. Built here so the host tape reader gets exactly what
    # `region_table_words` writes for the device.
    var r_h = List[Float64]()
    var rheights = region_half_heights(f)
    for i in range(len(f.regions)):
        r_h.append(rheights[i])
        r_site.append(rsites[i])
        r_x0.append(rects[i][0])
        r_y0.append(rects[i][1])
        r_x1.append(rects[i][2])
        r_y1.append(rects[i][3])

    # ── the tensors a kernel would be handed ──────────────────────────────
    comptime L_META = RowMajor[BATCH, METADATA_SIZE]
    comptime L_CUR = RowMajor[1, MODEL_CURRICULUM_SIZE]
    var meta = TensorImpl[DTYPE].alloc(BATCH * METADATA_SIZE)
    var cur = TensorImpl[DTYPE].alloc(MODEL_CURRICULUM_SIZE)
    for i in range(BATCH * METADATA_SIZE):
        meta.data[i] = Scalar[DTYPE](0)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for i in range(MODEL_CURRICULUM_SIZE):
        cur.data[i] = Scalar[DTYPE](cw[i])
    for k in range(TAPE_WORDS):
        meta.data[0 * METADATA_SIZE + META_IDX_TASK_PARAM_0 + k] = Scalar[DTYPE](tp0[k])
        meta.data[1 * METADATA_SIZE + META_IDX_TASK_PARAM_0 + k] = Scalar[DTYPE](tp1[k])

    var xpos = TensorImpl[DTYPE].alloc(BATCH * nb * 3)
    var xquat = TensorImpl[DTYPE].alloc(BATCH * nb * 4)
    var sxp = TensorImpl[DTYPE].alloc(BATCH * ns * 3)

    var total = 0
    var agreed = 0
    var t0_true = 0
    var t1_true = 0

    for ix in range(5):
        for iz in range(5):
            var px = 0.10 + Float64(ix) * 0.09
            var pz = 0.00 + Float64(iz) * 0.045

            # host arrays (identical content in both lanes)
            var hxp = List[Float64]()
            for _ in range(nb * 3):
                hxp.append(0.0)
            var hxq = List[Float64]()
            for _ in range(nb):
                hxq.append(0.0)
                hxq.append(0.0)
                hxq.append(0.0)
                hxq.append(1.0)
            var hsp = List[Float64]()
            for _ in range(ns * 3):
                hsp.append(0.0)
            var ts = rsites[0]
            hsp[ts * 3] = 0.25
            hsp[ts * 3 + 2] = 0.02
            for b in range(nb):
                hxp[b * 3] = px
                hxp[b * 3 + 2] = pz
            for b in range(nb):
                if String(fmd.body_names[b]).startswith("table_"):
                    hxp[b * 3] = 0.25
                    hxp[b * 3 + 1] = 0.0
                    hxp[b * 3 + 2] = 0.01
            # ⚠ `cube_a` PINNED — lane 0's goal is `Near(brick, cube_a)`, a
            # RELATIVE predicate, and the loop above puts every body at the
            # same swept point. Left unpinned it is constant TRUE at distance
            # zero and the FLIP check below reports 25 of 25. Same fix, same
            # reason, as `tests/tasks/test_task_tape.mojo`.
            for b in range(nb):
                if String(fmd.body_names[b]).startswith("cube_a_"):
                    hxp[b * 3] = 0.25
                    hxp[b * 3 + 1] = 0.0
                    hxp[b * 3 + 2] = 0.04
            for si in range(ns):
                if String(fmd.site_names[si]) == "robot_gripperframe":
                    hsp[si * 3] = px
                    hsp[si * 3 + 2] = pz

            for e in range(BATCH):
                for i in range(nb * 3):
                    xpos.data[e * nb * 3 + i] = Scalar[DTYPE](hxp[i])
                for i in range(nb * 4):
                    xquat.data[e * nb * 4 + i] = Scalar[DTYPE](hxq[i])
                for i in range(ns * 3):
                    sxp.data[e * ns * 3 + i] = Scalar[DTYPE](hsp[i])

            var mv = meta.tt["cpu", BATCH, METADATA_SIZE]()
            var cv = cur.tt["cpu", 1, MODEL_CURRICULUM_SIZE]()
            var xv = xpos.tt["cpu", BATCH, NB * 3]()
            var qv = xquat.tt["cpu", BATCH, NB * 4]()
            var sv = sxp.tt["cpu", BATCH, NS * 3]()

            var h0 = eval_tape(
                tp0, 0, hxp, hxq, hsp, r_site, r_x0, r_y0, r_x1, r_y1, r_h
            )
            var h1 = eval_tape(
                tp1, 0, hxp, hxq, hsp, r_site, r_x0, r_y0, r_x1, r_y1, r_h
            )
            var d0 = eval_tape_gpu[DTYPE, BATCH, NB, NS * 3](
                mv, cv, xv, qv, sv, 0
            )
            var d1 = eval_tape_gpu[DTYPE, BATCH, NB, NS * 3](
                mv, cv, xv, qv, sv, 1
            )
            total += 2
            if h0 == d0:
                agreed += 1
            if h1 == d1:
                agreed += 1
            if h0:
                t0_true += 1
            if h1:
                t1_true += 1

    print("  lane 0 (gather) satisfied in", t0_true, "of 25 states")
    print("  lane 1 (lift)   satisfied in", t1_true, "of 25 states")
    ta.check(t0_true > 0 and t0_true < 25, "lane 0's goal FLIPS in the sweep")
    ta.check(t1_true > 0 and t1_true < 25, "lane 1's goal FLIPS in the sweep")
    # ⚠⚠ THE TWO LANES MUST DISAGREE SOMEWHERE, or a loop that ignored `env`
    # and always read lane 0's tape would pass everything above.
    ta.check(
        t0_true != t1_true,
        "the two lanes run DIFFERENT goals (per-lane tape is read per lane)",
    )
    print("  ", agreed, "of", total, "evaluations agree")
    ta.check(agreed == total, "kernel loop == host loop on EVERY state")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "tape gpu parity: " + String(ta.failures) + " of "
            + String(ta.checks) + " failed"
        )
    print("=== PASS ===")

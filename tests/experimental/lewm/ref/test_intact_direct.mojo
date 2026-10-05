"""G8b-0 — INTACT's actor and Direct plan against torch, before any board.

docs/LEWM_REOPEN_PLAN.md P8. `tools/lewm/intact_reference.py` converted the
history-free PushT checkpoint (`INTACT-no-previous-action`, train seed 0) into
our names and ran, in float64, on the first fixture pairs:

  * the encoder on the start and goal frames (z0, zg);
  * the actor on (z0, zg − z0): mean and clamped log σ;
  * the 5-block Direct plan in both context pairings (`intact.mojo`):
    INTACT's own rollout (unaligned) and the training pairing (aligned).

Ours must match in float32, measured as max |ours − torch| / std(torch) per
quantity (the P2 gates' unit). Step 0 of both plans is the same action;
later steps carry the predictor's rollout.

    O=~/.cache/noeira/lewm_pusht/intact/dump_pusht_s0
    pixi run mojo run -I . tools/lewm/list_ref_params.mojo > $O/ours_names.tsv
    pixi run -e act-ref python tools/lewm/intact_reference.py --out $O --ckpt <weights_epoch_1.pt>
    pixi run -e apple mojo run -I . tests/experimental/lewm/ref/test_intact_direct.mojo $O
"""

from std.sys import argv
from std.os import getenv
from std.math import abs, sqrt
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_load import load_ref
from noeira.experimental.lewm.ref_rollout import RefEncoder, encode_ref, REF_EMB, REF_ACT
from noeira.experimental.lewm.paper_pairs import imagenet_from_hwc255, PAIR_HW
from noeira.experimental.lewm.intact import IntactDirect


comptime HORIZON = 5
comptime TOL = 1e-3
"""std units; float32 vs a float64 reference through a ViT + 5 predictor
steps (the P2 / P3 gates sit at ~1e-5 per stage)."""


def _err(got: List[Scalar[DT]], want: List[Scalar[DT]], off: Int, n: Int) -> Float64:
    var mu = 0.0
    for i in range(n):
        mu += Float64(want[off + i])
    mu /= Float64(n)
    var var_ = 0.0
    for i in range(n):
        var_ += (Float64(want[off + i]) - mu) ** 2
    var sd = sqrt(var_ / Float64(n))
    var worst = 0.0
    for i in range(n):
        worst = max(worst, abs(Float64(got[i]) - Float64(want[off + i])))
    return worst / max(sd, 1e-12)


def _run[target: StaticString](dump: String, fixture: String, ctx: Optional[DeviceContext]) raises -> Int:
    var rd = RefDump(dump)
    var fx = RefDump(fixture)
    var z0_ref = rd.get(String("intact.z0"))
    var zg_ref = rd.get(String("intact.zg"))
    var m_ref = rd.get(String("intact.mean0"))
    var s_ref = rd.get(String("intact.log_std0"))
    var pi_ref = rd.get(String("intact.plan_intact"))
    var pa_ref = rd.get(String("intact.plan_aligned"))
    var a_hist = rd.get(String("intact.a_hist"))
    var n = len(z0_ref) // REF_EMB
    var sp = fx.get(String("pairs.start_pixels"))
    var gp = fx.get(String("pairs.goal_pixels"))

    var enc = RefEncoder.make[target, Kaiming](ctx)
    _ = load_ref[target](enc, dump, String("emb.0."), ctx)
    var di = IntactDirect[target, HORIZON](dump, ctx)
    var e_z = 0.0
    var e_m = 0.0
    var e_s = 0.0
    var e_pi = 0.0
    var e_pa = 0.0
    for e in range(n):
        var z0 = encode_ref[target, 1](enc, imagenet_from_hwc255(sp, e * PAIR_HW * 3), ctx)
        var zg = encode_ref[target, 1](enc, imagenet_from_hwc255(gp, e * PAIR_HW * 3), ctx)
        e_z = max(e_z, max(_err(z0, z0_ref, e * REF_EMB, REF_EMB), _err(zg, zg_ref, e * REF_EMB, REF_EMB)))
        var m = List[Scalar[DT]]()
        for d in range(REF_EMB):
            m.append(zg[d] - z0[d])
        var o = di.actor_out(z0, m)
        var mean = List[Scalar[DT]]()
        var ls = List[Scalar[DT]]()
        for i in range(REF_ACT):
            mean.append(o[i])
            ls.append(min(Scalar[DT](2.0), max(Scalar[DT](-5.0), o[REF_ACT + i])))
        e_m = max(e_m, _err(mean, m_ref, e * REF_ACT, REF_ACT))
        e_s = max(e_s, _err(ls, s_ref, e * REF_ACT, REF_ACT))
        var pi = di.plan(z0, zg, a_hist, False)
        var pa = di.plan(z0, zg, a_hist, True)
        e_pi = max(e_pi, _err(pi, pi_ref, e * HORIZON * REF_ACT, HORIZON * REF_ACT))
        e_pa = max(e_pa, _err(pa, pa_ref, e * HORIZON * REF_ACT, HORIZON * REF_ACT))
    var worst = max(max(e_z, e_m), max(e_s, max(e_pi, e_pa)))
    var flag = String("") if worst <= TOL else String("  ✗")
    print("  --", target, ":", n, "pairs | z0/zg", e_z, "| actor mean", e_m, "log σ", e_s,
          "| plan (INTACT ctx)", e_pi, "| plan (aligned)", e_pa, flag)
    return 0 if worst <= TOL else 1


def main() raises:
    var dump = getenv("HOME") + "/.cache/noeira/lewm_pusht/intact/dump_pusht_s0"
    var fixture = getenv("HOME") + "/.cache/noeira/lewm_pusht/session_a/out/fixture"
    var args = argv()
    if len(args) > 1:
        dump = String(args[1])
    print("G8b-0  INTACT actor + Direct plan vs torch (std units, tol", TOL, ")")
    var fails = _run["cpu"](dump, fixture, None)
    var c = DeviceContext()
    fails += _run["gpu"](dump, fixture, Optional(c))
    if fails > 0:
        raise Error("FAIL G8b-0")
    print("PASS")

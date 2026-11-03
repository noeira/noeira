"""SuperMimic M3a: the per-lane G1 FK and its VJP (`deep_agents/supermimic/g1_fk`).

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/heads_oracle.py --out /tmp/sm_heads
    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_fk.mojo /tmp/sm_heads

WHAT EACH SECTION CAN CATCH
===========================
[1] the tree read from the engine: 40 bodies, one free root, 29 hinges with
    consecutive qpos / dof addresses, the rest fixed.
[2] float64 forward against MuJoCo's own FK on our XML (`xpos`, `xquat`,
    `mj_objectVelocity(mjOBJ_XBODY)`) and against the reproduction's torch
    `G1Kinematics` on the same model, 64 random states with NON-unit root
    quaternions. 1e-10 (plan M3; both references agree to ~1e-14).
[3] float32 forward against the same, 1e-5 relative to max(1, |x|).
[4] float64 VJP against torch autograd through the reproduction's FK, for
    random cotangents on all four outputs: 1e-10 relative.
[5] float64 VJP against CENTRAL DIFFERENCES of the forward (no torch): the
    directional derivative along random directions of (qpos, qvel), h = 1e-6,
    relative 1e-7. A VJP that agrees with autograd by sharing a mistake with
    it cannot also agree with finite differences of our own forward.
[6] [2]-[5] again with random hinge anchors (the oracle's `jp_` dump): our
    asset's anchors are all at the body origin, which zeroes one VJP term.
"""

from std.math import sqrt
from std.random import random_float64, seed
from std.sys import argv

from noeira.deep_agents.supermimic.g1_fk import (
    FK_DADR, FK_JPOS, FK_JTYPE, FK_QADR, FK_REC, G1_FK_FIXED, G1_FK_NB, G1_FK_NQ, G1_FK_NV, G1FKTree,
    g1_fk_lane, g1_fk_tree, g1_fk_vjp_lane,
)
from noeira.envs.robots import UnitreeG1
from noeira.io.npy import load_npy_2d_f64
from noeira.physics3d.joint_types import JNT_FREE, JNT_HINGE


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


@always_inline
def _p[dt: DType](mut l: List[Scalar[dt]], off: Int = 0) -> Pointer[Scalar[dt], MutAnyOrigin]:
    return l.unsafe_ptr().unsafe_offset(off).as_unsafe_any_origin()


def _cast[dt: DType](src: List[Float64], off: Int, n: Int) -> List[Scalar[dt]]:
    var o = List[Scalar[dt]](length=n, fill=Scalar[dt](0))
    for i in range(n):
        o[i] = Scalar[dt](src[off + i])
    return o^


struct FKOut[dt: DType](Movable):
    var pos: List[Scalar[Self.dt]]
    var quat: List[Scalar[Self.dt]]
    var lin: List[Scalar[Self.dt]]
    var ang: List[Scalar[Self.dt]]

    def __init__(out self):
        self.pos = List[Scalar[Self.dt]](length=G1_FK_NB * 3, fill=Scalar[Self.dt](0))
        self.quat = List[Scalar[Self.dt]](length=G1_FK_NB * 4, fill=Scalar[Self.dt](0))
        self.lin = List[Scalar[Self.dt]](length=G1_FK_NB * 3, fill=Scalar[Self.dt](0))
        self.ang = List[Scalar[Self.dt]](length=G1_FK_NB * 3, fill=Scalar[Self.dt](0))

    def __init__(out self, *, deinit move: Self):
        self.pos = move.pos^
        self.quat = move.quat^
        self.lin = move.lin^
        self.ang = move.ang^


def _fk[dt: DType](mut tree: G1FKTree[dt], mut qp: List[Scalar[dt]], mut qv: List[Scalar[dt]]) -> FKOut[dt] where dt.is_floating_point():
    var o = FKOut[dt]()
    g1_fk_lane[dt](tree.ptr(), _p(qp), _p(qv), _p(o.pos), _p(o.quat), _p(o.lin), _p(o.ang))
    return o^


def _err[dt: DType](got: List[Scalar[dt]], want: List[Float64], off: Int, rel: Bool) -> Float64:
    var e = 0.0
    for i in range(len(got)):
        var w = want[off + i]
        var d = abs(Float64(got[i]) - w)
        if rel:
            d /= max(1.0, abs(w))
        e = max(e, d)
    return e


def _loss(o: FKOut[DType.float64], c: List[List[Float64]], b: Int) -> Float64:
    """Σ <c, FK> for state b."""
    var s = 0.0
    for i in range(G1_FK_NB * 3):
        s += c[0][b * G1_FK_NB * 3 + i] * o.pos[i] + c[2][b * G1_FK_NB * 3 + i] * o.lin[i] + c[3][b * G1_FK_NB * 3 + i] * o.ang[i]
    for i in range(G1_FK_NB * 4):
        s += c[1][b * G1_FK_NB * 4 + i] * o.quat[i]
    return s


def suite(
    mut t: Tally, mut tree: G1FKTree[DType.float64], mut tree32: G1FKTree[DType.float32],
    dump: String, pre: String, tag: String,
) raises:
    var qpos = load_npy_2d_f64(dump + "/" + pre + "qpos.npy", G1_FK_NQ)
    var qvel = load_npy_2d_f64(dump + "/" + pre + "qvel.npy", G1_FK_NV)
    var B = qpos.rows
    var names = List[String]()
    names.append("pos")
    names.append("quat")
    names.append("lin")
    names.append("ang")
    var widths: List[Int] = [3, 4, 3, 3]
    var mj = List[List[Float64]]()
    var tr = List[List[Float64]]()
    var cot = List[List[Float64]]()
    for k in range(4):
        mj.append(load_npy_2d_f64(dump + "/" + pre + "mj_" + names[k] + ".npy", G1_FK_NB * widths[k]).values.copy())
        tr.append(load_npy_2d_f64(dump + "/" + pre + "t_" + names[k] + ".npy", G1_FK_NB * widths[k]).values.copy())
        cot.append(load_npy_2d_f64(dump + "/" + pre + "c_" + names[k] + ".npy", G1_FK_NB * widths[k]).values.copy())

    print("\n[2]" + tag + " float64 forward,", B, "states")
    var e_mj = List[Float64](length=4, fill=0.0)
    var e_tr = List[Float64](length=4, fill=0.0)
    var outs = List[FKOut[DType.float64]]()
    for b in range(B):
        var qp = _cast[DType.float64](qpos.values, b * G1_FK_NQ, G1_FK_NQ)
        var qv = _cast[DType.float64](qvel.values, b * G1_FK_NV, G1_FK_NV)
        var o = _fk(tree, qp, qv)
        e_mj[0] = max(e_mj[0], _err(o.pos, mj[0], b * G1_FK_NB * 3, False))
        e_mj[1] = max(e_mj[1], _err(o.quat, mj[1], b * G1_FK_NB * 4, False))
        e_mj[2] = max(e_mj[2], _err(o.lin, mj[2], b * G1_FK_NB * 3, False))
        e_mj[3] = max(e_mj[3], _err(o.ang, mj[3], b * G1_FK_NB * 3, False))
        e_tr[0] = max(e_tr[0], _err(o.pos, tr[0], b * G1_FK_NB * 3, False))
        e_tr[1] = max(e_tr[1], _err(o.quat, tr[1], b * G1_FK_NB * 4, False))
        e_tr[2] = max(e_tr[2], _err(o.lin, tr[2], b * G1_FK_NB * 3, False))
        e_tr[3] = max(e_tr[3], _err(o.ang, tr[3], b * G1_FK_NB * 3, False))
        outs.append(o^)
    for k in range(4):
        t.truth(e_mj[k] <= 1e-10 and e_tr[k] <= 1e-10,
            names[k] + ": max abs err vs MuJoCo " + String(e_mj[k]) + ", vs the reproduction " + String(e_tr[k]))

    print("\n[3]" + tag + " float32 forward")
    var e32 = 0.0
    for b in range(B):
        var qp = _cast[DType.float32](qpos.values, b * G1_FK_NQ, G1_FK_NQ)
        var qv = _cast[DType.float32](qvel.values, b * G1_FK_NV, G1_FK_NV)
        var o = _fk(tree32, qp, qv)
        e32 = max(e32, _err(o.pos, mj[0], b * G1_FK_NB * 3, True))
        e32 = max(e32, _err(o.quat, mj[1], b * G1_FK_NB * 4, True))
        e32 = max(e32, _err(o.lin, mj[2], b * G1_FK_NB * 3, True))
        e32 = max(e32, _err(o.ang, mj[3], b * G1_FK_NB * 3, True))
    t.truth(e32 <= 1e-5, "all four outputs within " + String(e32) + " of MuJoCo (relative to max(1, |x|))")

    print("\n[4]" + tag + " float64 VJP against torch autograd")
    var gq_ref = load_npy_2d_f64(dump + "/" + pre + "g_qpos.npy", G1_FK_NQ)
    var gv_ref = load_npy_2d_f64(dump + "/" + pre + "g_qvel.npy", G1_FK_NV)
    var e_vjp = 0.0
    var gq_all = List[Float64](length=B * G1_FK_NQ, fill=0.0)
    var gv_all = List[Float64](length=B * G1_FK_NV, fill=0.0)
    for b in range(B):
        var qp = _cast[DType.float64](qpos.values, b * G1_FK_NQ, G1_FK_NQ)
        var qv = _cast[DType.float64](qvel.values, b * G1_FK_NV, G1_FK_NV)
        var gp = _cast[DType.float64](cot[0], b * G1_FK_NB * 3, G1_FK_NB * 3)
        var gqt = _cast[DType.float64](cot[1], b * G1_FK_NB * 4, G1_FK_NB * 4)
        var gl = _cast[DType.float64](cot[2], b * G1_FK_NB * 3, G1_FK_NB * 3)
        var ga = _cast[DType.float64](cot[3], b * G1_FK_NB * 3, G1_FK_NB * 3)
        var gq = List[Float64](length=G1_FK_NQ, fill=0.0)
        var gv = List[Float64](length=G1_FK_NV, fill=0.0)
        g1_fk_vjp_lane[DType.float64](
            tree.ptr(), _p(qp), _p(qv), _p(outs[b].pos), _p(outs[b].quat), _p(outs[b].ang),
            _p(gp), _p(gqt), _p(gl), _p(ga), _p(gq), _p(gv),
        )
        for k in range(G1_FK_NQ):
            gq_all[b * G1_FK_NQ + k] = gq[k]
            e_vjp = max(e_vjp, abs(gq[k] - gq_ref.values[b * G1_FK_NQ + k]) / max(1.0, abs(gq_ref.values[b * G1_FK_NQ + k])))
        for k in range(G1_FK_NV):
            gv_all[b * G1_FK_NV + k] = gv[k]
            e_vjp = max(e_vjp, abs(gv[k] - gv_ref.values[b * G1_FK_NV + k]) / max(1.0, abs(gv_ref.values[b * G1_FK_NV + k])))
    t.truth(e_vjp <= 1e-10, "d/dqpos and d/dqvel: max rel err " + String(e_vjp))

    print("\n[5]" + tag + " float64 VJP against central differences of our forward")
    seed(5)
    var e_fd = 0.0
    var h = 1e-6
    for b in range(16):
        var dq = List[Float64](length=G1_FK_NQ, fill=0.0)
        var dv = List[Float64](length=G1_FK_NV, fill=0.0)
        for k in range(G1_FK_NQ):
            dq[k] = random_float64(-1.0, 1.0)
        for k in range(G1_FK_NV):
            dv[k] = random_float64(-1.0, 1.0)
        var lp = List[Float64]()
        for sgn in range(2):
            var s = 1.0 if sgn == 0 else -1.0
            var qp = _cast[DType.float64](qpos.values, b * G1_FK_NQ, G1_FK_NQ)
            var qv = _cast[DType.float64](qvel.values, b * G1_FK_NV, G1_FK_NV)
            for k in range(G1_FK_NQ):
                qp[k] += s * h * dq[k]
            for k in range(G1_FK_NV):
                qv[k] += s * h * dv[k]
            lp.append(_loss(_fk(tree, qp, qv), cot, b))
        var fd = (lp[0] - lp[1]) / (2.0 * h)
        var an = 0.0
        for k in range(G1_FK_NQ):
            an += gq_all[b * G1_FK_NQ + k] * dq[k]
        for k in range(G1_FK_NV):
            an += gv_all[b * G1_FK_NV + k] * dv[k]
        e_fd = max(e_fd, abs(fd - an) / max(1.0, abs(an)))
    t.truth(e_fd <= 1e-7, "16 random directions: max rel err " + String(e_fd))



def main() raises:
    var a = argv()
    var dump = String(a[1]) if len(a) > 1 else String("/tmp/sm_heads")
    var t = Tally()
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()

    print("\n[1] the tree")
    var tree = g1_fk_tree[DType.float64](env)
    var nh = 0
    var nfix = 0
    var addr_ok = True
    for i in range(G1_FK_NB):
        var jt = Int(tree.data[i * FK_REC + FK_JTYPE])
        if jt == JNT_HINGE:
            if Int(tree.data[i * FK_REC + FK_QADR]) != 7 + nh or Int(tree.data[i * FK_REC + FK_DADR]) != 6 + nh:
                addr_ok = False
            nh += 1
        elif jt == G1_FK_FIXED:
            nfix += 1
    t.truth(Int(tree.data[FK_JTYPE]) == JNT_FREE and nh == 29 and nfix == G1_FK_NB - 30 and addr_ok,
        "1 free root, " + String(nh) + " hinges at qpos 7.. / dof 6.., " + String(nfix) + " fixed bodies")

    var tree32 = g1_fk_tree[DType.float32](env)
    suite(t, tree, tree32, dump, String(""), String(" (our asset)"))

    # Our asset (and mjlab's) puts every hinge anchor at its body origin, so
    # the FK's `P_i − anchor` lever arm is zero and its VJP term untested: a
    # reversed cross product there passed every section above (planted). The
    # oracle's `jp_` dump uses random anchors; write the same into the tree.
    print("\n[6] the same sections with random hinge anchors (+-5 cm)")
    var jp = load_npy_2d_f64(dump + "/jp_jnt_pos.npy", 3)
    var k = 0
    for i in range(G1_FK_NB):
        if Int(tree.data[i * FK_REC + FK_JTYPE]) == JNT_HINGE:
            for c in range(3):
                tree.data[i * FK_REC + FK_JPOS + c] = jp.values[k * 3 + c]
                tree32.data[i * FK_REC + FK_JPOS + c] = Float32(jp.values[k * 3 + c])
            k += 1
    t.truth(k == jp.rows, String(k) + " hinge anchors written")
    suite(t, tree, tree32, dump, String("jp_"), String(" (random anchors)"))

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_supermimic_fk: " + String(t.fails) + " failed")

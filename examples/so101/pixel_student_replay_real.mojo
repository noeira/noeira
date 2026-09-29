"""A real arm run REPLAYED through the pixel student: real pictures against
sim pictures at the same arm pose — is the difference in what it SEES?

    pixi run mojo build -I . -D DAGGER_WINDOW -D DAGGER_PX_32 -D DAGGER_JOINT_VEL \\
        -D TASK_PPO_ACT_HIST=3 -o /tmp/px_replay examples/so101/pixel_student_replay_real.mojo
    /tmp/px_replay --ckpt projects/so101-tower/policies/pixel_bowl_h3.ckpt \\
        --rec /tmp/px_rec_h3b --brick 0.188,-0.198 --bowl 0.308,0.029

`pixel_student_deploy_real.mojo --record DIR` keeps, every 16 ticks, the two
POLICY VIEWS (the exact image planes the student got, upscaled x8 to 256 px
— rounding aside, recoverable) and a per-tick CSV of the joints, velocities
and actions. At each such tick this rebuilds the student's input four ways,
with the SAME joint planes (angles and velocities from the CSV row, the
action history from the three rows before):

    real       both cameras as the arm saw them  (must reproduce the logged action)
    sim        both cameras RENDERED at that arm pose, props at --brick / --bowl
    real-o     the real overhead, the sim wrist
    real-w     the sim overhead, the real wrist

and prints the four actions. Same joints, different pictures: a large real
vs sim gap says the policy reads the scene differently — a perception gap,
and the two mixes say which camera carries it. ⚠ The rebuilt scene holds only
until the arm first touches a prop; read the early ticks.
"""

from std.os.path import exists
from std.sys import argv

from max.gpu.host import DeviceContext

from noeira.io.png import load_png_file
from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import load_params
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.physics3d.fields import Data
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.pixel_student import (
    StudentNet, N_CAMS, IN_DIM, ACT, RENDER, OVERHEAD_RENDER_W,
    OVERHEAD_RENDER_H, OBS_PX, PLANE, check_pixel_manifest, render_to_planes,
    joints_to_planes, joint_vels_to_planes, HIST_WORDS, act_hist_to_planes,
    camera_names,
)
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, make_tower_model, make_tower_renderer, tower_cameras,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.utils.fmt import col

comptime NQ = So101TowerModel.NQ


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def _forward(
    mut net: StudentNet, ref xs: List[Scalar[DT]], mut x: Tensor, mut y: Tensor,
) raises -> List[Float64]:
    for k in range(IN_DIM):
        x.data[k] = xs[k]
    net.forward["cpu", 1](TensorRefs[1](x), y, None)
    var a = List[Float64](length=ACT, fill=0.0)
    for j in range(ACT):
        a[j] = Float64(y.data[j])
    return a^


def _line(tag: String, ref a: List[Float64]) -> String:
    var s = String("    ") + tag
    for j in range(ACT):
        s += " " + col(a[j], 6, 2)
    return s


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var ckpt = _arg(args, "--ckpt", "projects/so101-tower/policies/pixel_bowl_h3.ckpt")
    var rec = _arg(args, "--rec", "")
    var brick_xy = _arg(args, "--brick", "")
    var bowl_xy = _arg(args, "--bowl", "")
    if rec.byte_length() == 0 or brick_xy.byte_length() == 0 or bowl_xy.byte_length() == 0:
        raise Error("usage: --ckpt C --rec DIR --brick x,y --bowl x,y")
    var man_path = String(ckpt[byte = 0 : ckpt.byte_length() - 5]) + ".norm.json"
    if not exists(man_path):
        man_path = ckpt[byte = 0 : ckpt.rfind("/")] + "/norm.json"
    _ = check_pixel_manifest(man_path)

    # the run's per-tick rows: t_s, q0..5, qd0..5, a0..5, tgt0..5
    var rows = List[List[Float64]]()
    with open(rec + "/ticks.csv", "r") as fh:
        var lines = fh.read().split("\n")
        for i in range(1, len(lines)):
            var l = String(lines[i].strip())
            if l.byte_length() == 0:
                continue
            var r = List[Float64]()
            for w in l.split(","):
                r.append(Float64(String(w)))
            rows.append(r^)
    print("replay:", ckpt, "|", len(rows), "ticks in", rec)

    var ctx = DeviceContext()
    var fam = load_family("noeira/tasks/families/so101_tower.family")
    var fmd = parse_model_runtime(scene_path(fam))
    var cams = tower_cameras(fmd)
    var rm = make_tower_model(ctx)
    var rd = Data[RIG_DT, TOWER_MD, 1]()
    var r_w = make_tower_renderer[1, RENDER, RENDER, 1](ctx, fmd, rm)
    var r_o = make_tower_renderer[1, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, 1](ctx, fmd, rm)
    var jadr = List[Int]()
    var acc = 0
    for i in range(len(fmd.joints)):
        jadr.append(acc)
        acc += fmd.joints[i].nq
    var qa = List[Int]()
    for i in range(ACT):
        qa.append(jadr[fmd.actuators[i].joint_id])
    var q0 = posed_qpos[So101TowerPlacement](
        String("so101_tower_cube_in_bowl"), String("so101_tower"),
        So101TowerConfig.SLOT_RADIUS, seed=0,
    )
    var pb = bowl_xy.split(",")
    q0[So101TowerPlacement.free_qadr(0)] = Float64(String(pb[0]))
    q0[So101TowerPlacement.free_qadr(0) + 1] = Float64(String(pb[1]))
    var pk = brick_xy.split(",")
    q0[So101TowerPlacement.free_qadr(1)] = Float64(String(pk[0]))
    q0[So101TowerPlacement.free_qadr(1) + 1] = Float64(String(pk[1]))

    var net = StudentNet.make["cpu", Kaiming](None)
    load_params["cpu"](net, ckpt, None)
    var x = Tensor.alloc(IN_DIM)
    var y = Tensor.alloc(ACT)
    var names = camera_names()
    comptime S = 256 // OBS_PX
    var sum_rs = 0.0
    var n_cmp = 0
    var t = 0
    while t < len(rows):
        var po = rec + "/" + names[0] + "_policy_view_" + String(t) + ".png"
        if not exists(po):
            t += 16
            continue
        var r = rows[t].copy()
        var q = List[Float64](length=ACT, fill=0.0)
        var qd = List[Float64](length=ACT, fill=0.0)
        var logged = List[Float64](length=ACT, fill=0.0)
        for j in range(ACT):
            q[j] = r[1 + j]
            qd[j] = r[7 + j]
            logged[j] = r[13 + j]
        var hist = List[Float64](length=HIST_WORDS, fill=0.0)
        for k in range(HIST_WORDS // ACT):
            var tt = t - 1 - k
            if tt >= 0:
                for j in range(ACT):
                    hist[k * ACT + j] = rows[tt][13 + j]
        # REAL planes: the saved policy views, one sample per plane pixel
        var xr = List[Scalar[DT]](length=IN_DIM, fill=Scalar[DT](0))
        for k in range(N_CAMS):
            var img = load_png_file(rec + "/" + names[k] + "_policy_view_" + String(t) + ".png")
            for c in range(3):
                for oy in range(OBS_PX):
                    for ox in range(OBS_PX):
                        var b = Float64(img.pixels[((oy * S) * img.width + ox * S) * img.channels + c])
                        xr[(3 * k + c) * PLANE + oy * OBS_PX + ox] = Scalar[DT](b / 255.0 - 0.5)
        # SIM planes: the rebuilt scene at this arm pose
        var qs = q0.copy()
        for j in range(ACT):
            qs[qa[j]] = q[j]
        for k in range(NQ):
            rd.qpos.data[k] = Scalar[RIG_DT](qs[k])
        forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
        var xsim = List[Scalar[DT]](length=IN_DIM, fill=Scalar[DT](0))
        for k in range(N_CAMS):
            var rgb = List[Scalar[RIG_DT]]()
            var dd = List[Scalar[RIG_DT]]()
            var ss = List[Scalar[RIG_DT]]()
            if N_CAMS == 2 and k == 0:
                r_o.cam = cams[k]
                r_o.render_cpu(rd, rm, rgb, dd, ss)
                render_to_planes(rgb, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, k, xsim)
            else:
                r_w.cam = cams[k]
                r_w.render_cpu(rd, rm, rgb, dd, ss)
                render_to_planes(rgb, RENDER, RENDER, k, xsim)
        # the mixes: one camera real, the other sim
        var xmo = xsim.copy()
        var xmw = xsim.copy()
        for i in range(3 * PLANE):
            xmo[i] = xr[i]
            xmw[3 * PLANE + i] = xr[3 * PLANE + i]
        joints_to_planes(q, xr)
        joint_vels_to_planes(qd, xr)
        act_hist_to_planes(hist, xr)
        joints_to_planes(q, xsim)
        joint_vels_to_planes(qd, xsim)
        act_hist_to_planes(hist, xsim)
        joints_to_planes(q, xmo)
        joint_vels_to_planes(qd, xmo)
        act_hist_to_planes(hist, xmo)
        joints_to_planes(q, xmw)
        joint_vels_to_planes(qd, xmw)
        act_hist_to_planes(hist, xmw)
        var ar = _forward(net, xr, x, y)
        var asim = _forward(net, xsim, x, y)
        var amo = _forward(net, xmo, x, y)
        var amw = _forward(net, xmw, x, y)
        var d = 0.0
        for j in range(ACT):
            d += abs(ar[j] - asim[j])
        sum_rs += d / Float64(ACT)
        n_cmp += 1
        print("tick", t, "| q", col(q[0], 6, 2), col(q[1], 6, 2), col(q[2], 6, 2),
              col(q[3], 6, 2), col(q[4], 6, 2), col(q[5], 6, 2),
              "| mean |real - sim|", col(d / Float64(ACT), 5, 2))
        print(_line("logged ", logged))
        print(_line("real   ", ar))
        print(_line("sim    ", asim))
        print(_line("real-o ", amo))
        print(_line("real-w ", amw))
        t += 16
    if n_cmp > 0:
        print("replay: mean |real - sim| over", n_cmp, "ticks:", sum_rs / Float64(n_cmp))


"""The sim's cameras at a REAL arm pose — for checking the joint map by eye.

    pixi run mojo run -I . tools/so101/sim_view_at_pose.mojo SNAP_DIR [TAG]

`examples/so101/pixel_student_deploy_real.mojo --snap SNAP_DIR` writes each
real camera's undistorted frame and `pose<TAG>.txt`: the six joints it read,
in model radians through the rig's joint map. This renders the sim's overhead
and wrist cameras at exactly that arm pose (the props moved out of view —
their real positions are not recorded), 320 x 240, the real frames' 4:3
field at half size, into SNAP_DIR/sim_{overhead,wrist}<TAG>.png. Laid beside
the real frames, a joint mapped with the wrong sign or zero shows as a
different arm silhouette (overhead) or a rotated desk (wrist).
"""

from std.sys import argv
from max.gpu.host import DeviceContext

from noeira.io.png import save_png
from noeira.physics3d.fields import Data
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, make_tower_model, make_tower_renderer, tower_cameras,
    rig_byte,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime W = 320
comptime H = 240


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error("usage: sim_view_at_pose.mojo SNAP_DIR [TAG]")
    var dir = String(a[1])
    var tag = String(a[2]) if len(a) > 2 else String("")
    var txt = String("")
    with open(dir + "/pose" + tag + ".txt", "r") as f:
        txt = f.read()
    var pose = List[Float64]()
    for w in String(txt.strip()).split(" "):
        pose.append(Float64(String(w)))
    if len(pose) != 6:
        raise Error("pose" + tag + ".txt: expected 6 joints, got " + String(len(pose)))
    var ctx = DeviceContext()
    var f = load_family("noeira/tasks/families/so101_tower.family")
    var fmd = parse_model_runtime(scene_path(f))
    var cams = tower_cameras(fmd)
    var rm = make_tower_model(ctx)
    var rd = Data[RIG_DT, TOWER_MD, 1]()
    var r = make_tower_renderer[1, W, H, 1](ctx, fmd, rm)
    var q = posed_qpos[So101TowerPlacement](
        String("so101_tower_cube_in_bowl"), String("so101_tower"),
        So101TowerConfig.SLOT_RADIUS, seed=0,
    )
    var jadr = List[Int]()
    var acc = 0
    for i in range(len(fmd.joints)):
        jadr.append(acc)
        acc += fmd.joints[i].nq
    for j in range(6):
        q[jadr[fmd.actuators[j].joint_id]] = pose[j]
    for j in range(So101TowerPlacement.N_FREE):
        q[So101TowerPlacement.free_qadr(j) + 2] -= 20.0
    for k in range(So101TowerModel.NQ):
        rd.qpos.data[k] = Scalar[RIG_DT](q[k])
    forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
    var names: List[String] = ["overhead", "wrist"]
    for k in range(2):
        r.cam = cams[k]
        var rgb = List[Scalar[RIG_DT]]()
        var dd = List[Scalar[RIG_DT]]()
        var ss = List[Scalar[RIG_DT]]()
        r.render_cpu(rd, rm, rgb, dd, ss)
        var img = List[UInt8](length=W * H * 3, fill=UInt8(0))
        for p in range(W * H * 3):
            img[p] = rig_byte(Float64(rgb[p]))
        var out = dir + "/sim_" + names[k] + tag + ".png"
        save_png(out, img, W, H, 3)
        print("wrote", out)
    print("pose (rad):", txt.strip())

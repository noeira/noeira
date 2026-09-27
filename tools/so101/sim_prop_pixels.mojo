"""Where the sim's overhead camera sees each prop, over a grid of desk
positions — the samples a desk-plane homography is fitted to.

    pixi run mojo run -I . tools/so101/sim_prop_pixels.mojo > prop_pixels.csv

For the brick, then the bowl: the prop is moved (x, y at its resting
height) over a 7 x 7 grid covering the placement regions, the other prop
out of view, and the overhead camera traced at 320 x 240 (the real frame's
4:3 field at half size); the prop's pixel centroid is the mean of the pixels
that differ from the empty scene. CSV rows: prop, x, y, px, py. A real
frame's prop pixel mapped back through the fitted homography places that
prop in the sim — how a real scene is reproduced for the student's probe.
"""

from max.gpu.host import DeviceContext

from noeira.physics3d.fields import Data
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, make_tower_model, make_tower_renderer, tower_cameras,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime W = 320
comptime H = 240


def main() raises:
    var ctx = DeviceContext()
    var f = load_family("noeira/tasks/families/so101_tower.family")
    var fmd = parse_model_runtime(scene_path(f))
    var cams = tower_cameras(fmd)
    var rm = make_tower_model(ctx)
    var rd = Data[RIG_DT, TOWER_MD, 1]()
    var r = make_tower_renderer[1, W, H, 1](ctx, fmd, rm)
    r.cam = cams[0]
    var q0 = posed_qpos[So101TowerPlacement](
        String("so101_tower_cube_in_bowl"), String("so101_tower"),
        So101TowerConfig.SLOT_RADIUS, seed=0,
    )
    # the arm folded out of the way: the pose the sim episodes start in
    var dd = List[Scalar[RIG_DT]]()
    var ss = List[Scalar[RIG_DT]]()
    # empty scene
    var qe = q0.copy()
    for j in range(So101TowerPlacement.N_FREE):
        qe[So101TowerPlacement.free_qadr(j) + 2] -= 20.0
    for k in range(So101TowerModel.NQ):
        rd.qpos.data[k] = Scalar[RIG_DT](qe[k])
    forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
    var base = List[Scalar[RIG_DT]]()
    r.render_cpu(rd, rm, base, dd, ss)
    print("prop,x,y,px,py,npx")
    var names: List[String] = ["bowl", "brick"]  # slot order: bowl (0), brick (1)
    for j in range(So101TowerPlacement.N_FREE):
        var adr = So101TowerPlacement.free_qadr(j)
        for gy in range(7):
            for gx in range(7):
                var q = qe.copy()
                q[adr + 2] = q0[adr + 2]  # back at its resting height
                var x = 0.12 + 0.28 * Float64(gx) / 6.0
                var y = -0.24 + 0.46 * Float64(gy) / 6.0
                q[adr] = x
                q[adr + 1] = y
                for k in range(So101TowerModel.NQ):
                    rd.qpos.data[k] = Scalar[RIG_DT](q[k])
                forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
                var rgb = List[Scalar[RIG_DT]]()
                r.render_cpu(rd, rm, rgb, dd, ss)
                var sx = 0.0
                var sy = 0.0
                var n = 0
                for p in range(W * H):
                    var d = 0.0
                    for c in range(3):
                        d += abs(Float64(rgb[p * 3 + c]) - Float64(base[p * 3 + c]))
                    if d > 0.06:
                        sx += Float64(p % W)
                        sy += Float64(p // W)
                        n += 1
                if n > 3:
                    print(names[j] + "," + String(x) + "," + String(y) + ","
                          + String(sx / Float64(n)) + "," + String(sy / Float64(n))
                          + "," + String(n))

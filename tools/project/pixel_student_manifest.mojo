"""Write a pixel student run's manifest (`checkpoints/norm.json`) from THIS
build's constants and the run's own config — for runs made before the
driver wrote it (9052476e8..f7700f182), so `project-promote` carries it.

    pixi run mojo run -I . tools/project/pixel_student_manifest.mojo RUN_DIR
    # a 32x32 student: pixi run mojo run -I . -D DAGGER_PX_32 ...

Refuses a run whose recorded `obs_px` / `n_cams` differ from this build's:
the manifest describes the NETWORK the deploy will build, so it must be
written by a build of the same shape. Used by
`tools/project/promote_pixel_student.sh`.
"""

from std.os import makedirs
from std.os.path import exists
from std.sys import argv

from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.pixel_student import (
    N_CAMS, OBS_PX, write_pixel_manifest, check_pixel_manifest,
)
from noeira.tasks.so101_tower_xml import So101TowerModel


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error("usage: pixel_student_manifest.mojo RUN_DIR")
    var run_dir = String(a[1])
    var cfg = run_dir + "/metrics.config.kv"
    if not exists(cfg):
        raise Error("no " + cfg)
    var task = String("")
    var teacher = String("")
    var obs_px = -1
    var n_cams = -1
    var grip = False
    with open(cfg, "r") as f:
        for ln in f.read().split("\n"):
            var s = String(ln)
            var eq = s.find("=")
            if eq < 0:
                continue
            var k = String(s[byte = 0 : eq])
            var v = String(s[byte = eq + 1 :])
            if k == "task":
                task = v
            elif k == "teacher":
                teacher = v
            elif k == "obs_px":
                obs_px = Int(v)
            elif k == "n_cams":
                n_cams = Int(v)
            elif k == "gripper_sign":
                grip = v == "True" or v == "true" or v == "1"
    if obs_px < 0:
        obs_px = 16  # the driver wrote no obs_px before 1abcb1dce: 16 then
    if obs_px != OBS_PX or n_cams != N_CAMS:
        raise Error(
            "the run is " + String(obs_px) + "px x " + String(n_cams)
            + " cameras, this build " + String(OBS_PX) + "px x "
            + String(N_CAMS) + " — build with the run's defines"
        )
    makedirs(run_dir + "/checkpoints", exist_ok=True)
    var out = run_dir + "/checkpoints/norm.json"
    write_pixel_manifest(
        out, task, teacher, grip,
        Float64(So101TowerConfig.FRAME_SKIP) * So101TowerModel.TIMESTEP,
    )
    _ = check_pixel_manifest(out)
    print("wrote", out, "(", task, "| teacher", teacher, "|", OBS_PX, "px x",
          N_CAMS, "cameras )")

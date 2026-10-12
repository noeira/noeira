"""Sim twins of real SO-101 teleop frames, for S2's latent-alignment test.

noeira-docs/SO101_LEWM_PLAN.md S2 (1). To ask whether the world model's
encoder puts a REAL frame where it puts the SIM frame of the same scene, the
sim must be posed like each real frame: the follower's joints from the
teleop store, the cube and bowl where `tools/so101/real_scene_from_frame.py`
finds them on the episode's first overhead frame. Only an episode's first
`--rows` ticks are used: before the arm reaches the cube, the props have not
moved.

Two steps, with the Python prop finder between them:

    build/so101_twin --pngs DIR --real STORE
        writes DIR/ep_<e>.png: each episode's first overhead frame, doubled
        to the 640 x 480 grid the prop finder's homographies are fitted on
    pixi run python tools/so101/real_scene_from_frame.py DIR/ep_*.png > props.txt
    build/so101_twin --demo OUT.demo --real STORE --props props.txt --rows 60
        writes a .demo whose row r is real row r's pose (obs = qpos + zero
        qvel, the props at the found x, y and the scene's resting z / yaw);
        episodes whose props were not found are skipped; OUT.demo.map.txt
        pairs each twin episode with its real episode and first row
    tower_demo_rerender --demos OUT.demo --all-episodes --joint-zero follower --resize 112

The teleop store is `act_so101_import_dataset.mojo --undistort` output
(320 x 240, LeRobot units). ⚠ It was undistorted with the camera
calibration of its import date; a later recalibration shifts real against
sim.
"""

from std.sys import argv
from std.os import makedirs

from noeira.data.store import TrajectoryStore
from noeira.io.png import save_png
from noeira.deep_agents.demos.file import DemoSet, write_demo_file
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.so101_tower_rig import So101TowerUnits, RIG_JOINT_ZERO_FOLLOWER
from noeira.experimental.lewm.so101_data import So101Rig, JOINTS

comptime W = 320
comptime H = 240
comptime FULL = 2 * 3 * H * W
comptime NQ = So101TowerModel.NQ
comptime NV = So101TowerModel.NV
comptime TASK = "so101_tower_cube_in_bowl"


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var real = _arg(args, "--real", "")
    var pngs = _arg(args, "--pngs", "")
    var demo_out = _arg(args, "--demo", "")
    var props = _arg(args, "--props", "")
    var rows = Int(_arg(args, "--rows", "60"))
    if real.byte_length() == 0:
        raise Error("--real STORE is required")
    var st = TrajectoryStore(real)
    var buf = List[UInt8](length=FULL, fill=UInt8(0))

    if pngs.byte_length() > 0:
        makedirs(pngs, exist_ok=True)
        for e in range(st.n_episodes()):
            var r = Int(st.episodes.ep_offset[e])
            st.read_range[DType.uint8](
                String("images"), r, r + 1,
                rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](buf.unsafe_ptr()),
            )
            # slot 0 = overhead, CHW -> HWC doubled (nearest) to 640 x 480
            var hwc = List[UInt8](length=4 * H * W * 3, fill=UInt8(0))
            for y in range(2 * H):
                for x in range(2 * W):
                    for c in range(3):
                        hwc[(y * 2 * W + x) * 3 + c] = buf[c * H * W + (y // 2) * W + (x // 2)]
            save_png(pngs + "/ep_" + String(e) + ".png", hwc, 2 * W, 2 * H, 3)
        print("wrote", st.n_episodes(), "overhead frames to", pngs)
        return

    if demo_out.byte_length() == 0 or props.byte_length() == 0:
        raise Error("--demo OUT.demo needs --props props.txt (or use --pngs DIR)")
    # props.txt: the prop finder's stdout, one line per PNG:
    #   <path> --brick x,y --bowl x,y
    var bx = List[Float64](length=st.n_episodes(), fill=0.0)
    var by = List[Float64](length=st.n_episodes(), fill=0.0)
    var wx = List[Float64](length=st.n_episodes(), fill=0.0)
    var wy = List[Float64](length=st.n_episodes(), fill=0.0)
    var found = List[Bool](length=st.n_episodes(), fill=False)
    with open(props, "r") as f:
        for ln in f.read().split("\n"):
            var s = String(ln)
            var i0 = s.find("ep_")
            var ib = s.find("--brick ")
            var iw = s.find("--bowl ")
            if i0 < 0 or ib < 0 or iw < 0:
                continue
            var e = Int(String(s[byte = i0 + 3 : s.find(".png")]))
            var bpart = String(s[byte = ib + 8 :]).split(" ")[0]
            var wpart = String(s[byte = iw + 7 :]).split(" ")[0]
            var bxy = String(bpart).split(",")
            var wxy = String(wpart).split(",")
            bx[e] = Float64(String(bxy[0]))
            by[e] = Float64(String(bxy[1]))
            wx[e] = Float64(String(wxy[0]))
            wy[e] = Float64(String(wxy[1]))
            found[e] = True
    var rig = So101Rig()
    var units = So101TowerUnits(String(RIG_JOINT_ZERO_FOLLOWER))
    var qp = st.load_column[DType.float32](String("qpos"), max_bytes=1 << 30)
    var ds = DemoSet(NQ + NV, JOINTS)
    var obs = List[Float32](length=NQ + NV, fill=0)
    var act = List[Float32](length=JOINTS, fill=0)
    var kept = 0
    var twin_map = String("# twin episode -> real episode, real first row, rows\n")
    for e in range(st.n_episodes()):
        if not found[e]:
            continue
        var q0 = posed_qpos[So101TowerPlacement](
            String(TASK), String("so101_tower"), So101TowerConfig.SLOT_RADIUS, seed=UInt64(e),
        )
        q0[So101TowerPlacement.free_qadr(0)] = wx[e]
        q0[So101TowerPlacement.free_qadr(0) + 1] = wy[e]
        q0[So101TowerPlacement.free_qadr(1)] = bx[e]
        q0[So101TowerPlacement.free_qadr(1) + 1] = by[e]
        ds.begin_episode()
        var off = Int(st.episodes.ep_offset[e])
        var n = min(rows, Int(st.episodes.ep_len[e]))
        for r in range(off, off + n):
            for k in range(NQ):
                obs[k] = Float32(q0[k])
            for j in range(JOINTS):
                var q = units.lerobot_to_joint(j, Float64(qp[r * JOINTS + j]))
                obs[rig.qa[j]] = Float32(q)
                var mid = 0.5 * (rig.lo[j] + rig.hi[j])
                var half = 0.5 * (rig.hi[j] - rig.lo[j])
                act[j] = Float32((q - mid) / half)
            var nobs = obs.copy()
            ds.add(obs, act, 0.0, nobs, 0.0)
        ds.end_episode(success=False)
        twin_map += String(kept) + " " + String(e) + " " + String(off) + " " + String(n) + "\n"
        kept += 1
    write_demo_file(demo_out, ds)
    with open(demo_out + ".map.txt", "w") as f:
        f.write(twin_map)
    print("wrote", demo_out, "|", kept, "of", st.n_episodes(), "episodes (props found) |",
          ds.count(), "rows")

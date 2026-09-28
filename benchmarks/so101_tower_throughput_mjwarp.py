#!/usr/bin/env python3
"""MuJoCo Warp on the so101_tower cube-in-bowl scene — the third column for
`benchmarks/so101_tower_throughput.mojo`.

    # in a venv with mujoco + warp-lang + mujoco-warp (not the pixi env)
    python benchmarks/so101_tower_throughput_mjwarp.py [--only physics|camera] [--lanes 32,256,1024,4096]

Same protocol as the Mojo bench where MuJoCo Warp allows it:
- the composed scene `noeira/tasks/scenes/so101_tower.xml` (its own <option>:
  Newton, elliptic cone, impratio 10, 2 ms);
- per world: the brick in `desk_brick`, the bowl in `desk_bowl` (the family's
  rectangles around the `desk_surface` site), at least 0.135 m apart, random
  yaw; the arm at qpos0;
- a CONTROL step = 16 physics steps, captured once as a CUDA graph and
  replayed; the ctrl is uploaded before each control step from 8 pre-drawn
  tables of uniform targets over each actuator's ctrlrange, each held 16
  control steps (the Mojo bench's HOLD / N_TABLES);
- WARMUP control steps untimed, STEPS timed between two synchronisations;
- contacts per physics step (`nacon`) and solver iterations, sampled.

Rendering: `create_render_context` at 128x128 and 64x64, ONE camera active at
a time (wrist, overhead), geom groups 0, 2, 4 and 5 (the rig's CALIBRATED
visual set: props, the arm's visual meshes, the backdrop, the marker —
`so101_tower_rig.rig_visual_group_mask`; without 4/5 the wrist saw 0.52 of
its pixels hit against the Mojo renderer's 0.68), no shadows, one sample per
pixel;
`render` timed alone on posed worlds (after 40 random control steps), and
`refit_bvh` timed separately (the scene BVH refit MuJoCo Warp needs per frame).
"""

import argparse
import time

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw

SCENE = "noeira/tasks/scenes/so101_tower.xml"
FRAME_SKIP = 16
HOLD = 16
N_TABLES = 8
BRICK_RECT = (-0.22, -0.24, 0.05, 0.21)
BOWL_RECT = (-0.19, -0.23, 0.06, 0.19)
MIN_SEP = 0.135


def yaw_quat(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def placements(mjm, n, rng):
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    site = d.site("desk_surface").xpos.copy()
    bowl_adr = mjm.jnt_qposadr[mjm.joint("bowl_free").id]
    brick_adr = mjm.jnt_qposadr[mjm.joint("brick_free").id]
    q = np.tile(mjm.qpos0, (n, 1))
    for w in range(n):
        while True:
            bx = site[0] + rng.uniform(BRICK_RECT[0], BRICK_RECT[2])
            by = site[1] + rng.uniform(BRICK_RECT[1], BRICK_RECT[3])
            ox = site[0] + rng.uniform(BOWL_RECT[0], BOWL_RECT[2])
            oy = site[1] + rng.uniform(BOWL_RECT[1], BOWL_RECT[3])
            if np.hypot(bx - ox, by - oy) >= MIN_SEP:
                break
        q[w, brick_adr:brick_adr + 3] = [bx, by, site[2] + 0.0126]
        q[w, brick_adr + 3:brick_adr + 7] = yaw_quat(rng.uniform(-np.pi, np.pi))
        q[w, bowl_adr:bowl_adr + 3] = [ox, oy, site[2] + 0.001]
        q[w, bowl_adr + 3:bowl_adr + 7] = yaw_quat(rng.uniform(-np.pi, np.pi))
    return q


def ctrl_tables(mjm, n, rng):
    lo = mjm.actuator_ctrlrange[:, 0]
    hi = mjm.actuator_ctrlrange[:, 1]
    return [wp.array(rng.uniform(lo, hi, size=(n, mjm.nu)).astype(np.float32),
                     dtype=wp.float32) for _ in range(N_TABLES)]


def make(mjm, n, rng):
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=n, nconmax=48, njmax=256)  # PER WORLD (~9 contacts used)
    d.qpos.assign(placements(mjm, n, rng).astype(np.float32))
    return m, d


def bench_physics(mjm, n, warmup, steps, seed=11):
    rng = np.random.default_rng(seed)
    m, d = make(mjm, n, rng)
    tables = ctrl_tables(mjm, n, rng)

    def ctrl_step():
        for _ in range(FRAME_SKIP):
            mjw.step(m, d)

    wp.copy(d.ctrl, tables[0])
    ctrl_step()  # JIT
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        ctrl_step()
    graph = cap.graph

    for it in range(warmup):
        wp.copy(d.ctrl, tables[(it // HOLD) % N_TABLES])
        wp.capture_launch(graph)
    wp.synchronize()

    ncon = []
    niter = []
    t0 = time.perf_counter()
    for it in range(steps):
        wp.copy(d.ctrl, tables[((warmup + it) // HOLD) % N_TABLES])
        wp.capture_launch(graph)
    wp.synchronize()
    dt = time.perf_counter() - t0

    # contacts / iterations: sampled OUTSIDE the timed window, over a few
    # more control steps (reading them inside would sync every step)
    for it in range(20):
        wp.copy(d.ctrl, tables[((warmup + steps + it) // HOLD) % N_TABLES])
        wp.capture_launch(graph)
        wp.synchronize()
        ncon.append(int(d.nacon.numpy()[0]) / n)
        niter.append(float(np.mean(d.solver_niter.numpy())))
    env_sps = n * steps / dt
    print(f"RESULT side=mjwarp leg=physics n_envs={n} steps={steps} wall_s={dt:.3f} "
          f"us_per_batch_step={dt / steps * 1e6:.1f} env_steps_per_s={int(env_sps)} "
          f"physics_steps_per_s={int(env_sps * FRAME_SKIP)} "
          f"contacts_per_world={np.mean(ncon):.2f} solver_iters={np.mean(niter):.2f}",
          flush=True)


def bench_render(mjm, n, res, seed=11, pose_steps=40):
    rng = np.random.default_rng(seed)
    m, d = make(mjm, n, rng)
    tables = ctrl_tables(mjm, n, rng)
    for it in range(pose_steps):
        wp.copy(d.ctrl, tables[(it // HOLD) % N_TABLES])
        for _ in range(FRAME_SKIP):
            mjw.step(m, d)
    mjw.forward(m, d)
    wp.synchronize()
    out = []
    for cam in ("robot_wrist_cam", "tower_overhead_cam"):
        rc = mjw.create_render_context(
            mjm, nworld=n, cam_res=(res, res), render_rgb=True,
            render_depth=False, render_seg=False, use_shadows=False,
            enabled_geom_groups=[0, 2, 4, 5], cam_active=[cam], samples_per_pixel=1,
        )
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)  # JIT
        wp.synchronize()
        reps = 0
        t0 = time.perf_counter()
        while True:
            mjw.render(m, d, rc)
            reps += 1
            if reps >= 3:
                wp.synchronize()
                if time.perf_counter() - t0 > 0.3:
                    break
        ms = (time.perf_counter() - t0) / reps * 1e3
        reps = 0
        t0 = time.perf_counter()
        while True:
            mjw.refit_bvh(m, d, rc)
            reps += 1
            if reps >= 3:
                wp.synchronize()
                if time.perf_counter() - t0 > 0.2:
                    break
        refit_ms = (time.perf_counter() - t0) / reps * 1e3
        # Hit fraction over ALL worlds: a pixel that hit nothing is the packed
        # background (black, alpha 255). A camera looking at empty space
        # renders fast and measures nothing, so this is printed beside the
        # time (the Mojo bench prints its `seg >= 0` fraction).
        rgb = rc.rgb_data.numpy().astype(np.uint32)
        nonbg = float(np.mean(rgb != np.uint32(0xFF000000)))
        out.append(f"{cam.split('_')[1]}_ms={ms:.2f} {cam.split('_')[1]}_fps={int(n / (ms / 1e3))} "
                   f"{cam.split('_')[1]}_refit_ms={refit_ms:.2f} {cam.split('_')[1]}_hit={nonbg:.2f}")
    print(f"RESULT side=mjwarp leg=camera res={res}x{res} n_envs={n} " + " ".join(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lanes", default="32,256,1024,4096")
    a = ap.parse_args()
    wp.init()
    mjm = mujoco.MjModel.from_xml_path(SCENE)
    print("device:", wp.get_device().name, "| mujoco", mujoco.__version__,
          "| warp", wp.__version__, "| solver", mjm.opt.solver, "cone",
          mjm.opt.cone, "iters", mjm.opt.iterations, "ls", mjm.opt.ls_iterations,
          flush=True)
    lanes = [int(x) for x in a.lanes.split(",")]
    if a.only in ("", "physics"):
        for n in lanes:
            bench_physics(mjm, n, a.warmup, a.steps)
    if a.only in ("", "camera"):
        for n in lanes:
            for res in (128, 64):
                bench_render(mjm, n, res)


if __name__ == "__main__":
    main()

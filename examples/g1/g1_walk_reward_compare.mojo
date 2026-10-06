"""RoboParty's reward under two policies, on the same env and commands.

    pixi run mojo build -I . -D RP_ACTOR_LINVEL=1 examples/g1/g1_walk_reward_compare.mojo -o build/g1_walk_reward_compare
    ./build/g1_walk_reward_compare <mix-recipe run> <RP-recipe run>

Drives `UnitreeG1WalkRP` (eval mode) with (0) a walking policy of the
earlier mix recipe — its 70-D observation computed from the RP env's state
— and (1) an RoboParty-recipe policy, on the same random command sequences
(the training distribution, wz clipped to the mix's +-1). It prints, per
policy, the per-second value of every RoboParty term, and for the walker
the non-foot contact pairs.

What it answered (2026-10-06, G1_WALKER_PLAN §10): does the reward prefer
standing? s6's walk scored -3.29 / s against rp7's stand +1.17; the arm-on-
torso contacts alone cost -2.10 / s (fixed: `_arm_torso`).

Build with the `RP_ACTOR_LINVEL` the RP run was trained with.
"""
from std.math import abs, sqrt
from std.random import seed as seed_rng, random_float64
from std.sys import argv
from layout import Layout
from noeira.core.cont_action import ContAction
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.gpu.constants import METADATA_SIZE, CONTACT_SIZE, CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B, META_IDX_NUM_CONTACTS
from noeira.envs.robots.unitree_g1_walk_rp import UnitreeG1WalkRP, UnitreeG1WalkRPModel, g1r_host_reset, g1r_host_terms
from noeira.envs.robots.unitree_g1_walk_rp_config import (
    G1R_N_TERMS, G1R_OBS_DIM, G1R_O_CMD, G1R_CMD_VX, G1R_CMD_VY, G1R_CMD_WZ, G1R_CMD_TIMER,
    g1r_term_name, g1r_weight, g1r_command,
)
from noeira.envs.robots.unitree_g1_walk_config import g1_walk_obs, G1_WALK_OBS_DIM
from g1_walk_policy import WalkPolicy
from g1_walk_rp_policy import WalkRPPolicy

comptime NQ = UnitreeG1WalkRPModel.NQ
comptime NV = UnitreeG1WalkRPModel.NV
comptime E = UnitreeG1WalkRP[False]

def _mix_obs(mut env: E) -> List[Float64]:
    var o = TensorImpl[DType.float64].alloc(G1_WALK_OBS_DIM)
    g1_walk_obs[DType.float64, 1, NQ, NV, G1_WALK_OBS_DIM, False](
        env.d.qpos.lt["cpu", Layout.row_major(1, NQ)](), env.d.qvel.lt["cpu", Layout.row_major(1, NV)](),
        env.d.meta.lt["cpu", Layout.row_major(1, METADATA_SIZE)](), o.lt["cpu", Layout.row_major(1, G1_WALK_OBS_DIM)](), 0)
    var l = List[Float64]()
    for k in range(G1_WALK_OBS_DIM): l.append(Float64(o.data[k]))
    return l^

def run[WHICH: Int](mut mix: WalkPolicy, mut rp: WalkRPPolicy, episodes: Int) raises:
    seed_rng(11)
    var env = E()
    var terms = Array[Float64, G1R_N_TERMS](fill=0.0)
    var sums = List[Float64](length=G1R_N_TERMS, fill=0.0)
    var n = 0
    var falls = 0
    var pairs = Dict[String, Int]()
    for ep in range(episodes):
        _ = env.reset()
        g1r_host_reset(env.d, 1, False)
        var q = List[Float64](); var v = List[Float64](length=NV, fill=0.0)
        for i in range(NQ): q.append(Float64(env.d.qpos.data[i]))
        env.set_state(q, v)
        mix.reset(); rp.reset()
        var left = 0
        var o = env._get_obs()
        var orp = List[Float64]()
        for k in range(G1R_OBS_DIM): orp.append(Float64(o.data[k]))
        for k in range(1000):
            if left <= 0:
                var c = g1r_command(random_float64(), random_float64(), random_float64(), random_float64())
                # the mix policy was trained on wz in [-1, 1]
                var wz = max(-1.0, min(1.0, c[2]))
                env.d.meta.data[G1R_CMD_VX] = c[0]; env.d.meta.data[G1R_CMD_VY] = c[1]
                env.d.meta.data[G1R_CMD_WZ] = wz; env.d.meta.data[G1R_CMD_TIMER] = -1
                orp[G1R_O_CMD] = c[0]; orp[G1R_O_CMD + 1] = c[1]; orp[G1R_O_CMD + 2] = wz
                left = 100 + Int(random_float64() * 100.0)
            left -= 1
            var a: List[Float64]
            comptime if WHICH == 0:
                a = mix.act(_mix_obs(env))
            else:
                a = rp.act(orp)
            var act = ContAction[29]()
            for j in range(29): act[j] = a[j]
            var r = env.step(act)
            for kk in range(G1R_OBS_DIM): orp[kk] = Float64(r[0].data[kk])
            var done = g1r_host_terms(env.d, a, terms)
            comptime if WHICH == 0:
                var nc = Int(env.d.meta.data[META_IDX_NUM_CONTACTS])
                for c in range(nc):
                    var ba = Int(env.d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_A])
                    var bb = Int(env.d.contacts.data[c * CONTACT_SIZE + CONTACT_IDX_BODY_B])
                    var fa = (ba >= 6 and ba <= 11) or (ba >= 16 and ba <= 21)
                    var fb = (bb >= 6 and bb <= 11) or (bb >= 16 and bb <= 21)
                    if not (ba <= 0 and fb) and not (bb <= 0 and fa) and not (fa and fb):
                        var key = String(min(ba, bb)) + "-" + String(max(ba, bb))
                        pairs[key] = pairs.get(key, 0) + 1
            for t in range(G1R_N_TERMS): sums[t] += terms[t]
            n += 1
            if done:
                falls += 1
                break
    var tot = 0.0
    for t in range(G1R_N_TERMS):
        tot += g1r_weight(t) * sums[t] / Float64(n)
    for e in pairs.items():
        if e.value > 200:
            print("  non-foot contact pair", e.key, "in", e.value, "contact records")
    print("policy", "s6 (walks)" if WHICH == 0 else "rp (stands)", "| steps", n, "| falls", falls, "| RoboParty reward per second", tot)
    for t in range(G1R_N_TERMS):
        var m = sums[t] / Float64(n)
        print("    ", g1r_term_name(t), g1r_weight(t) * m)

def main() raises:
    var a = argv()
    var mix = WalkPolicy(String(a[1]))
    var rp = WalkRPPolicy(String(a[2]))
    run[0](mix, rp, 8)
    run[1](mix, rp, 8)

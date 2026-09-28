"""Every converted deep-agent trainer writes a v3 checkpoint, and restores it.

For each agent (CPU): train briefly so the weights AND the training scalars
(α, the step counter) have moved off their initial values, `save` → A; build
a FRESH agent, `save` → C, `load(A)`, `save` → B. Then:

  1. A starts with the `storage-ckpt v3` header (not v2 text).
  2. A carries the agent's `K` scalar sections (named below) — the state v2
     appended as `key=value` lines, or did not save at all.
  3. sha(B) == sha(A): every persisted field, scalars included, round-trips
     exactly. A `take_state` / `get_int` that silently kept the fresh value
     would re-save the fresh value and fail here.
  4. sha(C) != sha(A): the fresh agent really differed before the load, so
     (3) is not the identity of two untouched agents.

TD-MPC2 is not here: `test_tdmpc2_checkpoint.mojo` covers it, and on the
2026-09-22 toolchain both TD-MPC2 tests crash the compiler at HEAD (before
this change too).

Run: pixi run mojo run -I . tests/deep_agents/test_trainer_checkpoints_v3.mojo
"""

from std.random import seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.io.fileio import read_file_bytes
from noeira.io.sha256 import sha256_file
from noeira.deep_agents.ddpg.config import DDPG
from noeira.deep_agents.td3.config import TD3
from noeira.deep_agents.sac.config import SAC
from noeira.deep_agents.redq.config import SmallREDQ
from noeira.deep_agents.redq_ofe.config import REDQOFE6
from noeira.deep_agents.mbpo import MBPO
from noeira.deep_agents.ppo.config import PPO
from noeira.deep_agents.ppo_discrete.config import PPODiscrete
from noeira.envs.pendulum.pendulum_v1 import PendulumEnv
from noeira.envs.cartpole import CartPoleEnv


comptime OBS = 3
comptime ACT = 1
comptime H = 32
comptime BATCH = 32
comptime CAP = 4_096
comptime STEPS = 600
comptime WARM = 200
comptime ASCALE = Scalar[DT](2.0)
comptime DIR = "/tmp/noeira_ckpt_v3_"


def _has(bytes: List[UInt8], needle: String) -> Bool:
    var nb = needle.as_bytes()
    var n = len(nb)
    for i in range(len(bytes) - n + 1):
        var ok = True
        for j in range(n):
            if bytes[i + j] != nb[j]:
                ok = False
                break
        if ok:
            return True
    return False


def _check(name: String, keys: List[String]) raises:
    var a = String(DIR) + name + "_a.ckpt"
    var b = String(DIR) + name + "_b.ckpt"
    var c = String(DIR) + name + "_c.ckpt"
    var bytes = read_file_bytes(a)
    assert_true(_has(bytes, "storage-ckpt v3\n"), name + ": not a v3 file")
    for k in keys:
        assert_true(
            # No leading "\n": a `K` header follows the raw bytes of the
            # last tensor, not a line break.
            _has(bytes, "K " + k + "\n"), name + ": no `K " + k + "` section"
        )
    var ha = sha256_file(a)
    assert_true(sha256_file(b) == ha, name + ": save→load→save is not exact")
    assert_true(
        sha256_file(c) != ha,
        name + ": fresh agent already equal to the trained one — vacuous",
    )
    print("  " + name + ": v3, " + String(len(keys)) + " scalar key(s), exact round-trip")


def main() raises:
    seed(7)
    print("trainer checkpoints: v3 + K scalars round-trip (CPU)")
    var p = PendulumEnv[DT]()

    var ddpg = DDPG["cpu", OBS, ACT, BATCH, CAP, H](
        action_scale=ASCALE, learning_starts=WARM
    )
    _ = ddpg.train_single(p, total_timesteps=STEPS, print_every=STEPS)
    ddpg.save(String(DIR) + "ddpg_a.ckpt")
    var ddpg2 = DDPG["cpu", OBS, ACT, BATCH, CAP, H](action_scale=ASCALE)
    ddpg2.save(String(DIR) + "ddpg_c.ckpt")
    ddpg2.load(String(DIR) + "ddpg_a.ckpt")
    ddpg2.save(String(DIR) + "ddpg_b.ckpt")
    _check("ddpg", ["total_train_steps"])

    var td3 = TD3["cpu", OBS, ACT, BATCH, CAP, H](
        action_scale=ASCALE, learning_starts=WARM
    )
    _ = td3.train_single(p, total_timesteps=STEPS, print_every=STEPS)
    td3.save(String(DIR) + "td3_a.ckpt")
    var td32 = TD3["cpu", OBS, ACT, BATCH, CAP, H](action_scale=ASCALE)
    td32.save(String(DIR) + "td3_c.ckpt")
    td32.load(String(DIR) + "td3_a.ckpt")
    td32.save(String(DIR) + "td3_b.ckpt")
    _check("td3", ["total_train_steps"])

    var sac = SAC["cpu", OBS, ACT, BATCH, CAP, H](
        action_scale=ASCALE, learning_starts=WARM
    )
    _ = sac.train_single(p, total_timesteps=STEPS, print_every=STEPS)
    sac.save(String(DIR) + "sac_a.ckpt")
    var sac2 = SAC["cpu", OBS, ACT, BATCH, CAP, H](action_scale=ASCALE)
    sac2.save(String(DIR) + "sac_c.ckpt")
    sac2.load(String(DIR) + "sac_a.ckpt")
    sac2.save(String(DIR) + "sac_b.ckpt")
    _check("sac", ["alpha.value", "alpha.b1_pow", "total_train_steps"])

    var redq = SmallREDQ["cpu", OBS, ACT, BATCH, CAP, H](
        action_scale=ASCALE, learning_starts=WARM
    )
    _ = redq.train_single(p, total_timesteps=STEPS, print_every=STEPS)
    redq.save(String(DIR) + "redq_a.ckpt")
    var redq2 = SmallREDQ["cpu", OBS, ACT, BATCH, CAP, H](action_scale=ASCALE)
    redq2.save(String(DIR) + "redq_c.ckpt")
    redq2.load(String(DIR) + "redq_a.ckpt")
    redq2.save(String(DIR) + "redq_b.ckpt")
    _check("redq", ["alpha.value", "total_train_steps"])

    var ofe = REDQOFE6["cpu", OBS, ACT, BATCH, CAP, H, 8](
        action_scale=ASCALE, learning_starts=WARM
    )
    _ = ofe.train_single(p, total_timesteps=STEPS, print_every=STEPS)
    ofe.save(String(DIR) + "redq_ofe_a.ckpt")
    var ofe2 = REDQOFE6["cpu", OBS, ACT, BATCH, CAP, H, 8](action_scale=ASCALE)
    ofe2.save(String(DIR) + "redq_ofe_c.ckpt")
    ofe2.load(String(DIR) + "redq_ofe_a.ckpt")
    ofe2.save(String(DIR) + "redq_ofe_b.ckpt")
    _check("redq_ofe", ["alpha.value", "total_train_steps"])

    var mbpo = MBPO[
        "cpu", OBS, ACT, BATCH, CAP, 4 * CAP,
        N_ENS=3, N_ELITES=2, HIDDEN=H, DYN_HIDDEN=H,
    ](
        action_scale=ASCALE,
        learning_starts=WARM,
        model_train_freq=200,
        num_rollouts_per_step=16,
        dyn_batch_size=BATCH,
        dyn_max_epochs=2,
    )
    _ = mbpo.train_single(p, total_timesteps=STEPS, print_every=STEPS)
    mbpo.save(String(DIR) + "mbpo_a.ckpt")
    var mbpo2 = MBPO[
        "cpu", OBS, ACT, BATCH, CAP, 4 * CAP,
        N_ENS=3, N_ELITES=2, HIDDEN=H, DYN_HIDDEN=H,
    ](action_scale=ASCALE)
    mbpo2.save(String(DIR) + "mbpo_c.ckpt")
    mbpo2.load(String(DIR) + "mbpo_a.ckpt")
    mbpo2.save(String(DIR) + "mbpo_b.ckpt")
    _check("mbpo", ["alpha.value", "elites.n", "elites.0", "total_train_steps"])

    var ppo = PPO["cpu", OBS, ACT, 64, 16, 2, HIDDEN=H](action_scale=ASCALE)
    _ = ppo.train_single(p, total_timesteps=STEPS, print_every=STEPS)
    ppo.save(String(DIR) + "ppo_a.ckpt")
    var ppo2 = PPO["cpu", OBS, ACT, 64, 16, 2, HIDDEN=H](action_scale=ASCALE)
    ppo2.save(String(DIR) + "ppo_c.ckpt")
    ppo2.load(String(DIR) + "ppo_a.ckpt")
    ppo2.save(String(DIR) + "ppo_b.ckpt")
    _check("ppo", ["total_train_steps"])

    var cp = CartPoleEnv[DT]()
    var ppod = PPODiscrete["cpu", 4, 2, 64, 16, 2, HIDDEN=H]()
    _ = ppod.train(cp, total_timesteps=STEPS, print_every=STEPS)
    ppod.save(String(DIR) + "ppod_a.ckpt")
    var ppod2 = PPODiscrete["cpu", 4, 2, 64, 16, 2, HIDDEN=H]()
    ppod2.save(String(DIR) + "ppod_c.ckpt")
    ppod2.load(String(DIR) + "ppod_a.ckpt")
    ppod2.save(String(DIR) + "ppod_b.ckpt")
    _check("ppod", ["total_train_steps"])

    print("ALL TRAINER CHECKPOINTS V3 OK")

"""The G1 walker's networks under RoboParty's recipe, and the policy on the CPU.

    from g1_walk_rp_policy import WalkRPPolicy
    var pol = WalkRPPolicy(run_dir)       # <run>/ckpt + <run>/obs_norm.txt
    pol.reset()
    var a = pol.act(obs82)                # every control step, the env's obs

ONE DEFINITION: `g1_walk_rp_ppo_gpu.mojo` trains these types; the eval and
the room load them from here.

THE OBSERVATION is `run_ppo_vec`'s frame history (`_stack_k`), rebuilt on
the host exactly: newest first,

    [frame_0 .. frame_9 | priv_0 .. priv_9]
    frame = env[0:67] ++ the last EXECUTED action (29)
    priv  = env[67:82]

every slot filled with the first frame after a reset (zero action), then one
push per step; normalised `(x - mean) / sqrt(var + 1e-8)`, clipped +-10.
The actor reads the first 960 words (`Slice` at its head).
"""

from std.math import sqrt

from noeira.deep_agents.ppo import PPOAgent
from noeira.deep_agents.primitives.gaussian_head import GaussianHead
from noeira.deep_agents.training.obs_norm import RunningMeanStd
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT
from noeira.nn.primitives.activations import ELU
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.slice import Slice
from noeira.envs.robots.unitree_g1_walk_rp_config import (
    G1R_OBS_DIM, G1R_OBS_ACTOR, G1R_OBS_PRIV, G1R_ACTION_CLIP,
    g1_mirror_joint, g1r_mirror_env_obs,
)


comptime RP_ACT = 29
comptime RP_HIST = 1
comptime RP_FRAMES = 10
comptime RP_FRAME = G1R_OBS_ACTOR + RP_HIST * RP_ACT
comptime RP_ACTOR_OBS = RP_FRAMES * RP_FRAME
comptime RP_OBS = RP_ACTOR_OBS + RP_FRAMES * G1R_OBS_PRIV
comptime RP_OBS_CLIP: Float64 = 10.0
comptime H1 = 512
comptime H2 = 256
comptime H3 = 128
comptime RPActor = Sequential[
    Slice[RP_OBS, 0, RP_ACTOR_OBS],
    Linear[RP_ACTOR_OBS, H1], ELU[H1], Linear[H1, H2], ELU[H2],
    Linear[H2, H3], ELU[H3], GaussianHead[H3, RP_ACT],
]
comptime RP_HEAD_CHILD = 7
"""`RPActor.children[7]` is the GaussianHead (`set_log_std_init`)."""
comptime RPCritic = Sequential[
    Linear[RP_OBS, H1], ELU[H1], Linear[H1, H2], ELU[H2],
    Linear[H2, H3], ELU[H3], Linear[H3, 1],
]


def rp_mirror_maps() -> Tuple[List[Int], List[Float64], List[Int], List[Float64]]:
    """The policy observation's (1110 words) and the action's mirror maps,
    for `PPOActorLoss.enable_mirror`: every frame's env words through
    `g1r_mirror_env_obs`, its action words and the action through
    `g1_mirror_joint`; frames keep their slot."""
    var env_m = g1r_mirror_env_obs()
    var oi = List[Int](length=RP_OBS, fill=0)
    var os = List[Float64](length=RP_OBS, fill=1.0)
    for f in range(RP_FRAMES):
        for k in range(G1R_OBS_ACTOR):
            oi[f * RP_FRAME + k] = f * RP_FRAME + env_m[0][k]
            os[f * RP_FRAME + k] = env_m[1][k]
        for j in range(RP_ACT):
            var m = g1_mirror_joint(j)
            oi[f * RP_FRAME + G1R_OBS_ACTOR + j] = f * RP_FRAME + G1R_OBS_ACTOR + m[0]
            os[f * RP_FRAME + G1R_OBS_ACTOR + j] = m[1]
        for k in range(G1R_OBS_PRIV):
            var src = env_m[0][G1R_OBS_ACTOR + k] - G1R_OBS_ACTOR
            oi[RP_ACTOR_OBS + f * G1R_OBS_PRIV + k] = RP_ACTOR_OBS + f * G1R_OBS_PRIV + src
            os[RP_ACTOR_OBS + f * G1R_OBS_PRIV + k] = env_m[1][G1R_OBS_ACTOR + k]
    var ai = List[Int](length=RP_ACT, fill=0)
    var as_ = List[Float64](length=RP_ACT, fill=1.0)
    for j in range(RP_ACT):
        var m = g1_mirror_joint(j)
        ai[j] = m[0]
        as_[j] = m[1]
    return (oi^, os^, ai^, as_^)


struct WalkRPPolicy(Movable):
    var agent: PPOAgent["cpu", RPActor, RPCritic, RP_OBS, RP_ACT, 1, 1, 1, 1]
    var rms: RunningMeanStd
    var stack: List[Float64]
    var last: List[Float64]
    var fresh: Bool
    var obs: List[Scalar[DT]]
    var out: List[Scalar[DT]]

    def __init__(out self, run_dir: String) raises:
        self.agent = PPOAgent["cpu", RPActor, RPCritic, RP_OBS, RP_ACT, 1, 1, 1, 1](
            action_scale=Scalar[DT](G1R_ACTION_CLIP)
        )
        self.agent.load(run_dir + "/ckpt")
        self.rms = RunningMeanStd(RP_OBS)
        self.rms.load(run_dir + "/obs_norm.txt")
        self.stack = List[Float64](length=RP_OBS, fill=0.0)
        self.last = List[Float64](length=RP_ACT, fill=0.0)
        self.fresh = True
        self.obs = List[Scalar[DT]](length=RP_OBS, fill=0)
        self.out = List[Scalar[DT]](length=RP_ACT, fill=0)

    def reset(mut self):
        self.fresh = True
        for j in range(RP_ACT):
            self.last[j] = 0.0

    def _write_frame(mut self, f: Int, env_obs: List[Float64]):
        for k in range(G1R_OBS_ACTOR):
            self.stack[f * RP_FRAME + k] = env_obs[k]
        for j in range(RP_ACT):
            self.stack[f * RP_FRAME + G1R_OBS_ACTOR + j] = self.last[j]
        for k in range(G1R_OBS_PRIV):
            self.stack[RP_ACTOR_OBS + f * G1R_OBS_PRIV + k] = env_obs[G1R_OBS_ACTOR + k]

    def act(mut self, env_obs: List[Float64]) raises -> List[Float64]:
        if len(env_obs) != G1R_OBS_DIM:
            raise Error("WalkRPPolicy.act: expected 82 env words, got " + String(len(env_obs)))
        if self.fresh:
            for f in range(RP_FRAMES):
                self._write_frame(f, env_obs)
            self.fresh = False
        else:
            for f in range(RP_FRAMES - 1, 0, -1):
                for k in range(RP_FRAME):
                    self.stack[f * RP_FRAME + k] = self.stack[(f - 1) * RP_FRAME + k]
                for k in range(G1R_OBS_PRIV):
                    self.stack[RP_ACTOR_OBS + f * G1R_OBS_PRIV + k] = self.stack[
                        RP_ACTOR_OBS + (f - 1) * G1R_OBS_PRIV + k
                    ]
            self._write_frame(0, env_obs)
        for k in range(RP_OBS):
            var z = (self.stack[k] - self.rms.mean[k]) / sqrt(self.rms.var_[k] + 1e-8)
            if z > RP_OBS_CLIP:
                z = RP_OBS_CLIP
            elif z < -RP_OBS_CLIP:
                z = -RP_OBS_CLIP
            self.obs[k] = Scalar[DT](z)
        self.agent.select_greedy_action(self.obs, self.out)
        var a = List[Float64](length=RP_ACT, fill=0.0)
        for j in range(RP_ACT):
            a[j] = Float64(self.out[j])
            self.last[j] = a[j]
        return a^

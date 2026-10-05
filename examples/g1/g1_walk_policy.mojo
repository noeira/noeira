"""The G1 walker's networks, and the trained policy on the CPU.

    from g1_walk_policy import WalkPolicy
    var pol = WalkPolicy(run_dir)          # <run>/ckpt + <run>/obs_norm.txt
    pol.reset()                            # zero the last-action history
    var a = pol.act(obs70)                 # 29 actions, greedy, clamped +-2

ONE DEFINITION OF THE ARCHITECTURE: `g1_walk_ppo_gpu.mojo` trains these
types and everything that loads its checkpoint (the eval, the room) builds
them from here, so a width change cannot leave a loader behind.

The observation the policy sees is the env's 70 words then the last
EXECUTED action (29, the clamped one — `run_ppo_vec`'s `HIST = 1`), each
normalised `(x - mean) / sqrt(var + 1e-8)` and clipped to +-10 with the
run's `obs_norm.txt`. `act` keeps that history itself.
"""

from std.math import sqrt

from noeira.deep_agents.ppo import PPOAgent
from noeira.deep_agents.primitives.gaussian_head import GaussianHead
from noeira.deep_agents.training.obs_norm import RunningMeanStd
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT
from noeira.nn.primitives.activations import Swish
from noeira.nn.primitives.linear import Linear
from noeira.envs.robots.unitree_g1_walk_config import G1_WALK_OBS_DIM, G1_WALK_ACTION_CLIP


comptime WALK_ACT = 29
comptime WALK_HIST = 1
comptime WALK_OBS = G1_WALK_OBS_DIM + WALK_HIST * WALK_ACT
comptime WALK_ACTION_SCALE: Float64 = G1_WALK_ACTION_CLIP
comptime WALK_OBS_CLIP: Float64 = 10.0
comptime H1 = 512
comptime H2 = 256
comptime H3 = 128
comptime WalkActor = Sequential[
    Linear[WALK_OBS, H1], Swish[H1], Linear[H1, H2], Swish[H2],
    Linear[H2, H3], Swish[H3], GaussianHead[H3, WALK_ACT],
]
comptime WalkCritic = Sequential[
    Linear[WALK_OBS, H1], Swish[H1], Linear[H1, H2], Swish[H2],
    Linear[H2, H3], Swish[H3], Linear[H3, 1],
]


struct WalkPolicy(Movable):
    var agent: PPOAgent["cpu", WalkActor, WalkCritic, WALK_OBS, WALK_ACT, 1, 1, 1, 1]
    var rms: RunningMeanStd
    var last: List[Float64]
    var obs: List[Scalar[DT]]
    var out: List[Scalar[DT]]

    def __init__(out self, run_dir: String) raises:
        self.agent = PPOAgent["cpu", WalkActor, WalkCritic, WALK_OBS, WALK_ACT, 1, 1, 1, 1](
            action_scale=Scalar[DT](WALK_ACTION_SCALE)
        )
        self.agent.load(run_dir + "/ckpt")
        self.rms = RunningMeanStd(WALK_OBS)
        self.rms.load(run_dir + "/obs_norm.txt")
        self.last = List[Float64](length=WALK_ACT, fill=0.0)
        self.obs = List[Scalar[DT]](length=WALK_OBS, fill=0)
        self.out = List[Scalar[DT]](length=WALK_ACT, fill=0)

    def reset(mut self):
        for j in range(WALK_ACT):
            self.last[j] = 0.0

    def act(mut self, env_obs: List[Float64]) raises -> List[Float64]:
        """The greedy action for the env's 70-word observation; the
        returned list is what to execute (the actor's mean clamped +-2)."""
        if len(env_obs) != G1_WALK_OBS_DIM:
            raise Error("WalkPolicy.act: expected 70 env words, got " + String(len(env_obs)))
        for k in range(WALK_OBS):
            var x = env_obs[k] if k < G1_WALK_OBS_DIM else self.last[k - G1_WALK_OBS_DIM]
            var z = (x - self.rms.mean[k]) / sqrt(self.rms.var_[k] + 1e-8)
            if z > WALK_OBS_CLIP:
                z = WALK_OBS_CLIP
            elif z < -WALK_OBS_CLIP:
                z = -WALK_OBS_CLIP
            self.obs[k] = Scalar[DT](z)
        self.agent.select_greedy_action(self.obs, self.out)
        var a = List[Float64](length=WALK_ACT, fill=0.0)
        for j in range(WALK_ACT):
            a[j] = Float64(self.out[j])
            self.last[j] = a[j]
        return a^

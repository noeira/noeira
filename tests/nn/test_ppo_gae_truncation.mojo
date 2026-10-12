"""PPO's GAE handles episode ends right: a truncation bootstraps from the
ending episode's OWN final state, a termination from 0, and the recursion
never crosses an episode boundary — host (`PPOGAEStep`) and device
(`PPODeviceRollout.gae`).

Before 2026-10-05 GAE read only `term_buf`: a time-limit truncation
bootstrapped from the NEXT episode's first state and `last_gae` ran on across
the boundary (every family-driver run truncates every lane).

Fixture: T = 6, four envs, a Linear critic with known weights (so V(s) is
exact), and the four cases that matter —

    env 0  no episode end              (bootstraps the last row from V(s_T))
    env 1  truncated at t = 2          (V(its own final obs), cut at t = 2)
    env 2  terminated at t = 3         (0, cut at t = 3)
    env 3  truncated at t = 5 (last)   (V(its final obs))

The reference is the formula written out:

    nv_t  = (1 - term_t) V(next_obs_t)  if done_t or t = T - 1,  else val_{t+1}
    gae_t = r_t + gamma nv_t - val_t + gamma lambda (1 - done_t) gae_{t+1}

It also checks the BUG is gone: env 1's advantage at t = 2 must NOT depend on
row t = 3 (the next episode) — the fixture gives that row a huge reward.

    pixi run mojo run -I . tests/nn/test_ppo_gae_truncation.mojo
"""

from layout import Layout
from max.gpu.host import DeviceContext
from std.random import random_float64, seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.call import call_forward
from noeira.nn.core.initializer import Xavier
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.primitives.linear import Linear
from noeira.deep_agents.ppo.blocks.gae_step import PPOGAEStep
from noeira.deep_agents.ppo.blocks.device_rollout import PPODeviceRollout
from noeira.deep_agents.training.onpolicy_state import OnPolicyState


comptime OBS = 3
comptime T = 6
comptime N = 4
comptime RN = T * N
comptime Critic = Linear[OBS, 1]
comptime GAMMA: Scalar[DT] = 0.9
comptime LAMBDA: Scalar[DT] = 0.8
comptime W0: Scalar[DT] = 0.5
comptime W1: Scalar[DT] = -0.25
comptime W2: Scalar[DT] = 0.125
comptime B: Scalar[DT] = 0.1


def v_of(x0: Scalar[DT], x1: Scalar[DT], x2: Scalar[DT]) -> Scalar[DT]:
    return W0 * x0 + W1 * x1 + W2 * x2 + B


def fill[target: StaticString](
    mut st: OnPolicyState[OBS, 1, T, 8, N],
    ref rew: List[Scalar[DT]], ref val: List[Scalar[DT]],
    ref done: List[Scalar[DT]], ref term: List[Scalar[DT]],
    ref nobs: List[Scalar[DT]],
) raises:
    for i in range(RN):
        st.rew_buf.data[i] = rew[i]
        st.val_buf.data[i] = val[i]
        st.done_buf.data[i] = done[i]
        st.term_buf.data[i] = term[i]
    for i in range(RN * OBS):
        st.next_obs_buf.data[i] = nobs[i]
    comptime if target == "gpu":
        var c = st.ctx.value()
        st.rew_buf.upload_resident(c)
        st.val_buf.upload_resident(c)
        st.done_buf.upload_resident(c)
        st.term_buf.upload_resident(c)
        st.next_obs_buf.upload_resident(c)


def make_critic[target: StaticString](ctx: Optional[DeviceContext]) raises -> Critic:
    var c = Critic.make[target, Xavier](ctx)
    c.weight.val.data[0] = W0
    c.weight.val.data[1] = W1
    c.weight.val.data[2] = W2
    c.bias.val.data[0] = B
    comptime if target == "gpu":
        c.weight.val.upload_resident(ctx.value())
        c.bias.val.upload_resident(ctx.value())
    return c^


def main() raises:
    seed(5)
    print("--- PPO GAE: truncation / termination / episode boundaries ---")
    var rew = List[Scalar[DT]](length=RN, fill=Scalar[DT](0))
    var val = List[Scalar[DT]](length=RN, fill=Scalar[DT](0))
    var done = List[Scalar[DT]](length=RN, fill=Scalar[DT](0))
    var term = List[Scalar[DT]](length=RN, fill=Scalar[DT](0))
    var nobs = List[Scalar[DT]](length=RN * OBS, fill=Scalar[DT](0))
    for i in range(RN):
        rew[i] = Scalar[DT](random_float64(-1.0, 1.0))
        val[i] = Scalar[DT](random_float64(-1.0, 1.0))
    for i in range(RN * OBS):
        nobs[i] = Scalar[DT](random_float64(-2.0, 2.0))
    # env 1 truncated at t = 2; env 2 terminated at t = 3; env 3 truncated
    # at the last row
    done[2 * N + 1] = 1.0
    done[3 * N + 2] = 1.0
    term[3 * N + 2] = 1.0
    done[5 * N + 3] = 1.0
    # the row right after env 1's truncation belongs to its NEXT episode: a
    # huge reward there must not reach env 1's advantage at t = 2
    rew[3 * N + 1] = 1000.0

    # reference
    var ref_adv = List[Scalar[DT]](length=RN, fill=Scalar[DT](0))
    for e in range(N):
        var g: Scalar[DT] = 0.0
        for t in range(T - 1, -1, -1):
            var i = t * N + e
            var ended = done[i] > Scalar[DT](0.5)
            var nv: Scalar[DT]
            if ended or t == T - 1:
                nv = (Scalar[DT](1.0) - term[i]) * v_of(
                    nobs[i * OBS], nobs[i * OBS + 1], nobs[i * OBS + 2]
                )
            else:
                nv = val[(t + 1) * N + e]
            var cont = Scalar[DT](0.0) if ended else Scalar[DT](1.0)
            g = rew[i] + GAMMA * nv - val[i] + GAMMA * LAMBDA * cont * g
            ref_adv[i] = g

    # the bug's signature, independent of the reference
    var i_trunc = 2 * N + 1
    var trunc_expected = rew[i_trunc] + GAMMA * v_of(
        nobs[i_trunc * OBS], nobs[i_trunc * OBS + 1], nobs[i_trunc * OBS + 2]
    ) - val[i_trunc]
    assert_true(
        abs(ref_adv[i_trunc] - trunc_expected) < 1e-5,
        "reference: a truncated row is not its one-step bootstrapped delta",
    )

    for target_i in range(2):
        var host = target_i == 0
        var name = String("host") if host else String("device")
        var adv = List[Scalar[DT]](length=RN, fill=Scalar[DT](0))
        var ret = List[Scalar[DT]](length=RN, fill=Scalar[DT](0))
        if host:
            var st = OnPolicyState[OBS, 1, T, 8, N].make["cpu"](None)
            fill["cpu"](st, rew, val, done, term, nobs)
            var critic = make_critic["cpu"](None)
            var gae = PPOGAEStep[OBS, T, Critic].make["cpu"]()
            gae.step["cpu", 1, 8, N](st, critic, GAMMA, LAMBDA)
            for i in range(RN):
                adv[i] = st.adv_buf.data[i]
                ret[i] = st.ret_buf.data[i]
        else:
            var ctx = DeviceContext()
            var octx = Optional[DeviceContext](ctx)
            var st = OnPolicyState[OBS, 1, T, 8, N].make["gpu"](octx)
            fill["gpu"](st, rew, val, done, term, nobs)
            var critic = make_critic["gpu"](octx)
            # the trainer's device update: V of every next obs, then the kernel
            call_forward["gpu", RN](
                critic, TensorRefs[1](st.next_obs_buf), st.next_val_buf, octx
            )
            var dr = PPODeviceRollout()
            dr.gae(st, GAMMA, LAMBDA)
            st.adv_buf.download(ctx)
            st.ret_buf.download(ctx)
            for i in range(RN):
                adv[i] = st.adv_buf.data[i]
                ret[i] = st.ret_buf.data[i]
        # Float32 on the device (the critic GEMM may run TF32, the recursion
        # fused): judged relative to the advantage's size — the fixture's
        # next-episode reward makes some advantages ~1000.
        var worst: Scalar[DT] = 0.0
        for i in range(RN):
            var d = abs(adv[i] - ref_adv[i]) / (
                Scalar[DT](1.0) + abs(ref_adv[i])
            )
            if d > worst:
                worst = d
            assert_true(
                abs(ret[i] - (adv[i] + val[i])) < 1e-5,
                name + ": ret != adv + val",
            )
        print(" ", name, "max |adv - reference| / (1 + |reference|) =", worst)
        assert_true(worst < 5e-4, name + ": GAE disagrees with the reference")
        assert_true(
            abs(adv[i_trunc] - trunc_expected) < 1e-4,
            name + ": the truncated row leaked the next episode",
        )
    print("ALL PASSED")

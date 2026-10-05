"""PPO's mirror-symmetry loss (`deep_agents/ppo/mirror.mojo`) on the CPU.

    pixi run mojo run -I . tests/deep_agents/test_ppo_mirror.mojo

A small Gaussian actor (Linear 4->16, Tanh, GaussianHead 16->2) and a
4-word observation whose mirror swaps words 0 / 1 and negates word 2; the
action's mirror swaps its two words.

  1. INERT AT COEFF 0: two identical actors, one with the mirror pass on at
     coefficient 0, take the same PPO updates — their weights stay
     bit-identical (the pass adds exact zeros, and its forwards' caches are
     rewritten before the PPO vjp reads them).
  2. IT LEARNS SYMMETRY: with the PPO advantages at 0 (no policy-gradient
     signal), the mirror loss alone drives `mu(M s)` to `M mu(s)`: the loss
     falls 100x and the asymmetry on FRESH inputs falls with it.
  3. THE MAPS ARE CHECKED: a map that is not an involution is refused.
"""

from std.random import random_float64, seed
from std.math import sqrt

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.nn.core.initializer import Xavier
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import Tanh
from noeira.nn.optimizer.adam import Adam
from noeira.deep_agents.primitives.gaussian_head import GaussianHead
from noeira.deep_agents.ppo.actor_loss import PPOActorLoss

comptime OBS = 4
comptime ACT = 2
comptime B = 32
comptime Actor = Sequential[Linear[OBS, 16], Tanh[16], GaussianHead[16, ACT]]


def _fail(msg: String) raises:
    print("  FAIL:", msg)
    raise Error(msg)


def _maps() -> Tuple[List[Int], List[Float64], List[Int], List[Float64]]:
    var oi: List[Int] = [1, 0, 2, 3]
    var os: List[Float64] = [1.0, 1.0, -1.0, 1.0]
    var ai: List[Int] = [1, 0]
    var as_: List[Float64] = [1.0, 1.0]
    return (oi^, os^, ai^, as_^)


def _batch(mut s: Tensor, mut a: Tensor, mut olp: Tensor, mut adv: Tensor, adv_scale: Float64):
    for k in range(B * OBS):
        s.data[k] = Scalar[DT](random_float64() * 2.0 - 1.0)
    for k in range(B * ACT):
        a.data[k] = Scalar[DT](random_float64() * 2.0 - 1.0)
    for b in range(B):
        olp.data[b] = Scalar[DT](-2.0)
        adv.data[b] = Scalar[DT](adv_scale * (random_float64() * 2.0 - 1.0))


def _asym(mut actor: Actor) raises -> Float64:
    """Mean |mu(M s) - M mu(s)| over fresh inputs."""
    var s = Tensor.alloc(B * OBS)
    var sm = Tensor.alloc(B * OBS)
    var ao = Tensor.alloc(B * 2 * ACT)
    var aom = Tensor.alloc(B * 2 * ACT)
    for b in range(B):
        for k in range(OBS):
            s.data[b * OBS + k] = Scalar[DT](random_float64() * 2.0 - 1.0)
        sm.data[b * OBS + 0] = s.data[b * OBS + 1]
        sm.data[b * OBS + 1] = s.data[b * OBS + 0]
        sm.data[b * OBS + 2] = -s.data[b * OBS + 2]
        sm.data[b * OBS + 3] = s.data[b * OBS + 3]
    call_forward["cpu", B](actor, TensorRefs[Actor.ARITY](s), ao, None)
    call_forward["cpu", B](actor, TensorRefs[Actor.ARITY](sm), aom, None)
    var e = 0.0
    for b in range(B):
        e += abs(Float64(aom.data[b * 2 * ACT + 0] - ao.data[b * 2 * ACT + 1]))
        e += abs(Float64(aom.data[b * 2 * ACT + 1] - ao.data[b * 2 * ACT + 0]))
    return e / Float64(2 * B)


def _weights(mut actor: Actor) -> List[Float64]:
    var w = List[Float64]()
    ref l0 = actor.children[0]
    for k in range(len(l0.weight.val.data)):
        w.append(Float64(l0.weight.val.data[k]))
    ref h = actor.children[2]
    for k in range(len(h.weight.val.data)):
        w.append(Float64(h.weight.val.data[k]))
    return w^


def check_inert() raises:
    seed(5)
    var a1 = Actor.make["cpu", Xavier](ctx=None)
    seed(5)
    var a2 = Actor.make["cpu", Xavier](ctx=None)
    seed(5)
    var a3 = Actor.make["cpu", Xavier](ctx=None)
    var o3 = Adam(lr=Scalar[DT](1e-2))
    o3.adopt["cpu", M=Actor](a3, None)
    var l3 = PPOActorLoss[Actor, B].make["cpu"](clip_eps=0.2, entropy_coef=0.01)
    var o1 = Adam(lr=Scalar[DT](1e-2))
    o1.adopt["cpu", M=Actor](a1, None)
    var o2 = Adam(lr=Scalar[DT](1e-2))
    o2.adopt["cpu", M=Actor](a2, None)
    var l1 = PPOActorLoss[Actor, B].make["cpu"](clip_eps=0.2, entropy_coef=0.01)
    var l2 = PPOActorLoss[Actor, B].make["cpu"](clip_eps=0.2, entropy_coef=0.01)
    var m = _maps()
    l2.enable_mirror["cpu"](m[0], m[1], m[2], m[3], 0.0)
    # the control: the same pass at 0.5 must move the weights, or the
    # "inert" result above would be the shape of a pass that never ran
    l3.enable_mirror["cpu"](m[0], m[1], m[2], m[3], 0.5)
    var s = Tensor.alloc(B * OBS)
    var a = Tensor.alloc(B * ACT)
    var olp = Tensor.alloc(B)
    var adv = Tensor.alloc(B)
    for _ in range(20):
        _batch(s, a, olp, adv, 1.0)
        _ = l1.forward_backward["cpu"](a1, o1, s, a, olp, adv)
        _ = l2.forward_backward["cpu"](a2, o2, s, a, olp, adv)
        _ = l3.forward_backward["cpu"](a3, o3, s, a, olp, adv)
    var w1 = _weights(a1)
    var w2 = _weights(a2)
    var w3 = _weights(a3)
    var diff = 0.0
    var ctrl = 0.0
    for k in range(len(w1)):
        diff = max(diff, abs(w1[k] - w2[k]))
        ctrl = max(ctrl, abs(w1[k] - w3[k]))
    print("  inert: max |w_off - w_on(coeff 0)| after 20 updates =", diff,
          "| control at coeff 0.5:", ctrl)
    if diff != 0.0:
        _fail("the mirror pass at coefficient 0 changed the PPO update")
    if ctrl == 0.0:
        _fail("the mirror pass at 0.5 changed nothing: it is not running")


def check_learns() raises:
    seed(7)
    var actor = Actor.make["cpu", Xavier](ctx=None)
    var opt = Adam(lr=Scalar[DT](1e-2))
    opt.adopt["cpu", M=Actor](actor, None)
    var loss = PPOActorLoss[Actor, B].make["cpu"](clip_eps=0.2, entropy_coef=0.0)
    var m = _maps()
    loss.enable_mirror["cpu"](m[0], m[1], m[2], m[3], 1.0)
    var s = Tensor.alloc(B * OBS)
    var a = Tensor.alloc(B * ACT)
    var olp = Tensor.alloc(B)
    var adv = Tensor.alloc(B)
    var asym0 = _asym(actor)
    var first = -1.0
    var last = 0.0
    for it in range(400):
        _batch(s, a, olp, adv, 0.0)
        _ = loss.forward_backward["cpu"](actor, opt, s, a, olp, adv)
        last = loss.mirror.host_loss()
        if it == 0:
            first = last
    var asym1 = _asym(actor)
    print("  learns: mirror loss", first, "->", last, "| asymmetry on fresh inputs",
          asym0, "->", asym1)
    if not (last < first / 100.0) or not (asym1 < asym0 / 10.0):
        _fail("the mirror loss did not make the actor symmetric")


def check_refuses() raises:
    var loss = PPOActorLoss[Actor, B].make["cpu"]()
    var bad: List[Int] = [1, 2, 0, 3]          # a 3-cycle, not an involution
    var os: List[Float64] = [1.0, 1.0, 1.0, 1.0]
    var ai: List[Int] = [1, 0]
    var as_: List[Float64] = [1.0, 1.0]
    var refused = False
    try:
        loss.enable_mirror["cpu"](bad, os, ai, as_, 1.0)
    except e:
        refused = True
        print("  refuses:", e)
    if not refused:
        _fail("a non-involutive map was accepted")


def main() raises:
    print("1. inert at coefficient 0")
    check_inert()
    print("2. learns symmetry")
    check_learns()
    print("3. checks its maps")
    check_refuses()
    print("[PASS] ppo_mirror")

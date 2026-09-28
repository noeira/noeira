"""FB checkpoints are binary, and the text ones already on disk still load.

`FBTrainer.save_state` used to write `LegacyV2CheckpointWriter`'s v2 text format — one
DECIMAL FLOAT PER LINE, 23 bytes for a 4-byte number. On the G1 run that is a
500 MB `.ckpt` plus a 421 MB `.cpr` sidecar every checkpoint, ~86 GB over a
192 M-step run, and ~16.5 s of the step it lands on. It now writes v3 binary
through `BinaryCheckpointWriter`.

⚠⚠ THE READ MUST STAY BILINGUAL, AND THAT IS WHAT THIS FILE GATES. A run in
flight has v2 checkpoints on disk; a format change that could not read them
would make its own `--resume` useless — the failure would appear only at the
moment you needed the resume, hours later. `load_state` dispatches on the
`storage-ckpt v3` header, so both load, forever.

  [1] v3 round trip: save -> perturb -> load restores `B` and `pi_z`.
  [2] v2 round trip: a file written the OLD way still restores, through the
      same `load_state`.
  [3] NON-VACUITY, three ways. The trainer is perturbed between every save and
      load, so "restored" and "never changed" are distinguishable. The v2 file
      is written by this test with the old writer, so it is a real legacy file
      and not a v3 file relabelled. And the two files' SIZES are compared: if
      v3 were not actually binary they would match, and [1] would pass while
      testing nothing about the format.

Run: pixi run mojo run -I . tests/deep_agents/test_fb_checkpoint_formats.mojo
"""

from std.math import abs
from std.random import random_float64, seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU, Tanh
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.core.param import walk_params, ParamVisitorRef
from noeira.nn.core.checkpoint import LegacyV2CheckpointWriter
from noeira.io.fileio import file_size
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb import sample_z_uniform


comptime OBS: Int = 5
comptime ACT: Int = 3
comptime D: Int = 8
comptime BATCH: Int = 16
comptime HID: Int = 32
comptime SEED: Int = 20260911
comptime CK_V3: StaticString = "/tmp/test_fb_fmt_v3.ckpt"
comptime CK_V2: StaticString = "/tmp/test_fb_fmt_v2.ckpt"

comptime FNet = Sequential[Linear[OBS + ACT + D, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[Linear[OBS, HID], ReLU[HID], Linear[HID, D], LayerNorm[D]]
comptime ANet = Sequential[Linear[OBS + D, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH]


def _rt(n: Int) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    return t^


def _z(batch: Int) raises -> Tensor:
    var zl = sample_z_uniform[D](batch)
    var t = Tensor.alloc(batch * D)
    for i in range(batch * D):
        t.data[i] = zl[i]
    return t^


def _step(mut t: Trainer) raises:
    var s = _rt(BATCH * OBS); var a = _rt(BATCH * ACT)
    var sn = _rt(BATCH * OBS); var sp = _rt(BATCH * OBS)
    var z = _z(BATCH)
    t.load_batch(s, a, sn, sp, z)
    _ = t.train_step(want_loss=False)


def _save_v2(mut t: Trainer, path: String) raises:
    """`FBTrainer.save_state` EXACTLY as it was before v3 — the same writer,
    the same four prefixes, the same order. This is what makes [2] a real
    legacy-file test rather than a v3 file under another name."""
    var w = LegacyV2CheckpointWriter(save_moments=False)
    w.mode = 0
    walk_params["cpu"](t.bnet.online, w, t.ctx, "b")
    walk_params["cpu"](t.f1.online, w, t.ctx, "f1")
    walk_params["cpu"](t.f2.online, w, t.ctx, "f2")
    walk_params["cpu"](t.actor.online, w, t.ctx, "actor")
    w.mode = 1
    var s1 = ParamVisitorRef.of[type_of(w), "cpu"](w)
    t.bnet.online.for_each_state["cpu"](s1, t.ctx, "b")
    var s2 = ParamVisitorRef.of[type_of(w), "cpu"](w)
    t.f1.online.for_each_state["cpu"](s2, t.ctx, "f1")
    var s3 = ParamVisitorRef.of[type_of(w), "cpu"](w)
    t.f2.online.for_each_state["cpu"](s3, t.ctx, "f2")
    var s4 = ParamVisitorRef.of[type_of(w), "cpu"](w)
    t.actor.online.for_each_state["cpu"](s4, t.ctx, "actor")
    with open(path, "w") as f:
        f.write(w.content)


def _probe(mut t: Trainer, mut probe: Tensor, mut zp: Tensor,
           mut b: Tensor, mut a: Tensor) raises:
    t.backward_embed[BATCH](probe, b)
    t.act[BATCH](probe, zp, a)


def _maxdiff(a: Tensor, b: Tensor, n: Int) -> Float64:
    var m = Float64(0)
    for i in range(n):
        var e = abs(Float64(a.data[i]) - Float64(b.data[i]))
        if e > m:
            m = e
    return m


def main() raises:
    seed(SEED)
    print("FB checkpoint formats: v3 binary written, v2 text still read")
    var t = Trainer.make(lr=1e-3)
    for _ in range(15):
        _step(t)

    var probe = _rt(BATCH * OBS)
    var zp = _z(BATCH)
    var b0 = Tensor(); var a0 = Tensor()
    _probe(t, probe, zp, b0, a0)

    t.save_state(String(CK_V3))     # v3 binary, the new path
    _save_v2(t, String(CK_V2))      # v2 text, written the old way

    var n3 = file_size(String(CK_V3))
    var n2 = file_size(String(CK_V2))
    print("  v3 binary", n3, "B   v2 text", n2, "B   ratio", Float64(n2) / Float64(n3))
    # ---- [3] non-vacuity on the FORMAT ---------------------------------
    assert_true(
        n3 * 2 < n2,
        "the v3 file is not materially smaller than the v2 one (" + String(n3)
        + " vs " + String(n2) + ") — `save_state` is probably still writing"
        " text, so the round trips below would pass while testing nothing",
    )

    var ok = True
    for which in range(2):
        var path = String(CK_V3) if which == 0 else String(CK_V2)
        var label = String("v3 binary") if which == 0 else String("v2 text")
        # ⚠ perturb, or "restored" and "never changed" are one observation
        for _ in range(15):
            _step(t)
        var bm = Tensor(); var am = Tensor()
        _probe(t, probe, zp, bm, am)
        var moved = _maxdiff(b0, bm, BATCH * D)
        assert_true(
            moved > 1e-5,
            label + ": 15 steps did not move B, so the load cannot be shown"
            " to have restored anything",
        )
        t.load_state(path)
        var br = Tensor(); var ar = Tensor()
        _probe(t, probe, zp, br, ar)
        var db = _maxdiff(b0, br, BATCH * D)
        var da = _maxdiff(a0, ar, BATCH * ACT)
        print("  ", label, ": moved", moved, " after load |B diff|", db,
              " |pi_z diff|", da)
        if db > 1e-6 or da > 1e-6:
            ok = False

    assert_true(ok, "a checkpoint did not restore B / pi_z")
    print("FB_CKPT_FORMATS OK")

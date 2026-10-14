"""Checkpoint v3: the streaming writer against the in-memory one.

`save_params` / `save_params_multi` stream each section to `path + ".tmp"` and
rename it on `commit`, instead of building the whole file in memory first.

1. Byte identity: a model trained 3 Adam steps (moments populated), with `K`
   scalars, written by `save_params` and by `save_params_multi` (two models),
   equals byte for byte the in-memory writer's `content` for the same walk.
   CPU, and GPU (each value downloaded on the way out).
2. The chunk loop: the same file streamed with a 7-byte write chunk (every
   payload split, mostly at odd offsets) is still identical.
3. Atomicity: a streaming writer that walks the model and is NOT committed
   leaves the previous checkpoint at the path untouched; committing replaces
   it, and the file loads back.
4. Misuse raises: `commit` twice, and `commit` on an in-memory writer.

Run: pixi run -e apple mojo run -I . tests/nn/test_checkpoint_stream.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.checkpoint import (
    save_params, load_params, save_params_multi, write_model,
    BinaryCheckpointWriter, CheckpointScalars, _read_file_bytes,
)
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.optimizer.adam import Adam


comptime D = 4
comptime H = 6
comptime O = 3
comptime B = 5
comptime NET = Sequential[Linear[D, H], Linear[H, O]]
comptime SMALL = Sequential[Linear[D, O]]


def _train3[target: StaticString](
    mut net: NET, ctx: Optional[DeviceContext]
) raises:
    var opt = Adam(lr=1e-3)
    for step in range(3):
        var x = Tensor.alloc(B * D)
        var go = Tensor.alloc(B * O)
        for i in range(B * D):
            x.data[i] = Scalar[DT](((i + step) % 5) - 2) * 0.3
        for i in range(B * O):
            go.data[i] = Scalar[DT](((i * 3 + step) % 7) - 3) * 0.4
        var out = Tensor.alloc(B * O)
        var gi = Tensor.alloc(B * D)
        comptime if target == "gpu":
            x.upload(ctx.value()); go.upload(ctx.value())
        net.forward[target, B](TensorRefs[1](x), out, ctx)
        net.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
        opt.step[target](net, ctx)


def _scalars() raises -> CheckpointScalars:
    var sc = CheckpointScalars()
    sc.set_int("step", (1 << 24) + 3)
    sc.set("eps", 0.1)
    return sc^


def _ndiff(a: List[UInt8], b: List[UInt8]) -> Int:
    """Bytes that differ, counting a length mismatch as all of the tail."""
    var n = abs(len(a) - len(b))
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            n += 1
    return n


def _identity[target: StaticString](
    ctx: Optional[DeviceContext], tag: String
) raises -> Bool:
    var net = NET.make[target, Deterministic](ctx)
    _train3[target](net, ctx)
    var small = SMALL.make[target, Deterministic](ctx)

    # The in-memory sink, the reference.
    var ref1 = BinaryCheckpointWriter(True)
    write_model[target](ref1, net, ctx)
    ref1.write_scalars(_scalars())
    var ref2 = BinaryCheckpointWriter(True)
    write_model[target](ref2, net, ctx)
    write_model[target](ref2, small, ctx)
    ref2.write_scalars(_scalars())

    var p1 = "/tmp/ckpt_stream_" + tag + "_one.ckpt"
    var p2 = "/tmp/ckpt_stream_" + tag + "_multi.ckpt"
    save_params[target](net, p1, ctx, True, _scalars())
    save_params_multi[target](p2, ctx, True, net, small, scalars=_scalars())
    var d1 = _ndiff(_read_file_bytes(p1), ref1.content)
    var d2 = _ndiff(_read_file_bytes(p2), ref2.content)

    # The chunk loop, at 7 bytes a write.
    var p3 = "/tmp/ckpt_stream_" + tag + "_chunked.ckpt"
    var w = BinaryCheckpointWriter.to_file(p3, True)
    w.chunk = 7
    write_model[target](w, net, ctx)
    w.write_scalars(_scalars())
    var streamed = w.bytes_written
    w.commit()
    var d3 = _ndiff(_read_file_bytes(p3), ref1.content)

    print(
        "  [" + tag + "] bytes differing from the in-memory writer:"
        " save_params", d1, "of", len(ref1.content),
        "| save_params_multi", d2, "of", len(ref2.content),
        "| 7-byte chunks", d3, "(bytes_written", streamed, ")",
    )
    # Moments present, or the identity says nothing about them: 3 Adam steps
    # put m and v beside every value, so the file holds them three times.
    var n_vals = (D * H + H) + (H * O + O)
    var has_moments = len(ref1.content) > 3 * n_vals * 4
    return (
        d1 == 0 and d2 == 0 and d3 == 0 and has_moments
        and streamed == len(ref1.content)
    )


def _atomic() raises -> Bool:
    var path = String("/tmp/ckpt_stream_atomic.ckpt")
    var a = NET.make["cpu", Deterministic](None)
    save_params["cpu"](a, path, None)
    var before = _read_file_bytes(path)

    var b = NET.make["cpu", Deterministic](None)
    _train3["cpu"](b, None)
    var w = BinaryCheckpointWriter.to_file(path, True)
    write_model["cpu"](w, b, None)
    var untouched = _ndiff(_read_file_bytes(path), before) == 0
    w.commit()
    var replaced = _ndiff(_read_file_bytes(path), before) > 0

    var c = NET.make["cpu", Deterministic](None)
    _ = load_params["cpu"](c, path, None)
    var r = BinaryCheckpointWriter(True)
    write_model["cpu"](r, c, None)
    var q = BinaryCheckpointWriter(True)
    write_model["cpu"](q, b, None)
    # Loaded params (moments are loaded too) re-serialise to the same bytes.
    var loads = _ndiff(r.content, q.content) == 0
    print(
        "  atomic: target untouched before commit", untouched,
        "| replaced on commit", replaced, "| loads back", loads,
    )
    return untouched and replaced and loads


def _misuse() raises -> Bool:
    var twice = False
    var w = BinaryCheckpointWriter.to_file("/tmp/ckpt_stream_twice.ckpt", True)
    w.commit()
    try:
        w.commit()
    except:
        twice = True
    var mem = False
    var m = BinaryCheckpointWriter(True)
    try:
        m.commit()
    except:
        mem = True
    print("  misuse: commit twice raises", twice, "| commit in memory raises", mem)
    return twice and mem


def main() raises:
    print("Checkpoint v3: streaming writer vs in-memory writer")
    var ok_cpu = _identity["cpu"](None, "cpu")
    var ctx = DeviceContext()
    var ok_gpu = _identity["gpu"](Optional(ctx), "gpu")
    var ok_atomic = _atomic()
    var ok_misuse = _misuse()
    assert_true(ok_cpu and ok_gpu and ok_atomic and ok_misuse, "streaming checkpoint")
    print("PASSED")

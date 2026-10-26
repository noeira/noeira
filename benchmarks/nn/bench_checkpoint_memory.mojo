"""Peak memory of a checkpoint save: the streaming writer against the in-memory one.

A 1.07 GB model (four `Linear[8192, 8192]`, 268 M values, no moments) is saved
once per process, so each run's peak resident set is its own:

    pixi run mojo build -I . benchmarks/nn/bench_checkpoint_memory.mojo -o /tmp/bcm
    /usr/bin/time -l /tmp/bcm stream    # save_params: sections streamed to the file
    /usr/bin/time -l /tmp/bcm memory    # the whole file built in `content` first

`/usr/bin/time -l` (macOS) prints "maximum resident set size" in bytes; on Linux
use `/usr/bin/time -v` (kilobytes). `none` builds the model and saves nothing,
the baseline both are compared with.
"""

from std.sys import argv

from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.checkpoint import (
    save_params, write_model, BinaryCheckpointWriter, _write_file_bytes,
)
from noeira.io.fileio import remove_file
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential


comptime W = 8192
comptime NET = Sequential[Linear[W, W], Linear[W, W], Linear[W, W], Linear[W, W]]


def main() raises:
    var args = argv()
    var mode = String(args[1]) if len(args) > 1 else String("stream")
    var net = NET.make["cpu", Deterministic](None)
    var path = String("/tmp/bench_checkpoint_memory.ckpt")
    if mode == "stream":
        save_params["cpu"](net, path, None, False)
    elif mode == "memory":
        var w = BinaryCheckpointWriter(False)
        write_model["cpu"](w, net, None)
        _write_file_bytes(path, w.content)
    elif mode != "none":
        raise Error("mode: stream, memory or none")
    var values = 4 * (W * W + W)
    print(
        "mode", mode, "| model", values, "values,",
        Float64(values * 4) / 1e9, "GB",
    )
    if mode != "none":
        remove_file(path)

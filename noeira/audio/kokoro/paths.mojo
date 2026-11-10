"""Where the converted Kokoro checkpoint lives on this machine."""

from std.os.path import exists

from noeira.io.hf import noeira_cache


def kokoro_dir() raises -> String:
    """`~/.cache/noeira/kokoro` (or under `$NOEIRA_CACHE`): what
    `tools/kokoro/convert_kokoro_checkpoint.py` writes. Raises with that
    command when the checkpoint is not there."""
    var dir = noeira_cache() + "/kokoro"
    if not exists(dir + "/kokoro-v1_0.safetensors"):
        raise Error(
            "kokoro: no converted checkpoint in " + dir + ". Make it:"
            " pixi run -e detect python tools/kokoro/convert_kokoro_checkpoint.py"
        )
    return dir

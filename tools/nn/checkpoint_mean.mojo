"""The mean of a run's kept copies, as one checkpoint.

    pixi run checkpoint-mean --out mean.ckpt a.ckpt b.ckpt c.ckpt d.ckpt
    pixi run checkpoint-mean --list a.ckpt

The inputs are v3 checkpoints of ONE run (the same model saved at different
moments); the output has the same sections, each tensor the element-wise mean
of the inputs' and each scalar the last input's. `--out` is refused when the
file exists, unless `--force`. `--list` prints a file's sections and changes
nothing. See `noeira/nn/core/checkpoint_mean.mojo` for what a mean is good for
and what it is not (a copy to evaluate or deploy, not one to resume from).
"""

from std.sys import argv

from noeira.nn.core.checkpoint_mean import (
    checkpoint_layout, mean_checkpoints,
)


def _usage() -> String:
    return (
        "usage: checkpoint-mean --out OUT IN1 IN2 [IN3 ...] [--force]\n"
        "       checkpoint-mean --list IN"
    )


def _names(v: List[String]) -> String:
    if len(v) == 0:
        return String("none")
    var s = String("")
    for i in range(len(v)):
        if i > 0:
            s += ", "
        s += v[i]
    return s


def main() raises:
    var av = argv()
    var out = String("")
    var list_path = String("")
    var force = False
    var inputs = List[String]()
    var i = 1
    while i < len(av):
        var a = String(av[i])
        if a == "--out" or a == "--list":
            if i + 1 >= len(av):
                raise Error(a + " needs a path\n" + _usage())
            if a == "--out":
                out = String(av[i + 1])
            else:
                list_path = String(av[i + 1])
            i += 2
            continue
        if a == "--force":
            force = True
        elif a.startswith("--"):
            raise Error("unknown flag " + a + "\n" + _usage())
        else:
            inputs.append(a)
        i += 1

    if list_path.byte_length() > 0:
        if out.byte_length() > 0 or len(inputs) > 0:
            raise Error("--list takes one file and nothing else\n" + _usage())
        var secs = checkpoint_layout(list_path)
        var values = 0
        for k in range(len(secs)):
            print(secs[k].header)
            values += secs[k].nbytes // 4 if secs[k].kind != "K" else 0
        print(
            list_path + ": " + String(len(secs)) + " sections, "
            + String(values) + " values"
        )
        return

    if out.byte_length() == 0:
        raise Error(_usage())
    var rep = mean_checkpoints(out, inputs, force)
    print(
        out + ": the mean of " + String(rep.inputs) + " copies, "
        + String(rep.sections) + " sections, " + String(rep.values)
        + " values; the largest |mean - first input| is "
        + String(rep.max_shift)
    )
    print("  states that differ between the inputs (averaged): "
          + _names(rep.states_differing))
    print("  scalars that differ between the inputs (the last input's kept): "
          + _names(rep.scalars_differing))
    if rep.max_shift == 0.0:
        print("  ! the inputs hold the same tensors: this is the first"
              " input's copy, not a mean")

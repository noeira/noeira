"""Fit the tower's CHUNKED state policy to expert recordings.

    pixi run -e apple mojo run -I . examples/so101/tower_chunk_bc_fit.mojo \\
        --demos projects/so101-tower/demos/expert_lift_brick_jo_clean_600.demo \\
        --out projects/so101-tower/policies/chunk_bc/lift_brick_h32.ckpt \\
        [--max-episodes 0] [--val-episodes 20] [--epochs 40] [--lr 3e-4]

One row per recorded step of a successful episode: the 49-word state against
the `TOWER_CHUNK_H` actions recorded from that step on
(`BcDataset.add_demo_set_chunked`). `fit_bc` standardises the inputs, fits
`TowerChunkNet`, writes the checkpoint and its `.norm` sidecar, and re-scores
the reloaded file on the CPU.

`--max-episodes N` keeps the first N successful episodes of EACH file: the
way to get a deliberately weaker base for `tower_async_rl.mojo`, where the
question is what RL adds beside a frozen imperfect policy.

`--demos a.demo,b.demo` concatenates recordings; each holds out its own last
`--val-episodes` episodes.
"""

from std.sys import argv
from std.pathlib import Path

from noeira.deep_agents.bc.dataset import BcDataset
from noeira.deep_agents.bc.fit import fit_bc
from noeira.deep_agents.demos.file import read_demo_file
from noeira.tasks.so101_tower_chunk import (
    TOWER_ACT, TOWER_CHUNK_H, TOWER_CHUNK_OUT,
)
from noeira.tasks.so101_tower_xml import So101TowerModel

comptime OBS = So101TowerModel.OBS_DIM
comptime BATCH = 256


def _usage():
    print("usage: tower_chunk_bc_fit.mojo --demos F[,G] --out CKPT"
          " [--max-episodes N] [--val-episodes N] [--epochs N] [--lr X]")


def main() raises:
    var args = argv()
    var demos = String("")
    var out = String("")
    var max_episodes = 0
    var val_episodes = 20
    var epochs = 40
    var lr = 3.0e-4
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--help" or a == "-h":
            _usage()
            return
        if i + 1 >= len(args):
            _usage()
            raise Error("missing value for " + a)
        var v = String(args[i + 1])
        if a == "--demos":
            demos = v
        elif a == "--out":
            out = v
        elif a == "--max-episodes":
            max_episodes = Int(v)
        elif a == "--val-episodes":
            val_episodes = Int(v)
        elif a == "--epochs":
            epochs = Int(v)
        elif a == "--lr":
            lr = Float64(v)
        else:
            _usage()
            raise Error("unrecognised argument: " + a)
        i += 2
    if demos.byte_length() == 0 or out.byte_length() == 0:
        _usage()
        raise Error("--demos and --out are required")

    print("=" * 66)
    print("so101_tower — chunk BC fit: state", OBS, "->", TOWER_CHUNK_H, "x",
          TOWER_ACT, "actions")
    print("=" * 66)
    var data = BcDataset(OBS, TOWER_CHUNK_OUT)
    var parts = demos.split(",")
    for p in range(len(parts)):
        var path = String(String(parts[p]).strip())
        if path.byte_length() == 0:
            raise Error("--demos: empty element in '" + demos + "'")
        if not Path(path).exists():
            raise Error("--demos: no such file: " + path)
        var ds = read_demo_file(path)
        if ds.act_dim != TOWER_ACT:
            raise Error(
                path + ": actions of " + String(ds.act_dim) + " words, the"
                " tower's are " + String(TOWER_ACT)
            )
        var used = data.add_demo_set_chunked(
            ds, TOWER_CHUNK_H, val_episodes, max_episodes
        )
        print("  ", path, ":", ds.n_episodes(), "episodes,", ds.n_successes(),
              "successful,", used, "used")
    print("  rows  : train", data.n_tr, "| validation", data.n_va)
    var rep = fit_bc[OBS, TOWER_CHUNK_OUT, BATCH](
        data, epochs, lr, out, "tower-chunk-bc"
    )
    print("  best val MSE", rep.best_val, "| reloaded", rep.reloaded_val,
          "| predict-mean baseline", rep.mse_mean)

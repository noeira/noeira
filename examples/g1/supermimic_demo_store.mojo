# +--------------------------------------------------------------------------+ #
# | SuperMimic — gather chosen SEED clips into one small demo store
# +--------------------------------------------------------------------------+ #
"""Copy a list of (store, episode) clips into ONE `TrajectoryStore` in the
list's order, for `supermimic_motion_picker.mojo` on a machine that cannot
hold all of SEED (260 stores, ~14 GB; one take per motion is ~0.44 GB).

    pixi run mojo build -I . examples/g1/supermimic_demo_store.mojo -o /tmp/sm_demo_store
    /tmp/sm_demo_store --list demo_list.tsv --out seed_demo.h5

`--list` is `tools/supermimic/demo_catalog.py`'s `--out-list`: one
`store_path<TAB>episode` per line. Episode i of the output is line i, with
the same `ref` rows, a `motion_id` of i, the clip's name as task i, and
`fps` = `SEED_OUT_FPS` (every SEED store is written at it), the layout
`data/bones_seed_import.mojo` writes. Each source store is read ONCE (they
are ~55 MB; the list visits them in catalogue order, not store order).

⚠ SEED IS CONFIDENTIAL: the output is SEED data. Write it on a private
machine and keep it out of the repo.
"""

from std.sys import argv

from noeira.data.bones_seed import SEED_OUT_FPS, SEED_REF_DIM
from noeira.data.bones_seed_import import SEED_ENV_ID, SEED_MOTION_ID_COLUMN, SEED_REF_COLUMN
from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStore, TrajectoryStoreWriter
from noeira.io.fileio import rename_over


def _opt(args: List[String], name: String) raises -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return args[i + 1].copy()
    raise Error("give " + name + " PATH")


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(len(raw)):
        args.append(String(raw[i]))
    var list_path = _opt(args, String("--list"))
    var out = _opt(args, String("--out"))

    var stores = List[String]()
    var eps = List[Int]()
    with open(list_path, "r") as f:
        for line in f.read().split("\n"):
            var s = String(line).strip()
            if s.byte_length() == 0:
                continue
            var parts = String(s).split("\t")
            if len(parts) != 2:
                raise Error("bad list line: " + String(s))
            stores.append(String(parts[0]))
            eps.append(Int(String(parts[1])))
    var n = len(stores)
    print("clips:", n)

    # Each distinct store once, in sorted order; the clips it holds are copied out.
    var uniq = List[String]()
    for i in range(n):
        var seen = False
        for u in uniq:
            if u == stores[i]:
                seen = True
                break
        if not seen:
            uniq.append(stores[i].copy())
    var rows = List[List[Float32]](length=n, fill=List[Float32]())
    var names = List[String](length=n, fill=String(""))
    for ui in range(len(uniq)):
        var st = TrajectoryStore(uniq[ui].copy())
        var refc = st.load_column[DType.float32](String(SEED_REF_COLUMN))
        for i in range(n):
            if stores[i] != uniq[ui]:
                continue
            var e = eps[i]
            if e < 0 or e >= st.n_episodes():
                raise Error(uniq[ui] + " has no episode " + String(e))
            var s0 = st.episodes.start_of(e)
            var ln = st.episodes.length_of(e)
            var r = List[Float32](length=ln * SEED_REF_DIM, fill=Float32(0))
            for k in range(ln * SEED_REF_DIM):
                r[k] = refc[s0 * SEED_REF_DIM + k]
            rows[i] = r^
            names[i] = st.manifest.task_text(e)
        if (ui + 1) % 20 == 0 or ui == len(uniq) - 1:
            print("  read", ui + 1, "/", len(uniq), "stores")

    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String(SEED_REF_COLUMN), DType.float32, SEED_REF_DIM))
    cols.append(ColumnSpec(String(SEED_MOTION_ID_COLUMN), DType.int32, 1))
    var part = out + ".part"
    var w = TrajectoryStoreWriter(part, cols^, String(SEED_ENV_ID), 0, String(""))
    var total = 0
    for i in range(n):
        var ln = len(rows[i]) // SEED_REF_DIM
        var ids = List[Int32](length=ln, fill=Int32(i))
        w.append[DType.float32](
            String(SEED_REF_COLUMN),
            rows[i].unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]().as_unsafe_any_origin(),
            ln,
        )
        w.append[DType.int32](
            String(SEED_MOTION_ID_COLUMN),
            ids.unsafe_ptr().unsafe_bitcast[Scalar[DType.int32]]().as_unsafe_any_origin(),
            ln,
        )
        w.end_episode()
        w.add_task(i, names[i].copy())
        total += ln
    var fps = List[Float64](length=1, fill=SEED_OUT_FPS)
    w.write_vector[DType.float64](
        String("fps"), fps.unsafe_ptr().unsafe_bitcast[Scalar[DType.float64]]().as_unsafe_any_origin(), 1
    )
    w.close()
    rename_over(part, out.copy())
    print("wrote", out, ":", n, "clips,", total, "frames")

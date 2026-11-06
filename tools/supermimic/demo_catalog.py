"""One take per SEED motion: the clip list for the demo store + the viewer's catalogue.

    python3 tools/supermimic/demo_catalog.py --manifests DIR --metadata seed_metadata_v004.csv \
        --store-dir /workspace/seed/stores --out-list demo_list.tsv --out-catalog demo_catalog.tsv \
        [--benchmark benchmark.csv]

Inputs:
  * `--manifests`: the `seed_d_*.h5.manifest.csv` of every store, in the
    directory order the trainer and `supermimic_eval` read them (sorted).
    Kept rows (`keep == 1`) are the stores' episodes, in order, so the
    global clip index of the benchmark CSV is the running count of kept rows.
  * `--metadata`: SEED's `seed_metadata_v004.csv`. ⚠ JOINED ON THE G1 CSV
    STEM (`move_g1_path`), NOT `move_name`: a third of the clips (e.g.
    `confusion_001__A064`) are named by their file, which `move_name` spells
    differently (`neutral_confusion_R_001__A542`).
  * `--benchmark` (optional): `supermimic_eval`'s per-clip CSV; each picked
    take carries its normalised return and whether it failed.
  * `--only-stores` (optional): pick only from the stores present in that
    directory (a machine holding part of SEED). The global clip index still
    counts every manifest, so the benchmark rows stay aligned.

One take per `content_name` (the motion, across takes, actors and mirrors):
the first non-mirrored kept clip by (take name, actor). Both outputs are in
the catalogue's order (category, then motion name), which is also the demo
store's episode order — row i of the catalogue is episode i of the store.

Outputs (TSV, no quoting: tabs and newlines in the text are replaced):
  * `--out-list`: `store_path<TAB>episode` per pick, for
    `examples/g1/supermimic_demo_store.mojo`.
  * `--out-catalog`: header + `clip name category package frames
    eval_return eval_failed description` per pick, for
    `examples/g1/supermimic_motion_picker.mojo`.

⚠ SEED IS CONFIDENTIAL: both outputs are SEED-derived. Write them under a
private directory (e.g. `~/seed_private/`), never into the repo.
"""

import argparse
import csv
import os
import sys


def _clean(s):
    return " ".join(s.replace("\t", " ").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--out-list", required=True)
    ap.add_argument("--out-catalog", required=True)
    ap.add_argument("--benchmark", default="")
    ap.add_argument("--only-stores", default="",
                    help="pick only from the stores (*.h5) present in this directory")
    a = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    meta = {}
    with open(a.metadata, newline="") as f:
        rd = csv.reader(f)
        h = next(rd)
        col = {n: i for i, n in enumerate(h)}
        for r in rd:
            p = r[col["move_g1_path"]]
            meta[os.path.basename(p)[:-4] if p.endswith(".csv") else p] = r

    clips = []  # (global index, name, store file, episode, frames)
    files = sorted(x for x in os.listdir(a.manifests) if x.endswith(".manifest.csv"))
    for fn in files:
        store = fn[: -len(".manifest.csv")]
        ep = 0
        with open(os.path.join(a.manifests, fn), newline="") as f:
            rd = csv.reader(f)
            next(rd)
            for r in rd:
                if r[2] != "1":
                    continue
                clips.append((len(clips), r[0], store, ep, int(r[1])))
                ep += 1

    bench = {}
    if a.benchmark:
        with open(a.benchmark, newline="") as f:
            for r in csv.DictReader(f):
                bench[int(r["clip"])] = (float(r["norm_return"]), r["failed"] == "1", int(r["frames"]))

    have = None
    if a.only_stores:
        have = set(x for x in os.listdir(a.only_stores) if x.endswith(".h5"))
    best = {}
    for g, name, store, ep, frames in clips:
        if have is not None and store not in have:
            continue
        m = meta.get(name)
        if m is None:
            raise SystemExit("no metadata row for clip " + name)
        if m[col["is_mirror"]] == "True":
            continue
        key = m[col["content_name"]]
        rank = (m[col["take_name"]], m[col["take_actor"]])
        if key not in best or rank < best[key][0]:
            best[key] = (rank, g, name, store, ep, frames, m)

    picks = sorted(best.values(), key=lambda b: (b[6][col["category"]], b[6][col["content_name"]]))
    with open(a.out_list, "w") as fl, open(a.out_catalog, "w") as fc:
        fc.write("clip\tname\tcategory\tpackage\tframes\teval_return\teval_failed\tdescription\n")
        for _, g, name, store, ep, frames, m in picks:
            if g in bench and bench[g][2] != frames:
                raise SystemExit("benchmark row %d has %d frames, manifest %d" % (g, bench[g][2], frames))
            ret, failed = bench[g][:2] if g in bench else (float("nan"), False)
            desc = m[col["content_natural_desc_4"]] or m[col["content_natural_desc_1"]]
            fl.write("%s\t%d\n" % (os.path.join(a.store_dir, store), ep))
            fc.write("%d\t%s\t%s\t%s\t%d\t%.4f\t%d\t%s\n" % (
                g, name, _clean(m[col["category"]]), _clean(m[col["package"]]), frames, ret, int(failed),
                _clean(desc)))
    print("%d clips, %d motions picked -> %s, %s" % (len(clips), len(picks), a.out_list, a.out_catalog))


if __name__ == "__main__":
    main()

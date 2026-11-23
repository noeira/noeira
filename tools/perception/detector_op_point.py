#!/usr/bin/env python3
"""Precision and recall at a FIXED score threshold — a detector's own decision
point, where `detector_eval.py` reports AP and the best-F1 point — with
`detector_eval.py`'s own subset and matching rules (IoU 0.5, difficult
instances neither TP nor FP), per word: boxes (TP, FP, recall) and pictures
(how often the word is said present when it is there, and when it is not).

    pixi run -e detect python tools/perception/detector_op_point.py <words dir A> <threshold A> <words dir B> <threshold B>

The directories are `--detectors words:<dir>` outputs (`tools/perception/d1_frames.mojo`,
`tools/perception/sam3_frames.mojo`; docs/NN_MODEL_PORTS_PLAN.md §12.9, T5):
d1's own decision point is presence >= 0.5, SAM 3's default score 0.3.
"""
import os, sys
import h5py, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "tools/perception"))
import detector_eval as E

f = h5py.File(os.path.expanduser("~/.cache/noeira/datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat"), "r")
idx, _ = E.select(f, 120, 120, 0)
for d, thr in zip(sys.argv[1::2], map(float, sys.argv[2::2])):
    det = E.PrecomputedWords("cpu", d)
    st = {w: [0, 0, 0] for w in E.WORDS}  # tp, fp, n_gt
    img_hit = {w: [0, 0, 0, 0] for w in E.WORDS}  # word present & said, present & not, absent & said, absent & not
    for i in idx:
        img, lab, ins, dep = E.load(f, i)
        det.frame = i
        dets = det(img, list(E.WORDS))
        for w in E.WORDS:
            gts = E.instances_of(lab, ins, E.WORDS[w])
            gb = [E.box_of(m) for m, _ in gts]
            st[w][2] += sum(1 for _, dd in gts if not dd)
            used = [False] * len(gts)
            mine = sorted([(x[1], x[2]) for x in dets if x[0] == w and x[1] >= thr], key=lambda t: -t[0])
            for sc, b in mine:
                jb, ob = -1, 0.5
                for j, g in enumerate(gb):
                    o = E.iou(b, g)
                    if o >= ob and not used[j]:
                        jb, ob = j, o
                if jb >= 0:
                    used[jb] = True
                    if not gts[jb][1]:
                        st[w][0] += 1
                else:
                    st[w][1] += 1
            present = any(not dd for _, dd in gts)
            said = len(mine) > 0
            img_hit[w][(0 if present else 2) + (0 if said else 1)] += 1
    print(f"\n{det.name} at score >= {thr}")
    for w in E.WORDS:
        tp, fp, n = st[w]
        a, b, c, dd = img_hit[w]
        print(f"  {w:7s} boxes: P {tp / max(1, tp + fp):.2f} R {tp / max(1, n):.2f} ({tp} TP, {fp} FP, {n} GT) | "
              f"pictures: said present {a}/{a + b} when there, {c}/{c + dd} when absent")

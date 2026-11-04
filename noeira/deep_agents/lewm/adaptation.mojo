"""AdaJEPA's test-time adaptation windows for LeWM on PushT.

docs/LEWM_REOPEN_PLAN.md P7. The pairs, frames and shifts it adapts under
are the PushT protocol's (`noeira/envs/pusht/eval_pairs.mojo`).
"""

from noeira.nn.constants import DT
from noeira.envs.pusht.eval_pairs import imagenet_from_hwc255, PAIR_HW


comptime TTA_ACT = 10
"""One executed action block: frameskip 5 x 2 dims."""


def tta_windows[TB: Int, T: Int](
    frames: List[List[Scalar[DT]]], acts: List[List[Scalar[DT]]],
) -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    """An episode's newest TB windows of T consecutive frames, for the
    pretraining loss: pixels (TB, T, 3, 224, 224) ImageNet-normalised and
    actions (TB, T, 10).

    `frames[j]` is the observation after j executed blocks; `acts[j]` the
    block executed FROM frame j (it produced frame j + 1) — the dataset's
    alignment (`LewmPushTExpert`: action t is the dense span starting at frame
    t). Window b ends at frame n − 1 − b; with fewer than TB windows they are
    cycled. A window's LAST action is the one after its last frame: not yet
    taken for the newest window, so zero — the loss never reads it (the
    predictor sees the first T − 1 actions). Needs len(frames) >= T."""
    var n_win = len(frames) - T + 1
    var pix = List[Scalar[DT]](capacity=TB * T * 3 * PAIR_HW)
    var act = List[Scalar[DT]](capacity=TB * T * TTA_ACT)
    for b in range(TB):
        var k = len(frames) - T - (b % n_win)
        for t in range(T):
            var x = imagenet_from_hwc255(frames[k + t], 0)
            for v in x:
                pix.append(v)
            for j in range(TTA_ACT):
                act.append(acts[k + t][j] if k + t < len(acts) else Scalar[DT](0))
    return (pix^, act^)

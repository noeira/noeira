# +--------------------------------------------------------------------------+ #
# | Telling the artifact sink that a checkpoint just landed
# +--------------------------------------------------------------------------+ #
"""One place where "a checkpoint was written" becomes "upload it".

    trainer.save_state(checkpoint_path)
    announce_checkpoint(checkpoint_path, artifacts, run_dir)

## ⚠⚠ Why this is a function and not two lines at each site

`trainer.save_state(checkpoint_path)` appears **eighteen times** across four
driver files. Writing the offer inline at each of them is the defect shape this
tree pays for most often — a rule written in eighteen places drifts, and here
the drift is silent: a site that saves and forgets to announce produces an
artifact that simply never leaves the box, with nothing to see in the output.

So the rule lives here, and `tests/deep_agents/test_checkpoints_announce.mojo`
is a SOURCE gate that reads the four drivers and fails if any
`trainer.save_state(` is not immediately followed by an `announce_checkpoint(`.
That is the same shape as `tests/core/test_drivers_use_runcontext.mojo`, and it
exists because the previous version of this mistake — five hand-rolled `--tag`
blocks in the FB family — was found by accident rather than by a gate.

## ⚠ Why the offer is not inside `save_state`

`save_state` is a trait method on the trainers, implemented per algorithm and
defaulting to a no-op. Putting an upload behind it would make every trainer
depend on `io/artifact_sink` and would fire on paths that are not run
artifacts at all (a probe, a unit test's scratch file). The driver knows it is
running a RUN; the trainer does not.
"""

from ...io.artifact_sink import ArtifactSink, KIND_CHECKPOINT


def offered_path(path: String, run_dir: String) -> String:
    """The artifact path a checkpoint should be filed under, or "" to drop it.

    ⚠⚠ THE DECISION IS SPLIT OUT FROM THE EFFECT SO IT CAN BE GATED. A source
    gate can prove that all eighteen sites CALL `announce_checkpoint`; it
    cannot prove the call does anything, and a mutant that emptied the body
    survived exactly that gate. This function is pure, so the rule it encodes
    is checkable with no sink, no fixture and no network.

    ⚠ THE PATH IS MADE RELATIVE TO THE RUN DIRECTORY. The drivers work in
    absolute paths — `checkpoint_path` is whatever the caller passed — while
    the artifact's identity is its path WITHIN the run, because that is what
    the monitor keys on and what `run.kv` records.

    ⚠ A PATH OUTSIDE THE RUN DIRECTORY IS DROPPED. A driver still writing to a
    `comptime` constant (one that P0d did not retrofit) would otherwise land
    its checkpoint under a run it does not belong to, and a misfiled artifact
    is worse than an absent one — the whole value of the layer is that
    `run.kv` can be trusted.
    """
    if path.byte_length() == 0 or run_dir.byte_length() == 0:
        return String("")
    var prefix = run_dir + "/"
    if not path.startswith(prefix):
        return String("")
    return String(path[byte = prefix.byte_length() :])


def announce_checkpoint(
    path: String,
    artifacts: Optional[ArtifactSink],
    run_dir: String,
) raises:
    """Offer a just-written checkpoint to the sink. A no-op without one.

    ⚠ NEVER RAISES IN PRACTICE AND NEVER BLOCKS. `offer` is a memcpy onto a
    ring; the transfer happens on the sink's own thread. A driver must not pay
    for the dashboard being slow, and must not stop because it is down.

    The path decision — relative to the run, or dropped — is `offered_path`,
    which is pure and gated on its own.
    """
    if not artifacts:
        return
    var rel = offered_path(path, run_dir)
    if rel.byte_length() == 0:
        return
    var sink = artifacts.value()
    _ = sink.offer(rel, String(KIND_CHECKPOINT))


def checkpoint_retain(
    step: Int,
    ref written: List[Int],
    keep: Int,
    milestone: Int,
    best: Int,
) -> Bool:
    """Should the checkpoint at `step` survive a prune?

    ⚠ THIS EXISTS BECAUSE A RUN DIED OF A FULL DISK, not because a directory
    was untidy. The 2048/6 G1 run wrote 1.17 GB every 2000 batched steps and
    at 38 000 the 19th one cut off mid-write on a 60 GB box — 20 h of
    training lost with nothing wrong upstream of the filesystem
    (`docs/BFM_ZERO_G1_REPRODUCTION.md` §12.44 era).

    Three reasons to keep one, in priority order:

      * it is the BEST-scoring checkpoint (`best`, -1 for none);
      * it is on the milestone ladder (`step % milestone == 0`), which is what
        makes a long run auditable after the fact rather than only resumable;
      * it is among the last `keep` written, which is what makes it resumable.

    `written` is the ordered list of steps still on disk, oldest first, so
    "last `keep`" is positional rather than arithmetic — a run that skipped a
    checkpoint for want of space must still keep its most recent ones, and a
    rule like `step > last - keep * every` would quietly drop all of them.

    ⚠ ONE FUNCTION FOR BOTH THE PRUNE AND THE REPORT. The caller deletes with
    this predicate and then recounts with it; two copies of the rule would let
    the log claim a file that had just been removed.
    """
    if best >= 0 and step == best:
        return True
    if milestone > 0 and step % milestone == 0:
        return True
    if keep <= 0:
        return False
    var n = len(written)
    var first = n - keep if n > keep else 0
    for i in range(first, n):
        if written[i] == step:
            return True
    return False

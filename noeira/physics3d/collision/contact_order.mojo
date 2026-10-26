"""Put the contact array in MuJoCo's order.

`mj_broadphase` ends by SORTING its bodyflex pair list
(`bfsort(bfpair, buf, npair, NULL)`, `engine_collision_driver.c:1683`) by the
signature

    (min(b1, b2) << 16) + max(b1, b2)

and the driver then RELIES on that sortedness — it dedupes consecutive equal
signatures and walks `m->pair_signature` monotonically. Narrow phase runs body
pair by body pair in that order, so `d->contact` comes out ordered by body
pair. Ours did not: `broadphase_sap.mojo` says so in its own words, "THIS
SWEEP EMITS PAIRS IN AABB-SORT ORDER", and it additionally runs PLANES in a
separate phase BEFORE the sweep, which front-loads every plane contact
regardless of which body it is against. MuJoCo has no such split.

## Why an order is a correctness property and not a convention

The primal Newton solve is a global minimisation and does not care. Three
things do:

  * `mj_solNoSlip` is GAUSS-SEIDEL — each contact is solved against the state
    the ones before it have already moved, so the order IS part of the answer;
  * `solPGS` likewise;
  * anything comparing our contact array to MuJoCo's index by index.

Measured on `hello_robot_stretch_3` (board row #1, and the last row above
1e-9). Its contacts match MuJoCo to 1e-16 of position, depth and normal, and
with the noslip pass off the whole scene agrees to 5.9e-10 — but as shipped it
was 4.566e-05, and one contact carried all of it:

    excluded pair                            ncon   |d qpos|
    (baseline)                                  9   4.5659e-05
    link_aruco_right_base x wrist_pitch         8   3.8634e-14

That contact is `(27,87)`, body pair (5,19), and it makes a LARGE correction
in the sweep — its second tangential force goes 109 -> 11264. MuJoCo applies
it BEFORE the three `(34,87)` contacts of body pair (8,19), so theirs move; we
emitted it LAST, so ours never moved at all:

    MuJoCo   (9,87) (9,89) (27,87) (34,87) (34,87) (34,87)
    ours     (9,87) (9,89) (34,87) (34,87) (34,87) (27,87)

⚠⚠ THE CAUSAL PROOF IS ON THE REFERENCE'S SIDE, not ours. Deleting the
`bfsort` call from a local MuJoCo 3.10.0 build makes ITS order become exactly
the second line above, and moves ITS OWN qvel by 2.283242e-02 — against the
2.282963e-02 that separated the two engines. Ours-vs-MuJoCo then goes
4.63e-05 -> 1.50e-06, so the pair order is ~97% of that board row.

## What this sorts on, and what it does not

Primary and only key: the body pair `(min(bodyA, bodyB), max(bodyA, bodyB))`,
which is `add_pair`'s signature. The sort is STABLE, so contacts of one body
pair keep the order the narrow phase produced them in.

⚠ THE SECONDARY KEY IS NOT IMPLEMENTED. Within one body pair MuJoCo sorts
again, by `(geom[0], geom[1])` (`contactcompare`, :380), and this cannot: the
contact record carries the BODY pair and not the GEOM pair, and the narrow
phase emits from ~20 sites, most of them helpers that never see a geom id. So
a body pair contributing TWO DIFFERENT geom pairs keeps our sweep's relative
order for them rather than MuJoCo's. `hello_robot_stretch_3` has exactly that
case — body pair (1,19) contributes `(9,87)` and `(9,89)` — and our sweep
happens to emit those two in MuJoCo's order already, which is why the residual
after this sort is the 1.5e-06 the `bfsort` ablation predicted and not more.
Closing it means carrying the geom pair in the record; the measurement that
justifies paying for that does not exist yet.

⚠ `contactcompare`'s "undo this swapping" block is a NO-OP in 3.10.0 and must
not be ported as written: `pushPairArena` (:496) already guarantees
`type[g1] <= type[g2]`, which is the exact condition that block tests for, so
it never swaps. The stored pair IS the sort key there.

## The world body is spelled two ways

`detect_contacts` writes body 0 for the world and the SAP path writes -1 (see
`detect_contacts_auto`'s docstring on the two paths' differing record
conventions). Both mean "world", and MuJoCo's signature uses 0, so the key
normalises a negative id to 0 — otherwise every plane contact would sort to
one end on one path and the other end on the other, and the two paths would
disagree about the order for a reason that is pure bookkeeping.
"""

from ..fields.scratch import Scratch
from ..gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
)
from layout import TileTensor, TensorLayout


@always_inline
def sort_contacts_mujoco_order[
    DTYPE: DType,
    L_CONTACTS: TensorLayout,
](
    env: Int,
    contacts: TileTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    num_contacts: Int,
):
    """Reorder `contacts[env]` by body pair, stably. See the module docstring.

    INSERTION SORT, and that is not a placeholder. It is what makes the sort
    STABLE without a scratch buffer — the property the whole point rests on,
    since contacts of one body pair must keep the narrow phase's order — and
    `num_contacts` is bounded by `max_contacts`, which is 32-128 on every model
    in the tree. It also runs unchanged inside a GPU kernel: this is called
    from the PER-ENV function that both the CPU loop and the batched kernel
    share, so one implementation serves both and the two legs cannot drift.

    ⚠ THE RECORD IS MOVED WHOLE. A contact is `CONTACT_SIZE` consecutive
    floats and every one of them belongs to it, including the slots appended
    later (mixed solref/solimp at 23..29). Moving a named subset is how a
    reorder turns into a corruption that still looks like a plausible contact.
    """
    if num_contacts < 2:
        return

    # The sort key of one contact: the body pair as `add_pair` forms it.
    @always_inline
    def _key(c: Int, mut lo: Int, mut hi: Int) {imm}:
        var a = Int(rebind[Scalar[DTYPE]](contacts[env, c * CONTACT_SIZE + CONTACT_IDX_BODY_A]))
        var b = Int(rebind[Scalar[DTYPE]](contacts[env, c * CONTACT_SIZE + CONTACT_IDX_BODY_B]))
        # `-1` is the SAP path's spelling of the world body; `0` is
        # `detect_contacts`'. Normalised so the two paths cannot order the
        # same scene differently over pure bookkeeping.
        if a < 0:
            a = 0
        if b < 0:
            b = 0
        lo = a if a < b else b
        hi = b if a < b else a

    var rec = Scratch[Scalar[DTYPE], CONTACT_SIZE](
        CONTACT_SIZE, uninitialized=Scalar[DTYPE](0)
    )
    for i in range(1, num_contacts):
        var k_lo = 0
        var k_hi = 0
        _key(i, k_lo, k_hi)
        # Nothing to do unless `i` sorts before its predecessor — the common
        # case on a sweep that is already mostly ordered.
        var p_lo = 0
        var p_hi = 0
        _key(i - 1, p_lo, p_hi)
        if p_lo < k_lo or (p_lo == k_lo and p_hi <= k_hi):
            continue

        for t in range(CONTACT_SIZE):
            rec[t] = rebind[Scalar[DTYPE]](contacts[env, i * CONTACT_SIZE + t])

        # Walk left while the predecessor is STRICTLY greater. `<=` keeps
        # equal keys in their original order, which is what makes this stable.
        var j = i - 1
        while j >= 0:
            var j_lo = 0
            var j_hi = 0
            _key(j, j_lo, j_hi)
            if j_lo < k_lo or (j_lo == k_lo and j_hi <= k_hi):
                break
            for t in range(CONTACT_SIZE):
                contacts[env, (j + 1) * CONTACT_SIZE + t] = rebind[
                    Scalar[DTYPE]
                ](contacts[env, j * CONTACT_SIZE + t])
            j -= 1

        for t in range(CONTACT_SIZE):
            contacts[env, (j + 1) * CONTACT_SIZE + t] = rec[t]

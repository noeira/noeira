"""Reading the active mask INSIDE a kernel, and what it does to the obs. P3d.

`active.active_mask` is the HOST half — it takes a `TaskSpec` and a
`FamilySpec` and produces one float. This is the DEVICE half: it takes that
float and knows nothing about either.

The split is the same one `tape.mojo` / `gpu_eval.mojo` already make in this
package, and for the same two reasons. A kernel cannot hold a `TaskSpec`; and
keeping the spec parser out of the observation hook's compile unit keeps the
family config's instantiation small.

## ⚠⚠ NO `Float64` BELOW

Metal has no double. Everything is `Scalar[DTYPE]` or `Int`.

## WHAT AN INACTIVE SLOT LOOKS LIKE IN THE OBSERVATION

Zeros in its pose words, and a 0 in its own mask word.

⚠ THE ZERO IS NOT THE POINT — THE MASK WORD IS. Zeroing alone would be the
same convention the mask exists to remove, one constant swapped for another.
What makes it safe is that the constant is now ANNOUNCED: the policy is handed
"this slot is absent" as a value, and the pose words beside it are zeroed only
so that the parked pose — which is 50 m up, lateral, and FALLING, because
nothing reparks it — does not swamp the observation's scale.

⚠ AND THAT IS WHY THE ZEROING CANNOT BE DROPPED "because the mask says so".
An unnormalised +50 in an observation is not neutralised by a 0 elsewhere in
the vector; it dominates the first layer's activations whatever the mask bit
says. The two together are the fix, not either one.
"""

from noeira.core.tiles import Tile

# ⚠ A FREE JOINT'S WIDTH, NAMED ONCE. `reset.write_free_pose` writes seven
# `qpos` and `reset.write_free_vel_zero` six `qvel`; this zeroes the same two
# spans in the observation. Three sites, one rule —
# `_a_rule_written_inline_twice_drifts` is this tree's most recurring defect
# and a free joint's width is exactly the kind of constant it eats.
comptime FREE_JOINT_NQ: Int = 7
comptime FREE_JOINT_NV: Int = 6


@always_inline
def slot_active[DTYPE: DType](mask: Scalar[DTYPE], slot: Int) -> Bool:
    """Is family slot `slot` active in the lane whose mask word this is?

    ⚠ THE ONLY ARITHMETIC IS THE `Int()` NARROWING, and it is exact for every
    mask `active.active_mask` will write — that writer is where the width is
    checked against the float's mantissa.

    ⚠ A SLOT INDEX PAST THE MASK'S WIDTH READS FALSE, not garbage: the shift
    runs off the top of an `Int` and yields 0. "This family has no such slot"
    and "that slot is not active" are the same answer to a reader.
    """
    return ((Int(mask) >> slot) & 1) == 1


@always_inline
def write_free_slot_obs[
    DTYPE: DType, BATCH: Int, OBS_DIM: Int,
](
    obs: Tile[DTYPE, BATCH, OBS_DIM],
    env: Int,
    active: Bool,
    qadr: Int,
    dadr: Int,
    bit: Int,
):
    """One free slot's contribution to the observation.

    ⚠ THE THREE INDICES ARE INDICES INTO `obs`, ALREADY OFFSET — not into
    `qpos` and `qvel`. The caller knows where it laid the state out and this
    does not have to guess; passing raw joint addresses would make this
    function silently wrong for any layout that does not put `qpos` first.

    Args:
        obs: The observation tensor.
        env: This lane.
        active: Whether the slot is active this episode.
        qadr: Index in `obs` of the slot's first pose word (7 wide).
        dadr: Index in `obs` of the slot's first velocity word (6 wide).
        bit: Index in `obs` of this slot's own active word.
    """
    if not active:
        for k in range(FREE_JOINT_NQ):
            obs[env, qadr + k] = Scalar[DTYPE](0)
        for k in range(FREE_JOINT_NV):
            obs[env, dadr + k] = Scalar[DTYPE](0)
    obs[env, bit] = Scalar[DTYPE](1) if active else Scalar[DTYPE](0)


@always_inline
def write_free_slot_obs_host[DTYPE: DType](
    mut obs: List[Scalar[DTYPE]],
    active: Bool,
    qadr: Int,
    dadr: Int,
    bit: Int,
):
    """`write_free_slot_obs` over a `List` — the single-env CPU path.

    ⚠⚠ WRITTEN TWICE BECAUSE THE CONTAINER IS DIFFERENT, AND PINNED BY A GATE
    BECAUSE OF IT. `Phyics3dEnv`'s obs hook is handed a `List` and the batched
    one a `LayoutTensor`; there is no type that is both. The batched trainer
    writes a checkpoint the single-env eval loads, so a permutation between
    these two is a policy that works on one device and is nonsense on the
    other, with no error anywhere — the exact hazard `so_arm_reach_config`
    records. `tests/tasks/test_active_mask.mojo` runs both hooks on one state
    and demands the vectors be IDENTICAL. ABLATED: swapping qpos and qvel in
    the CPU hook fails that check and NOTHING else.

    ⚠ `obs` MUST ALREADY BE `bit + 1` LONG. The CPU hook appends the state
    words and the active words first, then calls this to overwrite; appending
    from here would put the active word before the poses it gates.
    """
    if not active:
        for k in range(FREE_JOINT_NQ):
            obs[qadr + k] = Scalar[DTYPE](0)
        for k in range(FREE_JOINT_NV):
            obs[dadr + k] = Scalar[DTYPE](0)
    obs[bit] = Scalar[DTYPE](1) if active else Scalar[DTYPE](0)

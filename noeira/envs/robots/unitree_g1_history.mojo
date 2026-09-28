"""`last_action` + `history_actor` — the 401 dims our actor never had.

## Why

`released/new_model/config.json` filters the observation dict PER NET:

    f             state, privileged_state, last_action, history_actor
    b             state, privileged_state
    actor         state,                   last_action, history_actor
    discriminator state, privileged_state

Our 527 is `state 64 | privileged 463`, so **B and D already match exactly**.
What differs is that our ACTOR sees the 463 privileged dims the reference's
actor does NOT, and is missing the 401 it does (docs §12.34). The reference's
actor input is 721 = state 64 + last_action 29 + history 372 + z 256.

A policy with no last action and no history cannot smooth its own output or
compensate for actuator lag, and ground-truth body positions do not substitute
for either. §12.34 measured our policy at 1.32-1.83x the RELEASED actor on the
same clips in an environment G2 proved identical to three decimals — so the
deficit is in the policy, and this is the only structural difference left.

## The layout, from the G2-validated oracle

`tools/g1/bfm_zero_tracking_oracle.py` transcribed this from the reference and
G2 passed with it (`ours = MuJoCo to three decimals on 37 of 39 segments`), so
it is the authority here, not the yaml.

    HIST_KEYS = (actions, base_ang_vel, dof_pos, dof_vel, projected_gravity)
    HIST_DIMS =  29       3             29       29       3                 = 93
    HIST_LEN  = 4                                             93 * 4        = 372

**KEY-MAJOR, newest first within a key** — `concatenate([hist[k].reshape(-1)
for k in HIST_KEYS])` over `(4, dim)` buffers whose row 0 is newest. NOT
step-major. Getting this backwards produces a vector of the right size and the
wrong meaning, which nothing but a value check can see.

⚠ `last_action` is the SCALED, CLIPPED action (`a * 5`, clipped to +-5) that
the PD chain consumed — not the actor's raw tanh output.

⚠ Two rules that decide what the policy sees at step t:
  * the RESET observation is NEVER pushed (`if self.t >= 1: self._push(...)`);
  * the push happens AFTER that step's history was read, and BEFORE
    `last_action` is updated — so the newest `actions` entry is the action
    applied at step t-1, not the one about to be applied.

## What this module is not

It holds no physics. `custom_extract_obs_gpu` produces the 527 from state; the
history is driver-side, because `last_action` is not a function of the
simulator's state and cannot be recovered from it.
"""

from max.gpu import global_idx

from noeira.nn.constants import DT
from noeira.data.resident import IDX_DT
from .unitree_g1_xml import UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM


comptime G1_HIST_LEN: Int = 4
comptime G1_N_ACT: Int = 29

# the five history keys, in the reference's SORTED key order, with the offset
# of each key's block inside the 372 and its per-step width
comptime G1_HIST_STEP: Int = 93          # 29 + 3 + 29 + 29 + 3
comptime G1_HIST_DIM: Int = 372          # G1_HIST_STEP * G1_HIST_LEN
comptime G1_LAST_ACTION_DIM: Int = 29
comptime G1_ACTOR_EXTRA: Int = 401       # last_action + history
comptime UNITREE_G1_FULL_OBS_DIM: Int = UNITREE_G1_OBS_DIM + G1_ACTOR_EXTRA

# key blocks inside the 372, key-major
comptime G1_H_ACTIONS: Int = 0                    # 4 * 29
comptime G1_H_ANGVEL: Int = 116                   # 4 * 3
comptime G1_H_DOFPOS: Int = 128                   # 4 * 29
comptime G1_H_DOFVEL: Int = 244                   # 4 * 29
comptime G1_H_GRAV: Int = 360                     # 4 * 3

# where each quantity lives inside the 64-D state block of the 527
# state = [dof_pos 29 | dof_vel 29 | projected_gravity 3 | base_ang_vel 3]
comptime G1_S_DOFPOS: Int = 0
comptime G1_S_DOFVEL: Int = 29
comptime G1_S_GRAV: Int = 58
comptime G1_S_ANGVEL: Int = 61


def g1_hist_key_offset(k: Int) -> Int:
    """Offset of history key `k` (0..4, sorted order) inside the 372."""
    if k == 0:
        return G1_H_ACTIONS
    elif k == 1:
        return G1_H_ANGVEL
    elif k == 2:
        return G1_H_DOFPOS
    elif k == 3:
        return G1_H_DOFVEL
    return G1_H_GRAV


def g1_hist_key_dim(k: Int) -> Int:
    """Per-step width of history key `k`."""
    if k == 0:
        return 29
    elif k == 1:
        return 3
    elif k == 2:
        return 29
    elif k == 3:
        return 29
    return 3


def g1_hist_key_state_offset(k: Int) -> Int:
    """Where key `k`'s CURRENT value sits in the 64-D state block.

    `actions` (k == 0) is not in the state at all — it comes from the stored
    `last_action` — and this returns -1 for it so a caller that forgets cannot
    silently read `dof_pos`.
    """
    if k == 0:
        return -1
    elif k == 1:
        return G1_S_ANGVEL
    elif k == 2:
        return G1_S_DOFPOS
    elif k == 3:
        return G1_S_DOFVEL
    return G1_S_GRAV


def g1_pack_full_obs_kernel[LANES: Int](
    obs: Pointer[Scalar[DT], MutAnyOrigin],        # LANES x 527
    last_action: Pointer[Scalar[DT], MutAnyOrigin],  # LANES x 29
    hist: Pointer[Scalar[DT], MutAnyOrigin],       # LANES x 372
    dst: Pointer[Scalar[DT], MutAnyOrigin],        # LANES x 928
):
    """`dst = [obs 527 | last_action 29 | history 372]`, element-parallel.

    The 527 stays FIRST and unchanged so `b` and `discriminator` keep reading
    `dst[0:527]` — the reference gives those two nets exactly `state +
    privileged_state`, which is what our 527 already is.
    """
    comptime W = UNITREE_G1_FULL_OBS_DIM
    var t = Int(global_idx.x)
    if t >= LANES * W:
        return
    var lane = t // W
    var k = t % W
    if k < UNITREE_G1_OBS_DIM:
        dst[unsafe_offset=t] = obs[unsafe_offset=lane * UNITREE_G1_OBS_DIM + k]
    elif k < UNITREE_G1_OBS_DIM + G1_LAST_ACTION_DIM:
        var j = k - UNITREE_G1_OBS_DIM
        dst[unsafe_offset=t] = last_action[unsafe_offset=lane * G1_N_ACT + j]
    else:
        var j = k - UNITREE_G1_OBS_DIM - G1_LAST_ACTION_DIM
        dst[unsafe_offset=t] = hist[unsafe_offset=lane * G1_HIST_DIM + j]


def g1_hist_push_kernel[LANES: Int](
    obs: Pointer[Scalar[DT], MutAnyOrigin],          # LANES x 527, this step's
    last_action: Pointer[Scalar[DT], MutAnyOrigin],  # LANES x 29, the PREVIOUS step's
    hist: Pointer[Scalar[DT], MutAnyOrigin],         # LANES x 372, in place
    live: Pointer[Scalar[DT], MutAnyOrigin],         # LANES, 0 = reset row, do not push
):
    """Shift every key's buffer by one step and write the newest entry.

    One thread per (lane, key, element-of-one-step) — `G1_HIST_STEP` threads
    per lane, each owning ONE element across all four steps, so the shift is
    a private read-then-write and needs no barrier.

    ⚠ `live[lane] == 0` skips the lane entirely: the RESET observation is
    never pushed. The caller sets it to 0 on the step after a reset and 1
    afterwards.

    ⚠ `last_action` must still hold the PREVIOUS step's value when this runs.
    The reference pushes before updating it, so the newest `actions` entry is
    the action applied at step t-1.
    """
    var t = Int(global_idx.x)
    if t >= LANES * G1_HIST_STEP:
        return
    var lane = t // G1_HIST_STEP
    if live[unsafe_offset=lane] == Scalar[DT](0.0):
        return
    var e = t % G1_HIST_STEP

    # resolve which key this element belongs to, and its index within the key
    var k = 0
    var within = e
    while k < 5:
        var d = g1_hist_key_dim(k)
        if within < d:
            break
        within -= d
        k += 1

    var base = lane * G1_HIST_DIM + g1_hist_key_offset(k)
    var d = g1_hist_key_dim(k)
    # newest first: shift 3<-2<-1<-0
    var j = G1_HIST_LEN - 1
    while j > 0:
        hist[unsafe_offset=base + j * d + within] = hist[
            unsafe_offset=base + (j - 1) * d + within
        ]
        j -= 1
    var so = g1_hist_key_state_offset(k)
    if so < 0:
        hist[unsafe_offset=base + within] = last_action[
            unsafe_offset=lane * G1_N_ACT + within
        ]
    else:
        hist[unsafe_offset=base + within] = obs[
            unsafe_offset=lane * UNITREE_G1_OBS_DIM + so + within
        ]


def g1_hist_reset_kernel[LANES: Int](
    last_action: Pointer[Scalar[DT], MutAnyOrigin],
    hist: Pointer[Scalar[DT], MutAnyOrigin],
    live: Pointer[Scalar[DT], MutAnyOrigin],
    mask: Pointer[Scalar[DT], MutAnyOrigin],   # LANES, non-zero = reset this lane
):
    """Zero a lane's `last_action` and history, and mark its next push skipped.

    `live` going to 0 is what implements "the reset observation is never
    pushed"; the driver sets it back to 1 after the first push is skipped.
    """
    var t = Int(global_idx.x)
    comptime W = G1_N_ACT + G1_HIST_DIM
    if t >= LANES * W:
        return
    var lane = t // W
    if mask[unsafe_offset=lane] == Scalar[DT](0.0):
        return
    var k = t % W
    if k < G1_N_ACT:
        last_action[unsafe_offset=lane * G1_N_ACT + k] = Scalar[DT](0.0)
    else:
        hist[unsafe_offset=lane * G1_HIST_DIM + (k - G1_N_ACT)] = Scalar[DT](0.0)
    if k == 0:
        live[unsafe_offset=lane] = Scalar[DT](0.0)


# ══════════════════════════════════════════════════════════════════════
# The TRAINING side: derive the 401 from the ring instead of storing it
# ══════════════════════════════════════════════════════════════════════
#
# The kernels above maintain the history incrementally during the ROLLOUT,
# where the actor needs it at action-selection time. Training needs the same
# 401 dims for a row sampled out of the replay ring — and they are already in
# the ring, exactly as `next_obs` was before §12.23:
#
#     last_action(row r)      = scale_clip(r_act[r - LANES])
#     history actions[j]      = scale_clip(r_act[r - (j+2)*LANES])
#     history <state key>[j]  = r_obs[r - (j+1)*LANES] at that key's offset
#
# all lane-aligned, because `ring_store_kernel` advances `pos` by exactly
# LANES per step. Storing them instead would cost 401 floats per row against
# the ONE `r_age` float this needs (docs §12.36): 4860 B/row against 3260.
#
# ⚠ Unlike `next_obs` this looks BACKWARD, so no sampling bound changes — a
# predecessor row is always already written. What it does need is the age.
#
# ⚠ THE AGE RULE, from the oracle's `if self.t >= 1: self._push(...)`:
#
#     last_action    valid iff age >= 1
#     history[j]     valid iff age >= j + 2      (BOTH the action and the
#                                                 state keys — a_0 exists at
#                                                 the reset step, and the
#                                                 reset observation is never
#                                                 pushed, which cancel)
#
# `age` is steps since this lane's last reset, capped at 5 (j = 3 on the
# action keys reaches 5 steps back). Everything above the cap is valid.


comptime G1_HIST_MAX_AGE: Int = 5


def g1_hist_gather_kernel[
    ROWS: Int, CAP: Int, LANES: Int, ACT: Int
](
    r_obs: Pointer[Scalar[DT], MutAnyOrigin],   # CAP x 527
    r_act: Pointer[Scalar[DT], MutAnyOrigin],   # CAP x ACT
    r_age: Pointer[Scalar[DT], MutAnyOrigin],   # CAP
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],  # ROWS, the drawn rows
    act_scale: Scalar[DT],                      # NORMALIZE_TO (5.0)
    act_clip: Scalar[DT],                       # ACTION_CLIP (5.0)
    dst: Pointer[Scalar[DT], MutAnyOrigin],     # ROWS x 401
):
    """`dst[i] = [last_action 29 | history 372]` for drawn row `idx[i]`.

    One thread per output element. `dst` is the 401-wide TAIL only; the caller
    writes `r_obs[idx]` into the 527-wide head separately (it is a plain
    `gather_rows_kernel`), which keeps `[0, 527)` of the batch byte-identical
    to what `b` and `discriminator` already consume.
    """
    var t = Int(global_idx.x)
    if t >= ROWS * G1_ACTOR_EXTRA:
        return
    var i = t // G1_ACTOR_EXTRA
    var k = t % G1_ACTOR_EXTRA
    var row = Int(idx[unsafe_offset=i])
    var age = Int(r_age[unsafe_offset=row])

    # how many steps back, and which source — resolved first, read once. The
    # helpers this replaced were nested `def`s, which a kernel cannot capture
    # (`Could not infer capture convention`).
    var steps = 0          # lane-aligned back-steps
    var elem = 0           # element within the key
    var from_action = False
    var state_off = -1

    if k < G1_LAST_ACTION_DIM:
        if age < 1:
            dst[unsafe_offset=t] = Scalar[DT](0.0)
            return
        steps = 1
        elem = k
        from_action = True
    else:
        var h = k - G1_LAST_ACTION_DIM      # 0 .. 371
        var key = 0
        var off = h
        while key < 5:
            var blk = g1_hist_key_dim(key) * G1_HIST_LEN
            if off < blk:
                break
            off -= blk
            key += 1
        var d = g1_hist_key_dim(key)
        var j = off // d                    # 0 = newest
        elem = off % d
        if age < j + 2:
            dst[unsafe_offset=t] = Scalar[DT](0.0)
            return
        state_off = g1_hist_key_state_offset(key)
        if state_off < 0:
            from_action = True
            steps = j + 2                   # actions lag the state by one push
        else:
            steps = j + 1

    var r = row - steps * LANES
    while r < 0:
        r += CAP

    if from_action:
        var a = r_act[unsafe_offset=r * ACT + elem] * act_scale
        if a > act_clip:
            a = act_clip
        if a < -act_clip:
            a = -act_clip
        dst[unsafe_offset=t] = a
    else:
        dst[unsafe_offset=t] = r_obs[
            unsafe_offset=r * UNITREE_G1_OBS_DIM + state_off + elem
        ]


def g1_build_tail_spec(mut spec: List[Int32]):
    """The 401-element spec `derive_tail_kernel` reads — built ONCE, on the host.

    Four int32 per output element: `kind` (0 = `r_obs`, 1 = `r_act` scaled and
    clipped), lane-aligned `steps` back, `src_off` within that row, and the
    `min_age` below which the element is zero.

    This is where the layout lives. The agent stays generic: it derives a tail
    it cannot interpret, which is what keeps one env's observation format out
    of code the walker path also runs (docs §12.36).

    ⚠ THE AGE RULE. `last_action` is valid from age 1; `history[j]` from age
    `j + 2` — the same bound for the action keys and the state keys, which is
    not obvious. The actions lag the state by one push (`buf[0] =
    last_action`, set at the END of the previous step) and the reset
    observation is never pushed; the two cancel exactly.

    ⚠ THE ACTION LAG. `history actions[j]` comes from `j + 2` steps back while
    `history <state key>[j]` comes from `j + 1`. Same `j`, different row.
    """
    spec.clear()
    # last_action 29: one step back, the whole action row, valid from age 1
    for e in range(G1_LAST_ACTION_DIM):
        spec.append(Int32(1))      # kind: action
        spec.append(Int32(1))      # steps back
        spec.append(Int32(e))
        spec.append(Int32(1))      # min_age
    # history 372, KEY-MAJOR, newest first within a key
    for key in range(5):
        var d = g1_hist_key_dim(key)
        var so = g1_hist_key_state_offset(key)
        for j in range(G1_HIST_LEN):
            for e in range(d):
                if so < 0:
                    spec.append(Int32(1))          # actions
                    spec.append(Int32(j + 2))      # ⚠ one further back
                    spec.append(Int32(e))
                else:
                    spec.append(Int32(0))          # a state key of r_obs
                    spec.append(Int32(j + 1))
                    spec.append(Int32(so + e))
                spec.append(Int32(j + 2))          # min_age, both kinds


def g1_scale_clip_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    src: Pointer[Scalar[DT], MutAnyOrigin],
    scale: Scalar[DT],
    clip: Scalar[DT],
):
    """`dst = clip(src * scale, +-clip)` — the action as the PD chain took it.

    The actor emits a `tanh` output in `[-1, 1]`; `unitree_g1_config`'s chain
    then does `a *= 5`, clips to `+-5`, and only afterwards scales by
    `action_scale * effort / kp`. `last_action` and the history's `actions`
    key hold the value AFTER the scale and clip — what the policy saw, not
    what the net emitted — so this runs between the two.
    """
    var i = Int(global_idx.x)
    if i >= N:
        return
    var v = src[unsafe_offset=i] * scale
    if v > clip:
        v = clip
    if v < -clip:
        v = -clip
    dst[unsafe_offset=i] = v

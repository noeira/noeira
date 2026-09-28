"""A task's episode, prepared for a HOST env: the posed `qpos` and the `meta` words.

    from noeira.tasks.posed_reset import posed_qpos, task_meta_words

    st.reset_qpos = posed_qpos[So101TowerPlacement](task, family, radius)
    var mw = task_meta_words(task, family, w_goal, w_reach, m_goal, m_reach)
    st.reset_meta_idx = mw[0].copy()
    st.reset_meta_val = mw[1].copy()

What a family's GPU driver does per lane at reset, for ONE lane on the host —
the two front ends that view a family (`examples/so101/tower_viewer_imgui.mojo`,
`examples/so101/tower_policy_viewer.mojo`) call these through
`ViewerState.reset_qpos` / `reset_meta_*`, and `run_view` applies them after
every reset. One implementation, so the viewers cannot pose a scene the
trainer would not.

`posed_qpos` reads everything from the family's PLACEMENT TABLE `P` — park
poses, region site frames (FK at rest, baked by the generator), slot addresses
— so no second model is built and no FK is run; the host sampler is the eval's
(`sampler.sample_placements`), at seed 0, lane 0. `task_meta_words` binds the
goal against the composed scene's names (one `parse_model_runtime`, seconds)
and returns the tape, the mask, the init words and the shaping words at the
`meta` indices the hooks read.
"""

from noeira.physics3d.gpu.constants import (
    META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE, META_IDX_INIT_REGION_0,
    META_IDX_SHAPE_W_GOAL,
)
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from .spec import load_family, load_task, validate_task_against_family, SLOT_FREE
from .family import scene_path, task_path
from .sampler import sample_placements, sample_base_qpos, RegionFrame, SampleReport
from .reset import reset_slots, SlotAddress
from .placement.table import PlacementTable
from .predicates import parse_goal, bind_goal
from .tape import encode_goal, TAPE_WORDS
from .active import active_mask, init_region_words
from .shaping import shaping_words, SHAPING_WORDS

comptime DT = DType.float64
comptime DEFAULT_ROOT = "noeira/tasks"
"""Where `family` / `task` names are looked up unless `root=` says otherwise."""


def posed_qpos[P: PlacementTable](
    task: String, family: String, fallback_radius: Float64,
    seed: UInt64 = 0,
    root: String = DEFAULT_ROOT,
) raises -> List[Float64]:
    """The composed scene's `qpos0` with the task's `init=` placements drawn.

    `fallback_radius` is what the host sampler uses for a free slot WITHOUT
    `slot_geom=` (every tabletop slot; no tower slot)."""
    var f = load_family(root + "/families/" + family + ".family")
    var t = load_task(task_path(f, task))
    validate_task_against_family(t, f)
    var q0 = List[Float64](length=P.NQ, fill=0.0)
    # the base asset's rest (and its per-episode draw) — the device reset's
    # first write, `placement/table.reset_task_slots`
    var rest = sample_base_qpos(f, seed, 0)
    for i in range(len(rest)):
        q0[i] = rest[i]
    for j in range(P.N_FREE):
        var adr = P.free_qadr(j)
        q0[adr] = Float64(P.free_park_x[DT](j))
        q0[adr + 1] = Float64(P.free_park_y[DT](j))
        q0[adr + 2] = Float64(P.free_park_z[DT](j))
        q0[adr + 3] = 1.0
    var frames = List[RegionFrame]()
    for r in range(P.N_REGIONS):
        frames.append(
            RegionFrame(
                Float64(P.region_site_x[DT](r)),
                Float64(P.region_site_y[DT](r)),
                Float64(P.region_site_z[DT](r)),
            )
        )
    var radii = List[Float64](length=len(f.slots), fill=fallback_radius)
    var addrs = List[SlotAddress]()
    var j = 0
    for si in range(len(f.slots)):
        if f.slots[si].kind == SLOT_FREE:
            addrs.append(SlotAddress(P.free_qadr(j), P.free_dadr(j)))
            j += 1
        else:
            addrs.append(SlotAddress(-1, -1))
    var rep = SampleReport()
    var placed = sample_placements(t, f, frames, radii, seed, 0, rep)
    var v0 = List[Float64](length=P.NV, fill=0.0)
    reset_slots(t, f, placed, addrs, q0, v0)
    return q0^


def task_meta_words(
    task: String, family: String,
    w_goal: Float64, w_reach: Float64, goal_margin: Float64,
    reach_margin: Float64,
    root: String = DEFAULT_ROOT,
) raises -> Tuple[List[Int], List[Float64]]:
    """The `meta` words a task's lane carries: tape, mask, init words, shaping.

    Indices and values, parallel lists, for `ViewerState.reset_meta_*`. The
    shaping words are the config's `SHAPE_W_GOAL`, `SHAPE_W_REACH`,
    `GOAL_MARGIN`, `REACH_MARGIN` — the caller hands them in from the type it
    knows, as the SAC driver does."""
    var f = load_family(root + "/families/" + family + ".family")
    var t = load_task(task_path(f, task))
    validate_task_against_family(t, f)
    var fmd = parse_model_runtime(scene_path(f))
    var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
    var tape = encode_goal(g)
    var mask = active_mask(t, f)
    var iw = init_region_words(t, f)
    var sw = shaping_words(w_goal, w_reach, goal_margin, reach_margin)
    var idx = List[Int]()
    var val = List[Float64]()
    for w in range(TAPE_WORDS):
        idx.append(META_IDX_TASK_PARAM_0 + w)
        val.append(tape[w])
    idx.append(META_IDX_TASK_ACTIVE)
    val.append(mask)
    for j in range(len(iw)):
        idx.append(META_IDX_INIT_REGION_0 + j)
        val.append(iw[j])
    for j in range(SHAPING_WORDS):
        idx.append(META_IDX_SHAPE_W_GOAL + j)
        val.append(sw[j])
    return (idx^, val^)

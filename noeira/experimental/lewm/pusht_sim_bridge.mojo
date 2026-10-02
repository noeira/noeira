"""Bridge: render a PushTEnv state into a LeWM-encoder input frame.

The frame is drawn the way stable-worldmodel 0.0.6 draws it
(`envs/pusht/render_swm.mojo`), the renderer of the LeWM dataset;
the LeWM world model encodes 224² CHW [0,1] frames (the offline HF dataset
layout). `sim_frame_chw_norm` renders one env state at the WM's resolution
and converts HWC [0,255] → CHW [0,1] in one host call, so a sim observation
can be fed straight into the frozen encoder.

Used by the sim-domain diagnostic (does the encoder, trained on HF renders,
transfer to the mojo renderer?) and — if it does — the closed-loop MPC
control loop. Host-side; the caller H2D-uploads the assembled (B, T·IMG_DIM)
window to the GPU world model.
"""

from std.memory import alloc, dealloc
from layout import Layout, LayoutTensor

from ...nn.constants import DT
from noeira.envs.pusht.render_swm import render_pusht_swm_at


def sim_frame_chw_norm[
    OUT: Int,
    do: MutOrigin = MutAnyOrigin,
](
    block_cx: Scalar[DT],
    block_cy: Scalar[DT],
    block_angle: Scalar[DT],
    agent_cx: Scalar[DT],
    agent_cy: Scalar[DT],
    dst_chw: Pointer[Scalar[DT], do],
) raises:
    """Render one PushT state at OUT×OUT and write CHW [0,1] (3·OUT·OUT) to
    `dst_chw` — the LeWM encoder's per-frame input layout."""
    comptime HW = OUT * OUT
    var tmp_a = alloc[Scalar[DT]]({count = HW * 3})
    var tmp = tmp_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()   # HWC [0,255]
    var pix = LayoutTensor[DT, Layout.row_major(OUT, OUT, 3), MutAnyOrigin](
        tmp.as_unsafe_any_origin()
    )
    # stable-worldmodel 0.0.6's drawing (render_swm.mojo): the frames the
    # LeWM dataset — and so every LeWM encoder — was trained on. The env's
    # own `render_pusht_rgb_at` (flat fills, no outline, no resize) put the
    # published encoder ~1.5 env steps of motion away from the real frame
    # (G4b, docs/LEWM_REOPEN_PLAN.md P4).
    render_pusht_swm_at[OUT](
        block_cx, block_cy, block_angle, agent_cx, agent_cy, pix
    )
    # HWC [0,255] → CHW [0,1]
    var inv = Scalar[DT](1.0 / 255.0)
    for c in range(3):
        for y in range(OUT):
            for x in range(OUT):
                dst_chw[unsafe_offset=c * HW + y * OUT + x] = (
                    tmp[unsafe_offset=(y * OUT + x) * 3 + c] * inv
                )
    dealloc(tmp_a^)

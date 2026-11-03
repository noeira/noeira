# +--------------------------------------------------------------------------+ #
# | BONES-SEED → BFM-Zero store rows (D1's "BFM-Zero retrained on SEED")
# +--------------------------------------------------------------------------+ #
"""A `SeedClip` through BFM-Zero's motion pipeline: `data/lafan.mojo`'s
`convert_frames` (steps 2–5) verbatim, exactly as `data/amass.mojo` does for
AMASS. Only step 1 (`data/bones_seed.mojo`'s CSV parse) is SEED's.

Kept apart from `bones_seed.mojo` so that SuperMimic's side, which needs no
FK at import time, does not pull the G1 engine in.

The float64 → float32 cast happens here, once: `convert_frames` runs
BFM-Zero's float32 pipeline from float32 inputs, as for LAFAN and AMASS.
"""

from noeira.data.bones_seed import SEED_RAW_DIM, SeedClip
from noeira.data.lafan import LafanRows, convert_frames
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF


def convert_seed_clip(clip: SeedClip, mut env: UnitreeG1[DType.float64]) raises -> LafanRows:
    """The clip's 50 Hz BFM-Zero store rows."""
    var fps = Int(clip.fps)
    if Float64(fps) != clip.fps:
        raise Error("bones_seed_store: convert_frames takes an integer fps, got " + String(clip.fps))
    var n = clip.n
    var root_pos = List[Float32](length=n * 3, fill=Float32(0))
    var root_quat = List[Float32](length=n * 4, fill=Float32(0))
    var dof = List[Float32](length=n * G1_N_DOF, fill=Float32(0))
    for f in range(n):
        var b = f * SEED_RAW_DIM
        for k in range(3):
            root_pos[f * 3 + k] = Float32(clip.raw[b + k])
        for k in range(4):
            root_quat[f * 4 + k] = Float32(clip.raw[b + 3 + k])
        for j in range(G1_N_DOF):
            dof[f * G1_N_DOF + j] = Float32(clip.raw[b + 7 + j])
    return convert_frames(fps, n, root_pos, root_quat, dof, env)

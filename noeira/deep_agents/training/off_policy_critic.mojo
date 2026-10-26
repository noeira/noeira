"""Off-policy critic-update helpers — reusable across SAC / TD3 / DDPG.

Phase 9B.4. Extracted from `SACTrainer` so the upcoming DDPG/TD3 trainers
can import a generic surface instead of duplicating the boilerplate.

Three free functions, all `CRITIC: Module`-generic:

  `concat_sa[OBS, ACT, B]`
      Pack (obs, act) → sa side-by-side. Used both for the critic
      target-y compute (next-state action) and the critic update
      (replay action).


Free functions (not methods) so the Mojo nightly aliasing analyzer
doesn't have to prove that `self.pair1.online` and `self.critic1_opt`
don't alias when both come from a `mut self` receiver. Caller passes
each `mut` arg distinctly. See memory:
`feedback_mojo_tile_tensor_generic_origin`.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.core.tiles import Tile


def concat_sa[
    OBS: Int, ACT: Int, B: Int
](
    obs: Tile[DT, B, OBS],
    act: Tile[DT, B, ACT],
    out_sa: Tile[DT, B, OBS + ACT],
):
    """`sa[b, :OBS] = obs[b]; sa[b, OBS:] = act[b]`. CPU only."""
    comptime SA = OBS + ACT
    for b in range(B):
        for d in range(OBS):
            out_sa[b, d] = obs[b, d]
        for j in range(ACT):
            out_sa[b, OBS + j] = act[b, j]


def concat_sa_kernel[
    OBS: Int, ACT: Int, B: Int
](
    obs: Tile[DT, B, OBS],
    act: Tile[DT, B, ACT],
    out_sa: Tile[DT, B, OBS + ACT],
):
    var idx = Int(global_idx.x)
    comptime SA = OBS + ACT
    var total = B * SA
    if idx < total:
        var b = idx // SA
        var d = idx % SA
        if d < OBS:
            out_sa[b, d] = obs[b, d]
        else:
            out_sa[b, d] = act[b, d - OBS]


def concat_sa_gpu[
    OBS: Int, ACT: Int, B: Int
](
    ctx: DeviceContext,
    obs: Tile[DT, B, OBS],
    act: Tile[DT, B, ACT],
    out_sa: Tile[DT, B, OBS + ACT],
) raises:
    """GPU `concat_sa` — one thread per [b, d] over the full SA shape."""
    comptime SA = OBS + ACT
    comptime total = B * SA
    comptime n_blocks = (total + TPB - 1) // TPB
    comptime kernel = concat_sa_kernel[OBS, ACT, B]
    ctx.enqueue_function[kernel](
        obs,
        act,
        out_sa,
        grid_dim=n_blocks,
        block_dim=TPB,
    )

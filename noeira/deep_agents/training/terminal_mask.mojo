"""Terminal-bootstrap mask shared by the SAC/DDPG/TD3 target-y blocks.

Each target-y graph computes only the BOOTSTRAP term (`γ·soft_v` for SAC,
`γ·Q'` for DDPG, `γ·min(Q1',Q2')` for TD3) into `mb_y`. `apply_terminal_mask`
then writes the full TD target in-place:

    mb_y[b] = r[b] + (1 − term[b]) · bootstrap[b]

dropping the bootstrap on natural termination (`term=1`) and keeping it on
time-limit truncation (`term=0`) — CleanRL semantics. For envs that never
terminate naturally (`term ≡ 0`) this reduces to `r + bootstrap`,
bit-identical to the previous in-graph `Add(r, bootstrap)` (multiply by an
exact 1.0 introduces no rounding).

Forward-only (target, never differentiated). Capture-safe (elementwise,
fixed buffers, no host work on the GPU path).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.core.tiles import Tile


def _mask_bootstrap_kernel[
    N: Int
](
    r: Tile[DT, N],
    term: Tile[DT, N],
    y: Tile[DT, N, 1],
):
    var i = Int(global_idx.x)
    if i < N:
        var nonterm: y.ElementType = 1.0 - term[i]
        y[i, 0] = r[i] + nonterm * y[i, 0]


def apply_terminal_mask[
    target: StaticString, N: Int
](
    ctx: Optional[DeviceContext],
    mb_r: Tile[DT, N],
    mb_term: Tile[DT, N],
    mb_y: Tile[DT, N, 1],
) raises:
    """In-place `mb_y[b] = r[b] + (1 − term[b]) · mb_y[b]` over N samples.

    `mb_y` enters holding the bootstrap term and exits holding the full TD
    target. `ctx` is required (and used) only on the GPU path."""
    comptime if target == "cpu":
        for b in range(N):
            var nonterm: mb_y.ElementType = 1.0 - mb_term[b]
            mb_y[b, 0] = mb_r[b] + nonterm * mb_y[b, 0]
    else:
        comptime n_blocks = (N + TPB - 1) // TPB
        ctx.value().enqueue_function[_mask_bootstrap_kernel[N]](
            mb_r,
            mb_term,
            mb_y,
            grid_dim=n_blocks,
            block_dim=TPB,
        )

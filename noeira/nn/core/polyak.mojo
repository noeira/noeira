"""Polyak soft-update — per-Tensor exponential moving average (storage surface).

`polyak_tensor` is the framework-agnostic per-Tensor soft-update used by the
`Module.polyak_from` overrides (Linear/Conv2D/… params; combinators recurse).
It is a generic nn helper — not SAC-specific — so it lives in `nn.core` rather
than in any agent's loss module. Both CPU + GPU.

  polyak:   dst = tau·src + (1-tau)·dst
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from .tensor import Tensor
from noeira.core.tiles import Tile


def _polyak_kernel[
    N: Int
](
    dst: Tile[DT, N],
    src: Tile[DT, N],
    tau: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = tau * rebind[Scalar[DT]](src[i]) + (
            Scalar[DT](1.0) - tau
        ) * rebind[Scalar[DT]](dst[i])


def polyak_tensor[
    target: StaticString, N: Int
](
    mut dst: Tensor,
    mut src: Tensor,
    tau: Scalar[DT],
    ctx: Optional[DeviceContext],
) raises:
    comptime if target == "cpu":
        for i in range(N):
            dst.data[i] = (
                tau * src.data[i] + (Scalar[DT](1.0) - tau) * dst.data[i]
            )
    else:
        var c = ctx.value()
        comptime nblk = (N + TPB - 1) // TPB
        c.enqueue_function[_polyak_kernel[N]](
            dst.tt["gpu", N](),
            src.tt["gpu", N](),
            tau,
            grid_dim=nblk,
            block_dim=TPB,
        )

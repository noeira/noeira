"""Smoke: nn AlphaZero net wrapped in the AZPredGPU adapter runs a batched
GPU prediction through the planner's PredictionGPU trait surface.

Validates the core Phase-A integration seam: an nn ``Module`` (self-contained
GPU params) satisfies ``planners.tree_search.PredictionGPU`` and the
``DT``↔``DT`` LayoutTensor/TileTensor bridge works end-to-end on device.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_pred_adapter_smoke.mojo
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.alphazero.nets import AZMLPNet
from noeira.deep_agents.zero.mcts_adapters import AZPredGPU


def main() raises:
    comptime OBS = 27   # TicTacToe canonical board
    comptime ACT = 9
    comptime H = 64
    comptime B = 4
    comptime Net = AZMLPNet[OBS, ACT, H]
    comptime PRED_OUT = ACT + 1

    var ctx = DeviceContext()
    var net = Net.make["gpu", Kaiming](Optional(ctx))
    var adapter = AZPredGPU[OBS, ACT, Net].make(net)

    # Input hidden = obs (B, OBS); fill with a constant on host, copy to device.
    var hidden_dev = ctx.enqueue_create_buffer[DT](B * OBS)
    var pred_dev = ctx.enqueue_create_buffer[DT](B * PRED_OUT)
    var h_host = ctx.enqueue_create_host_buffer[DT](B * OBS)
    ctx.synchronize()
    for i in range(B * OBS):
        h_host[i] = Scalar[DT](0.1)
    ctx.enqueue_copy(hidden_dev, h_host)
    ctx.synchronize()

    # The planner hands the adapter typed device views over its own buffers.
    var hidden_lt = tile[B, OBS](hidden_dev)
    var pred_lt = tile[B, PRED_OUT](pred_dev)

    adapter.predict_gpu[B](ctx, hidden_lt, pred_lt)

    var p_host = ctx.enqueue_create_host_buffer[DT](B * PRED_OUT)
    ctx.enqueue_copy(p_host, pred_dev)
    ctx.synchronize()

    var all_finite = True
    for i in range(B * PRED_OUT):
        var v = Float64(p_host[i])
        if v != v:  # NaN
            all_finite = False
    assert_true(all_finite, "AZ prediction output contained NaN")

    _ = net^  # keepalive: adapter holds a non-owning pointer into net
    print("AZ pred adapter smoke: OK (B=", B, " PRED_OUT=", PRED_OUT, ")")

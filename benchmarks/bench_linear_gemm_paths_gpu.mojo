"""Linear / LinearAct GPU time per call, MAX's GEMMs vs cuBLAS, at the shapes
noeira's agents run.

Build it once per path and compare the two tables row by row:

    pixi run mojo build -I . -D NN_GEMM_PATH=max    -D BL_PART=1 benchmarks/bench_linear_gemm_paths_gpu.mojo -o bl_max_1
    pixi run mojo build -I . -D NN_GEMM_PATH=cublas -D BL_PART=1 benchmarks/bench_linear_gemm_paths_gpu.mojo -o bl_cublas_1
    pixi run -e default ./bl_max_1 ; pixi run -e default ./bl_cublas_1        (and BL_PART=2, 3)

(`NN_GEMM_PATH`, `noeira/nn/core/cublas_gemm.mojo`: `max` = MAX's GEMMs,
padded where its dispatch needs it — the paths before cuBLAS; `cublas` =
`cublas_gemm` for every forward and backward GEMM; `auto` = the default rule.)

Per shape, two numbers, each the GPU time of ONE call measured inside a
captured CUDA graph of R back-to-back calls (how a training step runs them:
no launch overhead, kernels back to back), averaged over NREP replays:

  - fwd:     `forward` (GEMM + bias / bias+act, and the padding copies when
             the MAX path pads);
  - fwd+bwd: `forward` then `vjp` (bias grad, both backward GEMMs, the
             activation gate, the padding / transpose / accumulate kernels).

`capture_recast` is set, as the trainers do when they capture: the padded
MAX path then re-pads the weight on every call, as it does once per optimizer
step in training.

`BL_PART=4` times the bf16-flow `LinearAct` / `Linear` (bf16 activations,
fp32 master weights): `NN_GEMM_PATH=linmax` = MAX's GEMMs (the path before),
`auto` = cuBLAS. `-D NN_LT_BIAS=1` puts the fp32 `Linear` forward's bias in a
cuBLASLt epilogue (parts 1-3; `cublaslt_gemm.mojo`).

NVIDIA only (the `max` / `cublas` switch does nothing elsewhere). Run through
`pixi run` so the CUDA interceptor is preloaded (graph capture needs it).
"""

from std.time import perf_counter_ns
from std.random import seed, random_float64
from std.sys.defines import get_defined_int
from max.gpu.host import DeviceContext

from noeira.cuda.graph import CUDAGraph, maybe_capture_replay
from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.cublaslt_gemm import LT_BIAS
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.element_op import ElementOp
from noeira.nn.core.cublas_gemm import GEMM_PATH
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_act import LinearAct
from noeira.nn.primitives.ops.relu_op import ReLUOp
from noeira.nn.primitives.ops.tanh_op import TanhOp


comptime PART = get_defined_int["BL_PART", 1]()
"""Which third of the shape list (1: small RL nets, 2: mid-size RL, 3: large /
supervised); one binary each keeps the compile time down."""

comptime R = 20
"""Calls per captured graph."""
comptime NREP = 30
"""Timed replays per measurement."""


def _fill(mut t: Tensor, n: Int, ctx: DeviceContext) raises:
    t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64(-1, 1))
    t.upload(ctx)


def _replay_us[
    STEP: def () capturing raises -> None
](ctx: DeviceContext) raises -> Float64:
    var g = Optional[CUDAGraph](None)
    maybe_capture_replay[STEP, VERBOSE=False](g, ctx)  # runs, then captures
    if not g or not g.value().is_captured():
        print("  ! graph not captured: this row is timed EAGERLY (launch-bound)")
    for _ in range(3):
        maybe_capture_replay[STEP, VERBOSE=False](g, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(NREP):
        maybe_capture_replay[STEP, VERBOSE=False](g, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    return Float64(t1 - t0) / 1e3 / Float64(NREP * R)


def _row(kind: String, IN: Int, OUT: Int, B: Int, cub: Bool, f: Float64, fb: Float64):
    print(
        "ROW path=", GEMM_PATH, " lt_bias=", LT_BIAS, " kind=", kind, " in=", IN, " out=", OUT,
        " b=", B, " cublas_fwd=", cub, " fwd_us=", f, " fwdbwd_us=", fb,
        sep="",
    )


def bench_linear[IN: Int, OUT: Int, B: Int](ctx: DeviceContext) raises:
    comptime L = Linear[IN, OUT]
    seed(IN * 31 + OUT)
    var m = L.make["gpu", Kaiming](Optional(ctx))
    m.set_attr["capture_recast"](Scalar[DT](1.0))
    var x = Tensor()
    var go = Tensor()
    _fill(x, B * IN, ctx)
    _fill(go, B * OUT, ctx)
    var y = Tensor()
    var gi = Tensor()
    m.forward["gpu", B](TensorRefs[1](x), y, Optional(ctx))
    m.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(ctx))
    ctx.synchronize()

    def _f() capturing raises -> None:
        for _ in range(R):
            m.forward["gpu", B](TensorRefs[1](x), y, Optional(ctx))

    def _fb() capturing raises -> None:
        for _ in range(R):
            m.forward["gpu", B](TensorRefs[1](x), y, Optional(ctx))
            m.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(ctx))

    var f = _replay_us[_f](ctx)
    var fb = _replay_us[_fb](ctx)
    _row("Linear", IN, OUT, B, L.use_cublas_fwd[B](), f, fb)
    _ = m^
    _ = x^
    _ = go^
    _ = y^
    _ = gi^


def bench_act[IN: Int, OUT: Int, B: Int, OP: ElementOp](kind: String, ctx: DeviceContext) raises:
    comptime L = LinearAct[IN, OUT, OP]
    seed(IN * 31 + OUT)
    var m = L.make["gpu", Kaiming](Optional(ctx))
    m.set_attr["capture_recast"](Scalar[DT](1.0))
    var x = Tensor()
    var go = Tensor()
    _fill(x, B * IN, ctx)
    _fill(go, B * OUT, ctx)
    var y = Tensor()
    var gi = Tensor()
    m.forward["gpu", B](TensorRefs[1](x), y, Optional(ctx))
    m.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(ctx))
    ctx.synchronize()

    def _f() capturing raises -> None:
        for _ in range(R):
            m.forward["gpu", B](TensorRefs[1](x), y, Optional(ctx))

    def _fb() capturing raises -> None:
        for _ in range(R):
            m.forward["gpu", B](TensorRefs[1](x), y, Optional(ctx))
            m.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(ctx))

    var f = _replay_us[_f](ctx)
    var fb = _replay_us[_fb](ctx)
    _row(kind, IN, OUT, B, L.use_cublas_fwd[B](), f, fb)
    _ = m^
    _ = x^
    _ = go^
    _ = y^
    _ = gi^


comptime BF16 = DType.bfloat16


def _fill_bf(mut t: TensorImpl[BF16], n: Int, ctx: DeviceContext) raises:
    t = TensorImpl[BF16].alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64(-1, 1)).cast[BF16]()
    t.upload(ctx)


def bench_act_bf16[IN: Int, OUT: Int, B: Int, OP: ElementOp](kind: String, ctx: DeviceContext) raises:
    """`bench_act` for the bf16-flow `LinearAct[IN, OUT, OP, bf16]`."""
    comptime L = LinearAct[IN, OUT, OP, BF16]
    seed(IN * 31 + OUT)
    var m = L.make["gpu", Kaiming](Optional(ctx))
    m.set_attr["capture_recast"](Scalar[DT](1.0))
    var x = TensorImpl[BF16]()
    var go = TensorImpl[BF16]()
    _fill_bf(x, B * IN, ctx)
    _fill_bf(go, B * OUT, ctx)
    var y = TensorImpl[BF16]()
    var gi = TensorImpl[BF16]()
    m.forward["gpu", B](TensorRefs[1, ADT=BF16](x), y, Optional(ctx))
    m.vjp["gpu", B](TensorRefs[1, ADT=BF16](x), go, TensorRefs[1, ADT=BF16](gi), Optional(ctx))
    ctx.synchronize()

    def _f() capturing raises -> None:
        for _ in range(R):
            m.forward["gpu", B](TensorRefs[1, ADT=BF16](x), y, Optional(ctx))

    def _fb() capturing raises -> None:
        for _ in range(R):
            m.forward["gpu", B](TensorRefs[1, ADT=BF16](x), y, Optional(ctx))
            m.vjp["gpu", B](TensorRefs[1, ADT=BF16](x), go, TensorRefs[1, ADT=BF16](gi), Optional(ctx))

    var f = _replay_us[_f](ctx)
    var fb = _replay_us[_fb](ctx)
    _row(kind + "_bf16", IN, OUT, B, False, f, fb)
    _ = m^
    _ = x^
    _ = go^
    _ = y^
    _ = gi^


def main() raises:
    var ctx = DeviceContext()
    print("Linear / LinearAct GPU us per call, NN_GEMM_PATH =", GEMM_PATH, "| R =", R, "NREP =", NREP)
    comptime if PART == 1:
        bench_act[8, 64, 64, TanhOp]("LinearTanh", ctx)  # PPO LunarLander
        bench_act[8, 64, 16, TanhOp]("LinearTanh", ctx)  # PPO LL acting
        bench_act[8, 64, 1, TanhOp]("LinearTanh", ctx)  # PPO LL eval
        bench_act[64, 64, 64, TanhOp]("LinearTanh", ctx)  # PPO LL
        bench_linear[64, 4, 64](ctx)  # PPO LL actor head
        bench_linear[64, 1, 64](ctx)  # PPO LL critic head
        bench_linear[64, 1, 16384](ctx)  # PPO LL GAE
        bench_act[64, 64, 16384, TanhOp]("LinearTanh", ctx)  # PPO LL GAE
        bench_linear[11, 64, 64](ctx)  # PPO Hopper
        bench_linear[64, 64, 2048](ctx)  # PPO Hopper GAE
        bench_linear[3, 64, 256](ctx)  # Pendulum SAC
        bench_linear[3, 64, 1](ctx)  # Pendulum acting
        bench_linear[64, 64, 256](ctx)  # Pendulum SAC
        bench_act[17, 256, 256, ReLUOp]("LinearReLU", ctx)  # SAC HC actor
        bench_act[17, 256, 32, ReLUOp]("LinearReLU", ctx)  # SAC HC acting
        bench_act[17, 256, 1, ReLUOp]("LinearReLU", ctx)  # SAC eval
        bench_act[23, 256, 256, ReLUOp]("LinearReLU", ctx)  # SAC HC critic
        bench_act[256, 256, 256, ReLUOp]("LinearReLU", ctx)  # SAC trunk
        bench_act[256, 256, 32, ReLUOp]("LinearReLU", ctx)  # SAC acting
        bench_act[256, 256, 1, ReLUOp]("LinearReLU", ctx)  # SAC eval
        bench_linear[256, 6, 256](ctx)  # SAC actor head
        bench_linear[256, 6, 32](ctx)  # SAC acting head
        bench_linear[256, 1, 256](ctx)  # SAC critic head
        bench_act[27, 256, 256, ReLUOp]("LinearReLU", ctx)  # SAC Ant
        bench_act[45, 256, 256, ReLUOp]("LinearReLU", ctx)  # SAC Humanoid
        bench_act[4, 128, 64, ReLUOp]("LinearReLU", ctx)  # AZ/MuZero
        bench_act[6, 128, 64, ReLUOp]("LinearReLU", ctx)  # Rainbow Pong-6D
        bench_act[128, 128, 256, ReLUOp]("LinearReLU", ctx)  # Rainbow Pong-6D
        bench_act[128, 256, 64, ReLUOp]("LinearReLU", ctx)  # Pong RAM
        bench_act[130, 128, 128, ReLUOp]("LinearReLU", ctx)  # MuZero cartpole
        bench_linear[128, 51, 128](ctx)  # MuZero support
        bench_linear[128, 2, 1](ctx)  # MuZero search
        bench_linear[23, 200, 128](ctx)  # MBPO dyn
        bench_linear[200, 200, 128](ctx)  # MBPO dyn
        bench_linear[200, 36, 128](ctx)  # MBPO dyn
        bench_linear[4, 32, 256](ctx)  # DreamerV3 cartpole
        bench_linear[32, 32, 256](ctx)  # DreamerV3 cartpole
        bench_linear[32, 51, 256](ctx)  # DreamerV3 cartpole
        bench_act[27, 128, 64, ReLUOp]("LinearReLU", ctx)  # AZ tic-tac-toe
        bench_linear[128, 9, 64](ctx)  # AZ tic-tac-toe
    comptime if PART == 2:
        bench_act[223, 1024, 256, ReLUOp]("LinearReLU", ctx)  # SAC dm dog
        bench_act[1024, 1024, 256, ReLUOp]("LinearReLU", ctx)  # SAC dm dog
        bench_act[1024, 1024, 32, ReLUOp]("LinearReLU", ctx)  # dog acting
        bench_linear[1024, 38, 256](ctx)  # dog head
        bench_linear[1024, 1, 256](ctx)  # dog critic head
        bench_linear[17, 256, 256](ctx)  # TD-MPC2 enc
        bench_linear[256, 512, 256](ctx)  # TD-MPC2
        bench_linear[518, 512, 256](ctx)  # TD-MPC2 dyn
        bench_linear[518, 512, 1024](ctx)  # TD-MPC2 pi+Q
        bench_linear[518, 512, 268](ctx)  # TD-MPC2 MPPI
        bench_linear[518, 512, 2144](ctx)  # TD-MPC2 MPPI x8
        bench_linear[512, 512, 256](ctx)  # TD-MPC2
        bench_linear[512, 512, 268](ctx)  # TD-MPC2 MPPI
        bench_linear[512, 101, 256](ctx)  # TD-MPC2 reward
        bench_linear[512, 1, 1024](ctx)  # TD-MPC2 term
        bench_linear[512, 12, 268](ctx)  # TD-MPC2 pi
        bench_linear[99, 512, 6144](ctx)  # PPO G1
        bench_linear[99, 512, 1024](ctx)  # PPO G1 acting
        bench_linear[99, 512, 24576](ctx)  # PPO G1 GAE
        bench_linear[512, 256, 6144](ctx)  # PPO G1
        bench_linear[256, 128, 6144](ctx)  # PPO G1
        bench_linear[128, 1, 6144](ctx)  # PPO G1 critic
        bench_linear[960, 512, 6144](ctx)  # PPO G1 RP actor
        bench_linear[1110, 512, 6144](ctx)  # PPO G1 RP critic
        bench_linear[1345, 256, 2048](ctx)  # Craftax
        bench_linear[1345, 256, 32768](ctx)  # Craftax GAE
        bench_linear[256, 256, 2048](ctx)  # Craftax
        bench_linear[256, 17, 2048](ctx)  # Craftax head
        bench_linear[8268, 256, 2048](ctx)  # Craftax full
        bench_linear[32, 256, 512](ctx)  # PPO family
        bench_linear[256, 256, 4096](ctx)  # PPO family
        bench_linear[256, 1, 16384](ctx)  # PPO family GAE
        bench_linear[158, 1024, 1024](ctx)  # FB
        bench_linear[1024, 128, 1024](ctx)  # FB
        bench_linear[957, 2048, 1024](ctx)  # BFM F
        bench_linear[2048, 2048, 1024](ctx)  # BFM F
        bench_linear[2048, 1, 1024](ctx)  # BFM Q
        bench_linear[527, 256, 4096](ctx)  # BFM B-net
        bench_linear[2048, 29, 1024](ctx)  # BFM actor
        bench_act[2688, 256, 128, ReLUOp]("LinearReLU", ctx)  # MuZero C4
        bench_act[2560, 128, 128, ReLUOp]("LinearReLU", ctx)  # AZ C4
        bench_act[3136, 512, 32, ReLUOp]("LinearReLU", ctx)  # Rainbow CNN
        bench_act[3136, 512, 8, ReLUOp]("LinearReLU", ctx)  # Rainbow CNN acting
    comptime if PART == 3:
        bench_linear[3, 256, 1024](ctx)  # DreamerV3 pendulum
        bench_linear[256, 256, 1024](ctx)  # DreamerV3 pendulum
        bench_linear[1536, 256, 1024](ctx)  # DreamerV3 pendulum
        bench_linear[256, 255, 1024](ctx)  # DreamerV3 twohot
        bench_linear[4096, 512, 16](ctx)  # DreamerV3 Pong wm
        bench_linear[8704, 512, 16](ctx)  # DreamerV3 Pong wm
        bench_linear[1024, 512, 1024](ctx)  # DreamerV3 Pong
        bench_linear[5120, 512, 1024](ctx)  # DreamerV3 Pong
        bench_linear[1024, 4608, 1024](ctx)  # DreamerV3 Pong dec
        bench_linear[13824, 1024, 16](ctx)  # DreamerV3 car enc
        bench_linear[3072, 13824, 16](ctx)  # DreamerV3 car dec
        bench_linear[256, 256, 992](ctx)  # ACT
        bench_linear[256, 1024, 992](ctx)  # ACT ffn
        bench_linear[1024, 256, 992](ctx)  # ACT ffn
        bench_linear[256, 256, 2592](ctx)  # ACT
        bench_linear[6, 256, 960](ctx)  # ACT qpos
        bench_linear[256, 6, 960](ctx)  # ACT head
        bench_linear[960, 960, 1120](ctx)  # SmolVLA text
        bench_linear[960, 2560, 1120](ctx)  # SmolVLA text
        bench_linear[720, 960, 400](ctx)  # SmolVLA expert
        bench_linear[720, 2048, 400](ctx)  # SmolVLA expert
        bench_linear[2048, 720, 400](ctx)  # SmolVLA expert
        bench_linear[32, 720, 400](ctx)  # SmolVLA
        bench_linear[720, 32, 400](ctx)  # SmolVLA
        bench_linear[12288, 960, 128](ctx)  # SmolVLA connector
        bench_act[784, 256, 100, ReLUOp]("LinearReLU", ctx)  # MNIST MLP
        bench_act[256, 128, 100, ReLUOp]("LinearReLU", ctx)  # MNIST MLP
        bench_linear[128, 10, 100](ctx)  # MNIST head
        bench_linear[2048, 128, 100](ctx)  # CIFAR conv head
        bench_linear[384, 1152, 16384](ctx)  # GPT qkv
        bench_linear[1536, 384, 16384](ctx)  # GPT fc2
        bench_linear[384, 65, 16384](ctx)  # GPT lm head
        bench_linear[192, 576, 8192](ctx)  # ViT qkv
        bench_linear[768, 192, 8192](ctx)  # ViT fc2
        bench_linear[192, 10, 128](ctx)  # ViT head
        bench_linear[128, 65, 512](ctx)  # LSTM head
    comptime if PART == 4:
        bench_act_bf16[17, 256, 256, ReLUOp]("LinearReLU", ctx)  # SAC HC actor
        bench_act_bf16[256, 256, 256, ReLUOp]("LinearReLU", ctx)  # SAC trunk
        bench_act_bf16[256, 256, 32, ReLUOp]("LinearReLU", ctx)  # SAC acting
        bench_act_bf16[1024, 1024, 256, ReLUOp]("LinearReLU", ctx)  # SAC dm dog
        bench_act_bf16[64, 64, 16384, TanhOp]("LinearTanh", ctx)  # PPO LL GAE
        bench_act_bf16[784, 256, 100, ReLUOp]("LinearReLU", ctx)  # MNIST MLP
        bench_act_bf16[3136, 512, 32, ReLUOp]("LinearReLU", ctx)  # Rainbow CNN
        bench_act_bf16[384, 1536, 16384, TanhOp]("LinearTanh", ctx)  # GPT-size

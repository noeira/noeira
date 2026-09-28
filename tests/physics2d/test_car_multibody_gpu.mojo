"""GPU<->CPU parity for the multi-body car step (CarMBPhysicsKernel).

Runs the SAME initial cars + controls through:
  - CPU: CarDynamicsMB.step_single_env in a host loop
  - GPU: CarMBPhysicsKernel.step_gpu (one car per thread)
for K steps, then asserts the hull body states match within tolerance.

Run: pixi run -e apple mojo run -I . tests/physics2d/test_car_multibody_gpu.mojo
"""

from std.math import sqrt
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext
from noeira.physics2d import dtype
from noeira.physics2d.constants import IDX_X, IDX_Y, IDX_ANGLE, IDX_VX, IDX_VY, IDX_OMEGA
from noeira.physics2d.constants import BODY_STATE_SIZE, JOINT_DATA_SIZE
from noeira.physics2d.car import CarDynamicsMB, CarMBPhysicsKernel
from noeira.physics2d.car.constants import (
    FRICTION_LIMIT, GRASS_FRICTION, ROAD_FRICTION,
    CTRL_STEERING, CTRL_GAS, CTRL_BRAKE,
)
from noeira.nn.core.ptr import mptr

comptime BATCH = 8
comptime BOFF = 0
comptime FOFF = BOFF + CarDynamicsMB.NUM_BODIES * BODY_STATE_SIZE
comptime JOFF = FOFF + CarDynamicsMB.NUM_BODIES * 3
comptime ROFF = JOFF + CarDynamicsMB.NUM_JOINTS * JOINT_DATA_SIZE
comptime COFF = ROFF + CarDynamicsMB.NUM_WHEELS
comptime SSZ = COFF + 3
comptime K = 40
comptime DT = Scalar[dtype](0.02)


def init_all(state: LayoutTensor[dtype, Layout.row_major(BATCH, SSZ), MutAnyOrigin]):
    """Init BATCH cars at origin with per-env controls (mix of gas/steer)."""
    for e in range(BATCH):
        CarDynamicsMB.init_env[BATCH, SSZ, BOFF, FOFF, JOFF, ROFF](
            e, state, Scalar[dtype](0.0), Scalar[dtype](0.0), Scalar[dtype](0.0)
        )
        # Vary controls so different cars exercise different dynamics.
        var steer = Scalar[dtype](Float64(e % 3 - 1) * 0.5)  # -0.5, 0, +0.5, ...
        var gas = Scalar[dtype](0.5 + 0.5 * Float64(e % 2))   # 0.5 or 1.0
        state[e, COFF + CTRL_STEERING] = steer
        state[e, COFF + CTRL_GAS] = gas
        state[e, COFF + CTRL_BRAKE] = Scalar[dtype](0.0)


def main() raises:
    print("=== CarMBPhysicsKernel GPU<->CPU parity ===")
    var ctx = DeviceContext()
    var fric = Scalar[dtype](FRICTION_LIMIT) * Scalar[dtype](GRASS_FRICTION)

    # --- CPU reference -----------------------------------------------------
    var cbuf = List[Scalar[dtype]](capacity=BATCH * SSZ)
    for _ in range(BATCH * SSZ):
        cbuf.append(Scalar[dtype](0.0))
    var cstate = LayoutTensor[dtype, Layout.row_major(BATCH, SSZ), MutAnyOrigin](
        mptr(cbuf)
    )
    init_all(cstate)
    for _ in range(K):
        for e in range(BATCH):
            CarDynamicsMB.step_single_env[BATCH, SSZ, BOFF, FOFF, JOFF, ROFF, COFF](
                e, cstate, fric, DT
            )

    # --- GPU run -----------------------------------------------------------
    var host = ctx.enqueue_create_host_buffer[dtype](BATCH * SSZ)
    ctx.synchronize()
    var hstate = LayoutTensor[dtype, Layout.row_major(BATCH, SSZ), MutAnyOrigin](
        host.unsafe_ptr().as_unsafe_any_origin()
    )
    init_all(hstate)
    var dev = ctx.enqueue_create_buffer[dtype](BATCH * SSZ)
    ctx.enqueue_copy(dev, host)
    ctx.synchronize()
    for _ in range(K):
        CarMBPhysicsKernel.step_gpu[BATCH, SSZ, BOFF, FOFF, JOFF, ROFF, COFF](
            ctx, dev, fric, DT
        )
    ctx.synchronize()
    var gout = ctx.enqueue_create_host_buffer[dtype](BATCH * SSZ)
    ctx.enqueue_copy(gout, dev)
    ctx.synchronize()
    var gstate = LayoutTensor[dtype, Layout.row_major(BATCH, SSZ), MutAnyOrigin](
        gout.unsafe_ptr().as_unsafe_any_origin()
    )

    # --- compare hull body state across all envs ---------------------------
    var max_diff: Float64 = 0.0
    for e in range(BATCH):
        for idx in [IDX_X, IDX_Y, IDX_ANGLE, IDX_VX, IDX_VY, IDX_OMEGA]:
            var c = Float64(rebind[Scalar[dtype]](cstate[e, BOFF + idx]))
            var g = Float64(rebind[Scalar[dtype]](gstate[e, BOFF + idx]))
            var d = c - g
            if d < 0.0:
                d = -d
            if d > max_diff:
                max_diff = d
        var cs = sqrt(
            Float64(rebind[Scalar[dtype]](cstate[e, BOFF + IDX_VX])) ** 2
            + Float64(rebind[Scalar[dtype]](cstate[e, BOFF + IDX_VY])) ** 2
        )
        print(
            "env", e,
            " cpu_spd=", cs,
            " cpu(x,y,om)=", Float64(rebind[Scalar[dtype]](cstate[e, BOFF + IDX_X])),
            Float64(rebind[Scalar[dtype]](cstate[e, BOFF + IDX_Y])),
            Float64(rebind[Scalar[dtype]](cstate[e, BOFF + IDX_OMEGA])),
        )

    print("max |CPU - GPU| over hull state =", max_diff)
    # Same code both sides; only float reassociation/fast-math differs.
    if max_diff > 0.05:
        raise Error(
            String("GPU<->CPU parity exceeded tol: max_diff=", max_diff)
        )
    print("=== PASS: GPU matches CPU within tol ===")

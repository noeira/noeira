"""Constants for nn/.

DT is pinned to float32 at the framework scope; AMP overrides happen at
the per-call site via `POLICY: AMPPolicy` (see `core/amp.mojo`).

CPU_SIMD_W is the SIMD lane count for `DT` on the host CPU. Used by
hand-rolled SIMD elementwise paths (ReLU/Tanh/MSE/elementwise) since
Mojo nightly does not autovectorize scalar `ptr[i]` loops — manual
`while i + W <= N: ptr.load[width=W]` gives 3-5x over the scalar form.
See feedback_mojo_cpu_manual_simd_required.

TPB / TPB_REDUCE are the threads-per-block defaults for nn / deep_agents
GPU launches. TPB (128) is used for elementwise 1-D launches over N or
N*BATCH; TPB_REDUCE (64) is used by per-batch reduction kernels (one block
per batch element) such as MSE/CE forward. These values were chosen during
the nn bit-identity baselines (SAC -169.04118, MBPO -143.13, PPO -230.15)
and changing them invalidates those baselines. If hardware-adaptive tuning
is ever needed, gate via `default_accelerator().is_nvidia_gpu()` at the call site.
Shape-derived block sizes (e.g. `TPB = max(OBS, ACT)` in gpu_replay) stay
local to their kernel.
"""

from std.sys import simd_width_of

comptime DT = DType.float32
comptime CPU_SIMD_W = simd_width_of[DT]()
comptime TPB = 128
comptime TPB_REDUCE = 64

# Spatial tensor memory layout for conv/BN/pool. NCHW (channels-first) is the
# framework default and is bit-identical to the pre-LAYOUT code; NHWC
# (channels-last) makes the channel the inner/contiguous axis (coalesces the
# im2col gather + BN reduction — see docs/CHANNELS_LAST_NHWC_MIGRATION_PLAN.md).
# Threaded through the spatial primitives like ADT; default NCHW everywhere.
comptime LAYOUT_NCHW = 0
comptime LAYOUT_NHWC = 1

# A/B toggle for the grouped multi-tensor GPU optimizer (Adam step + zero_grad
# + polyak). When True (default), NVIDIA collapses the per-tensor/per-leaf
# launches into one grouped launch per optimizer/critic-pair; when False, the
# per-tensor path is used even on NVIDIA. No effect on CPU/Apple (always
# per-tensor — Metal can't deref host-captured device addresses in-kernel).
# Flip to False + rebuild to A/B the grouped path on NVIDIA.
comptime USE_GROUPED_GPU_OPTIMIZER = True

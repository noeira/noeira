"""SIGReg[DIM, SEQ_LEN, NUM_PROJ, KNOTS] — Sketch Isotropic Gaussian Regularizer
(storage surface).

Transformed from legacy `nn.primitives.SIGReg` (surface-only change). Epps-Pulley
Gaussianity statistic via random projections (Maes, Le Lidec et al., LeWM 2026):
enforces projected embeddings to look like N(0, I) samples in characteristic-
function space. The load-bearing anti-collapse term in the LeWM JEPA loss.

ARITY=1, PARAM-free. Input (BATCH, SEQ_LEN*DIM) interpreted as (B, T, D); output
(BATCH, 1) — the scalar `stat` replicated to every row so a 1/B grad seed gives
chain-rule effective seed 1 (matches the reference `.mean()`).

    A ~ N(0,I_D)^{D×P}, columns L2-normalized          (random projection)
    z[b,t,p]   = sum_d input[b,t*D+d] · A[d,p]
    cm[t,p,k]  = mean_b cos(z·t_k),  sm = mean_b sin(z·t_k)
    stat = (B/(T·P)) · sum_{t,p,k} w_k·[(cm−φ_k)² + sm²]
with t_k = k·3/(K−1), φ_k = exp(−t_k²/2), w_k = trap_k·φ_k.

Leaf-owned state (stable across forward→vjp within a step):
  * cache_z [BATCH, T·P]  — the projected z, reused by backward.
  * GPU workspace slabs (ws_a / ws_cm / ws_sm / ws_partials / ws_scalar /
    ws_dLdz) — separate owned `Tensor` fields (one buffer per slab), vs the
    legacy single pointer-sliced scratch. No `mptr`.
PRNG: A is regenerated each call. Base seed = the cache_z buffer address
(stable across fwd/bwd within a step, so backward sees forward's A). With
`resample` enabled each FORWARD additionally mixes a step counter into the seed
— fresh projections every training step, like the reference's per-forward
`torch.randn`; the matching vjp reuses the stored forward seed. Default off
(fixed A): required by fd-gradcheck and keeps existing runs bit-identical.

Numeric: GPU accumulates in DT (fp32); CPU uses Float64 reductions. Expect
~1e-3 relative agreement, not bit-exact, across CPU/GPU.

The CPU SIMD-free reduction loops and the GPU kernels (block.sum reductions, no
atomics) are carried over VERBATIM from the legacy op; the only kernel-signature
change is partials/stat/g raw pointers → 1-D LayoutTensor views (storage surface
never passes raw device pointers to kernels). The kernel BODY math is identical.
"""

from max.gpu import thread_idx, block_idx, block_dim, global_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import sin, cos, sqrt, log, exp, pi
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


struct SIGReg[DIM: Int, SEQ_LEN: Int, NUM_PROJ: Int, KNOTS: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.SEQ_LEN * Self.DIM)
    comptime OUT_DIM = 1

    # cache_z [BATCH, T*P] — leaf-owned, reused by backward.
    var cache_z: Tensor
    # GPU workspace slabs — separate owned Tensors (one buffer per slab), vs the
    # legacy single pointer-sliced scratch. GPU-only in practice (CPU uses local
    # InlineArrays). Lazily sized per batch.
    var ws_a: Tensor  # [DIM, NUM_PROJ]
    var ws_cm: Tensor  # [SEQ_LEN, NUM_PROJ*KNOTS]
    var ws_sm: Tensor  # [SEQ_LEN, NUM_PROJ*KNOTS]
    var ws_partials: Tensor  # [N_PARTIALS]
    var ws_scalar: Tensor  # [1] (forward stat / backward G)
    var ws_dLdz: Tensor  # [BATCH, SEQ_LEN*NUM_PROJ]
    # Per-forward projection RESAMPLING (reference parity). Default OFF: the
    # fd-gradcheck needs a deterministic f across forward calls, and every
    # existing run stays bit-identical.
    var resample: Bool
    # An INJECTED projection matrix [DIM, NUM_PROJ] (row-major, columns already
    # unit-norm) used instead of the PRNG draw, in forward AND vjp. Gating only:
    # it is how the LeWM reference gate replays torch's exact `A`
    # (`set_node_attr_buf["sig", "fixed_a"]`). None = the PRNG (every run).
    var fixed_a: Optional[DeviceBuffer[DT]]
    var _step_ctr: UInt64
    var _cur_seed: UInt64
    var _seed_valid: Bool

    def __init__(out self):
        self.cache_z = Tensor()
        self.ws_a = Tensor()
        self.ws_cm = Tensor()
        self.ws_sm = Tensor()
        self.ws_partials = Tensor()
        self.ws_scalar = Tensor()
        self.ws_dLdz = Tensor()
        self.resample = False
        self.fixed_a = None
        self._step_ctr = 0
        self._cur_seed = 0
        self._seed_valid = False

    def _forward_seed(mut self, base: UInt64) -> UInt64:
        """Seed for THIS forward; stored so the matching vjp reuses it."""
        if self.resample:
            self._step_ctr += 1
            # splitmix-style mix so consecutive steps decorrelate fully.
            self._cur_seed = base ^ (
                self._step_ctr * UInt64(0x9E3779B97F4A7C15)
            )
        else:
            self._cur_seed = base
        self._seed_valid = True
        return self._cur_seed

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """`resample` (0/1). ⚠ This override was LOST when the storage port
        replaced the legacy SIGReg (2026-06-23): `set_node_attr["sig",
        "resample"]` fell through to the trait's no-op, so every run since,
        the LeWM recipe included, kept ONE projection matrix for its whole
        life. Gate: tests/nn/test_sigreg_attrs.mojo."""
        comptime if ATTR == "resample":
            self.resample = value != Scalar[DT](0)

    def set_attr_buf[ATTR: StaticString](mut self, buf: DeviceBuffer[DT]):
        """`fixed_a`: inject the projection matrix (see the field)."""
        comptime if ATTR == "fixed_a":
            self.fixed_a = buf

    def _fill_a_cpu(
        self, seed: UInt64, a_ptr: Pointer[Scalar[DT], MutAnyOrigin]
    ) raises:
        """A for the CPU path: the injected matrix, else the PRNG draw."""
        if self.fixed_a:
            with self.fixed_a.value().map_to_host() as h:
                for i in range(Self.DIM * Self.NUM_PROJ):
                    a_ptr[unsafe_offset=i] = h[i]
        else:
            Self._generate_a_cpu(seed, a_ptr)

    def _fill_a_gpu(mut self, c: DeviceContext, seed: UInt64) raises:
        """A for the GPU path, into `ws_a`: the injected matrix, else the PRNG."""
        comptime D = Self.DIM
        comptime P = Self.NUM_PROJ
        comptime lay_a = Layout.row_major(D, P)
        if self.fixed_a:
            c.enqueue_copy(
                self.ws_a.dev.value().create_sub_buffer[DT](0, D * P),
                self.fixed_a.value().create_sub_buffer[DT](0, D * P),
            )
            return
        c.enqueue_function[_sr_gen_a_unnorm[D, P]](
            self.ws_a.lt["gpu", lay_a](),
            seed,
            grid_dim=((D * P + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )
        c.enqueue_function[_sr_norm_a[D, P]](
            self.ws_a.lt["gpu", lay_a](),
            grid_dim=((P + TPB - 1) // TPB,),
            block_dim=(TPB,),
        )

    def _backward_seed(self, base: UInt64) -> UInt64:
        return self._cur_seed if self._seed_valid else base

    # ── comptime knot grid / weights ──────────────────────────────────
    @always_inline
    @staticmethod
    def _t_step() -> Float64:
        return 3.0 / Float64(Self.KNOTS - 1)

    @always_inline
    @staticmethod
    def _t_k(k: Int) -> Float64:
        return Float64(k) * Self._t_step()

    @always_inline
    @staticmethod
    def _phi_k(k: Int) -> Float64:
        var tk = Self._t_k(k)
        return exp(-tk * tk * 0.5)

    @always_inline
    @staticmethod
    def _w_k(k: Int) -> Float64:
        var dt = Self._t_step()
        var trap = dt if (k == 0 or k == Self.KNOTS - 1) else 2.0 * dt
        return trap * Self._phi_k(k)

    # ── workspace sizing (elements) ───────────────────────────────────
    @always_inline
    @staticmethod
    def _n_partials() -> Int:
        return (Self.SEQ_LEN * Self.NUM_PROJ * Self.KNOTS + TPB - 1) // TPB

    # ── random projection (deterministic from seed) ───────────────────
    @staticmethod
    def _generate_a_cpu(
        seed: UInt64, a_ptr: Pointer[Scalar[DT], MutAnyOrigin]
    ):
        for d in range(Self.DIM):
            for p in range(Self.NUM_PROJ):
                var idx = d * Self.NUM_PROJ + p
                var rng1 = PhiloxRandom(seed=seed, offset=UInt64(2 * idx))
                var rng2 = PhiloxRandom(seed=seed, offset=UInt64(2 * idx + 1))
                var u1 = Float64(rng1.step_uniform()[0])
                var u2 = Float64(rng2.step_uniform()[0])
                if u1 < 1e-10:
                    u1 = 1e-10
                var g = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
                a_ptr[unsafe_offset=idx] = Scalar[DT](g)
        for p in range(Self.NUM_PROJ):
            var sum_sq = Float64(0.0)
            for d in range(Self.DIM):
                var v = Float64(a_ptr[unsafe_offset=d * Self.NUM_PROJ + p])
                sum_sq += v * v
            var norm = sqrt(sum_sq + 1e-12)
            for d in range(Self.DIM):
                var v = Float64(a_ptr[unsafe_offset=d * Self.NUM_PROJ + p])
                a_ptr[unsafe_offset=d * Self.NUM_PROJ + p] = Scalar[DT](v / norm)

    # ── factory ───────────────────────────────────────────────────────
    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SIGReg: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.KNOTS >= 2, "SIGReg: KNOTS must be >= 2"
        comptime if target != "cpu":
            if not ctx:
                raise Error("SIGReg.make[target='gpu']: ctx required")
        return Self()

    def _ensure_gpu(mut self, c: DeviceContext, batch: Int) raises:
        comptime D = Self.DIM
        comptime T = Self.SEQ_LEN
        comptime P = Self.NUM_PROJ
        comptime K = Self.KNOTS
        self.cache_z.ensure_gpu(c, batch * T * P)
        self.ws_a.ensure_gpu(c, D * P)
        self.ws_cm.ensure_gpu(c, T * P * K)
        self.ws_sm.ensure_gpu(c, T * P * K)
        self.ws_partials.ensure_gpu(c, Self._n_partials())
        self.ws_scalar.ensure_gpu(c, 1)
        self.ws_dLdz.ensure_gpu(c, batch * T * P)

    # ── forward ───────────────────────────────────────────────────────
    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime D = Self.DIM
        comptime T = Self.SEQ_LEN
        comptime P = Self.NUM_PROJ
        comptime K = Self.KNOTS

        comptime if target == "cpu":
            out.ensure(B * 1)
            self.cache_z.ensure(B * T * P)
            var cache = TileTensor(self.cache_z.data, row_major[B, T * P]())
            var input = TileTensor(in0.data, row_major[B, T * D]())
            var output_v = TileTensor(out.data, row_major[B, 1]())
            var seed = self._forward_seed(
                UInt64(Int(self.cache_z.data.unsafe_ptr()))
            )
            var a = Array[Scalar[DT], D * P](uninitialized=True)
            self._fill_a_cpu(
                seed,
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                    a.unsafe_ptr()
                ),
            )

            for b in range(B):
                for t in range(T):
                    for p in range(P):
                        var z = Scalar[DT](0)
                        for d in range(D):
                            z += input[b, t * D + d] * a[d * P + p]
                        cache[b, t * P + p] = z

            comptime NTPK = T * P * K
            var cm = Array[Scalar[DT], NTPK](uninitialized=True)
            var sm = Array[Scalar[DT], NTPK](uninitialized=True)
            for i in range(NTPK):
                cm[i] = Scalar[DT](0)
                sm[i] = Scalar[DT](0)
            var inv_b = Scalar[DT](1.0 / Float64(B))
            for b in range(B):
                for t in range(T):
                    for p in range(P):
                        var z = cache[b, t * P + p]
                        for k in range(K):
                            var tk = Scalar[DT](Self._t_k(k))
                            var arg = z * tk
                            var idx = (t * P + p) * K + k
                            cm[idx] += cos(arg) * inv_b
                            sm[idx] += sin(arg) * inv_b

            var prefactor = Scalar[DT](Float64(B) / Float64(T * P))
            var stat = Scalar[DT](0)
            for t in range(T):
                for p in range(P):
                    for k in range(K):
                        var idx = (t * P + p) * K + k
                        var phi = Scalar[DT](Self._phi_k(k))
                        var wk = Scalar[DT](Self._w_k(k))
                        var diff = cm[idx] - phi
                        stat += wk * (diff * diff + sm[idx] * sm[idx])
            stat *= prefactor
            for b in range(B):
                output_v[b, 0] = stat
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * 1)
            self._ensure_gpu(c, B)
            comptime N_PARTIALS = Self._n_partials()
            comptime lay_a = Layout.row_major(D, P)
            comptime lay_cmsm = Layout.row_major(T, P * K)
            comptime lay_in = Layout.row_major(B, T * D)
            comptime lay_cache = Layout.row_major(B, T * P)
            comptime lay_out = Layout.row_major(B, 1)
            comptime lay_part = Layout.row_major(N_PARTIALS)
            comptime lay_one = Layout.row_major(1)
            var seed = self._forward_seed(
                UInt64(Int(self.cache_z.dev.value().unsafe_ptr()))
            )

            self._fill_a_gpu(c, seed)
            # cache_z[B*T, P] = X @ A — kept hand-rolled (faster at M=B·T=96, the
            # real LeWM regime; see _sr_project note + bench). max_matmul only
            # wins forward at M≥~1.5k, which no LeWM config hits.
            c.enqueue_function[_sr_project[B, T, D, P]](
                in0.lt["gpu", lay_in](),
                self.ws_a.lt["gpu", lay_a](),
                self.cache_z.lt["gpu", lay_cache](),
                grid_dim=((B * T * P + TPB - 1) // TPB,),
                block_dim=(TPB,),
            )
            c.enqueue_function[_sr_cm_sm[B, T, P, K, N_PARTIALS, True]](
                self.cache_z.lt["gpu", lay_cache](),
                self.ws_cm.lt["gpu", lay_cmsm](),
                self.ws_sm.lt["gpu", lay_cmsm](),
                self.ws_partials.lt["gpu", lay_part](),
                grid_dim=(N_PARTIALS,),
                block_dim=(TPB,),
            )
            c.enqueue_function[_sr_final_reduce[N_PARTIALS]](
                self.ws_partials.lt["gpu", lay_part](),
                self.ws_scalar.lt["gpu", lay_one](),
                grid_dim=(1,),
                block_dim=(TPB,),
            )
            c.enqueue_function[_sr_broadcast_stat[B, T, P]](
                self.ws_scalar.lt["gpu", lay_one](),
                out.lt["gpu", lay_out](),
                grid_dim=((B + TPB - 1) // TPB,),
                block_dim=(TPB,),
            )

    # ── backward ──────────────────────────────────────────────────────
    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime D = Self.DIM
        comptime T = Self.SEQ_LEN
        comptime P = Self.NUM_PROJ
        comptime K = Self.KNOTS

        comptime if target == "cpu":
            gin.ensure(B * T * D)
            var cache = TileTensor(self.cache_z.data, row_major[B, T * P]())
            var go = TileTensor(grad_output.data, row_major[B, 1]())
            var gi = TileTensor(gin.data, row_major[B, T * D]())
            var seed = self._backward_seed(
                UInt64(Int(self.cache_z.data.unsafe_ptr()))
            )
            var a = Array[Scalar[DT], D * P](uninitialized=True)
            self._fill_a_cpu(
                seed,
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                    a.unsafe_ptr()
                ),
            )

            comptime NTPK = T * P * K
            var cm = Array[Scalar[DT], NTPK](uninitialized=True)
            var sm = Array[Scalar[DT], NTPK](uninitialized=True)
            for i in range(NTPK):
                cm[i] = Scalar[DT](0)
                sm[i] = Scalar[DT](0)
            var inv_b = Scalar[DT](1.0 / Float64(B))
            for b in range(B):
                for t in range(T):
                    for p in range(P):
                        var z = cache[b, t * P + p]
                        for k in range(K):
                            var tk = Scalar[DT](Self._t_k(k))
                            var arg = z * tk
                            var idx = (t * P + p) * K + k
                            cm[idx] += cos(arg) * inv_b
                            sm[idx] += sin(arg) * inv_b

            var G = Scalar[DT](0)
            for b in range(B):
                G += go[b, 0]
            var coef = G * Scalar[DT](2.0 / Float64(T * P))

            for b in range(B):
                for t in range(T):
                    var dLdz = Array[Scalar[DT], P](uninitialized=True)
                    for p in range(P):
                        var z = cache[b, t * P + p]
                        var acc = Scalar[DT](0)
                        for k in range(K):
                            var tk = Scalar[DT](Self._t_k(k))
                            var wk = Scalar[DT](Self._w_k(k))
                            var phi = Scalar[DT](Self._phi_k(k))
                            var idx = (t * P + p) * K + k
                            var arg = z * tk
                            var bracket = (
                                -(cm[idx] - phi) * sin(arg) + sm[idx] * cos(arg)
                            )
                            acc += wk * tk * bracket
                        dLdz[p] = coef * acc
                    for d in range(D):
                        var g = Scalar[DT](0)
                        for p in range(P):
                            g += a[d * P + p] * dLdz[p]
                        gi[b, t * D + d] = g
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * T * D)
            self._ensure_gpu(c, B)
            comptime N_PARTIALS = Self._n_partials()
            comptime lay_a = Layout.row_major(D, P)
            comptime lay_cmsm = Layout.row_major(T, P * K)
            comptime lay_cache = Layout.row_major(B, T * P)
            comptime lay_go = Layout.row_major(B, 1)
            comptime lay_dLdz = Layout.row_major(B, T * P)
            comptime lay_one = Layout.row_major(1)
            var seed = self._backward_seed(
                UInt64(Int(self.cache_z.dev.value().unsafe_ptr()))
            )

            self._fill_a_gpu(c, seed)
            c.enqueue_function[_sr_cm_sm[B, T, P, K, N_PARTIALS, False]](
                self.cache_z.lt["gpu", lay_cache](),
                self.ws_cm.lt["gpu", lay_cmsm](),
                self.ws_sm.lt["gpu", lay_cmsm](),
                self.ws_partials.lt["gpu", Layout.row_major(N_PARTIALS)](),
                grid_dim=(N_PARTIALS,),
                block_dim=(TPB,),
            )
            c.enqueue_function[_sr_reduce_g[B]](
                grad_output.lt["gpu", lay_go](),
                self.ws_scalar.lt["gpu", lay_one](),
                grid_dim=(1,),
                block_dim=(TPB,),
            )
            c.enqueue_function[_sr_dLdz[B, T, P, K]](
                self.cache_z.lt["gpu", lay_cache](),
                self.ws_cm.lt["gpu", lay_cmsm](),
                self.ws_sm.lt["gpu", lay_cmsm](),
                self.ws_dLdz.lt["gpu", lay_dLdz](),
                self.ws_scalar.lt["gpu", lay_one](),
                grid_dim=((B * T * P + TPB - 1) // TPB,),
                block_dim=(TPB,),
            )
            # grad_input[B*T, D] = dLdz[B*T, P] @ A[D, P]ᵀ  (contract over P;
            # A consumed transpose_b, no physical transpose — mirrors Linear's
            # grad_in = grad_out @ Wᵀ).
            var ga_dl_v = TileTensor(
                self.ws_dLdz.dev.value(), row_major[B * T, P]()
            )
            var ga_a_v = TileTensor(self.ws_a.dev.value(), row_major[D, P]())
            var ga_gi_v = TileTensor(gin.dev.value(), row_major[B * T, D]())
            max_matmul[transpose_b=True, target="gpu"](
                ga_gi_v, ga_dl_v, ga_a_v, c
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (SIGReg is param-free → they reflect to a no-op).


# ============================================================================
# Module-level GPU kernels — ported VERBATIM from legacy sigreg.mojo (already
# NVIDIA-validated). block.sum reductions (no atomics). Generic over DT. The
# only signature change is partials/stat/g raw pointers → 1-D LayoutTensor views
# (storage surface never passes raw device pointers to kernels); BODY identical.
# ============================================================================


def _sr_gen_a_unnorm[D: Int, P: Int](
    a_t: LayoutTensor[DT, Layout.row_major(D, P), MutAnyOrigin],
    seed: UInt64,
):
    var idx = Int(global_idx.x)
    if idx >= D * P:
        return
    var d_idx = idx // P
    var p_idx = idx % P
    var philox = PhiloxRandom(seed=seed, offset=UInt64(idx))
    var rand_vals = philox.step_uniform()
    var u1 = Float32(rand_vals[0]) + Float32(1e-8)
    var u2 = Float32(rand_vals[1])
    var mag = sqrt(Float32(-2.0) * log(u1))
    var g = mag * cos(u2 * Float32(6.283185307179586))
    a_t[d_idx, p_idx] = Scalar[DT](g)


def _sr_norm_a[D: Int, P: Int](
    a_t: LayoutTensor[DT, Layout.row_major(D, P), MutAnyOrigin],
):
    var p_idx = Int(global_idx.x)
    if p_idx >= P:
        return
    var sum_sq = Scalar[DT](0)
    for d in range(D):
        var v = rebind[Scalar[DT]](a_t[d, p_idx])
        sum_sq += v * v
    var inv_norm = Scalar[DT](1) / (sqrt(sum_sq) + Scalar[DT](1e-12))
    for d in range(D):
        var v = rebind[Scalar[DT]](a_t[d, p_idx])
        a_t[d, p_idx] = v * inv_norm


# Forward projection `Z = X @ A` keeps this hand-rolled scalar GEMM: at the real
# LeWM shapes M=B·T is tiny (16·6=96), so the grid (B·T·P ≈ 98k threads, short
# K=D loop) is already saturated and BEATS a tensor-core GEMM that underutilizes
# at M=96 (NVIDIA: naive 6.8/3.9µs vs max_matmul 7.9/7.9µs at pusht/pong; the GEMM
# only wins at M≥~1.5k). The backward `grad_in = dLdz @ Aᵀ` DID move to max_matmul
# (its naive grid is only B·T·D threads each with a long uncoalesced K=P loop →
# 4.4× win at P=1024, wash at P=256, never worse). See vjp() + the A/B microbench
# benchmarks/bench_storage_sigreg_gemm_gpu.mojo.
def _sr_project[BATCH: Int, T: Int, D: Int, P: Int](
    input_t: LayoutTensor[DT, Layout.row_major(BATCH, T * D), MutAnyOrigin],
    a_t: LayoutTensor[DT, Layout.row_major(D, P), MutAnyOrigin],
    cache_t: LayoutTensor[DT, Layout.row_major(BATCH, T * P), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * P:
        return
    var p_idx = idx % P
    var t_idx = (idx // P) % T
    var b = idx // (T * P)
    var z = Scalar[DT](0)
    for d in range(D):
        var xi = rebind[Scalar[DT]](input_t[b, t_idx * D + d])
        var aval = rebind[Scalar[DT]](a_t[d, p_idx])
        z += xi * aval
    cache_t[b, t_idx * P + p_idx] = z


def _sr_cm_sm[
    BATCH: Int, T: Int, P: Int, K: Int, N_PARTIALS: Int, INCLUDE_STAT: Bool
](
    cache_t: LayoutTensor[DT, Layout.row_major(BATCH, T * P), MutAnyOrigin],
    cm_t: LayoutTensor[DT, Layout.row_major(T, P * K), MutAnyOrigin],
    sm_t: LayoutTensor[DT, Layout.row_major(T, P * K), MutAnyOrigin],
    partials_t: LayoutTensor[DT, Layout.row_major(N_PARTIALS), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var dt = Scalar[DT](3.0 / Float64(K - 1))
    var inv_b = Scalar[DT](1.0 / Float64(BATCH))
    var contrib = Scalar[DT](0)
    if idx < T * P * K:
        var k_idx = idx % K
        var p_idx = (idx // K) % P
        var t_idx = idx // (K * P)
        var tk = dt * Scalar[DT](k_idx)
        var phi = exp(Scalar[DT](-0.5) * tk * tk)
        var cm_sum = Scalar[DT](0)
        var sm_sum = Scalar[DT](0)
        for b in range(BATCH):
            var z = rebind[Scalar[DT]](cache_t[b, t_idx * P + p_idx])
            var arg = z * tk
            cm_sum += cos(arg)
            sm_sum += sin(arg)
        var cm = cm_sum * inv_b
        var sm = sm_sum * inv_b
        cm_t[t_idx, p_idx * K + k_idx] = cm
        sm_t[t_idx, p_idx * K + k_idx] = sm
        comptime if INCLUDE_STAT:
            var trap = dt if (k_idx == 0 or k_idx == K - 1) else dt * Scalar[
                DT
            ](2.0)
            var wk = trap * phi
            var diff = cm - phi
            contrib = wk * (diff * diff + sm * sm)

    comptime if INCLUDE_STAT:
        var partial = block.sum[block_size=TPB, broadcast=False](
            val=SIMD[DT, 1](contrib)
        )
        if thread_idx.x == 0:
            partials_t[Int(block_idx.x)] = partial[0]


def _sr_final_reduce[N_PARTIALS: Int](
    partials_t: LayoutTensor[
        DT, Layout.row_major(N_PARTIALS), MutAnyOrigin
    ],
    out_t: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var v = Scalar[DT](0)
    var i = tid
    while i < N_PARTIALS:
        v += rebind[Scalar[DT]](partials_t[i])
        i += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=SIMD[DT, 1](v))
    if tid == 0:
        out_t[0] = total[0]


def _sr_broadcast_stat[BATCH: Int, T: Int, P: Int](
    stat_t: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
    output_t: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var prefactor = Scalar[DT](Float64(BATCH) / Float64(T * P))
    output_t[b, 0] = rebind[Scalar[DT]](stat_t[0]) * prefactor


def _sr_reduce_g[BATCH: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    g_t: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
):
    var b = Int(thread_idx.x)
    var v = Scalar[DT](0)
    if b < BATCH:
        v = rebind[Scalar[DT]](grad_output[b, 0])
    var total = block.sum[block_size=TPB, broadcast=False](val=SIMD[DT, 1](v))
    if b == 0:
        g_t[0] = total[0]


def _sr_dLdz[BATCH: Int, T: Int, P: Int, K: Int](
    cache_t: LayoutTensor[DT, Layout.row_major(BATCH, T * P), MutAnyOrigin],
    cm_t: LayoutTensor[DT, Layout.row_major(T, P * K), MutAnyOrigin],
    sm_t: LayoutTensor[DT, Layout.row_major(T, P * K), MutAnyOrigin],
    dLdz_t: LayoutTensor[DT, Layout.row_major(BATCH, T * P), MutAnyOrigin],
    g_t: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * P:
        return
    var p_idx = idx % P
    var t_idx = (idx // P) % T
    var b = idx // (T * P)
    var z = rebind[Scalar[DT]](cache_t[b, t_idx * P + p_idx])
    var dt = Scalar[DT](3.0 / Float64(K - 1))
    var acc = Scalar[DT](0)
    for k in range(K):
        var tk = dt * Scalar[DT](k)
        var phi = exp(Scalar[DT](-0.5) * tk * tk)
        var trap = dt if (k == 0 or k == K - 1) else dt * Scalar[DT](2.0)
        var wk = trap * phi
        var cm = rebind[Scalar[DT]](cm_t[t_idx, p_idx * K + k])
        var sm = rebind[Scalar[DT]](sm_t[t_idx, p_idx * K + k])
        var arg = z * tk
        var bracket = -(cm - phi) * sin(arg) + sm * cos(arg)
        acc += wk * tk * bracket
    var G = rebind[Scalar[DT]](g_t[0])
    var coef = G * Scalar[DT](2.0 / Float64(T * P))
    dLdz_t[b, t_idx * P + p_idx] = coef * acc

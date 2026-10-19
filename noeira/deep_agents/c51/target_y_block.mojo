"""C51TargetYBlock — categorical-projected target distribution (STORAGE).

Bellemare, Dabney & Munos 2017 (Algorithm 1). Operates on the distributional
Q-net output `[B, NA · N_ATOMS]` (`Q_target.forward(sp)`) and produces a
per-batch target distribution `m [B, N_ATOMS]`.

Pipeline (standard branch):
  1. logits_t = Q_target.forward(sp)               [B, NA · N_ATOMS]
  2. Per (b, a): softmax over N_ATOMS              p_t[b, a, k]
  3. Per (b, a): expected Q = Σ_k p_t[b, a, k]·z_k  exp_q[b, a]
  4. a*[b] = argmax_a exp_q[b, a]
  5. p_target[b, k] = p_t[b, a*[b], k]              [B, N_ATOMS]
  6. Projection: Tz_k = clip(r + γ^n·z_k·(1−d), V_min, V_max);
     scatter p_target onto ⌊bj⌋/⌈bj⌉ bins → m.

Double-C51 branch (`DOUBLE=True`): argmax a* from Q_online's expected-Q;
distribution sampled from Q_target's softmax at a*.

STORAGE migration (Stage 5): `Scratch`/`TargetStorage`/`init_scratch_auto`/
TileTensor gone — scratch are owned `nn.storage.Tensor`s; the Q nets forward
through the storage `Module` surface over `TensorRefs`. The two BATCH-parallel
kernels (argmax / projection, no atomics) are unchanged; views are built via
`.lt[target, layout]()`. Output is written into the trainer's `_mb_m` Tensor
([B · N_ATOMS], NOT the BATCH-wide `state.mb_y`). CPU + GPU.
"""

from std.math import exp as fexp, log as flog
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.call import call_forward
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.core.tiles import RowMajor, Tile


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — module-level (one thread per BATCH row, no atomics).
# ──────────────────────────────────────────────────────────────────────


def _c51_argmax_action_kernel[
    BATCH: Int, NA: Int, N_ATOMS: Int,
](
    logits: Tile[DT, BATCH, NA * N_ATOMS],
    z: Tile[DT, N_ATOMS],
    best_a_out: Tile[DT, BATCH],
):
    """Per row b: pick `argmax_a Σ_k softmax(logits[b, a, ·])_k · z_k`."""
    var b = Int(global_idx.x)
    if b < BATCH:
        var best_a: Int = 0
        var best_eq: Scalar[DT] = Scalar[DT](0.0)
        for a in range(NA):
            var base = a * N_ATOMS
            var mx = rebind[Scalar[DT]](logits[b, base])
            for i in range(1, N_ATOMS):
                var v = rebind[Scalar[DT]](logits[b, base + i])
                if v > mx:
                    mx = v
            var s_exp: Scalar[DT] = Scalar[DT](0.0)
            for i in range(N_ATOMS):
                s_exp = s_exp + fexp(
                    rebind[Scalar[DT]](logits[b, base + i]) - mx
                )
            var eq: Scalar[DT] = Scalar[DT](0.0)
            for i in range(N_ATOMS):
                var p = (
                    fexp(rebind[Scalar[DT]](logits[b, base + i]) - mx)
                    / s_exp
                )
                eq = eq + p * rebind[Scalar[DT]](z[i])
            if a == 0 or eq > best_eq:
                best_eq = eq
                best_a = a
        best_a_out[b] = Scalar[DT](best_a)


def _c51_project_kernel[
    BATCH: Int, NA: Int, N_ATOMS: Int,
](
    logits_t: Tile[DT, BATCH, NA * N_ATOMS],
    best_a: Tile[DT, BATCH],
    r: Tile[DT, BATCH],
    d: Tile[DT, BATCH],
    z: Tile[DT, N_ATOMS],
    m_out: Tile[DT, BATCH, N_ATOMS],
    gamma_n: Scalar[DT],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    dz: Scalar[DT],
):
    """Bellemare Algorithm 1 — one thread per b. Zeroes m[b, *] then
    scatter-adds (l_idx, u_idx) bin pairs from softmax(logits_t[b, a*]·z) for
    the chosen action a*. No atomics needed."""
    var b = Int(global_idx.x)
    if b < BATCH:
        for k in range(N_ATOMS):
            m_out[b, k] = Scalar[DT](0.0)

        var a_star = Int(rebind[Scalar[DT]](best_a[b]))
        var tbase = a_star * N_ATOMS

        var mxt = rebind[Scalar[DT]](logits_t[b, tbase])
        for i in range(1, N_ATOMS):
            var v = rebind[Scalar[DT]](logits_t[b, tbase + i])
            if v > mxt:
                mxt = v
        var st_exp: Scalar[DT] = Scalar[DT](0.0)
        for i in range(N_ATOMS):
            st_exp = st_exp + fexp(
                rebind[Scalar[DT]](logits_t[b, tbase + i]) - mxt
            )

        var rb = rebind[Scalar[DT]](r[b])
        var nonterm = Scalar[DT](1.0) - rebind[Scalar[DT]](d[b])
        for j in range(N_ATOMS):
            var p_tgt = (
                fexp(rebind[Scalar[DT]](logits_t[b, tbase + j]) - mxt)
                / st_exp
            )
            var tz = rb + gamma_n * rebind[Scalar[DT]](z[j]) * nonterm
            if tz < v_min:
                tz = v_min
            if tz > v_max:
                tz = v_max
            var bj = (tz - v_min) / dz
            var l_idx = Int(bj)
            var u_idx = l_idx + 1
            if u_idx >= N_ATOMS:
                u_idx = N_ATOMS - 1
            if l_idx >= N_ATOMS:
                l_idx = N_ATOMS - 1
            if l_idx == u_idx:
                m_out[b, l_idx] = rebind[Scalar[DT]](m_out[b, l_idx]) + p_tgt
            else:
                var u_w = bj - Scalar[DT](l_idx)
                var l_w = Scalar[DT](1.0) - u_w
                m_out[b, l_idx] = (
                    rebind[Scalar[DT]](m_out[b, l_idx]) + p_tgt * l_w
                )
                m_out[b, u_idx] = (
                    rebind[Scalar[DT]](m_out[b, u_idx]) + p_tgt * u_w
                )


struct C51TargetYBlock[
    Q_NET: Module,
    BATCH: Int,
    OBS: Int,
    NA: Int,
    N_ATOMS: Int,
    DOUBLE: Bool = False,
](Defaultable & Movable & Deinitable):
    """Owns `_logits_t`/`_logits_on` (Q forward outputs), `_z` (atom support),
    `_best_a` (argmax action), and the γ^n / V_min / V_max / Δz scalars."""

    var _logits_t: Tensor       # [B · NA · N_ATOMS]
    var _logits_on: Tensor      # [B · NA · N_ATOMS] (Double only)
    var _z: Tensor              # [N_ATOMS] atom support
    var _best_a: Tensor         # [B]

    var gamma_n: Scalar[DT]
    var v_min: Scalar[DT]
    var v_max: Scalar[DT]
    var dz: Scalar[DT]

    def __init__(out self):
        self._logits_t = Tensor()
        self._logits_on = Tensor()
        self._z = Tensor()
        self._best_a = Tensor()
        self.gamma_n = Scalar[DT](0.99)
        self.v_min = Scalar[DT](-10.0)
        self.v_max = Scalar[DT](10.0)
        self.dz = Scalar[DT](0.0)

    @staticmethod
    def make[
        target: StaticString,
    ](
        gamma: Scalar[DT] = Scalar[DT](0.99),
        nstep: Int = 1,
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Bakes γ^nstep into the bootstrap discount + sets the atom support
        `z_k = V_min + k·(V_max − V_min)/(N_ATOMS−1)`."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "C51TargetYBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS
        ), "C51TargetYBlock: Q_NET.IN_DIM must equal OBS"
        comptime assert (
            Self.Q_NET.OUT_DIM == Self.NA * Self.N_ATOMS
        ), "C51TargetYBlock: Q_NET.OUT_DIM must equal NA · N_ATOMS"
        comptime assert (
            Self.N_ATOMS >= 2
        ), "C51TargetYBlock: N_ATOMS must be ≥ 2 (Δz needs N_ATOMS−1)"
        var b = Self()
        var g_n: Scalar[DT] = Scalar[DT](1.0)
        for _ in range(nstep):
            g_n = g_n * gamma
        b.gamma_n = g_n
        b.v_min = v_min
        b.v_max = v_max
        b.dz = (v_max - v_min) / Scalar[DT](Self.N_ATOMS - 1)

        comptime SZ = Self.BATCH * Self.NA * Self.N_ATOMS
        comptime if target == "cpu":
            b._logits_t = Tensor.alloc(SZ)
            b._logits_on = Tensor.alloc(SZ)
            b._z = Tensor.alloc(Self.N_ATOMS)
            b._best_a = Tensor.alloc(Self.BATCH)
        else:
            var c = ctx.value()
            b._logits_t = Tensor.alloc_gpu(c, SZ)
            b._logits_on = Tensor.alloc_gpu(c, SZ)
            b._z = Tensor.alloc_gpu(c, Self.N_ATOMS)
            b._best_a = Tensor.alloc_gpu(c, Self.BATCH)

        # Bake the z support — CPU `data` always (the trainer's action selection
        # reads it at greedy-eval). On GPU also upload (kernels read it).
        b._z.ensure(Self.N_ATOMS)  # ensure host List exists even on GPU
        for k in range(Self.N_ATOMS):
            b._z.data[k] = v_min + Scalar[DT](k) * b.dz
        comptime if target == "gpu":
            b._z.upload(ctx.value())
        return b^

    def z(mut self) -> ref [MutAnyOrigin] Tensor:
        """Expose the z support Tensor — used by the trainer's action selection
        + the Q-update block to compute expected Q for argmax (host `.data`)."""
        return self._z

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut q_target: Self.Q_NET,
        mut q_online: Self.Q_NET,
        mut mb_sp: Tensor,
        mut mb_r: Tensor,
        mut mb_d: Tensor,
        mut mb_m: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Writes `mb_m` [B, N_ATOMS] target distribution in-place. `q_online`
        is ignored on the standard path (DOUBLE=False)."""
        comptime ROW = Self.NA * Self.N_ATOMS

        # Step 1: Q_target(sp) → _logits_t.
        call_forward[target, Self.BATCH, POLICY=POLICY](
            q_target, TensorRefs[Self.Q_NET.ARITY](mb_sp), self._logits_t, ctx
        )

        # Step 2 (DOUBLE only): Q_online(sp) → _logits_on.
        comptime if Self.DOUBLE:
            call_forward[target, Self.BATCH, POLICY=POLICY](
                q_online, TensorRefs[Self.Q_NET.ARITY](mb_sp), self._logits_on, ctx
            )

        comptime if target == "gpu":
            var c = ctx.value()
            comptime LZ = RowMajor[Self.N_ATOMS]
            comptime LB = RowMajor[Self.BATCH]
            comptime LROW = RowMajor[Self.BATCH, ROW]
            comptime LM = RowMajor[Self.BATCH, Self.N_ATOMS]
            comptime n_blocks_b = (Self.BATCH + TPB - 1) // TPB

            # Argmax: source is _logits_on if DOUBLE else _logits_t.
            comptime arg_kernel = _c51_argmax_action_kernel[
                Self.BATCH, Self.NA, Self.N_ATOMS,
            ]
            comptime if Self.DOUBLE:
                c.enqueue_function[arg_kernel](
                    self._logits_on.tt["gpu", Self.BATCH, ROW](),
                    self._z.tt["gpu", Self.N_ATOMS](),
                    self._best_a.tt["gpu", Self.BATCH](),
                    grid_dim=n_blocks_b, block_dim=TPB,
                )
            else:
                c.enqueue_function[arg_kernel](
                    self._logits_t.tt["gpu", Self.BATCH, ROW](),
                    self._z.tt["gpu", Self.N_ATOMS](),
                    self._best_a.tt["gpu", Self.BATCH](),
                    grid_dim=n_blocks_b, block_dim=TPB,
                )

            # Project Q_target softmax(a*) onto z support → m.
            comptime proj_kernel = _c51_project_kernel[
                Self.BATCH, Self.NA, Self.N_ATOMS,
            ]
            c.enqueue_function[proj_kernel](
                self._logits_t.tt["gpu", Self.BATCH, ROW](),
                self._best_a.tt["gpu", Self.BATCH](),
                mb_r.tt["gpu", Self.BATCH](),
                mb_d.tt["gpu", Self.BATCH](),
                self._z.tt["gpu", Self.N_ATOMS](),
                mb_m.tt["gpu", Self.BATCH, Self.N_ATOMS](),
                self.gamma_n, self.v_min, self.v_max, self.dz,
                grid_dim=n_blocks_b, block_dim=TPB,
            )
            return

        # ── CPU path. ──────────────────────────────────────────────────
        # Zero output m first (we scatter-add into it).
        for k in range(Self.BATCH * Self.N_ATOMS):
            mb_m.data[k] = Scalar[DT](0.0)

        # Step 3: argmax action per row. Source = Q_online (Double) or Q_target.
        # Inlined (not a helper taking a `ref` into a self-field — that would
        # alias `self`); all reads go through `self.` so it's one borrow.
        var best_a_per_row = List[Int](length=Self.BATCH, fill=0)
        for b in range(Self.BATCH):
            var best_a: Int = 0
            var best_eq: Scalar[DT] = Scalar[DT](0.0)
            for a in range(Self.NA):
                var base = b * ROW + a * Self.N_ATOMS
                var mxa: Scalar[DT]
                comptime if Self.DOUBLE:
                    mxa = self._logits_on.data[base]
                else:
                    mxa = self._logits_t.data[base]
                for i in range(1, Self.N_ATOMS):
                    var vi: Scalar[DT]
                    comptime if Self.DOUBLE:
                        vi = self._logits_on.data[base + i]
                    else:
                        vi = self._logits_t.data[base + i]
                    if vi > mxa:
                        mxa = vi
                var s_exp: Scalar[DT] = Scalar[DT](0.0)
                for i in range(Self.N_ATOMS):
                    comptime if Self.DOUBLE:
                        s_exp = s_exp + fexp(self._logits_on.data[base + i] - mxa)
                    else:
                        s_exp = s_exp + fexp(self._logits_t.data[base + i] - mxa)
                var eq: Scalar[DT] = Scalar[DT](0.0)
                for i in range(Self.N_ATOMS):
                    var pi: Scalar[DT]
                    comptime if Self.DOUBLE:
                        pi = fexp(self._logits_on.data[base + i] - mxa) / s_exp
                    else:
                        pi = fexp(self._logits_t.data[base + i] - mxa) / s_exp
                    eq = eq + pi * self._z.data[i]
                if a == 0 or eq > best_eq:
                    best_eq = eq
                    best_a = a
            best_a_per_row[b] = best_a

        # Per-row projection from Q_target's softmax at a*.
        for b in range(Self.BATCH):
            var best_a = best_a_per_row[b]
            var tbase = b * ROW + best_a * Self.N_ATOMS
            var mxt = self._logits_t.data[tbase]
            for i in range(1, Self.N_ATOMS):
                if self._logits_t.data[tbase + i] > mxt:
                    mxt = self._logits_t.data[tbase + i]
            var st_exp: Scalar[DT] = Scalar[DT](0.0)
            for i in range(Self.N_ATOMS):
                st_exp = st_exp + fexp(self._logits_t.data[tbase + i] - mxt)

            var r = mb_r.data[b]
            var nonterm = Scalar[DT](1.0) - mb_d.data[b]
            for j in range(Self.N_ATOMS):
                var p_tgt_j = fexp(self._logits_t.data[tbase + j] - mxt) / st_exp
                var tz = r + self.gamma_n * self._z.data[j] * nonterm
                if tz < self.v_min:
                    tz = self.v_min
                if tz > self.v_max:
                    tz = self.v_max
                var bj = (tz - self.v_min) / self.dz
                var l_idx = Int(bj)
                var u_idx = l_idx + 1
                if u_idx >= Self.N_ATOMS:
                    u_idx = Self.N_ATOMS - 1
                if l_idx >= Self.N_ATOMS:
                    l_idx = Self.N_ATOMS - 1
                if l_idx == u_idx:
                    mb_m.data[b * Self.N_ATOMS + l_idx] = (
                        mb_m.data[b * Self.N_ATOMS + l_idx] + p_tgt_j
                    )
                else:
                    var u_w = bj - Scalar[DT](l_idx)
                    var l_w = Scalar[DT](1.0) - u_w
                    mb_m.data[b * Self.N_ATOMS + l_idx] = (
                        mb_m.data[b * Self.N_ATOMS + l_idx] + p_tgt_j * l_w
                    )
                    mb_m.data[b * Self.N_ATOMS + u_idx] = (
                        mb_m.data[b * Self.N_ATOMS + u_idx] + p_tgt_j * u_w
                    )


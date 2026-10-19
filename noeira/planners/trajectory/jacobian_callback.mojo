"""Jacobian rollout callback traits — gradient-aware model contract for iLQR.

iLQR (and DDP, future MPC variants) needs more than the gradient-free
``RolloutCallback{CPU,GPU}`` exposes: at each step it needs the
linearization of the dynamics around the current trajectory plus the
quadratic approximation of the per-step cost. These traits are the
opt-in extension surface where a world model exposes those derivatives.

Why a separate trait and not a single mega-``RolloutCallback`` with
optional methods? Two reasons:

1. CEM / MPPI only need ``(z, u) → (z', r)`` — they should not be forced
   to implement Jacobian methods. Mojo's trait system is structural, so
   an extension trait is the cleanest way to gate the gradient surface.
2. Computing the linearization is the heaviest method on the contract.
   Splitting it lets a single world model implement *only* the
   gradient-free surface for MPPI and *both* surfaces for iLQR without
   coupling the two paths in the agent.

iLQR convention: **cost-space** (minimization). The callback returns
``cost`` (typically ``-reward``) and a quadratic cost expansion
``l(z, u) ≈ l(z̄, ū) + l_z·δz + l_u·δu + ½δz·l_zz·δz + ½δu·l_uu·δu
+ δz·l_zu·δu``. Translating from reward-space to cost-space is the
adapter's job, not the planner's.

Layout convention for matrices passed as flat ``List[Float64]`` / 2-D
``LayoutTensor`` (CPU / GPU surfaces):

* ``A``      — ``LATENT × LATENT``, row-major (Jacobian ``∂f/∂z``).
* ``B``      — ``LATENT × ACTION``, row-major (Jacobian ``∂f/∂u``).
* ``l_z``    — ``LATENT`` vector.
* ``l_u``    — ``ACTION`` vector.
* ``l_zz``   — ``LATENT × LATENT``, row-major (symmetric).
* ``l_uu``   — ``ACTION × ACTION``, row-major (symmetric, PSD for a
  well-posed problem; the planner adds Levenberg-Marquardt μ·I).
* ``l_zu``   — ``LATENT × ACTION``, row-major (cross-Hessian, ``∂²l/∂z∂u``).

Terminal-state expansion writes ``V_z`` (length ``LATENT``) and ``V_zz``
(``LATENT × LATENT`` row-major) and returns the scalar terminal cost.

Two traits, mirroring ``RolloutCallback{CPU,GPU}``: implementors pick
one or both. Stub callbacks for unit tests typically implement CPU only;
production world-model adapters implement GPU; CPU↔GPU parity tests
implement both.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT as dtype
from noeira.core.tiles import Tile


trait RolloutJacobianCallbackCPU(Deinitable):
    """CPU-side gradient-aware rollout contract for iLQR.

    All operations are scalar (B = 1) — the planner loops over horizon
    timesteps and calls the callback once per step. Buffers passed in
    must already be sized to their natural dims (caller resizes; callee
    fills).
    """

    comptime LATENT_DIM: Int
    comptime ACTION_DIM: Int

    def step_cpu(
        mut self,
        z: List[Float64],
        u: List[Float64],
        mut z_next_out: List[Float64],
    ) raises -> Float64:
        """Advance one step: write ``z'`` into ``z_next_out`` (length
        ``LATENT_DIM``) and return the scalar per-step ``cost``. ``z``
        has length ``LATENT_DIM``, ``u`` has length ``ACTION_DIM``.
        Lower cost is better (iLQR minimizes).
        """
        ...

    def linearize_cpu(
        mut self,
        z: List[Float64],
        u: List[Float64],
        mut A_out: List[Float64],
        mut B_out: List[Float64],
        mut l_z_out: List[Float64],
        mut l_u_out: List[Float64],
        mut l_zz_out: List[Float64],
        mut l_uu_out: List[Float64],
        mut l_zu_out: List[Float64],
    ) raises:
        """Write the dynamics Jacobians and quadratic cost expansion at
        the operating point ``(z, u)``. See the module docstring for the
        layout convention. All output buffers are caller-allocated with
        the natural sizes (``LATENT*LATENT``, ``LATENT*ACTION``,
        ``LATENT``, ``ACTION``, ``LATENT*LATENT``, ``ACTION*ACTION``,
        ``LATENT*ACTION``); the callback fills them in-place.
        """
        ...

    def terminal_cpu(
        mut self,
        z: List[Float64],
        mut V_z_out: List[Float64],
        mut V_zz_out: List[Float64],
    ) raises -> Float64:
        """Terminal cost expansion: return scalar ``Φ(z)`` and write its
        gradient (``LATENT``) and Hessian (``LATENT × LATENT``) into the
        caller-supplied buffers. The iLQR backward pass seeds the
        Riccati recursion with these values.
        """
        ...


trait RolloutJacobianCallbackGPU(Deinitable):
    """GPU-side gradient-aware rollout contract for iLQR.

    Operates on row-major ``LayoutTensor`` views with a method-level
    ``B`` comptime parameter. The planner orchestrates ``HORIZON``
    linearize calls without intervening sync; implementors must not add
    sync points inside these methods (consistent with the
    ``RolloutCallbackGPU`` performance contract).

    The 4-D matrix outputs (``A_out``, ``B_out``, ``l_zz_out``,
    ``l_uu_out``, ``l_zu_out``) follow the same row-major layout as the
    CPU surface, lifted to a leading batch dim: ``A_out`` is
    ``(B, LATENT_DIM, LATENT_DIM)``, etc.
    """

    comptime LATENT_DIM: Int
    comptime ACTION_DIM: Int

    def step_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        u: Tile[dtype, B, Self.ACTION_DIM],
        z_next_out: Tile[dtype, B, Self.LATENT_DIM],
        cost_out: Tile[dtype, B],
    ) raises:
        """Advance one step for the full batch and write per-row scalar
        costs into ``cost_out``. Convention: lower cost is better.
        """
        ...

    def linearize_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        u: Tile[dtype, B, Self.ACTION_DIM],
        A_out: Tile[dtype, B, Self.LATENT_DIM, Self.LATENT_DIM],
        B_out: Tile[dtype, B, Self.LATENT_DIM, Self.ACTION_DIM],
        l_z_out: Tile[dtype, B, Self.LATENT_DIM],
        l_u_out: Tile[dtype, B, Self.ACTION_DIM],
        l_zz_out: Tile[dtype, B, Self.LATENT_DIM, Self.LATENT_DIM],
        l_uu_out: Tile[dtype, B, Self.ACTION_DIM, Self.ACTION_DIM],
        l_zu_out: Tile[dtype, B, Self.LATENT_DIM, Self.ACTION_DIM],
    ) raises:
        """Write dynamics Jacobians and quadratic cost expansion at every
        operating point in the batch. See module docstring for layout.
        """
        ...

    def terminal_gpu[
        B: Int
    ](
        mut self,
        ctx: DeviceContext,
        z: Tile[dtype, B, Self.LATENT_DIM],
        V_z_out: Tile[dtype, B, Self.LATENT_DIM],
        V_zz_out: Tile[dtype, B, Self.LATENT_DIM, Self.LATENT_DIM],
        cost_out: Tile[dtype, B],
    ) raises:
        """Terminal cost expansion for the batch: scalar ``Φ`` per row +
        gradient + Hessian. Seeds the Riccati backward pass.
        """
        ...

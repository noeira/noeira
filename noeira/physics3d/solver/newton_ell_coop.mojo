"""The blocked Newton kernel's ELLIPTIC leg — its per-contact and cooperative
pieces, over the threadgroup-memory row layout.

WHY THIS FILE EXISTS (PERFORMANCE.md §13.53, 2026-09-15). `solve_newton`
routed the ELLIPTIC cone on every device to the one-thread-per-env kernel,
whose launch is `NS_TPB = 1`: 256 LIBERO lanes were 256 single-thread blocks
on 170 SMs, and the solver was 60% of the step at 23.6 ms per substep — for
1.27 Newton iterations per solve. The cost was the per-solve SETUP (rows,
Hessian build, factor) on one thread, which is exactly what
`_newton_blocked_fields_kernel` parallelises across the block for the
pyramidal cone. This module is the elliptic cone's half of that kernel.

THE ROW LAYOUT. The kernel keeps one constraint-row list in `Je_sh`
(`[row * nv + i]`), `De_sh`, `bias_e_sh`, ... A pyramidal contact owns
`2*(dim-1)` edge rows there. An ELLIPTIC contact `c` owns `RPC = NT + 1`
rows at `row0 = c * RPC`: the NORMAL row at `row0` and tangential row `t` at
`row0 + 1 + t`, `t < ntc[c]` (the contact's own `dim-1`; the rest are zero).
Per row, `De_sh` holds `D_n` / `D_t[t]`, `bias_e_sh` holds `pos_bias` /
`bt[t]`, and `fr_e_sh` — an array only this leg allocates — holds
`con->friction[t]` on the tangential rows. Per contact, `mu_sh`, `ntc_sh`,
`cact_sh` (penetrating) and `cs_sh` (the zone, `ELL_*`) sit in MC-sized
arrays. `RPC <= 2*NT` for `NT >= 1`, so the contact rows fit the `ME` the
pyramidal edge count already sizes `Je_sh` by — `je_budget` is unchanged.

⚠⚠ THE CONE ARITHMETIC IS `elliptic_cone.mojo`'s AND IS NOT REPEATED HERE.
Its helpers take NT-sized `Scratch` arrays indexed from a `base`; this leg
copies a contact's `nt` tangential entries out of the shared rows into
NT-sized locals (`base = 0`), calls the helper, and copies the forces back.
NT is at most 5 and the copies are exact, so the answer is the per-env
leg's bit for bit and the zone / force / cost / Hessian / line rules have
ONE spelling in the tree (`_a_rule_written_inline_twice_drifts`).

⚠ THE SUMMATION ORDERS ARE THE PER-ENV LEG'S, ON PURPOSE. `_newton_solve_env`
(the GPU per-env kernel runs it dense, `TREE_AWARE=False`) accumulates
`qfrc` per DOF as `sum_c (Jn*fn + sum_t Jt*ft)` — grouped per contact —
then the scalar rows, then the equality rows; `jar` per row as
`bias + sum_i J*qacc` ascending; the Hessian entry as
`M + rows + sum_c sum_k J_k[i] * (sum_j Hb[k,j] J_j[j'])` with
`ell_add_contact_hessian`'s zero skips. `_ell_recompute_coop` and the
kernel's entry loop keep every one of those orders, which is what lets
`test_newton_blocked_elliptic` gate blocked-GPU against per-env-GPU at
EQUALITY rather than a tolerance. The one known departure is the dense
rows' ORDER when a model has tendon or connect/weld rows: the per-env leg
adds joint-limit and friction-dof rows before them and the blocked row list
puts friction-dof rows last, so on such a model an entry both kinds touch
can differ in its last bit. LIBERO and the gate fixture have neither.
"""

from layout import TileTensor, TensorLayout
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from ..fields.dims import DimsLike
from ..fields.scratch import Scratch
from ..constraints.scalar_rows import (
    scalar_row_state,
    scalar_row_force,
    scalar_row_cost,
)
from .elliptic_cone import (
    ell_state_force,
    ell_row_cost,
    ell_hessian_block,
    ell_line_eval,
    ELL_SATISFIED,
)


@always_inline
def _ell_tangent_locals[
    DTYPE: DType, NT: Int, L_ROW: TensorLayout
](
    nt: Int,
    row0: Int,
    src: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    mut dst: Scratch[Scalar[DTYPE], NT],
):
    """This contact's `nt` tangential entries of a per-row shared array
    (`row0 + 1 + t`) into an NT-sized local; the tail zeroed."""
    for t in range(NT):
        if t < nt:
            dst[t] = rebind[Scalar[DTYPE]](src[row0 + 1 + t])
        else:
            dst[t] = Scalar[DTYPE](0)


@always_inline
def _ell_cost_at_jar[
    DTYPE: DType, NT: Int
](
    nt: Int,
    jar_n: Scalar[DTYPE],
    jar_t: Scratch[Scalar[DTYPE], NT],
    mu: Scalar[DTYPE],
    D_n: Scalar[DTYPE],
    D_t: Scratch[Scalar[DTYPE], NT],
    fr: Scratch[Scalar[DTYPE], NT],
) -> Scalar[DTYPE]:
    """One contact's cost at a TRIAL `jar` — the warm-start comparison's
    term: classify (`ell_state_force`, forces discarded), then price
    (`ell_row_cost`). The per-env leg calls the same pair."""
    var f_n = Scalar[DTYPE](0)
    var f_t = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    var zone = ell_state_force[DTYPE, NT, NT](
        nt, 0, jar_n, jar_t, mu, D_n, D_t, fr, f_n, f_t
    )
    return ell_row_cost[DTYPE, NT, NT](
        zone, nt, 0, jar_n, jar_t, mu, D_n, D_t, fr
    )


@always_inline
def _ell_contact_cost[
    DTYPE: DType, NT: Int, L_ROW: TensorLayout, L_FR: TensorLayout, L_C: TensorLayout
](
    c: Int,
    jar_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    De_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    fr_e_sh: TileTensor[DTYPE, L_FR, MutAnyOrigin, address_space=AddressSpace.SHARED],
    mu_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    ntc_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    cs_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
) -> Scalar[DTYPE]:
    """The contact's primal cost at the CURRENT `jar` and zone — the total
    cost's term (`_total_cost` in the per-env leg). The caller skips
    inactive contacts, as that leg does."""
    comptime RPC = NT + 1
    var row0 = c * RPC
    var nt = Int(rebind[Scalar[DTYPE]](ntc_sh[c]))
    var jar_t = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    var D_t = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    var fr = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    _ell_tangent_locals[DTYPE, NT](nt, row0, jar_sh, jar_t)
    _ell_tangent_locals[DTYPE, NT](nt, row0, De_sh, D_t)
    _ell_tangent_locals[DTYPE, NT](nt, row0, fr_e_sh, fr)
    return ell_row_cost[DTYPE, NT, NT](
        Int(rebind[Scalar[DTYPE]](cs_sh[c])),
        nt,
        0,
        rebind[Scalar[DTYPE]](jar_sh[row0]),
        jar_t,
        rebind[Scalar[DTYPE]](mu_sh[c]),
        rebind[Scalar[DTYPE]](De_sh[row0]),
        D_t,
        fr,
    )


@always_inline
def _ell_contact_line_eval[
    DTYPE: DType, NT: Int, L_ROW: TensorLayout, L_FR: TensorLayout, L_C: TensorLayout
](
    c: Int,
    a: Scalar[DTYPE],
    jar_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    Jv_e_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    De_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    fr_e_sh: TileTensor[DTYPE, L_FR, MutAnyOrigin, address_space=AddressSpace.SHARED],
    mu_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    ntc_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    mut cost: Scalar[DTYPE],
    mut d0: Scalar[DTYPE],
    mut d1: Scalar[DTYPE],
):
    """`PrimalEval`'s term for one contact at `alpha = a`, through
    `ell_line_eval` on the current `jar` rows and the `J * search` rows the
    cooperative matvec left in `Jv_e_sh`."""
    comptime RPC = NT + 1
    var row0 = c * RPC
    var nt = Int(rebind[Scalar[DTYPE]](ntc_sh[c]))
    var jar_t = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    var Js_t = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    var D_t = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    var fr = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    _ell_tangent_locals[DTYPE, NT](nt, row0, jar_sh, jar_t)
    _ell_tangent_locals[DTYPE, NT](nt, row0, Jv_e_sh, Js_t)
    _ell_tangent_locals[DTYPE, NT](nt, row0, De_sh, D_t)
    _ell_tangent_locals[DTYPE, NT](nt, row0, fr_e_sh, fr)
    ell_line_eval[DTYPE, NT, NT](
        nt,
        0,
        a,
        rebind[Scalar[DTYPE]](jar_sh[row0]),
        jar_t,
        rebind[Scalar[DTYPE]](Jv_e_sh[row0]),
        Js_t,
        rebind[Scalar[DTYPE]](mu_sh[c]),
        rebind[Scalar[DTYPE]](De_sh[row0]),
        D_t,
        fr,
        cost,
        d0,
        d1,
    )


@always_inline
def _ell_contact_hb[
    DTYPE: DType, NT: Int, HN: Int, L_ROW: TensorLayout, L_FR: TensorLayout, L_C: TensorLayout,
    L_HB: TensorLayout,
](
    c: Int,
    jar_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    De_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    fr_e_sh: TileTensor[DTYPE, L_FR, MutAnyOrigin, address_space=AddressSpace.SHARED],
    mu_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    ntc_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    cs_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    hb_sh: TileTensor[DTYPE, L_HB, MutAnyOrigin, address_space=AddressSpace.SHARED],
):
    """The contact's `(NT+1)^2` Hessian block (`ell_hessian_block`, PSD
    projection included) into `hb_sh[c*HN ..]` — one thread per contact,
    read by every thread in the entry loop. A SATISFIED contact writes a
    zero block; the entry loop skips it anyway, as the per-env leg does."""
    comptime RPC = NT + 1
    var row0 = c * RPC
    var nt = Int(rebind[Scalar[DTYPE]](ntc_sh[c]))
    var jar_t = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    var D_t = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    var fr = Scratch[Scalar[DTYPE], NT](NT, fill=Scalar[DTYPE](0))
    _ell_tangent_locals[DTYPE, NT](nt, row0, jar_sh, jar_t)
    _ell_tangent_locals[DTYPE, NT](nt, row0, De_sh, D_t)
    _ell_tangent_locals[DTYPE, NT](nt, row0, fr_e_sh, fr)
    var Hb = Array[Scalar[DTYPE], HN](fill=Scalar[DTYPE](0))
    ell_hessian_block[DTYPE, NT, NT, HN](
        Int(rebind[Scalar[DTYPE]](cs_sh[c])),
        nt,
        0,
        rebind[Scalar[DTYPE]](jar_sh[row0]),
        jar_t,
        rebind[Scalar[DTYPE]](mu_sh[c]),
        rebind[Scalar[DTYPE]](De_sh[row0]),
        D_t,
        fr,
        Hb,
    )
    for k in range(HN):
        hb_sh[c * HN + k] = Hb[k]


@always_inline
def _ell_entry_contact_term[
    DTYPE: DType, NT: Int, HN: Int, L_JE: TensorLayout, L_C: TensorLayout, L_HB: TensorLayout,
    JE_AS: AddressSpace,
](
    c: Int,
    i: Int,
    j: Int,
    nv: Int,
    Je_sh: TileTensor[DTYPE, L_JE, MutAnyOrigin, address_space=JE_AS],
    ntc_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    hb_sh: TileTensor[DTYPE, L_HB, MutAnyOrigin, address_space=AddressSpace.SHARED],
    mut h: Scalar[DTYPE],
):
    """Add contact `c`'s `J^T Hb J` to Hessian entry `(i, j)`, in EXACTLY
    `ell_add_contact_hessian`'s order: for `k` ascending, `JH_k[j] = sum_j'
    Hb[k,j'] * J_j'[j]` (skipping zero `Hb` entries), then `h += J_k[i] *
    JH_k[j]` (skipping a zero `J_k[i]`). Recomputed per entry rather than
    staged per contact: `JH` for every contact is `MC * RPC * NV` scalars,
    too much threadgroup memory, and the recompute is `RPC^2` multiply-adds.
    """
    comptime RPC = NT + 1
    comptime ZERO = Scalar[DTYPE](0)
    var row0 = c * RPC
    var nt = Int(rebind[Scalar[DTYPE]](ntc_sh[c]))
    for k in range(nt + 1):
        var jki = rebind[Scalar[DTYPE]](Je_sh[(row0 + k) * nv + i])
        if jki == ZERO:
            continue
        var jh = ZERO
        for jj in range(nt + 1):
            var hb = rebind[Scalar[DTYPE]](hb_sh[c * HN + k * RPC + jj])
            if hb == ZERO:
                continue
            jh += hb * rebind[Scalar[DTYPE]](Je_sh[(row0 + jj) * nv + j])
        h += jki * jh


@no_inline
def _ell_recompute_coop[
    DTYPE: DType,
    NT: Int,
    D: DimsLike,
    L_JE: TensorLayout,
    L_ROW: TensorLayout,
    L_FR: TensorLayout,
    L_V: TensorLayout,
    L_C: TensorLayout,
    JE_AS: AddressSpace = AddressSpace.SHARED,
](
    tid: Int,
    n_threads: Int,
    nc: Int,
    dense0: Int,
    num_edges: Int,
    dims: D,
    Je_sh: TileTensor[DTYPE, L_JE, MutAnyOrigin, address_space=JE_AS],
    De_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    bias_e_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    kind_e_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    R_e_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    floss_e_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    state_e_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    fr_e_sh: TileTensor[DTYPE, L_FR, MutAnyOrigin, address_space=AddressSpace.SHARED],
    mu_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    ntc_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    cact_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    cs_sh: TileTensor[DTYPE, L_C, MutAnyOrigin, address_space=AddressSpace.SHARED],
    qacc_sh: TileTensor[DTYPE, L_V, MutAnyOrigin, address_space=AddressSpace.SHARED],
    jar_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    force_sh: TileTensor[DTYPE, L_ROW, MutAnyOrigin, address_space=AddressSpace.SHARED],
    qfrc_sh: TileTensor[DTYPE, L_V, MutAnyOrigin, address_space=AddressSpace.SHARED],
):
    """Cooperative `jar` / zone / force / `qfrc` recompute at `qacc_sh` —
    the elliptic twin of `_recompute_jfq_coop`.

    Phase A, one thread per CONTACT: the `1 + nt` row `jar`s (`bias +
    sum_i J*qacc`, ascending), the zone and the forces through
    `ell_state_force`; an inactive contact zeroes its rows and is
    SATISFIED, as in the per-env leg's Step 3. One thread per DENSE row
    (`[dense0, num_edges)`): `scalar_row_state` / `scalar_row_force`,
    verbatim from the pyramidal twin. Phase B, one thread per DOF: `qfrc`
    with the per-env grouping — each non-SATISFIED contact's
    `Jn*fn + sum_t Jt*ft` added as ONE term, then the dense rows in row
    order.
    """
    comptime RPC = NT + 1
    comptime ZERO = Scalar[DTYPE](0)
    var nv = dims.get_nv()

    # ── Phase A: contacts ─────────────────────────────────────────────────
    for c in range(tid, nc, n_threads):
        var row0 = c * RPC
        if rebind[Scalar[DTYPE]](cact_sh[c]) == ZERO:
            for k in range(RPC):
                jar_sh[row0 + k] = ZERO
                force_sh[row0 + k] = ZERO
            cs_sh[c] = Scalar[DTYPE](ELL_SATISFIED)
            continue
        var nt = Int(rebind[Scalar[DTYPE]](ntc_sh[c]))
        # `jar` for the live rows; the dead tangential rows stay 0 — the
        # per-env leg never writes them either.
        for k in range(RPC):
            if k > nt:
                jar_sh[row0 + k] = ZERO
                continue
            var jr = rebind[Scalar[DTYPE]](bias_e_sh[row0 + k])
            for i in range(nv):
                jr += rebind[Scalar[DTYPE]](
                    Je_sh[(row0 + k) * nv + i]
                ) * rebind[Scalar[DTYPE]](qacc_sh[i])
            jar_sh[row0 + k] = jr
        var jar_t = Scratch[Scalar[DTYPE], NT](NT, fill=ZERO)
        var D_t = Scratch[Scalar[DTYPE], NT](NT, fill=ZERO)
        var fr = Scratch[Scalar[DTYPE], NT](NT, fill=ZERO)
        var f_t = Scratch[Scalar[DTYPE], NT](NT, fill=ZERO)
        _ell_tangent_locals[DTYPE, NT](nt, row0, jar_sh, jar_t)
        _ell_tangent_locals[DTYPE, NT](nt, row0, De_sh, D_t)
        _ell_tangent_locals[DTYPE, NT](nt, row0, fr_e_sh, fr)
        var f_n = ZERO
        var cs = ell_state_force[DTYPE, NT, NT](
            nt,
            0,
            rebind[Scalar[DTYPE]](jar_sh[row0]),
            jar_t,
            rebind[Scalar[DTYPE]](mu_sh[c]),
            rebind[Scalar[DTYPE]](De_sh[row0]),
            D_t,
            fr,
            f_n,
            f_t,
        )
        cs_sh[c] = Scalar[DTYPE](cs)
        force_sh[row0] = f_n
        for t in range(NT):
            if t < nt:
                force_sh[row0 + 1 + t] = f_t[t]
            else:
                force_sh[row0 + 1 + t] = ZERO

    # ── Phase A': the dense rows — `_recompute_jfq_coop`'s first phase ──
    for e in range(dense0 + tid, num_edges, n_threads):
        var jr = rebind[Scalar[DTYPE]](bias_e_sh[e])
        for i in range(nv):
            jr += rebind[Scalar[DTYPE]](Je_sh[e * nv + i]) * rebind[
                Scalar[DTYPE]
            ](qacc_sh[i])
        jar_sh[e] = jr
        var st = scalar_row_state[DTYPE](
            Int(rebind[Scalar[DTYPE]](kind_e_sh[e])),
            jr,
            rebind[Scalar[DTYPE]](R_e_sh[e]),
            rebind[Scalar[DTYPE]](floss_e_sh[e]),
        )
        state_e_sh[e] = Scalar[DTYPE](st)
        force_sh[e] = scalar_row_force[DTYPE](
            st,
            jr,
            rebind[Scalar[DTYPE]](De_sh[e]),
            rebind[Scalar[DTYPE]](floss_e_sh[e]),
        )
    barrier()

    # ── Phase B: `qfrc` per dof, grouped per contact as the per-env leg ──
    for i in range(tid, nv, n_threads):
        var q = ZERO
        for c in range(nc):
            if Int(rebind[Scalar[DTYPE]](cs_sh[c])) == ELL_SATISFIED:
                continue
            var row0 = c * RPC
            var nt = Int(rebind[Scalar[DTYPE]](ntc_sh[c]))
            var acc = rebind[Scalar[DTYPE]](
                Je_sh[row0 * nv + i]
            ) * rebind[Scalar[DTYPE]](force_sh[row0])
            for t in range(nt):
                acc += rebind[Scalar[DTYPE]](
                    Je_sh[(row0 + 1 + t) * nv + i]
                ) * rebind[Scalar[DTYPE]](force_sh[row0 + 1 + t])
            q += acc
        for e in range(dense0, num_edges):
            q += rebind[Scalar[DTYPE]](Je_sh[e * nv + i]) * rebind[
                Scalar[DTYPE]
            ](force_sh[e])
        qfrc_sh[i] = q

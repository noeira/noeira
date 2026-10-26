"""`DYN1`/`DYN2` + `rl1`/`rl2` — a tensor view whose EXTENTS are runtime.

TileTensor form (LayoutTensor migration, docs/MOJO_1_2_MAX_26_7_UPGRADE.md §5):
`DYN2` is now the TYPE of `row_major(Coord(rows, cols))` with runtime extents
and `rl2` returns that layout VALUE; `t.lt_dyn["cpu", DYN2](rl)` is
`t.tt_dyn["cpu", DYN2](rl)`. The kernels it feeds are generic over
`L: TensorLayout`, so one body still serves both legs. The text below
predates the migration and still describes the design.

This is phase 3a. `Scratch` (2b.2) solved the function-local buffers; this
solves the other half: the `LayoutTensor` views a dispatcher builds over
`Data`/`Model` tensors. Today they read

    comptime L_NV = Layout.row_major(BATCH, D.NV)
    var qvel_v = d.qvel.lt["cpu", L_NV]()

and `D.NV` is `DIM_POISON` on a dynamic provider, so the layout is not
spellable there at all. The runtime form is

    var rl_nv = rl2(BATCH, dm.get_nv())
    var qvel_v = d.qvel.lt_dyn["cpu", DYN2](rl_nv)

with `dm = d.dims` — the provider VALUE the container now carries.

## Why this is not merely tolerable but FASTER

§15.1 measured it on the Newton solve, static provider both sides, arm B
verified bit-identical to arm A:

| walker2d PYR | hopper PYR | walker2d ELL | hopper ELL |
|---|---|---|---|
| 0.855 | 0.796 | 0.934 | 0.921 |

⇒ **Runtime layouts are ~10-20% FASTER than comptime ones on the shipped
leg.** Unexplained; recorded as measured. It is why 3a is worth doing on its
own, ahead of and independent of any dynamic-leg work — unlike `Scratch`,
where the two legs genuinely want different containers and §15.2 makes
keeping both a RULE, layouts want exactly one kind.

⚠ THAT MEASUREMENT IS THE SOLVE ONLY. FK, collision and the integrators have
different access patterns and are INFERRED, not measured. Each package
carries its own A/B.

## ⚠⚠ THE FAILURE MODE IS SILENT, UNLIKE 2b.1's

A wrong `D.NV` in a comptime layout is a compile error or a poison value. A
wrong extent in a `RuntimeLayout` is a legal layout over the WRONG MEMORY: it
compiles, it runs, and it reads a neighbouring row. That is the cap-as-stride
class from 2b.2, which no gate in the tree could see and which was found by
static audit three times.

So the conversion is a TOKEN-LEVEL TRANSCRIPTION and not a rewrite:

    Layout.row_major(BATCH, D.NBODY * 3)  ->  rl2(BATCH, dm.get_nbody() * 3)

expression tree preserved, one substitution per dimension. Anything that
re-derives an extent by hand is how a wrong one gets in.

## The GPU branches keep their comptime layouts

Decision 3: kernels stay comptime. A kernel signature spells
`LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin]`, and a
runtime provider cannot cross into one anyway (a captured provider reads 0
and the output is silently zeroed). So a dispatcher with both branches keeps
`comptime L_*` for the GPU side and gains `var rl_*` for the CPU side. They
are not redundant; they serve different legs.
"""

from noeira.core.tiles import DYN1, DYN2, rl1, rl2

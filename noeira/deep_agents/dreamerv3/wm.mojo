"""DreamerV3 world-model graphs — the trainer's RSSM surface (PR5c Step 2+).

Wraps the validated `ComputeGraph` pieces (PR5c Step 1 + the chunk-1/2/3
spikes) into the graph types the trainer drives. Two graphs:

  * `WMObserveGraph[...]` — single-step `observe`: core (`ActionSquash` /
    `BlockGroupAssemble` / `GRUGate` + `Sequential` dynin/dynhid/dyngru) →
    `Concat[deter,token]` → obs-MLP → `post` logits. The new deter (`nd`)
    is the carry; read it via `node_out_ptr["nd"]`. Graph OUTPUT = `post`.

  * `WMLossGraph[...]` — the full single-step WM loss (dyn, rep, recon,
    rew, con) with the carry (`nd`, `stoch_new`) appended as identity
    passthrough columns of the output. Output layout `[B, 5+DETER+SC]`:
        cols 0..4         = [dyn, rep, recon, rew, con]
        cols 5..5+DETER   = nd            (carry deter)
        cols 5+DETER..end = stoch_new     (carry stoch, ST-sampled)
    The passthrough columns let the trainer inject the next timestep's
    carry gradient back into `nd`/`stoch_new` during BPTT (the graph then
    accumulates it with the loss-path grads — exactly the manual oracle's
    `grad_new_deter` threading, now framework-routed). Linear/BlockLinear
    vjp accumulate param grads, so a `zero_grad` once + T backward passes
    accumulates the BPTT gradient.

The activation is `GELU` here to keep parity with the PR4/5b fixtures;
the size1m/dmc config swaps it to SiLU (PR5c Step 5).

Param load + checkpoint use the `ParamVisitor` name-walk
(`for_each_param`); the node-index access in the spikes is test-only.
"""

from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import InputSlot, Node
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.block_linear import BlockLinear
from noeira.nn.primitives.rms_norm import RMSNorm
from noeira.nn.primitives.elementwise import Elementwise
from noeira.nn.primitives.ops.gelu_tanh_op import GELUTanhOp
from noeira.nn.primitives.concat import Concat
from noeira.nn.core.element_op import ElementOp
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer, Zero
from .rssm_ops import (
    ActionSquash, BlockGroupAssemble, GRUGate, StraightThroughSample,
)
from .onehot_kl import OneHotKLLoss
from .wm_loss_ops import SymlogMSELoss, TwoHotLoss, BinaryLoss
from .nets import DreamerDecoder, DreamerRewardMLP, DreamerContMLP


# ──────────────────────────────────────────────────────────────────────
# Head-loss mini-graphs — each takes the carry (nd, stoch_new) as input
# slots + a target, runs a standalone head Module + its loss op, and
# routes the loss gradient back to nd/stoch_new (read via grad_input_ptr).
# DreamerOpt-walkable (own params); reused by both the WM-loss BPTT and
# (decoder excepted) the imagination AC. Validated as the Step-1 mini-graphs.
# ──────────────────────────────────────────────────────────────────────


# `DEC` is the decoder Module TYPE (default = MLP `DreamerDecoder`). The pixel
# path passes `DreamerDecoderCNN[SC+DETER, C, H, W, BASE, A]` (same IN=SC+DETER,
# OUT=OBS=C*H*W) — a deconv reconstruction instead of an MLP, no other change.
comptime DecLossGraph[
    SC: Int, DETER: Int, OBS: Int, DEC_U: Int, A: ElementOp = GELUTanhOp,
    DEC: Module = DreamerDecoder[SC + DETER, OBS, DEC_U, A],
    SIGMOID: Bool = False,  # True → sigmoid+MSE recon (bounded [0,1] pixels)
] = ComputeGraph[
    InputSlot["stoch_new", SC],
    InputSlot["nd", DETER],
    InputSlot["rtgt", OBS],
    Node["decin", Concat[SC, DETER],                   "stoch_new", "nd"],
    Node["dec",   DEC,                                 "decin"],
    Node["recon", SymlogMSELoss[OBS, SIGMOID],         "dec", "rtgt"],
]


comptime RewLossGraph[
    DETER: Int, SC: Int, HU: Int, BINS: Int, A: ElementOp = GELUTanhOp,
    OUT_INIT: Initializer = Zero,  # reward-head output-layer init (nets.mojo)
] = ComputeGraph[
    InputSlot["nd", DETER],
    InputSlot["stoch_new", SC],
    InputSlot["rtgt", 1],
    Node["feat", Concat[DETER, SC],                    "nd", "stoch_new"],
    Node["rew",  DreamerRewardMLP[DETER + SC, HU, BINS, A, OUT_INIT], "feat"],
    Node["rewl", TwoHotLoss[BINS],                     "rew", "rtgt"],
]


comptime ConLossGraph[DETER: Int, SC: Int, HU: Int, A: ElementOp = GELUTanhOp] = ComputeGraph[
    InputSlot["nd", DETER],
    InputSlot["stoch_new", SC],
    InputSlot["ctgt", 1],
    Node["feat", Concat[DETER, SC],                    "nd", "stoch_new"],
    Node["con",  DreamerContMLP[DETER + SC, HU, A],       "feat"],
    Node["conl", BinaryLoss,                           "con", "ctgt"],
]


# ──────────────────────────────────────────────────────────────────────
# WMImagineGraph — single imagination step: core → prior → ST sample →
# feat. No obs/token path (the latent rolls forward on the prior). Output
# = feat = concat([nd, stoch_new]) for the policy/value heads; the carry
# (nd, stoch_new) is read via node_out_ptr.
# ──────────────────────────────────────────────────────────────────────


comptime WMImagineGraph[
    DETER: Int, H: Int, STOCH: Int, CLASSES: Int, BLOCKS: Int, ACT: Int,
    A: ElementOp = GELUTanhOp,
] = ComputeGraph[
    InputSlot["deter", DETER],
    InputSlot["stoch", STOCH * CLASSES],
    InputSlot["action", ACT],
    Node["a",    ActionSquash[ACT],                                  "action"],
    Node["x0",   Sequential[Linear[DETER, H], RMSNorm[H], Elementwise[H, A]],   "deter"],
    Node["x1",   Sequential[Linear[STOCH * CLASSES, H], RMSNorm[H], Elementwise[H, A]], "stoch"],
    Node["x2",   Sequential[Linear[ACT, H], RMSNorm[H], Elementwise[H, A]],     "a"],
    Node["dhin", BlockGroupAssemble[DETER, H, BLOCKS], "deter", "x0", "x1", "x2"],
    Node["h",    Sequential[BlockLinear[DETER + 3 * H * BLOCKS, DETER, BLOCKS], RMSNorm[DETER], Elementwise[DETER, A]], "dhin"],
    Node["gru",  BlockLinear[DETER, 3 * DETER, BLOCKS],              "h"],
    Node["nd",   GRUGate[DETER, BLOCKS],                             "gru", "deter"],
    Node["pr0",   Sequential[Linear[DETER, H], RMSNorm[H], Elementwise[H, A]], "nd"],
    Node["pr1",   Sequential[Linear[H, H], RMSNorm[H], Elementwise[H, A]],     "pr0"],
    Node["prior", Linear[H, STOCH * CLASSES],                        "pr1"],
    Node["stoch_new", StraightThroughSample[STOCH, CLASSES],         "prior"],
    Node["feat",  Concat[DETER, STOCH * CLASSES],                    "nd", "stoch_new"],
]


# ──────────────────────────────────────────────────────────────────────
# WMObserveGraph — single-step observe → post logits (carry = node "nd").
# ──────────────────────────────────────────────────────────────────────


comptime WMObserveGraph[
    DETER: Int, H: Int, STOCH: Int, CLASSES: Int, BLOCKS: Int,
    ACT: Int, TOKEN: Int, A: ElementOp = GELUTanhOp,
] = ComputeGraph[
    InputSlot["deter", DETER],
    InputSlot["stoch", STOCH * CLASSES],
    InputSlot["action", ACT],
    InputSlot["tokens", TOKEN],
    Node["a",    ActionSquash[ACT],                                  "action"],
    Node["x0",   Sequential[Linear[DETER, H], RMSNorm[H], Elementwise[H, A]],   "deter"],
    Node["x1",   Sequential[Linear[STOCH * CLASSES, H], RMSNorm[H], Elementwise[H, A]], "stoch"],
    Node["x2",   Sequential[Linear[ACT, H], RMSNorm[H], Elementwise[H, A]],     "a"],
    Node["dhin", BlockGroupAssemble[DETER, H, BLOCKS], "deter", "x0", "x1", "x2"],
    Node["h",    Sequential[BlockLinear[DETER + 3 * H * BLOCKS, DETER, BLOCKS], RMSNorm[DETER], Elementwise[DETER, A]], "dhin"],
    Node["gru",  BlockLinear[DETER, 3 * DETER, BLOCKS],              "h"],
    Node["nd",   GRUGate[DETER, BLOCKS],                             "gru", "deter"],
    Node["obsin",  Concat[DETER, TOKEN],                             "nd", "tokens"],
    Node["obshid", Sequential[Linear[DETER + TOKEN, H], RMSNorm[H], Elementwise[H, A]], "obsin"],
    Node["post",   Linear[H, STOCH * CLASSES],                       "obshid"],
]


# ──────────────────────────────────────────────────────────────────────
# WMCoreGraph — RSSM dyn/rep loss + carry passthrough (no decoder/heads).
#   output [B, 2 + DETER + STOCH*CLASSES]:
#     cols 0,1          = [dyn, rep]
#     cols 2..2+DETER   = nd          (carry deter)
#     cols 2+DETER..end = stoch_new   (carry stoch, ST-sampled from POST)
# The trainer runs the standalone decoder/reward/cont/value/policy Modules
# on feat=concat(nd, stoch_new); their grads to feat sum back into the
# carry passthrough columns (+ the next-timestep BPTT grad) when seeding
# this graph's vjp. Keeps the heads as reusable, DreamerOpt-walkable
# Modules (shared by WM-loss + imagination).
# ──────────────────────────────────────────────────────────────────────


comptime WMCoreGraph[
    DETER: Int, H: Int, STOCH: Int, CLASSES: Int, BLOCKS: Int,
    ACT: Int, TOKEN: Int, A: ElementOp = GELUTanhOp,
] = ComputeGraph[
    InputSlot["deter", DETER],
    InputSlot["stoch", STOCH * CLASSES],
    InputSlot["action", ACT],
    InputSlot["tokens", TOKEN],
    Node["a",    ActionSquash[ACT],                                  "action"],
    Node["x0",   Sequential[Linear[DETER, H], RMSNorm[H], Elementwise[H, A]],   "deter"],
    Node["x1",   Sequential[Linear[STOCH * CLASSES, H], RMSNorm[H], Elementwise[H, A]], "stoch"],
    Node["x2",   Sequential[Linear[ACT, H], RMSNorm[H], Elementwise[H, A]],     "a"],
    Node["dhin", BlockGroupAssemble[DETER, H, BLOCKS], "deter", "x0", "x1", "x2"],
    Node["h",    Sequential[BlockLinear[DETER + 3 * H * BLOCKS, DETER, BLOCKS], RMSNorm[DETER], Elementwise[DETER, A]], "dhin"],
    Node["gru",  BlockLinear[DETER, 3 * DETER, BLOCKS],              "h"],
    Node["nd",   GRUGate[DETER, BLOCKS],                             "gru", "deter"],
    Node["obsin",  Concat[DETER, TOKEN],                             "nd", "tokens"],
    Node["obshid", Sequential[Linear[DETER + TOKEN, H], RMSNorm[H], Elementwise[H, A]], "obsin"],
    Node["post",   Linear[H, STOCH * CLASSES],                       "obshid"],
    Node["pr0",   Sequential[Linear[DETER, H], RMSNorm[H], Elementwise[H, A]], "nd"],
    Node["pr1",   Sequential[Linear[H, H], RMSNorm[H], Elementwise[H, A]],     "pr0"],
    Node["prior", Linear[H, STOCH * CLASSES],                        "pr1"],
    Node["kl",    OneHotKLLoss[STOCH, CLASSES],                      "post", "prior"],
    Node["stoch_new", StraightThroughSample[STOCH, CLASSES],         "post"],
    Node["out",   Concat[2, DETER, STOCH * CLASSES],   "kl", "nd", "stoch_new"],
]


# ──────────────────────────────────────────────────────────────────────
# WMLossGraph — full single-step WM loss + carry passthrough.
#   output [B, 5 + DETER + STOCH*CLASSES]
# ──────────────────────────────────────────────────────────────────────


comptime WMLossGraph[
    DETER: Int, H: Int, STOCH: Int, CLASSES: Int, BLOCKS: Int,
    ACT: Int, TOKEN: Int, OBS: Int, DEC_U: Int, HU: Int, BINS: Int,
    A: ElementOp = GELUTanhOp,
] = ComputeGraph[
    InputSlot["deter", DETER],
    InputSlot["stoch", STOCH * CLASSES],
    InputSlot["action", ACT],
    InputSlot["tokens", TOKEN],
    InputSlot["recon_target", OBS],
    InputSlot["rew_target", 1],
    InputSlot["con_target", 1],
    Node["a",    ActionSquash[ACT],                                  "action"],
    Node["x0",   Sequential[Linear[DETER, H], RMSNorm[H], Elementwise[H, A]],   "deter"],
    Node["x1",   Sequential[Linear[STOCH * CLASSES, H], RMSNorm[H], Elementwise[H, A]], "stoch"],
    Node["x2",   Sequential[Linear[ACT, H], RMSNorm[H], Elementwise[H, A]],     "a"],
    Node["dhin", BlockGroupAssemble[DETER, H, BLOCKS], "deter", "x0", "x1", "x2"],
    Node["h",    Sequential[BlockLinear[DETER + 3 * H * BLOCKS, DETER, BLOCKS], RMSNorm[DETER], Elementwise[DETER, A]], "dhin"],
    Node["gru",  BlockLinear[DETER, 3 * DETER, BLOCKS],              "h"],
    Node["nd",   GRUGate[DETER, BLOCKS],                             "gru", "deter"],
    Node["obsin",  Concat[DETER, TOKEN],                             "nd", "tokens"],
    Node["obshid", Sequential[Linear[DETER + TOKEN, H], RMSNorm[H], Elementwise[H, A]], "obsin"],
    Node["post",   Linear[H, STOCH * CLASSES],                       "obshid"],
    Node["pr0",   Sequential[Linear[DETER, H], RMSNorm[H], Elementwise[H, A]], "nd"],
    Node["pr1",   Sequential[Linear[H, H], RMSNorm[H], Elementwise[H, A]],     "pr0"],
    Node["prior", Linear[H, STOCH * CLASSES],                        "pr1"],
    Node["kl",    OneHotKLLoss[STOCH, CLASSES],                      "post", "prior"],
    Node["stoch_new", StraightThroughSample[STOCH, CLASSES],         "post"],
    Node["decin", Concat[STOCH * CLASSES, DETER],                    "stoch_new", "nd"],
    Node["dec",   DreamerDecoder[STOCH * CLASSES + DETER, OBS, DEC_U, A], "decin"],
    Node["recon", SymlogMSELoss[OBS],                                "dec", "recon_target"],
    Node["feat",  Concat[DETER, STOCH * CLASSES],                    "nd", "stoch_new"],
    Node["rew",   DreamerRewardMLP[DETER + STOCH * CLASSES, HU, BINS, A], "feat"],
    Node["rewl",  TwoHotLoss[BINS],                                  "rew", "rew_target"],
    Node["con",   DreamerContMLP[DETER + STOCH * CLASSES, HU, A],       "feat"],
    Node["conl",  BinaryLoss,                                        "con", "con_target"],
    # losses → [B,5]; then append carry passthrough (nd, stoch_new). Graph
    # nodes cap at ARITY=4, so the assemble is two-level (4-way + 3-way).
    Node["lossvec", Concat[2, 1, 1, 1],   "kl", "recon", "rewl", "conl"],
    Node["out",     Concat[5, DETER, STOCH * CLASSES],
         "lossvec", "nd", "stoch_new"],
]

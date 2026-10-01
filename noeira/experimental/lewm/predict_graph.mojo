"""LeWMPredictGraph — predictor-from-latents (the MPC inference path), storage.

The loss graph couples encoder→predictor (its context always comes from
encoding pixels). Autoregressive MPC needs the predictor to run on ARBITRARY
latent context (the rolling buffer of predicted latents), so this is a
separate graph:

  latent_ctx (B, H·EMB) ─BiasAdd(pos)→ x_pe ─┐
  actions (B, T·ACT) ─ActionEmbedder→ ─Slice[0:H]→ ctx_a ─┴ARPredictor→PredProj→ pred

Its param-bearing nodes — `act_emb`, `x_pe`, `pred_raw`, `pred` — are the
SAME module types, with the SAME `for_each_param` names, as the matching
nodes in `LeWMLossGraph` (a suffix of its param set, minus the encoder
`emb`). So `LeWMPredictor.sync_from_named` copies the trained weights by
NAME from the trainer's exported dict — exact, order-independent.

Storage surface: the graph owns its activation pool (no Scratch / TargetStorage).
The public `forward` keeps its raw-`TileTensor` facade (the MPC kernels feed it
host/device pointers); inside, inputs are bridged into the graph's storage
`set_input` and the `pred` output is copied back to the caller's tile. The only
residual unsafe is the device-pointer ↔ `DeviceBuffer` bridge at this raw
boundary (GPU ABI interop), not the framework.
"""

from std.collections import Dict
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.nn import (
    Tensor,
    ParamVisitor,
    Kaiming,
    ComputeGraph,
    InputSlot,
    Node,
    Slice,
    BiasAdd,
)
from .encoder import ActionEmbedder, ARPredictor, PredProj


comptime LeWMPredictGraph[
    EMB: Int,
    T: Int,
    ACT: Int,
    SMOOTHED: Int,
    AE_MLP: Int,
    H: Int,
    PRED_HEADS: Int,
    PRED_FF: Int,
    DEPTH: Int,
    PRED_PROJ_H: Int,
    PRED_DIM_HEAD: Int = 0,
] = ComputeGraph[
    InputSlot["latent_ctx", H * EMB],
    InputSlot["actions", T * ACT],
    Node["act_emb", ActionEmbedder[T, ACT, SMOOTHED, EMB, AE_MLP], "actions"],
    Node["ctx_a", Slice[T * EMB, 0, H * EMB], "act_emb"],
    Node["x_pe", BiasAdd[H * EMB], "latent_ctx"],
    Node[
        "pred_raw",
        ARPredictor[EMB, PRED_HEADS, H, PRED_FF, DEPTH, PRED_DIM_HEAD],
        "x_pe",
        "ctx_a",
    ],
    Node["pred", PredProj[H, EMB, PRED_PROJ_H], "pred_raw"],
]


# Import visitor: copy each param/state from a name→values dict (CPU direct /
# GPU host-fill + upload). Missing names are left at their init (shouldn't
# happen if the predict-graph node names match the trainer's).
struct _NamedImportVisitor(ParamVisitor):
    var d: Dict[String, List[Scalar[DT]]]
    var missing: Int

    def __init__(out self, var d: Dict[String, List[Scalar[DT]]]):
        self.d = d^
        self.missing = 0

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if name not in self.d:
            self.missing += 1
            return
        ref vals = self.d[name]
        param.ensure(N)
        for i in range(N):
            param.data[i] = vals[i]
        param.n = N
        # Advance `version` so `Linear`'s version-gated `w_pad` / `w_bf` pick
        # up the new weight (see `_LoadVisitor` in trainer.mojo).
        param.version += 1
        comptime if target == "gpu":
            param.upload_resident(ctx.value())


struct LeWMPredictor[
    EMB: Int,
    T: Int,
    ACT: Int,
    SMOOTHED: Int,
    AE_MLP: Int,
    H: Int,
    PRED_HEADS: Int,
    PRED_FF: Int,
    DEPTH: Int,
    PRED_PROJ_H: Int,
    BATCH: Int,
    target: StaticString = "cpu",
    PRED_DIM_HEAD: Int = 0,
](Movable & Deinitable):
    # PRED_DIM_HEAD added last (after target) so existing positional call
    # sites (Pong, default 0 ⇒ EMB/PRED_HEADS) are unchanged; >0 selects the
    # paper's expanded predictor attention to match a paper-width WM.
    comptime PG = LeWMPredictGraph[
        Self.EMB,
        Self.T,
        Self.ACT,
        Self.SMOOTHED,
        Self.AE_MLP,
        Self.H,
        Self.PRED_HEADS,
        Self.PRED_FF,
        Self.DEPTH,
        Self.PRED_PROJ_H,
        Self.PRED_DIM_HEAD,
    ]
    comptime HE = Self.H * Self.EMB
    comptime ACTIN = Self.T * Self.ACT

    var graph: Self.PG
    var ctx: Optional[DeviceContext]
    # Owned graph-output buffer (graph.forward writes here; copied to caller).
    var out_buf: Tensor

    def __init__(out self):
        self.graph = Self.PG()
        self.ctx = None
        self.out_buf = Tensor()

    @staticmethod
    def make(ctx: Optional[DeviceContext] = None) raises -> Self:
        var p = Self()
        p.graph = Self.PG.make[Self.target, Kaiming](ctx=ctx)
        p.ctx = ctx
        comptime if Self.target == "gpu":
            p.out_buf = Tensor.alloc_gpu(ctx.value(), Self.BATCH * Self.HE)
        else:
            p.out_buf = Tensor.alloc(Self.BATCH * Self.HE)
        return p^

    def sync_from_named(
        mut self,
        var d: Dict[String, List[Scalar[DT]]],
    ) raises:
        """Overwrite the predictor's params AND state (BatchNorm running
        stats) with the trainer's snapshot, matched by `for_each_param` /
        `for_each_state` name. State sync makes eval-mode BN at planning
        normalize identically to the trainer's warmed running stats."""
        var v = _NamedImportVisitor(d^)
        self.graph.for_each_param[Self.target, _NamedImportVisitor](v, self.ctx)
        self.graph.for_each_state[Self.target, _NamedImportVisitor](v, self.ctx)

    def set_bn_training(mut self, training: Bool) raises:
        """Flip PredProj's BatchNorm (node "pred") between training and
        eval mode. Planning runs eval (running stats, synced from the
        trainer) — training-mode BN would normalize over the CEM candidate
        batch, coupling candidate scores."""
        var v = Scalar[DT](1.0) if training else Scalar[DT](0.0)
        self.graph.set_node_attr["pred", "training"](v)

    def _seed_input[
        slot_name: StaticString, N: Int
    ](
        mut self,
        src: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises:
        """Bridge a raw input tile into the named graph input slot (storage
        `set_input` copies it into the pool). CPU: copy into a host `List`;
        GPU: wrap the device pointer in a non-owning `DeviceBuffer`."""
        var t = Tensor()
        comptime if Self.target == "cpu":
            t.data = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
            for i in range(N):
                t.data[i] = rebind[Scalar[DT]](src.ptr[unsafe_offset=i])
            t.n = N
        else:
            var c = self.ctx.value()
            var sp = rebind[Pointer[Scalar[DT], MutAnyOrigin]](src.ptr)
            t.dev = DeviceBuffer[DT](c, sp, N, owning=False)
            t.n = N
        self.graph.set_input[slot_name, Self.BATCH](t, self.ctx)

    def forward(
        mut self,
        latent_ctx: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
        actions: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
        mut pred_out: TileTensor[
            mut=True,
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            ...,
        ],
    ) raises:
        """Run the predictor: (latent_ctx, actions) → pred (B, H·EMB),
        written to `pred_out`."""
        self._seed_input["latent_ctx", Self.BATCH * Self.HE](latent_ctx)
        self._seed_input["actions", Self.BATCH * Self.ACTIN](actions)
        self.graph.forward[Self.BATCH, Self.target](self.out_buf, self.ctx)
        comptime N = Self.BATCH * Self.HE
        comptime if Self.target == "cpu":
            for i in range(N):
                pred_out.ptr[unsafe_offset=i] = self.out_buf.data[i]
        else:
            var c = self.ctx.value()
            var dp = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                pred_out.ptr
            )
            var db = DeviceBuffer[DT](c, dp, N, owning=False)
            c.enqueue_copy(db, self.out_buf.dev.value())

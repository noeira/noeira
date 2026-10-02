"""G2a — the reference-exact LeWM forward against torch, stage by stage.

docs/LEWM_REOPEN_PLAN.md P2. The published checkpoint (quentinll/lewm-pusht)
runs through `ref_model.mojo`'s modules and through torch (float64,
`tools/lewm/dump_lewm_reference.py`). Each STAGE is fed the reference's own
input for that stage and compared with the reference's output, so an error
shows where it is made instead of compounding; the whole encoder also runs
end to end. Weights load by the loss graph's walk names (`ref_load.load_ref`
with the node prefix), so a stage that loads is also a check that the graph
names what the converter mapped.

Unit: max |ours - torch| / std(torch). torch's OWN float32 error on these
stages is 1e-6..1e-5 of that (`--noise`); the gate allows TOL.

Needs the dump:
    pixi run -e act-ref python tools/lewm/dump_lewm_reference.py --out /tmp/lewm_ref
    pixi run mojo run -I . tools/lewm/list_ref_params.mojo > /tmp/lewm_ref/ours_names.tsv
    pixi run -e act-ref python tools/lewm/convert_ref_to_ours.py --dump /tmp/lewm_ref
Run:
    pixi run -e apple mojo run -I . tests/experimental/lewm/ref/test_ref_forward.mojo
"""

from std.sys import argv
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn import Sequential, Tokenwise, LayerNorm, BiasAdd, LearnedTokens
from noeira.nn.models.vit import PatchEmbed
from noeira.experimental.lewm.ref_model import (
    ViTBlockHF,
    ProjectorRef,
    LeWMEncoderRef,
    ActionEmbedderRef,
    ConditionalTransformerBlockRef,
    HF_VIT_LN_EPS,
    TORCH_LN_EPS,
)
from noeira.experimental.lewm.ref_load import load_ref, ref_input, std_err


comptime TOL = 1e-4

# published config (config.json): ViT-tiny/14 at 224, EMB 192, predictor 6 x 16 x 64
comptime IMG = 224
comptime PATCH = 14
comptime NP = (IMG // PATCH) * (IMG // PATCH)
comptime SEQ = NP + 1
comptime HID = 192
comptime EMB = 192
comptime PROJ_H = 2048
comptime N_ENC = 8   # encoder section frames
comptime B = 4       # action / predictor sections
comptime T_ACT = 4
comptime H = 3


struct Gate(Movable):
    var worst: Float64
    var failed: List[String]

    def __init__(out self):
        self.worst = 0.0
        self.failed = List[String]()

    def check(mut self, label: String, err: Float64):
        var flag = String("  ") if err <= TOL else String("✗ ")
        print("   ", flag, label, " ", err, sep="")
        self.worst = max(self.worst, err)
        if err > TOL:
            self.failed.append(label)


def _encoder[target: StaticString](
    dump: String, ctx: Optional[DeviceContext], mut g: Gate
) raises:
    comptime Embed = Sequential[
        PatchEmbed[3, IMG, IMG, PATCH, HID, NP],
        LearnedTokens[NP, 1, HID, True, 0.02],
        BiasAdd[SEQ * HID],
    ]
    var embed = Embed.make[target, Kaiming](ctx)
    _ = load_ref[target](embed, dump, String("emb.0."), ctx)
    var pix = ref_input[target](dump, String("enc.pixels"), ctx)
    var out = Tensor.alloc(N_ENC * SEQ * HID)
    embed.forward[target, N_ENC](TensorRefs[1](pix), out, ctx)
    g.check(String("enc  patch+cls+pos"), std_err[target](dump, String("enc.vit_embeddings"), out, ctx))

    var blk = ViTBlockHF[HID, 3, SEQ, 4 * HID].make[target, Kaiming](ctx)
    for i in range(12):
        _ = load_ref[target](blk, dump, String("emb.0.3.") + String(i) + ".", ctx)
        var src = String("enc.vit_embeddings") if i == 0 else String("enc.vit_layer") + String(i - 1)
        var x = ref_input[target](dump, src, ctx)
        var y = Tensor.alloc(N_ENC * SEQ * HID)
        blk.forward[target, N_ENC](TensorRefs[1](x), y, ctx)
        g.check(String("enc  vit block ") + String(i), std_err[target](dump, String("enc.vit_layer") + String(i), y, ctx))

    var ln = Tokenwise[SEQ, LayerNorm[HID, DT, HF_VIT_LN_EPS]].make[target, Kaiming](ctx)
    _ = load_ref[target](ln, dump, String("emb.0.4."), ctx)
    var x11 = ref_input[target](dump, String("enc.vit_layer11"), ctx)
    var lno = Tensor.alloc(N_ENC * SEQ * HID)
    ln.forward[target, N_ENC](TensorRefs[1](x11), lno, ctx)
    g.check(String("enc  final LN"), std_err[target](dump, String("enc.vit_final_ln"), lno, ctx))

    var proj = ProjectorRef[HID, PROJ_H, EMB].make[target, Kaiming](ctx)
    _ = load_ref[target](proj, dump, String("emb.0.6."), ctx)
    proj.set_attr["training"](Scalar[DT](0.0))
    var cls = ref_input[target](dump, String("enc.cls"), ctx)
    var emb = Tensor.alloc(N_ENC * EMB)
    proj.forward[target, N_ENC](TensorRefs[1](cls), emb, ctx)
    g.check(String("enc  projector (BN eval)"), std_err[target](dump, String("enc.emb"), emb, ctx))

    var enc = LeWMEncoderRef[3, IMG, PATCH, HID, 3, 12, EMB, PROJ_H].make[target, Kaiming](ctx)
    _ = load_ref[target](enc, dump, String("emb.0."), ctx)
    enc.set_attr["training"](Scalar[DT](0.0))
    var e2e = Tensor.alloc(N_ENC * EMB)
    enc.forward[target, N_ENC](TensorRefs[1](pix), e2e, ctx)
    g.check(String("enc  END TO END pixels -> emb"), std_err[target](dump, String("enc.emb"), e2e, ctx))


def _action[target: StaticString](
    dump: String, ctx: Optional[DeviceContext], mut g: Gate
) raises:
    var ae = ActionEmbedderRef[T_ACT, 10, EMB].make[target, Kaiming](ctx)
    _ = load_ref[target](ae, dump, String("act_emb."), ctx)
    var a = ref_input[target](dump, String("act.in"), ctx)
    var out = Tensor.alloc(B * T_ACT * EMB)
    ae.forward[target, B](TensorRefs[1](a), out, ctx)
    g.check(String("act  embedder"), std_err[target](dump, String("act.act_emb"), out, ctx))


def _predictor[target: StaticString](
    dump: String, ctx: Optional[DeviceContext], mut g: Gate
) raises:
    var pe = BiasAdd[H * EMB].make[target, Kaiming](ctx)
    _ = load_ref[target](pe, dump, String("x_pe."), ctx)
    var emb = ref_input[target](dump, String("pred.emb"), ctx)
    var xp = Tensor.alloc(B * H * EMB)
    pe.forward[target, B](TensorRefs[1](emb), xp, ctx)
    g.check(String("pred position embedding"), std_err[target](dump, String("pred.x_pos"), xp, ctx))

    var blk = ConditionalTransformerBlockRef[EMB, 16, H, 2048, 64].make[target, Kaiming](ctx)
    for i in range(6):
        _ = load_ref[target](blk, dump, String("pred_raw.") + String(i) + ".", ctx)
        var ins = TensorPack[2]()
        var src = String("pred.x_pos") if i == 0 else String("pred.blk") + String(i - 1)
        var xv = ref_input[target](dump, src, ctx)
        var cv = ref_input[target](dump, String("pred.act_emb"), ctx)
        ins[0].ensure(B * H * EMB)
        ins[1].ensure(B * H * EMB)
        for q in range(B * H * EMB):
            ins[0].data[q] = xv.data[q]
            ins[1].data[q] = cv.data[q]
        comptime if target == "gpu":
            ins[0].upload(ctx.value())
            ins[1].upload(ctx.value())
        var y = Tensor.alloc(B * H * EMB)
        blk.forward[target, B](TensorRefs[2](ins[0], ins[1]), y, ctx)
        g.check(String("pred AdaLN block ") + String(i), std_err[target](dump, String("pred.blk") + String(i), y, ctx))

    var ln = Tokenwise[H, LayerNorm[EMB, DT, TORCH_LN_EPS]].make[target, Kaiming](ctx)
    _ = load_ref[target](ln, dump, String("pred_ln."), ctx)
    var b5 = ref_input[target](dump, String("pred.blk5"), ctx)
    var lno = Tensor.alloc(B * H * EMB)
    ln.forward[target, B](TensorRefs[1](b5), lno, ctx)
    g.check(String("pred final LN"), std_err[target](dump, String("pred.final_ln"), lno, ctx))

    var pp = Tokenwise[H, ProjectorRef[EMB, PROJ_H, EMB]].make[target, Kaiming](ctx)
    _ = load_ref[target](pp, dump, String("pred."), ctx)
    pp.set_attr["training"](Scalar[DT](0.0))
    var fl = ref_input[target](dump, String("pred.final_ln"), ctx)
    var pred = Tensor.alloc(B * H * EMB)
    pp.forward[target, B](TensorRefs[1](fl), pred, ctx)
    g.check(String("pred pred_proj (BN eval)"), std_err[target](dump, String("pred.pred"), pred, ctx))


def main() raises:
    var dump = String("/tmp/lewm_ref")
    var args = argv()
    if len(args) > 1:
        dump = String(args[1])
    print("G2a  LeWM reference forward, stage by stage (max |ours-torch| / std(torch); TOL", TOL, ")")
    var g = Gate()
    print("  -- cpu")
    _encoder["cpu"](dump, None, g)
    _action["cpu"](dump, None, g)
    _predictor["cpu"](dump, None, g)
    var c = DeviceContext()
    print("  -- gpu")
    _encoder["gpu"](dump, Optional(c), g)
    _action["gpu"](dump, Optional(c), g)
    _predictor["gpu"](dump, Optional(c), g)
    print("  worst", g.worst)
    if len(g.failed) > 0:
        raise Error("FAIL G2a: " + String(len(g.failed)) + " stage(s) above TOL")
    print("PASS")

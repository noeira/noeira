"""PEFT's LoRA files: `adapter_model.safetensors` and `adapter_config.json`.

    var w = PeftAdapterWriter(r=16, lora_alpha=32.0)
    w.add["gpu"](String("model.layers.0.self_attn.q_proj"), q_adapter, ctx)
    ...
    w.save(String("out/adapter"))           # PeftAdapterWriter(..., bf16=True)

    var cfg = read_peft_config(String("out/adapter"))
    var st = SafeTensors(String("out/adapter/adapter_model.safetensors"))
    load_peft_lora["gpu"](st, cfg, String("model.layers.0.self_attn.q_proj"),
                          q_adapter, ctx)

## The format

Per adapted linear at the torch module path P, the safetensors file holds
`base_model.model.P.lora_A.weight` [R, IN] and `base_model.model.P.lora_B.weight`
[OUT, R]: torch's [out, in], the transposes of noeira's A [IN, R] and B [R, OUT].
The config carries `peft_type = "LORA"`, `r`, `lora_alpha`, `use_rslora` and
`target_modules`. Module paths are the TORCH model's: a caller maps each of its
layers to the path the torch model gives it (SmolVLA: `names.mojo`).

`target_modules` is written as the full paths, not leaf names. PEFT matches a
target against a module name by equality or by a ".target" suffix, so a leaf
name such as `q_proj` would also adapt every other `q_proj` of the torch model
(the vision tower's, in SmolVLA), which this adapter has no
weights for.

## What the reader refuses

Every config field that changes what the numbers mean: DoRA, `fan_in_fan_out`
(weights stored [in, out]), a bias (`bias` other than "none", `lora_bias`),
per-module ranks or alphas (`rank_pattern`, `alpha_pattern`),
`modules_to_save`, `layer_replication`, `target_parameters`, QALoRA. Loading
past one of them gives an adapter that runs and computes something else.
"""

from std.os import makedirs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.io.safetensors import SafeTensors, SafeTensorsWriter
from noeira.io.json import (
    J_NULL, J_BOOL, J_STRING, J_ARRAY, J_OBJECT, JsonDoc, JsonWriter, load_json,
)
from noeira.io.fileio import write_text_atomic
from .lora import LoRAAdapter


comptime PEFT_PREFIX = "base_model.model."


def peft_key(path: String, which: String) -> String:
    """`base_model.model.<path>.<which>.weight`, `which` = lora_A | lora_B."""
    return String(PEFT_PREFIX) + path + "." + which + ".weight"


struct PeftLoRAConfig(Movable):
    var r: Int
    var lora_alpha: Float64
    var use_rslora: Bool
    var target_modules: List[String]
    var base_model_name_or_path: String

    def __init__(
        out self,
        r: Int,
        lora_alpha: Float64,
        use_rslora: Bool = False,
        base_model_name_or_path: String = String(""),
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora
        self.target_modules = List[String]()
        self.base_model_name_or_path = base_model_name_or_path


# ── reading ────────────────────────────────────────────────────────────


def _neutral(ref doc: JsonDoc, node: Int) raises -> Bool:
    """Absent, null, false, "none", {} or []."""
    if node < 0:
        return True
    var k = doc.kind_of(node)
    if k == J_NULL:
        return True
    if k == J_BOOL:
        return not doc.boolean(node)
    if k == J_STRING:
        return doc.string(node) == "none"
    if k == J_ARRAY or k == J_OBJECT:
        return doc.size(node) == 0
    return False


def read_peft_config(dir: String) raises -> PeftLoRAConfig:
    """`dir/adapter_config.json`, refusing what this reader cannot honour
    (module docstring)."""
    var path = dir + "/adapter_config.json"
    var doc = load_json(path)
    var root = doc.root()
    var pt = doc.field(root, "peft_type")
    if pt < 0 or doc.string(pt) != "LORA":
        raise Error(path + ": peft_type is not \"LORA\"")
    var refused = List[String]()
    refused.append("use_dora")
    refused.append("fan_in_fan_out")
    refused.append("bias")
    refused.append("lora_bias")
    refused.append("rank_pattern")
    refused.append("alpha_pattern")
    refused.append("modules_to_save")
    refused.append("layer_replication")
    refused.append("target_parameters")
    refused.append("use_qalora")
    for i in range(len(refused)):
        if not _neutral(doc, doc.field(root, refused[i])):
            raise Error(
                path + ": '" + refused[i] + "' is set; this reader supports"
                " plain LoRA only and would load an adapter that computes"
                " something else"
            )
    var r_node = doc.field(root, "r")
    var a_node = doc.field(root, "lora_alpha")
    if r_node < 0 or a_node < 0:
        raise Error(path + ": no 'r' or no 'lora_alpha'")
    var cfg = PeftLoRAConfig(doc.integer(r_node), doc.number(a_node))
    var rs = doc.field(root, "use_rslora")
    if rs >= 0 and doc.kind_of(rs) == J_BOOL:
        cfg.use_rslora = doc.boolean(rs)
    var tm = doc.field(root, "target_modules")
    if tm >= 0 and doc.kind_of(tm) == J_ARRAY:
        for i in range(doc.size(tm)):
            cfg.target_modules.append(doc.string(doc.at(tm, i)))
    elif tm >= 0 and doc.kind_of(tm) == J_STRING:
        cfg.target_modules.append(doc.string(tm))  # a regex, in PEFT
    var bm = doc.field(root, "base_model_name_or_path")
    if bm >= 0 and doc.kind_of(bm) == J_STRING:
        cfg.base_model_name_or_path = doc.string(bm)
    return cfg^


def load_peft_lora[
    IN: Int, OUT: Int, R: Int, PAD_R: Int, RSLORA: Bool, //,
    target: StaticString,
](
    ref st: SafeTensors,
    ref cfg: PeftLoRAConfig,
    path: String,
    mut a: LoRAAdapter[IN, OUT, R, PAD_R, RSLORA],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Load the adapter PEFT saved for the torch module `path` into `a`:
    A and B transposed into noeira's layout, the pad left zero, α from the
    config. Raises on any rank, shape or scale-rule mismatch."""
    if cfg.r != R:
        raise Error(
            "load_peft_lora: the file has r = " + String(cfg.r)
            + ", the adapter R = " + String(R)
        )
    if cfg.use_rslora != RSLORA:
        raise Error(
            "load_peft_lora: use_rslora is " + String(cfg.use_rslora)
            + " in the file and " + String(RSLORA) + " in the adapter: the"
            " scale would be α / R where α / √R was trained, or the reverse"
        )
    if a.merged:
        raise Error("load_peft_lora: the adapter is merged; unmerge it first")
    var ka = peft_key(path, "lora_A")
    var kb = peft_key(path, "lora_B")
    if not st.has(ka) or not st.has(kb):
        raise Error("load_peft_lora: no '" + ka + "' / '" + kb + "' in the file")
    var sa = st.shape(ka)
    var sb = st.shape(kb)
    if len(sa) != 2 or sa[0] != R or sa[1] != IN:
        raise Error(
            "load_peft_lora: " + ka + " is " + st.shape_str(ka) + ", expected ["
            + String(R) + ", " + String(IN) + "]"
        )
    if len(sb) != 2 or sb[0] != OUT or sb[1] != R:
        raise Error(
            "load_peft_lora: " + kb + " is " + st.shape_str(kb) + ", expected ["
            + String(OUT) + ", " + String(R) + "]"
        )
    var va = st.read_f32(ka)
    var vb = st.read_f32(kb)
    comptime RP = LoRAAdapter[IN, OUT, R, PAD_R, RSLORA].R_PAD
    for i in range(IN):
        for j in range(RP):
            a.lora_A.val.data[i * RP + j] = (
                Scalar[DT](va[j * IN + i]) if j < R else Scalar[DT](0)
            )
    for j in range(RP):
        for o in range(OUT):
            a.lora_B.val.data[j * OUT + o] = (
                Scalar[DT](vb[o * R + j]) if j < R else Scalar[DT](0)
            )
    a.alpha = cfg.lora_alpha
    comptime if target == "gpu":
        a.lora_A.val.upload_resident(ctx.value())
        a.lora_B.val.upload_resident(ctx.value())
    a.lora_A.val.version += 1
    a.lora_B.val.version += 1


# ── writing ────────────────────────────────────────────────────────────


struct PeftAdapterWriter(Movable):
    """Collect adapters by torch module path, then write a PEFT adapter
    directory. Every adapter shares the config's r, α and scale rule, as one
    PEFT config without `rank_pattern` / `alpha_pattern` requires."""

    var st: SafeTensorsWriter
    var cfg: PeftLoRAConfig
    var bf16: Bool

    def __init__(
        out self,
        r: Int,
        lora_alpha: Float64,
        use_rslora: Bool = False,
        bf16: Bool = False,
        base_model_name_or_path: String = String(""),
    ):
        self.st = SafeTensorsWriter()
        self.cfg = PeftLoRAConfig(r, lora_alpha, use_rslora, base_model_name_or_path)
        self.bf16 = bf16

    def add[
        IN: Int, OUT: Int, R: Int, PAD_R: Int, RSLORA: Bool, //,
        target: StaticString,
    ](
        mut self,
        path: String,
        mut a: LoRAAdapter[IN, OUT, R, PAD_R, RSLORA],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`a`'s logical A and B under `path`, transposed to PEFT's layout.
        On the GPU the values are read back from the device first."""
        if R != self.cfg.r or RSLORA != self.cfg.use_rslora:
            raise Error(
                "PeftAdapterWriter.add: '" + path + "' has R = " + String(R)
                + ", rslora " + String(RSLORA) + "; the file's config says r = "
                + String(self.cfg.r) + ", rslora " + String(self.cfg.use_rslora)
            )
        if a.alpha != self.cfg.lora_alpha:
            raise Error(
                "PeftAdapterWriter.add: '" + path + "' has α = " + String(a.alpha)
                + ", the file's config " + String(self.cfg.lora_alpha)
            )
        comptime if target == "gpu":
            a.lora_A.val.download(ctx.value())
            a.lora_B.val.download(ctx.value())
        comptime RP = LoRAAdapter[IN, OUT, R, PAD_R, RSLORA].R_PAD
        var at = List[Float32](capacity=R * IN)
        for j in range(R):
            for i in range(IN):
                at.append(Float32(a.lora_A.val.data[i * RP + j]))
        var bt = List[Float32](capacity=OUT * R)
        for o in range(OUT):
            for j in range(R):
                bt.append(Float32(a.lora_B.val.data[j * OUT + o]))
        var sa = List[Int]()
        sa.append(R)
        sa.append(IN)
        var sb = List[Int]()
        sb.append(OUT)
        sb.append(R)
        if self.bf16:
            self.st.add_bf16(peft_key(path, "lora_A"), sa, at, R * IN)
            self.st.add_bf16(peft_key(path, "lora_B"), sb, bt, OUT * R)
        else:
            self.st.add_f32(peft_key(path, "lora_A"), sa, at, R * IN)
            self.st.add_f32(peft_key(path, "lora_B"), sb, bt, OUT * R)
        self.cfg.target_modules.append(path)

    def config_json(self) raises -> String:
        """`adapter_config.json`: the fields PEFT writes for a plain LoRA, the
        refused ones at their neutral values (`read_peft_config`)."""
        var paths = self.cfg.target_modules.copy()
        sort(paths)
        var w = JsonWriter()
        w.begin_object()
        w.key("alpha_pattern")
        w.begin_object()
        w.end_object()
        w.member("base_model_name_or_path", self.cfg.base_model_name_or_path)
        w.member("bias", String("none"))
        w.key("fan_in_fan_out")
        w.boolean(False)
        w.key("inference_mode")
        w.boolean(True)
        w.key("init_lora_weights")
        w.boolean(True)
        w.key("lora_alpha")
        if self.cfg.lora_alpha == Float64(Int(self.cfg.lora_alpha)):
            w.integer(Int(self.cfg.lora_alpha))
        else:
            w.number(self.cfg.lora_alpha)
        w.key("lora_bias")
        w.boolean(False)
        w.member("lora_dropout", Float64(0.0))
        w.key("modules_to_save")
        w.null()
        w.member("peft_type", String("LORA"))
        w.member("r", self.cfg.r)
        w.key("rank_pattern")
        w.begin_object()
        w.end_object()
        w.key("target_modules")
        w.begin_array()
        for i in range(len(paths)):
            w.string(paths[i])
        w.end_array()
        w.key("task_type")
        w.null()
        w.key("use_dora")
        w.boolean(False)
        w.key("use_rslora")
        w.boolean(self.cfg.use_rslora)
        w.end_object()
        return w.done()

    def save(self, dir: String) raises:
        """`dir/adapter_model.safetensors` and `dir/adapter_config.json`."""
        if self.st.size() == 0:
            raise Error("PeftAdapterWriter.save: no adapter was added")
        makedirs(dir, exist_ok=True)
        self.st.save(dir + "/adapter_model.safetensors")
        write_text_atomic(dir + "/adapter_config.json", self.config_json())

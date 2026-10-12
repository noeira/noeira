# +--------------------------------------------------------------------------+ #
# | nn <-> safetensors, by walked name
# +--------------------------------------------------------------------------+ #
"""Save and load an `nn` model as `.safetensors`.

    from noeira.nn.core.safetensors_io import save_safetensors, load_safetensors

    save_safetensors[target](model, String("model.safetensors"), ctx)
    var n = load_safetensors[target](model, String("model.safetensors"), ctx)

## This is the INTEROP format, not the resume format

`checkpoint.mojo`'s v3 stays the checkpoint: it carries Adam's per-param `m`/`v`
moments, which is what makes a run resumable, and safetensors has no place to
put them that another framework would understand. What safetensors buys is the
other direction — weights that PyTorch, JAX, or anything on the Hub can read,
and weights from any of them that we can read.

`save_safetensors` therefore writes params + states and NOT moments. A model
saved here and loaded here is the same model; it is not the same training run.

## Names

The dotted `for_each_param` / `for_each_state` path ("0.weight",
"1.running_mean"), verbatim. Params and states share one namespace — they
already have disjoint names, and `SafeTensorsWriter` raises on a collision
rather than letting one shadow the other.

⚠ **Loading is BY NAME, and that is not a preference.** A safetensors producer
picks its own key order — a torchvision ResNet18 file leads with
`bn1.num_batches_tracked` — so anything positional reads the right sizes into
the wrong tensors on a file that is completely valid.

## Shapes

Written as rank-1 `[N]`. Our `Param[NAME, DECAY, SIZE]` carries a flat size and
nothing else, so a real shape would have to be invented here, and inventing it
wrongly produces a file that loads and is silently transposed.

That is the right trade for OUR files. It is the wrong one for a file another
framework has to reshape, and that case is `torch_names.mojo`: an export that a
`load_state_dict` can consume needs their names, their layout AND their shapes,
all three of which are per-architecture facts that no generic walk can know.
"""

from max.gpu.host import DeviceContext

from noeira.io.safetensors import SafeTensors, SafeTensorsWriter
from noeira.nn.constants import DT
from .param import ParamVisitor, ParamWalkable, ParamVisitorRT, walk_params, ParamVisitorRef
from .tensor import Tensor


struct SafeTensorsSaver(ParamVisitor, ParamVisitorRT):
    """Collects every visited tensor into a `SafeTensorsWriter` as rank-1 f32.

    ⚠ GPU params are DOWNLOADED first. Without that the file holds whatever
    the host mirror last had, which for a model trained on device is the
    initialisation — a checkpoint of the random weights, saved without error.
    """

    var writer: SafeTensorsWriter
    var count: Int

    def __init__(out self):
        self.writer = SafeTensorsWriter()
        self.count = 0

    def __init__(out self, *, deinit move: Self):
        self.writer = move.writer^
        self.count = move.count

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        var shape: List[Int] = [n]
        self.writer.add_f32(String(name), shape, param.data, n)
        self.count += 1

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
struct SafeTensorsLoader(ParamVisitor, ParamVisitorRT):
    """Fills every visited tensor from `file[name]`.

    Records `loaded` / `missing` so the caller can decide, rather than assuming
    a silent partial load was fine. A model left half-random loads without
    error and then reports "the weights did not help", which is the failure
    this bookkeeping exists to make impossible.
    """

    var file: SafeTensors
    var loaded: List[String]
    var missing: List[String]

    def __init__(out self, var file: SafeTensors):
        self.file = file^
        self.loaded = List[String]()
        self.missing = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.file = move.file^
        self.loaded = move.loaded^
        self.missing = move.missing^

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if not self.file.has(name):
            self.missing.append(String(name))
            return
        var vals = self.file.read_f32(name)
        if len(vals) != n:
            raise Error(
                "load_safetensors: '" + name + "' holds " + String(len(vals))
                + " values but the model's param has " + String(n)
                + " (shape " + self.file.shape_str(name) + ")"
            )
        fill_param(param, vals, ctx)
        self.loaded.append(String(name))

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
def fill_param(
    mut param: Tensor, ref vals: List[Float32], ctx: Optional[DeviceContext]
) raises:
    """Host fill, then upload when the module lives on a device.

    ⚠ The upload is not optional and its absence is silent: the host tensor
    would hold the loaded weights, every device kernel would keep reading the
    old ones, and the run would look like the load simply did not matter."""
    param.ensure(len(vals))
    for i in range(len(vals)):
        param.data[i] = Scalar[DT](vals[i])
    if ctx:
        param.upload_resident(ctx.value())
    # ⚠ A load is a write, so it advances `version`, as the checkpoint readers
    # do: `Linear` caches derived copies of its weight (`w_pad`, `w_bf`) gated
    # on it. Without the bump, a forward run before this load keeps serving
    # the old weight. A trained layer hid that until its first optimizer step;
    # a frozen layer is never stepped and would serve it forever.
    param.version += 1


def save_safetensors[
    target: StaticString, M: ParamWalkable
](
    mut model: M,
    var path: String,
    ctx: Optional[DeviceContext] = None,
    include_state: Bool = True,
    var framework: String = String("noeira"),
) raises -> Int:
    """Write every param (and, by default, every state) to `path`.

    Returns the tensor count. `framework` lands in `__metadata__` — the format
    reserves that member for exactly this and consumers ignore what they do not
    recognise."""
    var s = SafeTensorsSaver()
    walk_params[target](model, s, ctx)
    if include_state:
        var _sref1 = ParamVisitorRef.of[type_of(s), target](s)
        model.for_each_state[target](_sref1, ctx)
    if s.count == 0:
        raise Error(
            "save_safetensors: the walk visited no tensors — '" + path
            + "' would have been an empty file"
        )
    s.writer.add_metadata(String("producer"), framework^)
    s.writer.save(path^)
    return s.count


def load_safetensors[
    target: StaticString, M: ParamWalkable
](
    mut model: M,
    var path: String,
    ctx: Optional[DeviceContext] = None,
    include_state: Bool = True,
    strict: Bool = True,
) raises -> Int:
    """Fill every param (and, by default, every state) from `path`.

    Returns the number filled. With `strict` (the default) a tensor the file
    does not name is an error: a partial load is the failure mode that looks
    like a working one."""
    var pl = SafeTensorsLoader(SafeTensors(String(path)))
    walk_params[target](model, pl, ctx)
    var n = len(pl.loaded)
    var missing = len(pl.missing)
    var first_missing = String("") if missing == 0 else pl.missing[0]

    if include_state:
        var sl = SafeTensorsLoader(SafeTensors(String(path)))
        var _sref2 = ParamVisitorRef.of[type_of(sl), target](sl)
        model.for_each_state[target](_sref2, ctx)
        n += len(sl.loaded)
        if len(sl.missing) > 0 and missing == 0:
            first_missing = sl.missing[0]
        missing += len(sl.missing)

    if strict and missing > 0:
        raise Error(
            "load_safetensors: '" + path + "' does not name " + String(missing)
            + " of the model's tensors, first '" + first_missing
            + "' — pass strict=False to load the rest anyway"
        )
    if n == 0:
        raise Error(
            "load_safetensors: '" + path + "' matched NOTHING in this model."
            " The file's names and the model's walk have nothing in common,"
            " so every weight would have kept its initialisation"
        )
    return n

"""List every Param and State of the reference-exact LeWM loss graph, with its
size, in walk order — the names `tools/lewm/convert_ref_to_ours.py` maps the
checkpoint onto.

    pixi run mojo run -I . tools/lewm/list_ref_params.mojo > /tmp/lewm_ref/ours_names.tsv

Output: `P<TAB>name<TAB>size` / `S<TAB>name<TAB>size` lines. Paper width
(`config.json` of quentinll/lewm-pusht), T = 4 (history 3 + 1 prediction).
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Kaiming
from noeira.experimental.lewm.ref_model import LeWMLossGraphRef


comptime Graph = LeWMLossGraphRef[
    3, 224, 14, 192, 3, 12, 192, 2048,  # IN_CH IMG PATCH HIDDEN HEADS LAYERS EMB PROJ_H
    4, 10, 3, 1,                        # T ACT H N_PREDS
    16, 64, 2048, 6,                    # PRED_HEADS PRED_DIM_HEAD PRED_FF DEPTH
    1024, 17,                           # SIG_PROJ SIG_KNOTS
]


struct _List(ParamVisitor):
    var kind: String

    def __init__(out self, kind: String):
        self.kind = kind

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        print(self.kind, "\t", name, "\t", N, sep="")


def main() raises:
    var g = Graph.make["cpu", Kaiming](None)
    var p = _List(String("P"))
    g.for_each_param["cpu"](p, None)
    var s = _List(String("S"))
    g.for_each_state["cpu"](s, None)

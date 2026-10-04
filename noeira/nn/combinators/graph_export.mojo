"""ComputeGraph exporters — `GraphVisitor` conformers that render a storage
`ComputeGraph`'s topology (run via `graph.describe[Exporter](name, exporter)`).

  - `TextExporter`          — a readable indented table; container nodes
    (a `Node` wrapping a `Sequential`) expand their children underneath.
  - `MermaidExporter`       — a `graph TD` Mermaid diagram (InputSlots render as
    stadiums; containers as subgraphs of their chained children).
  - `FusionReportExporter`  — a static fusion census: classifies each node
    (matmul / pointwise / memory / rng / loss / input / other) by display label,
    computes fan-out, and finds maximal connected regions of pointwise nodes —
    the chains a single fused element-wise kernel could collapse. Pure static
    analysis off the comptime topology; no kernels, no timing.

Transformed from the legacy `nn/combinators/graph_export.mojo`. Node labels come
from `Module.display_label` (e.g. "Linear", "ReLU", "Concat") and container
expansion from `Module.display_steps`; read `.out` for the rendered String.
"""

from ..core.graph_visitor import GraphVisitor


def _pad(s: String, width: Int) -> String:
    """Right-pad `s` with spaces to at least `width` bytes (ASCII names)."""
    var out = s
    var n = s.byte_length()
    while n < width:
        out += " "
        n += 1
    return out


def _idx2(i: Int) -> String:
    """Zero-padded 2-digit node index."""
    if i < 10:
        return "0" + String(i)
    return String(i)


def _is_pointwise(label: String) -> Bool:
    """Element-wise ops a single fused kernel can chain (same width in/out, one
    scalar op per element, derivative known per-op)."""
    var pw = [
        String("ReLU"), String("Tanh"), String("Sigmoid"), String("Mish"),
        String("GELU"), String("GELUTanh"), String("SiLU"), String("Symlog"), String("StopGrad"),
        String("Scale"), String("Min"), String("Sub"), String("Add"),
        String("Binary"),
    ]
    for i in range(len(pw)):
        if label == pw[i]:
            return True
    return False


def _fusion_category(label: String, kind: Int) -> String:
    if kind == 0:
        return String("input")
    if _is_pointwise(label):
        return String("pointwise")
    if label == String("Linear") or label == String("LinearAct"):
        return String("matmul")
    if label == String("Concat") or label == String("Slice"):
        return String("memory")
    if label == String("RSample") or label == String("STSample"):
        return String("rng")
    var loss = [
        String("MSEPlain"), String("SymlogMSE"), String("TwoHot"),
        String("TwoHotDecode"), String("OneHotKL"), String("TDMPC2TwoHot"),
        String("MSE"),
    ]
    for i in range(len(loss)):
        if label == loss[i]:
            return String("loss")
    return String("other")


def _uf_find(mut parent: List[Int], x: Int) -> Int:
    var r = x
    while parent[r] != r:
        r = parent[r]
    var c = x
    while parent[c] != c:
        var nxt = parent[c]
        parent[c] = r
        c = nxt
    return r


def _uf_union(mut parent: List[Int], a: Int, b: Int):
    var ra = _uf_find(parent, a)
    var rb = _uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra


struct TextExporter(GraphVisitor):
    """Buffers node rows + container children, formats an indented table in
    `end()`. Read the result from `.out`."""

    var out: String
    var gname: String
    var names: List[String]
    var labels: List[String]
    var outs: List[Int]
    var ins: List[String]  # accumulated "src1, src2, ..." per node
    var subs: List[List[String]]  # container children ("label →out") per node

    def __init__(out self):
        self.out = String("")
        self.gname = String("")
        self.names = List[String]()
        self.labels = List[String]()
        self.outs = List[Int]()
        self.ins = List[String]()
        self.subs = List[List[String]]()

    def begin(mut self, graph_name: String, n_nodes: Int) raises:
        self.gname = graph_name

    def node(
        mut self, idx: Int, name: String, label: String, kind: Int,
        out_dim: Int,
    ) raises:
        self.names.append(name)
        self.labels.append(label)
        self.outs.append(out_dim)
        self.ins.append(String(""))
        self.subs.append(List[String]())

    def node_inner(
        mut self, parent: String, step_idx: Int, label: String, out_dim: Int
    ) raises:
        self.subs[len(self.subs) - 1].append(label + " →" + String(out_dim))

    def edge(
        mut self, dst: String, src: String, slot: Int, in_dim: Int
    ) raises:
        var last = len(self.ins) - 1
        if self.ins[last].byte_length() > 0:
            self.ins[last] = self.ins[last] + ", " + src
        else:
            self.ins[last] = src

    def end(mut self) raises:
        var maxname = 4
        var maxlabel = 5
        for i in range(len(self.names)):
            if self.names[i].byte_length() > maxname:
                maxname = self.names[i].byte_length()
            if self.labels[i].byte_length() > maxlabel:
                maxlabel = self.labels[i].byte_length()

        var s = String("ComputeGraph")
        if self.gname.byte_length() > 0:
            s += ' "' + self.gname + '"'
        s += "  (" + String(len(self.names)) + " nodes)\n"

        for i in range(len(self.names)):
            s += "  #" + _idx2(i) + "  " + _pad(self.names[i], maxname)
            s += "  " + _pad(self.labels[i], maxlabel)
            if self.ins[i].byte_length() > 0:
                s += _pad("  <- " + self.ins[i], maxname + 28)
            else:
                s += _pad("", maxname + 28)
            s += "  -> " + String(self.outs[i]) + "\n"
            # Container children, indented under the node.
            for j in range(len(self.subs[i])):
                s += _pad("", maxname + 10) + "└ " + self.subs[i][j] + "\n"
        self.out = s


struct MermaidExporter(GraphVisitor):
    """Buffers nodes + edges, renders a `graph TD` Mermaid diagram in `end()`.
    Read the result from `.out`."""

    var out: String
    var gname: String
    var names: List[String]
    var labels: List[String]
    var kinds: List[Int]
    var outs: List[Int]
    var subs: List[List[String]]  # container children ("label →out") per node
    var esrc: List[String]
    var edst: List[String]
    var ein: List[Int]

    def __init__(out self):
        self.out = String("")
        self.gname = String("")
        self.names = List[String]()
        self.labels = List[String]()
        self.kinds = List[Int]()
        self.outs = List[Int]()
        self.subs = List[List[String]]()
        self.esrc = List[String]()
        self.edst = List[String]()
        self.ein = List[Int]()

    def begin(mut self, graph_name: String, n_nodes: Int) raises:
        self.gname = graph_name

    def node(
        mut self, idx: Int, name: String, label: String, kind: Int,
        out_dim: Int,
    ) raises:
        self.names.append(name)
        self.labels.append(label)
        self.kinds.append(kind)
        self.outs.append(out_dim)
        self.subs.append(List[String]())

    def node_inner(
        mut self, parent: String, step_idx: Int, label: String, out_dim: Int
    ) raises:
        self.subs[len(self.subs) - 1].append(label + " →" + String(out_dim))

    def edge(
        mut self, dst: String, src: String, slot: Int, in_dim: Int
    ) raises:
        self.esrc.append(src)
        self.edst.append(dst)
        self.ein.append(in_dim)

    def end(mut self) raises:
        var s = String("")
        if self.gname.byte_length() > 0:
            s += "%% " + self.gname + "\n"
        s += "graph TD\n"

        for i in range(len(self.names)):
            var name = self.names[i]
            var label = self.labels[i]
            var od = String(self.outs[i])
            if self.kinds[i] == 0:
                # InputSlot — stadium.
                s += "  " + name + '(["' + name + "<br/>in " + od + '"])\n'
            elif len(self.subs[i]) > 0:
                # Container — subgraph holding the chained children.
                s += "  subgraph " + name + " [" + name + ": " + label + "]\n"
                var nsub = len(self.subs[i])
                for j in range(nsub):
                    s += (
                        "    " + name + "__" + String(j) + '["'
                        + self.subs[i][j] + '"]\n'
                    )
                for j in range(nsub - 1):
                    s += (
                        "    " + name + "__" + String(j) + " --> "
                        + name + "__" + String(j + 1) + "\n"
                    )
                s += "  end\n"
            else:
                # Leaf — box labelled op name + output width.
                s += (
                    "  " + name + '["' + name + "<br/>" + label
                    + " &rarr; " + od + '"]\n'
                )

        for e in range(len(self.esrc)):
            s += (
                "  " + self.esrc[e] + " -->|" + String(self.ein[e]) + "| "
                + self.edst[e] + "\n"
            )
        self.out = s


struct FusionReportExporter(GraphVisitor):
    """Static fusion census — node categories, fan-out, and maximal pointwise
    regions (the chains a single fused element-wise kernel could collapse).
    Read the result from `.out`."""

    var out: String
    var gname: String
    var names: List[String]
    var labels: List[String]
    var kinds: List[Int]
    var outs: List[Int]
    var esrc: List[String]
    var edst: List[String]

    def __init__(out self):
        self.out = String("")
        self.gname = String("")
        self.names = List[String]()
        self.labels = List[String]()
        self.kinds = List[Int]()
        self.outs = List[Int]()
        self.esrc = List[String]()
        self.edst = List[String]()

    def begin(mut self, graph_name: String, n_nodes: Int) raises:
        self.gname = graph_name

    def node(
        mut self, idx: Int, name: String, label: String, kind: Int,
        out_dim: Int,
    ) raises:
        self.names.append(name)
        self.labels.append(label)
        self.kinds.append(kind)
        self.outs.append(out_dim)

    def edge(
        mut self, dst: String, src: String, slot: Int, in_dim: Int
    ) raises:
        self.esrc.append(src)
        self.edst.append(dst)

    def end(mut self) raises:
        var n = len(self.names)

        # name -> index
        var name2idx = Dict[String, Int]()
        for i in range(n):
            name2idx[self.names[i]] = i

        # categories + fan-out + pointwise mask
        var cats = List[String]()
        var pointwise = List[Bool]()
        var fanout = List[Int]()
        for i in range(n):
            cats.append(_fusion_category(self.labels[i], self.kinds[i]))
            pointwise.append(cats[i] == String("pointwise"))
            fanout.append(0)
        for e in range(len(self.esrc)):
            if self.esrc[e] in name2idx:
                fanout[name2idx[self.esrc[e]]] += 1

        # union pointwise->pointwise edges
        var parent = List[Int]()
        for i in range(n):
            parent.append(i)
        for e in range(len(self.esrc)):
            if self.esrc[e] in name2idx and self.edst[e] in name2idx:
                var sidx = name2idx[self.esrc[e]]
                var didx = name2idx[self.edst[e]]
                if pointwise[sidx] and pointwise[didx]:
                    _uf_union(parent, sidx, didx)

        # per-node table
        var s = String("FusionReport")
        if self.gname.byte_length() > 0:
            s += ' "' + self.gname + '"'
        s += "  (" + String(n) + " nodes)\n"
        var maxname = 4
        var maxlabel = 5
        for i in range(n):
            if self.names[i].byte_length() > maxname:
                maxname = self.names[i].byte_length()
            if self.labels[i].byte_length() > maxlabel:
                maxlabel = self.labels[i].byte_length()
        for i in range(n):
            s += "  #" + _idx2(i) + "  " + _pad(self.names[i], maxname)
            s += "  " + _pad(self.labels[i], maxlabel)
            s += "  " + _pad(cats[i], 10)
            s += "  out=" + _pad(String(self.outs[i]), 5)
            s += "  fanout=" + String(fanout[i]) + "\n"

        # category histogram
        var cat_order = [
            String("matmul"), String("pointwise"), String("memory"),
            String("rng"), String("loss"), String("input"), String("other"),
        ]
        s += "  -- categories --\n"
        for c in range(len(cat_order)):
            var cnt = 0
            for i in range(n):
                if cats[i] == cat_order[c]:
                    cnt += 1
            if cnt > 0:
                s += "    " + _pad(cat_order[c], 10) + " " + String(cnt) + "\n"

        # fusable pointwise regions (size >= 2)
        var seen = Dict[Int, Bool]()
        var total_saved = 0
        var n_regions = 0
        s += "  -- fusable pointwise regions --\n"
        for i in range(n):
            if not pointwise[i]:
                continue
            var r = _uf_find(parent, i)
            if r in seen:
                continue
            seen[r] = True
            # collect members of this root
            var members = String("")
            var size = 0
            for j in range(n):
                if pointwise[j] and _uf_find(parent, j) == r:
                    if size > 0:
                        members += ", "
                    members += self.names[j]
                    size += 1
            if size >= 2:
                n_regions += 1
                total_saved += size - 1
                s += (
                    "    {" + members + "}  size=" + String(size)
                    + " -> 1 kernel (saves " + String(size - 1) + ")\n"
                )

        # memory-glue eliminable count
        var mem = 0
        for i in range(n):
            if cats[i] == String("memory"):
                mem += 1

        s += "  -- summary --\n"
        s += (
            "    pointwise-fusion: " + String(n_regions)
            + " region(s), " + String(total_saved)
            + " launch(es) eliminable\n"
        )
        s += "    memory-glue nodes (Concat/Slice): " + String(mem) + "\n"
        self.out = s

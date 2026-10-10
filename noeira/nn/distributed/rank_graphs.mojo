"""RankGraphs — one CUDA graph per GPU for a step enqueued over N contexts.

A data-parallel step is enqueued phase by phase across ranks (every rank's
forward and vjp, then every rank's collective, then every rank's optimizer),
so no single capture window holds "rank r's step". `RankGraphs` therefore
opens a capture on EVERY context's stream at once, runs the step code
unchanged, and closes them all: each stream records only what was enqueued on
it, so each GPU ends up with exactly its own rank's step, collectives
included.

    var g = RankGraphs(pg.graph_ctxs())
    for it in ...:
        def _step() capturing raises -> None:
            job.step()          # mention ONE struct that owns everything
        g.run[_step]()          # first call: eager + capture; then replay

Replay launches every rank's graph back to back WITHOUT waiting on any of
them. That is not an optimisation, it is what MAX's collectives require: every
rank's collective kernel must be launched before any can finish (they meet on
the Signal barriers), so a wait on rank 0's graph before rank 1's is launched
deadlocks. `CUDAGraph.replay()` waits; this uses `replay_on_mojo_stream`.

Why the capture survives replay: the comm kernels keep their barrier counters
and Lamport generation in device memory (`Signal.self_counter`,
`Signal.lamport_state`, `comm/sync.mojo`), not in kernel arguments, so a
replayed collective advances them like an enqueued one. The P2P paths allocate
nothing. The NAIVE (host-staged, no P2P) allreduce allocates its staging
buffers per call (`allreduce.mojo:634`), which aborts a capture — callers
capture compute only there and run the collective eagerly between two graphs.

Each context is made current (`push_context`) around every driver call,
because the replay stream and the instantiated graph belong to the current
context and one host thread drives all N.

On non-NVIDIA `CUDAGraph` is a compile-time no-op, so `run` runs the step
eagerly every call: bit-identical to the uncaptured loop (the Mac path).
"""

from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext

from noeira.cuda.graph import CUDAGraph


comptime _FRESH = 0
comptime _CAPTURED = 1
comptime _DISABLED = 2


struct RankGraphs(Movable):
    var ctxs: List[DeviceContext]
    """One per DISTINCT device context (the shared simulator has one)."""
    var graphs: List[CUDAGraph]
    var state: Int
    var verbose: Bool

    def __init__(out self, ctxs: List[DeviceContext], verbose: Bool = True):
        self.ctxs = ctxs.copy()
        self.graphs = List[CUDAGraph]()
        self.state = _FRESH
        self.verbose = verbose

    def is_captured(self) -> Bool:
        return self.state == _CAPTURED

    def num_nodes(self) -> List[Int]:
        var out = List[Int]()
        for i in range(len(self.graphs)):
            out.append(self.graphs[i].num_nodes())
        return out^

    def run[STEP: def () capturing raises -> None](mut self) raises:
        """Run one step: eager + capture on the first call, replay after.

        `STEP` must enqueue the same work on the same buffers every call and do
        no host reads (a D2H in the window breaks the capture). The first call
        runs `STEP` once eagerly — module loads, autotuning and first-use
        allocations happen outside the window — then records a second pass
        that does NOT execute, so every call is exactly one update."""
        comptime if not has_nvidia_gpu_accelerator():
            STEP()
            return
        if self.state == _CAPTURED:
            for i in range(len(self.ctxs)):
                with self.ctxs[i].push_context():
                    self.graphs[i].replay_on_mojo_stream()
            return
        if self.state == _DISABLED:
            STEP()
            return

        STEP()
        for i in range(len(self.ctxs)):
            self.ctxs[i].synchronize()
        for i in range(len(self.ctxs)):
            with self.ctxs[i].push_context():
                self.graphs.append(
                    CUDAGraph(self.ctxs[i], stream_of_context=True)
                )
            if self.graphs[i].is_disabled():
                # The step already ran for this call; run eagerly from now on.
                self.state = _DISABLED
                return
        for i in range(len(self.ctxs)):
            with self.ctxs[i].push_context():
                self.graphs[i].begin_capture()
        try:
            STEP()
        except e:
            # Close every capture before the error unwinds: a buffer freed
            # while its stream still captures aborts the process
            # (STREAM_CAPTURE_UNSUPPORTED in ~DeviceBuffer) and hides `e`.
            for i in range(len(self.ctxs)):
                try:
                    with self.ctxs[i].push_context():
                        self.graphs[i].end_capture()
                except:
                    pass
            self.state = _DISABLED
            raise e^
        for i in range(len(self.ctxs)):
            with self.ctxs[i].push_context():
                self.graphs[i].end_capture()
        self.state = _CAPTURED
        if self.verbose:
            var s = String("[RankGraphs] captured")
            for i in range(len(self.graphs)):
                s += " gpu" + String(i) + "=" + String(self.graphs[i].num_nodes())
            print(s, "nodes")

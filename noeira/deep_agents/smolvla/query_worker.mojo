# +--------------------------------------------------------------------------+ #
# | The SmolVLA query, on a thread that is allowed to block
# +--------------------------------------------------------------------------+ #
"""`SmolVLAQueryWorker` — one observation in, one action chunk out, off the
control thread.

⚠⚠ WHY A THREAD, WHEN THE QUERY IS ALL ENQUEUES. `SmolVLAPolicy.start_action`
submits the whole query without synchronising, and that was supposed to be
enough: submit, keep commanding the arm, collect later. MEASURED ON AN ORIN IT
IS NOT. A query is ~6970 kernel launches; once the driver's pending-launch
queue fills, `cuLaunchKernel` stops being asynchronous and blocks until slots
free. Submission cost **619.5 ms of a 663.7 ms query** — 93% — so the control
loop was stuck inside the submission and the arm froze anyway.

A thread is the fix that does not depend on driver queue depth: this worker is
ALLOWED to block. The control loop keeps the serial bus and the 30 Hz grid; the
worker keeps the GPU.

## The split, and why it falls this way

    control thread          worker thread
    --------------          -------------
    serial bus              DeviceContext + the 3.2 GB policy
    the 30 Hz grid          the query
    cameras + resize        —
    clamps and writes       —

The cameras stay with the control loop on purpose: their reader is a
single-consumer ring, and moving the consumer would leave the loop unable to
tell a stale frame from a fresh one. The cost is the observation build (~68 ms,
two grid steps) still landing on the control thread — small against the 620 ms
this removes, and the ACT loop's answer (resize on the camera thread) closes it
later without touching any of this.

## The protocol

One request ring (images + pose), one response ring (the chunk in robot
units), and cells for state and timings. Exactly one producer and one consumer
per ring, which is what `SpscRing` requires:

    control -> `req`: [pose: RDIM float32][images: N_CAM*3*IMG*IMG float32]
    worker  -> `rsp`: [chunk: CHUNK*RDIM float32]

⚠ THE WORKER OWNS ITS OWN `images`/`noise` TENSORS. The request's bytes are
copied into them before the query starts, so the control thread may refill its
own tensor the moment `end_push` returns — the two never share a buffer.

⚠ NOTHING HERE MAY RAISE (`BackgroundWorker`'s contract — pthread has no
exception channel). Every failure is caught, printed, and latched into
`QW_STATE`, which the control thread must check before arming anything.
"""

from std.math import cos, log, sin, sqrt
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.core.concurrent.block import SharedBlock
from noeira.core.concurrent.ring import SharedRing
from noeira.core.concurrent.worker import (
    BackgroundWorker,
    POLL_DID_WORK,
    POLL_IDLE,
    WorkerCtl,
)
from noeira.deep_agents.smolvla.expert import EXPERT_FF
from noeira.deep_agents.smolvla.finetune import load_trainables
from noeira.deep_agents.smolvla.heads import (
    SMOLVLA_ACTION_DIM,
    SMOLVLA_EXPERT_W,
    SMOLVLA_STATE_DIM,
)
from noeira.deep_agents.smolvla.policy import SmolVLAPolicy
from noeira.deep_agents.smolvla.tasks import TaskTokens
from noeira.deep_agents.smolvla.text import (
    SMOLLM_DIM,
    SMOLLM_KV_W,
    SMOLLM_LAYERS,
)
from noeira.io.hf import hf_download_file, HF_MODEL
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.tensor import Tensor
from noeira.nn.primitives.linear import Linear
from noeira.vision.resize_pad import SIGLIP_INPUT


# ── cells ───────────────────────────────────────────────────────────────────
# ⚠ SPACED EIGHT APART, one per cache line, for the same reason
# `camera_thread.mojo` spaces its own: two counters in one line make every
# worker write invalidate the reader's line and vice versa.

comptime QW_STATE: Int = 0
"""0 = starting, 1 = ready, 2 = failed. Latched; never goes backwards."""
comptime QW_QUERY_US: Int = 8
"""Last query, microseconds — submission, execution and the copy back."""
comptime QW_SUBMIT_US: Int = 16
"""Of which, the submission. On the Orin this is ~93% and is exactly what the
control thread is no longer paying."""
comptime QW_SERVED: Int = 24
"""Chunks published. The control thread's own count should match it."""
comptime QW_DROPPED: Int = 32
"""Chunks computed and thrown away because the control thread had not taken
the previous one. Non-zero means the loop is asking for chunks faster than it
consumes them, which is a `lead` that is too large — not a GPU problem."""
comptime QW_POLLS: Int = 40
"""The worker's OWN poll counter, published from inside `poll`.

⚠ NOT REDUNDANT with `BackgroundThread.polls()`. That counter is written by
the drive loop; this one is written by the worker body. If the drive loop's
count climbs while this one does not, the worker is wedged before its first
statement; if both climb while no chunk appears, the two threads are holding
DIFFERENT rings."""
comptime QW_N_CELLS: Int = 48

comptime QW_STARTING: Int64 = 0
comptime QW_READY: Int64 = 1
comptime QW_FAILED: Int64 = 2


struct SmolVLAQueryWorker[
    N_CAM: Int,
    N_LANG: Int,
    CHUNK: Int,
    STEPS: Int,
    RDIM: Int,
    target: StaticString,
    IMG: Int = SIGLIP_INPUT,
](BackgroundWorker):
    """Builds the policy on its own thread, then answers one request at a time.

    Built cheap and started expensive: the constructor only records paths, so
    the 3.2 GB of layers, the checkpoint and the warm-up all happen in
    `on_start` — on the worker thread, where a `DeviceContext` is created and
    from then on used by nobody else.
    """

    comptime Pol = SmolVLAPolicy[
        Self.N_CAM, Self.N_LANG, Self.CHUNK, Self.STEPS, 1
    ]
    comptime IMG_ELEMS: Int = Self.N_CAM * 3 * Self.IMG * Self.IMG
    comptime REQ_BYTES: Int = (Self.RDIM + Self.IMG_ELEMS) * 4
    comptime RSP_BYTES: Int = Self.CHUNK * Self.RDIM * 4
    comptime XN: Int = Self.CHUNK * SMOLVLA_ACTION_DIM

    var req: SharedRing
    var rsp: SharedRing
    var cells: SharedBlock
    var base_repo: String
    var ckpt: String
    var stats_path: String
    var tasks_path: String
    var task_index: Int
    var warmups: Int
    var fused_vision: Bool
    """Route the SigLIP towers through the fused attention kernel (the
    deploy's default; `--no-fused-vision` turns it off for an A/B)."""

    var pol: Optional[Self.Pol]
    var ctx: Optional[DeviceContext]
    var ids: List[Int]
    var images: Tensor
    var noise: Tensor
    var pose: List[Float32]
    var act: List[Float32]
    var served: Int

    def __init__(
        out self,
        req: SharedRing,
        rsp: SharedRing,
        cells: SharedBlock,
        base_repo: String,
        ckpt: String,
        stats_path: String,
        tasks_path: String,
        task_index: Int,
        warmups: Int,
        fused_vision: Bool = True,
    ):
        self.req = req
        self.rsp = rsp
        self.cells = cells
        self.base_repo = base_repo
        self.ckpt = ckpt
        self.stats_path = stats_path
        self.tasks_path = tasks_path
        self.task_index = task_index
        self.warmups = warmups
        self.fused_vision = fused_vision
        self.pol = None
        self.ctx = None
        self.ids = List[Int]()
        self.images = Tensor()
        self.noise = Tensor()
        self.pose = List[Float32](length=Self.RDIM, fill=Float32(0))
        self.act = List[Float32]()
        self.served = 0

    def __init__(out self, *, deinit move: Self):
        self.req = move.req
        self.rsp = move.rsp
        self.cells = move.cells
        self.base_repo = move.base_repo^
        self.ckpt = move.ckpt^
        self.stats_path = move.stats_path^
        self.tasks_path = move.tasks_path^
        self.task_index = move.task_index
        self.warmups = move.warmups
        self.fused_vision = move.fused_vision
        self.pol = move.pol^
        self.ctx = move.ctx^
        self.ids = move.ids^
        self.images = move.images^
        self.noise = move.noise^
        self.pose = move.pose^
        self.act = move.act^
        self.served = move.served

    def _build(mut self) raises:
        """Everything that can fail, in one place, so `on_start` is the only
        thing that has to be non-raising."""
        comptime if Self.target != "cpu":
            self.ctx = DeviceContext()
            print(
                "worker       " + String(self.ctx.value().name())
                + "  (the query thread owns this device)"
            )
        var base = hf_download_file(
            self.base_repo, String("model.safetensors"), HF_MODEL
        )
        print("             building every layer (3.2 GB host+device) ...")
        var p = Self.Pol.make[Self.target, Deterministic](self.ctx)
        p.load[Self.target](base, self.ctx)

        # ⚠ THE FINE-TUNE IS NOT SELF-CONTAINED — the trainable set only, so
        # the base loads first and this goes over it.
        var sp_frozen = Linear[SMOLVLA_STATE_DIM, SMOLLM_DIM].make[
            Self.target, Deterministic
        ](self.ctx)
        load_trainables[
            Self.target, SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF,
            SMOLLM_DIM, SMOLLM_KV_W, SMOLVLA_ACTION_DIM,
        ](
            self.ckpt, p.expert, p.action_in, p.time_mlp_in, p.time_mlp_out,
            p.action_out, sp_frozen, self.ctx,
        )
        p.load_stats(self.stats_path)
        if p.stats.action_dim() != Self.RDIM or p.stats.state_dim() != Self.RDIM:
            raise Error(
                "smolvla query worker: " + self.stats_path + " describes a "
                + String(p.stats.state_dim()) + "-state / "
                + String(p.stats.action_dim()) + "-action robot, this build is "
                + String(Self.RDIM) + "/" + String(Self.RDIM)
            )
        # Inference only from here: the towers' attention fused (see
        # `SmolVLAPolicy.set_fused_vision_attention`) unless asked not to.
        p.set_fused_vision_attention(self.fused_vision)
        # Nothing trains here: every gradient buffer goes back.
        var n_frozen = p.freeze_all[Self.target](self.ctx)
        print("             " + String(n_frozen) + " values frozen")
        self.pol = p^

        var tasks = TaskTokens(self.tasks_path)
        self.ids = tasks.for_index(self.task_index)
        if len(self.ids) != Self.N_LANG:
            raise Error(
                "smolvla query worker: instruction " + String(self.task_index)
                + " is " + String(len(self.ids)) + " tokens, this build is "
                + String(Self.N_LANG)
            )

        self.images.ensure(Self.IMG_ELEMS)
        self.noise.ensure(Self.XN)
        comptime if Self.target != "cpu":
            self.images.ensure_gpu(self.ctx.value(), Self.IMG_ELEMS)
            self.noise.ensure_gpu(self.ctx.value(), Self.XN)

        # ⚠ THE FIRST QUERY COMPILES KERNELS — 3.9 s on an Orin, against 664 ms
        # warm. Paid here, before the control thread is told READY, it is a
        # slow start; paid in the loop it is one command arriving seconds late
        # with the arm energised.
        _qw_fill_noise(self.noise, Self.XN, 12345, self.ctx)
        var first_ms = 0.0
        var warm_ms = 0.0
        for w in range(self.warmups + 1):
            var t0 = perf_counter_ns()
            self.pol.value().select_action[Self.target](
                self.images, self.ids, self.pose, self.noise, self.act,
                self.ctx,
            )
            var dt = Float64(perf_counter_ns() - t0) / 1e6
            if w == 0:
                first_ms = dt
            else:
                warm_ms = dt
        print(
            "query        " + String(Int(warm_ms * 10.0) / 10) + " ms warm ("
            + "first " + String(Int(first_ms * 10.0) / 10) + " ms, "
            + String(Self.target) + ", on the query thread)"
        )

    def on_start(mut self, ctl: WorkerCtl):
        try:
            self._build()
            self.cells.release_store(QW_STATE, QW_READY)
        except e:
            # ⚠ PRINTED HERE AND LATCHED THERE. The control thread sees only a
            # number, and a number cannot say which file was missing.
            print("⚠⚠ the query thread failed to start: " + String(e))
            self.cells.release_store(QW_STATE, QW_FAILED)

    def poll(mut self, ctl: WorkerCtl) -> Int:
        self.cells.relaxed_store(
            QW_POLLS, self.cells.relaxed_load(QW_POLLS) + 1
        )
        if self.cells.relaxed_load(QW_STATE) != QW_READY:
            return POLL_IDLE
        var claim = self.req.begin_pop()
        if not claim.ok():
            return POLL_IDLE
        try:
            var src = claim.data().unsafe_bitcast[Float32]()
            for j in range(Self.RDIM):
                self.pose[j] = src[unsafe_offset=j]
            for i in range(Self.IMG_ELEMS):
                self.images.data[i] = Scalar[DT](
                    src[unsafe_offset = Self.RDIM + i]
                )
            self.req.end_pop()

            comptime if Self.target != "cpu":
                self.images.upload(self.ctx.value())
            # ⚠ FRESH NOISE EVERY QUERY. Flow matching integrates FROM a sample
            # of x_1; one reused sample makes every chunk a deterministic
            # function of the observation and throws the policy's action
            # distribution away.
            _qw_fill_noise(
                self.noise, Self.XN, self.served * 7919 + 13, self.ctx
            )

            var t0 = perf_counter_ns()
            self.pol.value().start_action[Self.target](
                self.images, self.ids, self.pose, self.noise, self.ctx
            )
            var t_sub = perf_counter_ns()
            comptime if Self.target != "cpu":
                self.ctx.value().synchronize()
            self.pol.value().finish_action[Self.target](self.act)
            var t1 = perf_counter_ns()

            var push = self.rsp.begin_push()
            if push.ok():
                var dst = push.data().unsafe_bitcast[Float32]()
                for i in range(Self.CHUNK * Self.RDIM):
                    dst[unsafe_offset=i] = self.act[i]
                self.rsp.end_push(Self.RSP_BYTES)
            else:
                # ⚠ THE RING IS FULL, AND ONLY THE CONSUMER MAY POP. The
                # control thread has not taken the previous chunk, so the
                # fresher one is lost — blocking here would stall the next
                # request behind it. Counted, because silence would read as a
                # slow GPU rather than a loop that over-requests.
                self.cells.release_store(
                    QW_DROPPED, self.cells.relaxed_load(QW_DROPPED) + 1
                )

            self.served += 1
            self.cells.relaxed_store(
                QW_QUERY_US, Int64((t1 - t0) // 1000)
            )
            self.cells.relaxed_store(
                QW_SUBMIT_US, Int64((t_sub - t0) // 1000)
            )
            self.cells.release_store(QW_SERVED, Int64(self.served))
            return POLL_DID_WORK
        except e:
            print("⚠⚠ the query thread failed mid-run: " + String(e))
            self.cells.release_store(QW_STATE, QW_FAILED)
            return POLL_IDLE

    def on_stop(mut self, ctl: WorkerCtl):
        pass


def _qw_fill_noise(
    mut noise: Tensor, n: Int, seed: Int, ctx: Optional[DeviceContext]
) raises:
    """x_1 ~ N(0,1), freshly drawn, from a LOCAL LCG.

    Box-Muller over a per-worker stream: this runs on the query thread and must
    neither perturb nor be perturbed by anything else's RNG."""
    var s = UInt64(seed * 2 + 1)
    for i in range(0, n, 2):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u1 = Float64(
            (s >> 11) & UInt64(0x1FFFFFFFFFFFFF)
        ) / 9.007199254740992e15
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u2 = Float64(
            (s >> 11) & UInt64(0x1FFFFFFFFFFFFF)
        ) / 9.007199254740992e15
        if u1 < 1e-12:
            u1 = 1e-12
        var r = sqrt(-2.0 * log(u1))
        var a = 6.283185307179586 * u2
        noise.data[i] = Scalar[DT](r * cos(a))
        if i + 1 < n:
            noise.data[i + 1] = Scalar[DT](r * sin(a))
    if ctx:
        noise.upload(ctx.value())

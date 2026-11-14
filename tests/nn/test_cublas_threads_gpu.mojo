"""Two models on two threads, each with its own `DeviceContext`, calling cuBLAS at the same time: every result exact.

    pixi run -e nvidia mojo build -I . tests/nn/test_cublas_threads_gpu.mojo -o build/t_cublas_threads
    ./build/t_cublas_threads [GEMMs a thread]
    # the old path, for the before/after: -D NOEIRA_CUBLAS_MAX_HANDLE (MAX's handle, one per device)

WHY THIS EXISTS. The room demo runs its models in one process, each on its own
thread with its own context (`noeira.ai.local_service`), and one start in ten
lost d1 in its warm-up (`cublas_gemm: [29x2048] K=7680 ... EXECUTION_FAILED`).
MAX keeps ONE cuBLAS handle per device and re-attaches it to the caller's
stream on every call, and noeira's workspace was keyed the same way
(`noeira/nn/core/cublas_gemm.mojo`'s note): two threads can then launch each
other's GEMMs on the wrong stream, into one workspace. The fault is too rare
to show at start-up on demand; this makes the two threads collide on purpose.

Each "model" here is a `LocalModel` on a `LocalWorker`, as d1 and the reply
model are. Thread `i` fills `a` with `i + 1` and `b` with ones, so every
element of `a @ b^T` is exactly `K (i + 1)` in fp32; each GEMM is followed by
a copy back and a check of every element. Two shapes alternate: d1's
warm-up GEMM (29 x 2048, K = 7680) and a prefill-sized one (256 x 2048,
K = 2048). PASS: both threads finish with no wrong element and no error.
"""

from std.sys import argv, exit
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.ai.local_service import LocalCall, LocalModel, LocalWorker
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread
from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm, CUBLAS_MAX_HANDLE


struct GemmModel(LocalModel):
    var which: Int
    var reps: Int
    var ctx: Optional[DeviceContext]

    def __init__(out self, which: Int, reps: Int):
        self.which = which
        self.reps = reps
        self.ctx = None

    def load(mut self) raises:
        self.ctx = DeviceContext()

    def warm(mut self) raises:
        pass

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        """Runs `reps` GEMMs, checks each; replies `<gemms> <wrong elements> <first error>`."""
        var ctx = self.ctx.value()
        var v = Float32(self.which + 1)
        var ms: List[Int] = [29, 256]
        var ns: List[Int] = [2048, 2048]
        var ks: List[Int] = [7680, 2048]
        var a = ctx.enqueue_create_buffer[DT](256 * 7680)
        var b = ctx.enqueue_create_buffer[DT](2048 * 7680)
        var y = ctx.enqueue_create_buffer[DT](256 * 2048)
        var h = ctx.enqueue_create_host_buffer[DT](256 * 2048)
        a.enqueue_fill(v)
        b.enqueue_fill(Float32(1.0))
        ctx.synchronize()
        var wrong = 0
        var done = 0
        var err = String("")
        for r in range(self.reps):
            var s = r % 2
            var m = ms[s]
            var n = ns[s]
            var k = ks[s]
            try:
                y.enqueue_fill(Float32(-1.0))
                cublas_gemm[False, True, False](ctx, y, a, b, m, n, k, 0.0)
                ctx.enqueue_copy(h, y)
                ctx.synchronize()
            except e:
                err = String(e)
                break
            var want = Float32(k) * v
            for i in range(m * n):
                if h[i] != want:
                    wrong += 1
            done += 1
        var out = String(done) + " " + String(wrong) + " " + err
        var bytes = List[UInt8]()
        for x in out.as_bytes():
            bytes.append(x)
        return bytes^


def report(mut c: LocalCall, i: Int, reps: Int) -> Bool:
    """Thread `i`'s reply, printed; True when every GEMM ran and was exact."""
    var reply: String
    try:
        var b = c.take(String("thread ") + String(i))
        reply = String(unsafe_from_utf8=Span(b))
    except e:
        reply = String("0 0 ") + String(e)
    var p = reply.split(" ", 2)
    var done = 0
    var wrong = 0
    try:
        done = Int(String(p[0]))
        wrong = Int(String(p[1]))
    except:
        pass
    var err = String(p[2]) if len(p) > 2 else String("")
    var ok = done == reps and wrong == 0 and err == ""
    print("  thread", i, ":", done, "of", reps, "GEMMs,", wrong, "wrong elements", ("| " + err) if err != "" else String(""),
          "ok" if ok else "FAIL")
    return ok


def main() raises:
    var args = argv()
    var reps = Int(String(args[1])) if len(args) > 1 else 300
    print("cuBLAS handle:", "MAX's, one per device (the old path)" if CUBLAS_MAX_HANDLE else "noeira's, one per stream")
    var c0 = LocalCall(1024, 1024)
    var c1 = LocalCall(1024, 1024)
    var t0 = BackgroundThread(LocalWorker(GemmModel(0, reps), c0))
    var t1 = BackgroundThread(LocalWorker(GemmModel(1, reps), c1))
    while c0.loading() or c1.loading():
        _ = sleep_us(1000)
    var go = List[UInt8]()
    go.append(1)
    var start = perf_counter_ns()
    c0.begin(go, String("thread 0"))
    c1.begin(go, String("thread 1"))
    while not (c0.poll() and c1.poll()):
        _ = sleep_us(500)
    var fails = 0
    if not report(c0, 0, reps):
        fails += 1
    if not report(c1, 1, reps):
        fails += 1
    print("  ", round(Float64(perf_counter_ns() - start) / 1e9, 2), "s")
    t0.stop()
    t1.stop()
    print("ALL PASS" if fails == 0 else String(fails) + " FAILED")
    if fails != 0:
        exit(1)

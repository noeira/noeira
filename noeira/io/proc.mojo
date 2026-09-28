# +--------------------------------------------------------------------------+ #
# | Read the stdout of a child process, without Python
# +--------------------------------------------------------------------------+ #
"""A thin `popen` / `fread` / `pclose` wrapper.

Mojo's standard library has no subprocess API, and the alternative — driving a
child through Mojo's embedded CPython — is what `nn/datasets/lewm_pusht.mojo`
documents as leaking every large `PythonObject` until the enclosing function
returns. `popen` is three libc calls with a stable ABI.

    var p = Pipe(quote_arg(path) ... )
    while True:
        var n = p.read_into(buf, count)
        ...
    p.close()          # raises on a non-zero child exit status

    var w = WritePipe(...)          # the other direction: we feed its stdin
    w.write_all(frame_ptr, n)
    w.close()                       # WAITS, so the container trailer lands

⚠ **`popen` RUNS THE STRING THROUGH `/bin/sh`.** Every interpolated path must
go through `quote_arg`, which single-quotes it and REJECTS an embedded single
quote or newline outright. Inside POSIX single quotes every byte is literal
except `'` itself, so rejecting that one character makes the quoting total.
Building a command with bare interpolation would let a filename execute
arbitrary shell — and dataset paths come from a downloaded repo.

⚠ **`pclose` RETURNS A WAIT STATUS, NOT AN EXIT CODE.** A child that exits 1
yields 256. Ignoring it is how a truncated decode reads as a short file rather
than an error, so `close()` decodes the status and raises with the command in
the message.

⚠ **CLOSING EARLY BREAKS THE CHILD'S OUTPUT PIPE, AND THAT IS NOT A FAILURE.**
`pclose` closes the read end, so the child's next write fails. A caller that
deliberately stops reading — the video decoder abandoning a file once it has
the frames it needs — passes `allow_broken_pipe=True`. Treating that as an
error would make every mid-file rollover raise; treating EVERY failure as fine
would hide an `ffmpeg` that actually crashed, so exactly two outcomes are
excused, and only when asked:

  - **signal 13**, `SIGPIPE`'s default action, for a child that leaves the
    signal alone; and
  - **exit 224**, for one that does not. `ffmpeg` calls
    `signal(SIGPIPE, SIG_IGN)` in `term_init()` UNCONDITIONALLY, so it never
    dies of the signal — it gets `EPIPE`, returns `AVERROR(EPIPE)` = `-32`
    from `main`, and `exit` truncates that to `224`.

⚠ **THE SIGNAL ARM ALONE WAS DEAD CODE FOR OUR ONLY CALLER.** It was written
first and looked right for two months; every abandoned decode was really
raising `child exited 224`, and the importer only got away with it because
each decoder happened to reach EOF before its rollover. `-32 & 0xFF` is not a
number you predict from the signal story — measure the child you actually
spawn before believing which arm fires.
"""

from noeira.core.bytes import string_from_bytes

from std.ffi import external_call


comptime SIGPIPE = 13
comptime EPIPE_EXIT = 224
"""`AVERROR(EPIPE)` (-32) returned from `main` and truncated by `exit`."""


def quote_arg(s: String) raises -> String:
    """Single-quote `s` for `/bin/sh`, refusing what cannot be quoted."""
    for i in range(s.byte_length()):
        var c = Int(s.as_bytes()[i])
        if c == 0x27:
            raise Error(
                "proc: refusing to shell-quote a path containing a single"
                " quote: " + s
            )
        if c == 0x0A or c == 0x0D or c == 0:
            raise Error(
                "proc: refusing to shell-quote a path containing a newline or"
                " NUL byte"
            )
    return "'" + s + "'"


struct Pipe(Movable):
    """A running child process whose stdout is being read.

    Its stderr is inherited, so a failing `ffmpeg` prints where it can be seen
    rather than into a buffer nobody reads.
    """

    var _fp: Int
    """`FILE*` as an address. Zero once closed."""
    var command: String
    var closed: Bool

    def __init__(out self, var command: String) raises:
        var mode = String("r")
        var fp = external_call["popen", Int](
            command.as_c_string_span().ptr(),
            mode.as_c_string_span().ptr(),
        )
        if fp == 0:
            raise Error("proc: popen failed for: " + command)
        self._fp = fp
        self.command = command^
        self.closed = False

    def __init__(out self, *, deinit move: Self):
        self._fp = move._fp
        self.command = move.command^
        self.closed = move.closed

    def __deinit__(deinit self):
        # Best effort: a Pipe dropped without `close()` still must not leak the
        # child. Errors are unreportable here, so `close()` stays the way to
        # learn the exit status.
        if not self.closed and self._fp != 0:
            _ = external_call["pclose", Int32](self._fp)

    def read_into(
        mut self, dst: Pointer[Scalar[DType.uint8], MutAnyOrigin], count: Int
    ) raises -> Int:
        """Read up to `count` bytes. Returns the count read; 0 means EOF.

        ⚠ A PIPE READ IS SHORT WHENEVER IT FEELS LIKE IT. `fread` on a pipe
        already loops internally until it has the full request or hits EOF, so
        a short return here really is the end of the stream — but callers still
        have to treat "fewer bytes than a frame" as EOF rather than as an
        error, which is why this returns the count instead of raising.
        """
        if self.closed:
            raise Error("proc: read from a closed pipe")
        if count <= 0:
            return 0
        return external_call["fread", Int](dst, Int(1), count, self._fp)

    def close(mut self, allow_broken_pipe: Bool = False) raises -> Int:
        """Wait for the child and return its exit code. Raises if it failed.

        `allow_broken_pipe` excuses BOTH ways a child reacts to the read end
        going away — signal 13 and exit 224 — because which one it is depends
        on whether the child ignores `SIGPIPE`, which is not ours to know. See
        the module docstring. Pass it only when this side stopped reading on
        purpose.
        """
        if self.closed:
            return 0
        var status = Int(external_call["pclose", Int32](self._fp))
        self.closed = True
        self._fp = 0
        if status < 0:
            raise Error("proc: pclose failed for: " + self.command)
        # POSIX wait status: low 7 bits are the signal, next 8 the exit code.
        var sig = status & 0x7F
        if sig != 0 and not (allow_broken_pipe and sig == SIGPIPE):
            raise Error(
                "proc: child killed by signal " + String(sig) + ": "
                + self.command
            )
        if sig != 0:
            return 0
        var code = (status >> 8) & 0xFF
        if code != 0 and not (allow_broken_pipe and code == EPIPE_EXIT):
            raise Error(
                "proc: child exited " + String(code) + ": " + self.command
            )
        if code != 0:
            return 0
        return code


def run_capture(var command: String, max_bytes: Int = 1 << 16) raises -> String:
    """Run a command and return its stdout as text. For small outputs only.

    Raises if the output would exceed `max_bytes` rather than returning a
    truncated string — a half-read JSON listing parses as a syntax error at a
    meaningless offset, which is a much worse thing to debug than a cap.
    """
    var p = Pipe(String(command))
    var ob = List[UInt8]()
    var chunk = List[UInt8](unsafe_uninit_length=4096)
    var got = 0
    var overflow = False
    while True:
        var n = p.read_into(
            chunk.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
            .as_unsafe_any_origin(),
            4096,
        )
        if n <= 0:
            break
        if got + n > max_bytes:
            overflow = True
            break
        # ⚠ BYTES — see `core/bytes.mojo`. This is a subprocess's stdout;
        # `ffmpeg` and `curl` both emit non-ASCII in diagnostics.
        for i in range(n):
            ob.append(chunk[i])
        got += n
    _ = p.close(overflow)
    if overflow:
        raise Error(
            "proc: output exceeded " + String(max_bytes) + " bytes: " + command
        )
    return string_from_bytes(ob)


# ══════════════════════════════════════════════════════════════════════════
# Writing INTO a child process
# ══════════════════════════════════════════════════════════════════════════


def run_system(command: String) raises -> Int:
    """Run `command` through `/bin/sh` in the FOREGROUND, stdin/stdout/stderr
    inherited, and return its exit status (the `wait` status's high byte).

    For launching something a person watches — a training run started by
    `project-rerun --exec` — where `run_capture` would swallow the output
    until it ended.
    """
    var c = command.copy()
    var st = external_call["system", Int32](c.as_c_string_span().ptr())
    _ = c^
    if st == -1:
        raise Error("proc: system() could not start /bin/sh")
    return Int((st >> 8) & 0xFF)


def _ignore_sigpipe():
    """Make a write to a dead pipe return an error instead of killing us.

    ⚠ THIS IS NOT OPTIONAL, AND ITS ABSENCE IS INVISIBLE UNTIL IT ISN'T.
    Writing to a pipe whose reader has exited raises `SIGPIPE`, whose DEFAULT
    disposition is to terminate the process — so an `ffmpeg` that dies on its
    first frame (a bad codec argument, a full disk, an odd frame width) would
    take the whole training or viewer process down with it, with no message
    and no stack.

    Python installs `SIG_IGN` for `SIGPIPE` at startup, which is why
    `imageio`'s writer never had to think about this and why the failure only
    appears once the Python layer is removed. With the handler ignored,
    `fwrite` returns short and sets `EPIPE`, and `write_all` below turns that
    into an error naming the command.

    `SIG_IGN` is the constant `(void (*)(int)) 1`. Calling this repeatedly is
    harmless.
    """
    _ = external_call["signal", Int](Int(SIGPIPE), Int(1))


struct WritePipe(Movable):
    """A child process this side is FEEDING through its stdin.

    The mirror of `Pipe`. Its stdout and stderr are inherited, so an `ffmpeg`
    that complains is visible rather than swallowed.
    """

    var _fp: Int
    var command: String
    var closed: Bool
    var written: Int

    def __init__(out self, var command: String) raises:
        _ignore_sigpipe()
        var mode = String("w")
        var fp = external_call["popen", Int](
            command.as_c_string_span().ptr(),
            mode.as_c_string_span().ptr(),
        )
        if fp == 0:
            raise Error("proc: popen(w) failed for: " + command)
        self._fp = fp
        self.command = command^
        self.closed = False
        self.written = 0

    def __init__(out self, *, deinit move: Self):
        self._fp = move._fp
        self.command = move.command^
        self.closed = move.closed
        self.written = move.written

    def __deinit__(deinit self):
        if not self.closed and self._fp != 0:
            _ = external_call["pclose", Int32](self._fp)

    def write_all(
        mut self, src: Pointer[Scalar[DType.uint8], MutAnyOrigin], count: Int
    ) raises:
        """Write exactly `count` bytes, or raise.

        A short write means the child is gone — with `SIGPIPE` ignored that is
        reported here instead of terminating the process.
        """
        if self.closed:
            raise Error("proc: write to a closed pipe: " + self.command)
        if count <= 0:
            return
        var done = 0
        while done < count:
            var n = external_call["fwrite", Int](
                src.unsafe_offset(done), Int(1), count - done, self._fp
            )
            if n <= 0:
                raise Error(
                    "proc: the child stopped reading after "
                    + String(self.written + done) + " bytes — it exited early."
                    " Its own message is above, if it printed one. Command: "
                    + self.command
                )
            done += n
        self.written += count

    def close(mut self) raises -> Int:
        """Close stdin and WAIT for the child to finish.

        ⚠ WAITING IS THE POINT, not a courtesy. An encoder writes its
        container trailer (an MP4's `moov` atom, a GIF's terminator) only when
        its input ends, so returning before the child exits hands the caller a
        truncated, unplayable file that looks finished.
        """
        if self.closed:
            return 0
        var status = Int(external_call["pclose", Int32](self._fp))
        self.closed = True
        self._fp = 0
        if status < 0:
            raise Error("proc: pclose failed for: " + self.command)
        var sig = status & 0x7F
        if sig != 0:
            raise Error(
                "proc: child killed by signal " + String(sig) + ": "
                + self.command
            )
        var code = (status >> 8) & 0xFF
        if code != 0:
            raise Error(
                "proc: child exited " + String(code) + ": " + self.command
            )
        return code

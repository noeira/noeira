# +--------------------------------------------------------------------------+ #
# | Chunked, atomic whole-file I/O
# +--------------------------------------------------------------------------+ #
"""`read_file_bytes` / `write_file_atomic` — the two rules every large blob
this repo writes has to obey, in one place.

## The two rules

**Chunking.** A single `write(2)` silently stops at `0x7FFFF000` (~2 GiB) on
Linux and returns the short count. Ignoring that return is how checkpoint v2
produced TRUNCATED files that loaded without complaint at DreamerV3 size200m.
`read(2)` has the same cap. So neither direction may issue one call for the
whole payload; both loop in `_CHUNK`-sized pieces.

**Atomicity.** The payload lands in `path + ".tmp"` and is `rename(2)`d over
`path` only once fully written. `rename` is atomic within a filesystem, so a
crash mid-save cannot destroy the previous good file — which for a checkpoint
is the difference between losing one save and losing the run.

## Why it is here and not in `nn/core/checkpoint.mojo`

It was there first, and correctly. It moved when `io/safetensors.mojo` needed
the same two rules: a second transcription of a rule is a second thing to keep
in step, and this repo's most frequent defect shape by a distance is a rule
written inline twice. `checkpoint.mojo` keeps its `_write_file_bytes` /
`_read_file_bytes` names as one-line forwards, so nothing that imported them
changed.
"""

from std.ffi import external_call
from std.memory import unsafe_memcpy
from std.os.path import exists
from std.sys import CompilationTarget

comptime _CHUNK = 1 << 30
"""1 GiB — comfortably below every single-syscall I/O cap."""


def rename_over(var src: String, var dst: String) raises:
    """`rename(2)`: atomic within a filesystem, which is the whole reason the
    download and the checkpoint both write `<dst>.part`/`<dst>.tmp` first."""
    var rc = external_call["rename", Int32](
        src.as_c_string_span().ptr(),
        dst.as_c_string_span().ptr(),
    )
    if rc != 0:
        raise Error("rename failed: " + src + " -> " + dst)


def remove_file(var path: String) raises:
    """`unlink(2)`. A file that was already gone is NOT an error — every
    caller here means "make sure it is not there", not "delete this"."""
    var rc = external_call["unlink", Int32](
        path.as_c_string_span().ptr()
    )
    if rc != 0 and exists(path):
        raise Error("cannot remove " + path)


def parent_dir(path: String) -> String:
    """The directory part of `path`, or "." when it has none."""
    var b = path.as_bytes()
    var cut = -1
    for i in range(path.byte_length()):
        if Int(b[i]) == 0x2F:  # "/"
            cut = i
    if cut <= 0:
        return String(".") if cut < 0 else String("/")
    return String(path[byte=0:cut])


def write_file_atomic(var path: String, ref content: List[UInt8]) raises:
    """Write `content` to `path` via `path + ".tmp"` + `rename(2)`."""
    var tmp = path + ".tmp"
    with open(tmp, "w") as f:
        var off = 0
        while off < len(content):
            var take = len(content) - off
            if take > _CHUNK:
                take = _CHUNK
            # A bounded slice rather than a raw pointer + a separate length:
            # the chunk bound is then CHECKED against the buffer instead of
            # asserted by the caller.
            f.write_bytes(Span(content)[off : off + take])
            off += take
    rename_over(tmp^, path^)


def write_text_atomic(var path: String, text: String) raises:
    """Write `text` to `path` atomically. The String half of the call above.

    ⚠⚠ THIS EXISTS BECAUSE THE THREE-LINE COPY WAS WRITTEN FIVE TIMES. Every
    caller with a String to persist reached for the same loop —
    `robot/so101/calibration`, `data/lerobot_push`, `data/lerobot_write` twice,
    and a private `_bytes` in `tools/hf/probe_push`. That is the shape
    `_a_rule_written_inline_twice_drifts` names as this repo's most frequent
    defect, and the drift it invites here is real: one copy re-derived
    `as_bytes()` inside the loop, so the cost was quadratic in the span lookup
    for anyone who copied that one next.

    ⚠ IT IS A COPY, NOT A VIEW, AND THAT IS FORCED. `write_file_atomic` takes an
    owned `List[UInt8]`; a `Span` over the String's storage would not outlive
    the chunked write below it.
    """
    var b = List[UInt8]()
    var src = text.as_bytes()
    for i in range(text.byte_length()):
        b.append(src[i])
    write_file_atomic(path^, b)


def read_file_bytes(path: String) raises -> List[UInt8]:
    """Read the whole file, looping until `read_bytes` returns nothing."""
    var out = List[UInt8]()
    with open(path, "r") as f:
        while True:
            var chunk = f.read_bytes(_CHUNK)
            if len(chunk) == 0:
                break
            var old = len(out)
            out.resize(old + len(chunk), 0)
            unsafe_memcpy(
                dest=out.unsafe_ptr().unsafe_offset(old),
                src=chunk.unsafe_ptr(),
                count=len(chunk),
            )
    return out^


def read_file_range(path: String, offset: Int, count: Int) raises -> List[UInt8]:
    """`count` bytes starting at `offset`. Raises on a short read — a caller
    asking for a tensor's bytes wants the tensor, not a prefix of it."""
    if offset < 0 or count < 0:
        raise Error(
            "read_file_range: negative offset/count (" + String(offset) + ", "
            + String(count) + ")"
        )
    var out = List[UInt8]()
    out.reserve(count)
    with open(path, "r") as f:
        _ = f.seek(offset)
        var got = 0
        while got < count:
            var want = count - got
            if want > _CHUNK:
                want = _CHUNK
            var chunk = f.read_bytes(want)
            if len(chunk) == 0:
                raise Error(
                    "read_file_range: '" + path + "' ended after "
                    + String(got) + " of " + String(count)
                    + " bytes from offset " + String(offset)
                )
            var old = len(out)
            out.resize(old + len(chunk), 0)
            unsafe_memcpy(
                dest=out.unsafe_ptr().unsafe_offset(old),
                src=chunk.unsafe_ptr(),
                count=len(chunk),
            )
            got += len(chunk)
    return out^


def release_file_cache(var path: String):
    """Tell the system it may drop the pages of `path` it keeps in memory
    (`posix_fadvise(..., POSIX_FADV_DONTNEED)` over the whole file). Linux
    only; nothing elsewhere, and nothing on a failure: it is advice.

    For a loader that reads gigabytes ONCE and keeps its own copy. A file read
    stays in the page cache until the system needs the memory back, and on a
    machine whose host and device share one memory (a Jetson) that moment is
    the device's next allocation: with 9.6 GB of weights read by four models
    loading together, free memory fell to 112 MB and one start in six lost a
    model to a cuBLAS INTERNAL_ERROR (docs/ROBOT_BRAIN_PLAN.md §16). Call it
    after each piece is on the device; the pages another reader still needs
    are simply read again.
    """
    comptime if CompilationTarget.is_linux():
        var mode = String("rb")
        var fp = external_call["fopen", Int](
            path.as_c_string_span().ptr(), mode.as_c_string_span().ptr()
        )
        if fp == 0:
            return
        var fd = external_call["fileno", Int32](fp)
        # (fd, offset, len, advice): len 0 = to the end of the file; DONTNEED = 4
        _ = external_call["posix_fadvise", Int32](fd, Int(0), Int(0), Int32(4))
        _ = external_call["fclose", Int32](fp)


def file_size(path: String) raises -> Int:
    var f = open(path, "r")
    var n = f.seek(0, 2)
    f.close()
    return Int(n)


def stdout_is_tty() -> Bool:
    """Whether stdout is a terminal.

    A redrawn, in-place table is right on a terminal and wrong everywhere
    else: piped to a file there is no cursor to move and every redraw appends,
    which is worse than the scrolling it replaced.
    """
    return external_call["isatty", Int32](Int32(1)) == 1


def stdin_is_tty() -> Bool:
    """Whether stdin is a terminal.

    ⚠ THE DISTINCTION MATTERS FOR `StdinReader.has_input`. On a terminal,
    "input is waiting" means a key was pressed. On a PIPE that has reached its
    end, `poll` returns POLLHUP — also a ready event — and `line()` then
    returns an empty string, which is indistinguishable from the operator
    pressing Enter. A tool that uses a keypress as "stop now" therefore stops
    IMMEDIATELY when run as `printf '\n' | tool`, on its first check. Gate the
    keypress affordance on this and scripted runs behave.
    """
    return external_call["isatty", Int32](Int32(0)) == 1


struct StdinReader(Movable):
    """Line-at-a-time stdin for an interactive tool.

    ⚠ **THIS EXISTS BECAUSE `input()` CANNOT BE LINKED ALONGSIDE
    `core/concurrent`, AND NEITHER CAN `read`.** Two separate symbol
    collisions, both reported against a stdlib file with none of your code in
    the message, so they look like toolchain bugs rather than collisions:

        # input()  ->  std/sys/_libc.mojo declares `free` with allockind /
        #              alloc-family; core/concurrent/ring.mojo declares the
        #              same symbol WITHOUT them.
        error: existing function with conflicting attributes ... func = "free"

        # external_call["read", Int](...)  ->  std/ffi already declares `read`
        error: existing function with conflicting signature

    `fdopen` + `fgetc` are declared by nothing else, so they link cleanly.
    Any interactive tool that also uses a background thread hits this, which
    is why the workaround lives here rather than in one example.

    ⚠ THE `FILE*` IS NEVER `fclose`d. It wraps fd 0; closing it would close
    the process's standard input. Opening it once per prompt instead would
    leak a stdio buffer each time, which is why this is a struct the caller
    holds rather than a free function.
    """

    var _fp: Int

    def __init__(out self) raises:
        var mode = String("r")
        var fp = external_call["fdopen", Int](
            Int32(0), mode.as_c_string_span().ptr()
        )
        if fp == 0:
            raise Error("fileio: cannot open stdin")
        self._fp = fp

    def __init__(out self, *, deinit move: Self):
        self._fp = move._fp

    def has_input(self) -> Bool:
        """True when a line is waiting, without blocking to find out.

        ⚠ `poll(2)`, NOT `fcntl`. Making stdin non-blocking would be the
        obvious route and `fcntl(fd, cmd, ...)` is C-VARIADIC — on Apple
        arm64 variadic arguments go on the stack while `external_call` emits
        a fixed prototype, so the flags argument arrives as garbage. `poll`
        has a fixed signature and needs no shim.

        `struct pollfd` is `{int fd; short events; short revents}` — eight
        bytes, laid out here as two `Int32`s with `events` in the low half of
        the second.
        """
        var pfd = Array[Int32, 2](fill=0)
        pfd[0] = Int32(0)  # fd 0
        pfd[1] = Int32(0x0001)  # events = POLLIN
        var rc = external_call["poll", Int32](
            Pointer(to=pfd[0]), UInt64(1), Int32(0)
        )
        return rc > 0

    def discard_pending(self):
        """Throw away anything already typed but not yet read.

        ⚠ WITHOUT THIS, A KEYSTROKE DURING A LONG OPERATION SILENTLY ANSWERS
        THE NEXT PROMPT. In a real recording session the operator pressed `q`
        while an episode was still running; the tty buffered it, and the
        "keep this episode?" prompt read it as the answer and ended the run
        three episodes into five. Nothing was lost, but the run stopped for a
        reason that was invisible.

        `tcflush(0, TCIFLUSH)` is the discard, and it is a no-op on a pipe —
        which is right: a scripted `printf '...' | tool` MUST keep its input.
        """
        _ = external_call["tcflush", Int32](Int32(0), Int32(1))  # TCIFLUSH

    def line(mut self, max_bytes: Int = 1024) raises -> String:
        """One line, without its newline. Empty at EOF."""
        var buf = List[UInt8]()
        while len(buf) < max_bytes:
            var c = Int(external_call["fgetc", Int32](self._fp))
            if c < 0:
                break  # EOF
            if c == 0x0A:
                break
            if c == 0x0D:
                continue  # a CRLF terminal
            buf.append(UInt8(c))
        if len(buf) == 0:
            return String("")
        buf.append(0)
        return String(unsafe_from_utf8_ptr=buf.unsafe_ptr())

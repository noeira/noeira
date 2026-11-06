# +--------------------------------------------------------------------------+ #
# | noeira serial — a raw-mode tty over libc
# +--------------------------------------------------------------------------+ #
"""`SerialPort`: open a tty, put it in 8N1 raw mode at an arbitrary baud, and
read/write bytes against a deadline.

Deliberately knows nothing about servos. The Feetech packet layer sits on top
in `noeira/robot/feetech/`, and the same fd plumbing is what a native socket
sink would reuse (`docs/RERUN_ASSESSMENT.md` §5.5).

Two facts that cost an afternoon each, both measured 2026-08-25 on Darwin
25.5 / arm64 against a real SO-101 bus:

* **Use `/dev/cu.*`, not `/dev/tty.*`.** The `tty.` callin device blocks in
  `open` waiting for carrier detect. `cu.` is the callout device and does not.
  (pyserial hides this by always passing `O_NONBLOCK`; so do we, but the
  device choice still matters for anything that later clears the flag.)
* **`tcsetattr` REJECTS a literal 1 000 000 with EINVAL** on macOS, even
  though BSD's `Bxxx` constants *are* the baud numbers (`B230400 == 230400`).
  Only `IOSSIOSPEED` gets there — see `native.mojo` for why that needs C.
"""

from std.ffi import external_call
from std.sys import CompilationTarget
from std.time import perf_counter_ns

from noeira.io.serial.native import (
    baud_constant, layout_from_headers, layout_names, set_speed,
)

# ═══════════════════════════════════════════════════════════════════════════
# libc constants — Darwin/arm64 and Linux/glibc
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠⚠ ALMOST NOTHING IS SHARED BETWEEN THE TWO. It is not only the struct
# layout: `tcflag_t` is 64-bit on Darwin and 32-bit on Linux, VMIN/VTIME swap
# places, O_NOCTTY and O_NONBLOCK differ, EAGAIN is 35 against 11, AT_FDCWD is
# -2 against -100, errno lives behind a different symbol, and a Linux Bxxx is
# a small ORDINAL where a BSD one is the baud number itself. A table that got
# any one of these right by accident is the dangerous case, which is why
# `native/nra_serial.c` reports all of them from the real headers and
# `assert_layout` refuses to open a port that disagrees.

comptime IS_MAC = CompilationTarget.is_macos()

comptime AT_FDCWD = -2 if IS_MAC else -100
comptime O_RDWR = 2
comptime O_NOCTTY = 131072 if IS_MAC else 0o400
comptime O_NONBLOCK = 4 if IS_MAC else 0o4000

comptime TCSANOW = 0
comptime TCIOFLUSH = 3 if IS_MAC else 2

# `struct termios`, from a C `offsetof` probe rather than assumed.
#
#   Darwin: 4 x tcflag_t(8) | cc_t[NCCS=20] | 4 pad | c_ispeed | c_ospeed = 72
#   Linux:  4 x tcflag_t(4) | c_line(1) | cc_t[NCCS=32] | 3 pad
#                                               | c_ispeed | c_ospeed  = 60
#
# A wrong offset here is SILENT — it writes into c_cc and the port merely
# misbehaves — so these are gated twice: `assert_layout` checks them against
# the shim's `offsetof` at every open, and
# `tests/robot/test_serial_termios_layout.mojo` checks the speed offsets
# against libc's own `cfgetospeed` on a pty.
comptime TERMIOS_SIZE = 72 if IS_MAC else 60
comptime OFF_IFLAG = 0
comptime OFF_OFLAG = 8 if IS_MAC else 4
comptime OFF_CFLAG = 16 if IS_MAC else 8
comptime OFF_LFLAG = 24 if IS_MAC else 12
comptime OFF_CC = 32 if IS_MAC else 17
comptime OFF_ISPEED = 56 if IS_MAC else 52
comptime OFF_OSPEED = 64 if IS_MAC else 56
comptime TCFLAG_SIZE = 8 if IS_MAC else 4
comptime NCCS = 20 if IS_MAC else 32
# ⚠ THESE TWO ARE SWAPPED BETWEEN THE PLATFORMS, not merely different.
comptime VMIN = 16 if IS_MAC else 6
comptime VTIME = 17 if IS_MAC else 5

comptime CSIZE = 0x300 if IS_MAC else 0o60
comptime CS8 = 0x300 if IS_MAC else 0o60
comptime CLOCAL = 0x8000 if IS_MAC else 0o4000
comptime CREAD = 0x800 if IS_MAC else 0o200
comptime PARENB = 0x1000 if IS_MAC else 0o400
comptime CSTOPB = 0x400 if IS_MAC else 0o100
comptime CRTSCTS = 0x30000 if IS_MAC else 0x80000000

comptime EINTR = 4
comptime EAGAIN = 35 if IS_MAC else 11


def errno() -> Int32:
    # ⚠ DIFFERENT SYMBOL, SAME MEANING. glibc has no `__error`; a build that
    # picked the wrong one fails to link rather than misreporting, which is
    # the one merciful failure in this file.
    comptime if IS_MAC:
        return external_call["__error", Pointer[Int32, MutAnyOrigin]]()[]
    else:
        return external_call[
            "__errno_location", Pointer[Int32, MutAnyOrigin]
        ]()[]


def _read_flag(p: Pointer[UInt8, MutAnyOrigin], off: Int) -> UInt64:
    """One `tcflag_t`, whatever width this platform gives it."""
    comptime if TCFLAG_SIZE == 8:
        return p.unsafe_offset(off).unsafe_bitcast[UInt64]()[]
    else:
        return UInt64(p.unsafe_offset(off).unsafe_bitcast[UInt32]()[])


def _write_flag(p: Pointer[UInt8, MutAnyOrigin], off: Int, v: UInt64):
    comptime if TCFLAG_SIZE == 8:
        p.unsafe_offset(off).unsafe_bitcast[UInt64]()[] = v
    else:
        p.unsafe_offset(off).unsafe_bitcast[UInt32]()[] = UInt32(v)


def _open_advice(e: Int) -> String:
    """What errno actually means for a tty, on THIS platform.

    ⚠ THE OLD MESSAGE GAVE macOS ADVICE ON EVERY PLATFORM ("prefer /dev/cu.*
    over /dev/tty.*"), which on the board is not merely useless — it points
    away from the answer. EACCES on Linux is a GROUP problem and nothing about
    the cable, and it is the first thing a new board hits.
    """
    comptime if IS_MAC:
        if e == 13:
            return String(
                "permission denied. Another process may hold the port, or the"
                " device needs different permissions."
            )
        if e == 2:
            return String(
                "no such device — is the arm plugged in? Prefer /dev/cu.* over"
                " /dev/tty.*: the tty.* callin device blocks in open() waiting"
                " for carrier detect."
            )
        if e == 16:
            return String("the port is busy — another process has it open.")
        return String("is the arm plugged in?")
    else:
        if e == 13:
            return String(
                "permission denied (EACCES). On Linux a tty belongs to the"
                " `dialout` group, and this user is not in it:\n"
                "      sudo usermod -aG dialout $USER\n"
                "  then log out and back in (or `newgrp dialout` for this"
                " shell only — a fresh login is what makes it stick).\n"
                "  `ls -l " + String("$(readlink -f <the port>)") + "` shows"
                " the owning group."
            )
        if e == 2:
            return String(
                "no such device (ENOENT). On the board these are udev"
                " symlinks: check `ls -l /dev/soarm_*` and"
                " /etc/udev/rules.d/99-soarm.rules"
                " (docs/JETSON_DEPLOYMENT.md §3)."
            )
        if e == 16:
            return String(
                "the port is busy (EBUSY) — another process has it open."
                " ModemManager grabs new ttyACM devices on some distros:"
                " `systemctl status ModemManager`."
            )
        return String("is the arm plugged in?")


def raw_speed_at(p: Pointer[UInt8, MutAnyOrigin], off: Int) -> Int:
    """The raw `speed_t` at `off` — a Bxxx ORDINAL on Linux, the baud on BSD.

    `speed_t` is as wide as `tcflag_t` on both platforms, so one width test
    serves. Raw on purpose: the gate compares this against `cfgetospeed`, and
    decoding either side first would make them agree for the wrong reason.
    """
    comptime if TCFLAG_SIZE == 8:
        return Int(p.unsafe_offset(off).unsafe_bitcast[UInt64]()[])
    else:
        return Int(p.unsafe_offset(off).unsafe_bitcast[UInt32]()[])


def expected_layout() -> List[Int]:
    """This file's constants, in the shim table's order."""
    return [
        TERMIOS_SIZE, OFF_IFLAG, OFF_OFLAG, OFF_CFLAG, OFF_LFLAG, OFF_CC,
        TCFLAG_SIZE, NCCS, VMIN, VTIME, CSIZE, CS8, CLOCAL, CREAD, PARENB,
        CSTOPB, CRTSCTS, TCSANOW, TCIOFLUSH, O_NOCTTY, O_NONBLOCK, EAGAIN,
        AT_FDCWD,
    ]


def assert_layout() raises:
    """Refuse to touch a tty if our constants disagree with `<termios.h>`.

    ⚠ AT EVERY OPEN, NOT ONLY IN A TEST. A port is opened once per process and
    the check is a dlsym plus one call, so the cost is nothing next to what it
    prevents: an offset that is wrong by eight bytes configures the port with
    garbage, opens anyway, and surfaces as unexplained bus timeouts. The names
    come back with the numbers so the message says WHICH field is wrong.
    """
    var actual = layout_from_headers()
    var want = expected_layout()
    var names = layout_names()
    var bad = String("")
    for i in range(len(want)):
        if actual[i] != want[i]:
            bad += (
                "\n  " + names[i] + ": libc says " + String(actual[i])
                + ", port.mojo has " + String(want[i])
            )
    if bad.byte_length() > 0:
        raise Error(
            "serial: `struct termios` on this platform does not match the"
            " layout `noeira/io/serial/port.mojo` was written for."
            + bad
            + "\nThe C headers are right and the Mojo constants are wrong."
            " Fix them there — do NOT relax this check."
        )


# ═══════════════════════════════════════════════════════════════════════════
# SerialPort
# ═══════════════════════════════════════════════════════════════════════════


struct SerialPort(Movable):
    """An open tty in raw mode. Closes itself when it goes out of scope."""

    var fd: Int32
    var baud: Int
    var _path: String

    def __init__(out self, var path: String, baud: Int = 1000000) raises:
        """Open and configure. Raises rather than returning a bad fd, so a
        caller that got a `SerialPort` has a usable one."""
        # ⚠⚠ THE LAYOUT IS CHECKED AGAINST THE HEADERS BEFORE THE FIRST
        # `open`, NOT ASSUMED. This used to be a `comptime assert` pinning the
        # whole file to Darwin, with the note "do not guess the offsets" — the
        # right instinct, because a wrong offset writes into `c_cc` and the
        # port still opens. Guessing is still not allowed; the difference is
        # that `assert_layout` MEASURES, so Linux is supported on the same
        # terms Darwin always was rather than on a remembered table.
        assert_layout()

        self._path = path^
        self.baud = baud
        # ⚠ `openat`, NOT `open`, and the reason is a Mojo linking trap.
        # `external_call` re-declares a C symbol per module, and a SECOND
        # declaration of the same symbol with a different signature fails at
        # LLVM lowering — "existing function with conflicting signature". The
        # stdlib already declares `open` (`std/io/file.mojo:141`), and this
        # module lands in the same binary as it whenever anything touches the
        # filesystem, which the viewer does. Matching its signature exactly
        # did NOT resolve it. `openat` is the same call with an explicit
        # directory fd, nothing else declares it, and AT_FDCWD makes it
        # identical to `open` for an absolute path.
        #
        # The mode argument is harmless whichever way it travels: C `openat`
        # is variadic, so on Apple arm64 it lands in a register the callee
        # will not read — and the kernel only reads `mode` under O_CREAT,
        # which is never set here.
        self.fd = external_call["openat", Int32](
            Int32(AT_FDCWD),
            self._path.as_c_string_span().ptr(),
            Int32(O_RDWR | O_NOCTTY | O_NONBLOCK),
            Int32(0),
        )
        if self.fd < 0:
            var e = errno()
            raise Error(
                "serial: open(" + self._path + ") failed, errno=" + String(e)
                + " — " + _open_advice(Int(e))
            )
        try:
            self._configure()
        except e:
            _ = external_call["close", Int32](self.fd)
            self.fd = -1
            raise e

    def __deinit__(deinit self):
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)

    def _configure(mut self) raises:
        var tio = Array[UInt8, TERMIOS_SIZE](fill=0)
        var p = tio.unsafe_ptr()
        if external_call["tcgetattr", Int32](self.fd, p) != 0:
            raise Error("serial: tcgetattr failed, errno=" + String(errno()))

        external_call["cfmakeraw", NoneType](p)
        var cflag = p.unsafe_offset(OFF_CFLAG).unsafe_bitcast[UInt64]()
        cflag[] = (cflag[] & ~UInt64(CSIZE)) | UInt64(CS8 | CLOCAL | CREAD)
        cflag[] = cflag[] & ~UInt64(PARENB | CSTOPB | CRTSCTS)
        # Fully non-blocking: `read` returns whatever is there, right now.
        # The deadline lives in `read_bytes`, not in the driver, so one
        # timeout policy governs the whole stack.
        p[unsafe_offset = OFF_CC + VMIN] = 0
        p[unsafe_offset = OFF_CC + VTIME] = 0

        # ⚠ THE TWO PLATFORMS REACH THE SAME BAUD BY DIFFERENT ROUTES.
        # Darwin's termios tops out at B230400 and REJECTS a literal 1000000
        # with EINVAL, so the struct is parked at a speed it accepts and
        # IOSSIOSPEED sets the real one afterwards. Linux has B1000000 and
        # takes it here, with no ioctl at all — but as an ORDINAL (4104), not
        # as the number, which is also what `cfgetospeed` hands back.
        comptime if IS_MAC:
            _ = external_call["cfsetispeed", Int32](p, UInt64(9600))
            _ = external_call["cfsetospeed", Int32](p, UInt64(9600))
        else:
            var code = baud_constant(self.baud)
            if code < 0:
                raise Error(
                    "serial: this libc has no Bxxx constant for "
                    + String(self.baud)
                    + " baud. A rate outside the standard table needs"
                    " TCSETS2/BOTHER, which the shim does not implement yet."
                )
            if (
                external_call["cfsetispeed", Int32](p, UInt32(code)) != 0
                or external_call["cfsetospeed", Int32](p, UInt32(code)) != 0
            ):
                raise Error(
                    "serial: cfsetspeed rejected " + String(self.baud)
                    + " (B=" + String(code) + "), errno=" + String(errno())
                )
        if external_call["tcsetattr", Int32](self.fd, Int32(TCSANOW), p) != 0:
            raise Error("serial: tcsetattr failed, errno=" + String(errno()))

        if set_speed(self.fd, self.baud) != 0:
            raise Error(
                "serial: could not set "
                + String(self.baud)
                + " baud, errno="
                + String(errno())
            )

        # Non-vacuity: confirm the driver took the speed rather than trusting
        # a 0 return. A shim that silently no-ops would otherwise leave the
        # port at 9600 and every read would time out with no explanation.
        if self.speed() != self.baud:
            raise Error(
                "serial: baud readback is "
                + String(self.speed())
                + ", asked for "
                + String(self.baud)
            )

    def speed(mut self) -> Int:
        """The output line speed the driver reports, IN BITS PER SECOND.

        ⚠ NOT THE RAW FIELD ON LINUX. There `c_ospeed` holds the `Bxxx`
        ordinal (4104 for 1 Mbaud), so returning it raw would make the
        readback check in `_configure` compare 4104 against 1000000 and refuse
        every port. The units are the caller's, on both platforms.
        """
        var tio = Array[UInt8, TERMIOS_SIZE](fill=0)
        var p = tio.unsafe_ptr()
        _ = external_call["tcgetattr", Int32](self.fd, p)
        var raw = raw_speed_at(p.as_unsafe_any_origin(), OFF_OSPEED)
        comptime if IS_MAC:
            # BSD's Bxxx constants ARE the baud numbers, and IOSSIOSPEED
            # writes the true rate here, so the field needs no decoding.
            return raw
        else:
            # The inverse of `baud_constant`, over the rates anything on this
            # bus uses. Unknown ordinals come back as themselves, which the
            # caller's comparison then reports rather than hiding.
            for b in [
                Int(9600), Int(19200), Int(38400), Int(57600), Int(115200),
                Int(230400), Int(460800), Int(500000), Int(576000),
                Int(921600), Int(1000000), Int(1152000), Int(1500000),
                Int(2000000), Int(3000000), Int(4000000),
            ]:
                try:
                    if baud_constant(b) == raw:
                        return b
                except:
                    break
            return raw

    def flush(mut self):
        """Discard everything queued in BOTH directions.

        Call before a request whose reply you intend to parse: a half-read
        packet from a previous timeout would otherwise be mistaken for this
        one's header.
        """
        _ = external_call["tcflush", Int32](self.fd, Int32(TCIOFLUSH))

    def write_bytes[
        mut: Bool, //, o: Origin[mut=mut]
    ](mut self, buf: Span[UInt8, o]) raises -> Int:
        # ⚠ `Int32` fd, like `read` below: since Mojo 1.2.0 the stdlib declares
        # `write` with an `Int32` fd, and an `Int` fd fails at LLVM lowering
        # in any binary that also writes a file.
        var n = external_call["write", Int](
            Int32(self.fd), buf.unsafe_ptr(), len(buf)
        )
        if n < 0:
            raise Error("serial: write failed, errno=" + String(errno()))
        return n

    def read_bytes[
        o: MutOrigin
    ](
        mut self, buf: Span[UInt8, o], want: Int, timeout_ms: Int
    ) raises -> Int:
        """Fill up to `want` bytes, returning early only on the deadline.

        Returns how many arrived — a short read is normal and is the caller's
        to interpret, because only the packet layer knows how long the reply
        should have been.
        """
        if want > len(buf):
            raise Error(
                "serial: read_bytes wants "
                + String(want)
                + " into a "
                + String(len(buf))
                + "-byte span"
            )
        var deadline = perf_counter_ns() + timeout_ms * 1_000_000
        var got = 0
        while got < want:
            # ⚠ `Int32` fd: the stdlib declares `read` with an `Int32` fd since Mojo 1.2.0 (an `Int` fd fails at LLVM lowering,
            # "existing function with conflicting signature", in any binary that also reads a file).
            # Before 1.2 it was an `index` fd (`std/io/file_descriptor.mojo:117`).
            var n = external_call["read", Int](
                Int32(self.fd), buf.unsafe_ptr().unsafe_offset(got), want - got
            )
            if n > 0:
                got += n
                continue
            if n < 0:
                var e = errno()
                if e != Int32(EAGAIN) and e != Int32(EINTR):
                    raise Error("serial: read failed, errno=" + String(e))
            if perf_counter_ns() >= deadline:
                break
            # 100 us: two orders of magnitude under a 1 Mbaud packet's own
            # transit time, so the poll never dominates the round trip.
            _ = external_call["usleep", Int32](UInt32(100))
        return got

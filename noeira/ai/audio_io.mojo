# +--------------------------------------------------------------------------+ #
# | Microphone and speaker for demos — through ffmpeg / afplay
# +--------------------------------------------------------------------------+ #
"""Record a clip, play a clip. Prototype-grade, on purpose.

    record_wav("/tmp/cmd.wav", seconds=4.0)     # blocks while recording
    play_wav("/tmp/reply.wav")                  # blocks while playing
    speak_local("Cube placed.")                 # OS voice, no API

    var mic = MicCapture.start()                # continuous, 16 kHz mono
    while running:
        var chunk = mic.read()                  # NEVER blocks; [] if nothing new
        if rms(chunk) > 0.02: ...               # the caller's VAD
    mic.stop()

Both shell out: `ffmpeg` (already required on PATH for the video decoder)
records, `afplay` (macOS) / `aplay` (Linux) plays. That is the shortest path
to a push-to-talk demo and nothing more: no streaming, no voice-activity
detection, fixed-length takes.

⚠ THE REAL-TIME PATH IS SDL3, NOT THIS. `noeira/render/sdl/sdl_audio.mojo`
already binds SDL3's audio streams (capture + playback), and the renderer
already runs an SDL event loop; a demo that must keep drawing while it
listens belongs there. This module is for the first version of a demo.

⚠ macOS ASKS FOR MICROPHONE PERMISSION the first time, on behalf of the
TERMINAL app, not of this binary. What a DENIAL looks like is not verified
here (no one denied it on this machine): expect either an ffmpeg error or a
clip of exact zeros — `MicCapture.digital_silence()` tells the second case
apart from a quiet room.
"""

from std.ffi import external_call
from std.sys import CompilationTarget
from std.time import perf_counter_ns

from noeira.core.bytes import string_from_byte_span
from noeira.io.fileio import read_file_bytes, remove_file
from noeira.io.proc import Pipe, quote_arg, run_system


def record_wav(
    path: String,
    seconds: Float64,
    sample_rate: Int = 16000,
    var device: String = String(""),
) raises:
    """Record `seconds` of mono 16-bit audio from the default input.

    `device`: macOS avfoundation index as `":1"` (list them with
    `ffmpeg -f avfoundation -list_devices true -i ""`); Linux ALSA name.
    """
    var input: String
    comptime if CompilationTarget.is_macos():
        if device.byte_length() == 0:
            device = String(":default")
        input = "-f avfoundation -i " + quote_arg(device)
    else:
        if device.byte_length() == 0:
            device = String("default")
        input = "-f alsa -i " + quote_arg(device)
    var cmd = (
        "ffmpeg -hide_banner -loglevel error -y " + input
        + " -t " + String(seconds) + " -ac 1 -ar " + String(sample_rate)
        + " -sample_fmt s16 " + quote_arg(path)
    )
    var rc = run_system(cmd)
    if rc != 0:
        raise Error("record_wav: ffmpeg exited " + String(rc) + ": " + cmd)


def play_wav(path: String) raises:
    var cmd: String
    comptime if CompilationTarget.is_macos():
        cmd = "afplay " + quote_arg(path)
    else:
        cmd = "aplay -q " + quote_arg(path)
    var rc = run_system(cmd)
    if rc != 0:
        raise Error("play_wav: exited " + String(rc) + ": " + cmd)


def speak_local(text: String, var voice: String = String("")) raises:
    """Say `text` through the OS voice: `say` (macOS) / `espeak` (Linux).

    No key, no network, ~instant — the placeholder TTS for a demo until a
    `TextToSpeech` service is wired. ⚠ `say` picks the SYSTEM LANGUAGE's
    voice by default, so English text on a French Mac comes out with a French
    accent; pass `voice` ("Daniel", "Samantha" — `say -v '?'` lists them).
    """
    var cmd: String
    comptime if CompilationTarget.is_macos():
        cmd = String("say ")
        if voice.byte_length() > 0:
            cmd += "-v " + quote_arg(voice) + " "
    else:
        cmd = String("espeak ")
        if voice.byte_length() > 0:
            cmd += "-v " + quote_arg(voice) + " "
    cmd += quote_arg(text)
    var rc = run_system(cmd)
    if rc != 0:
        raise Error("speak_local: exited " + String(rc))


# ═══════════════════════════════════════════════════════════════════════════
# Continuous capture — for voice-activity detection in a real-time loop
# ═══════════════════════════════════════════════════════════════════════════


def rms(ref samples: List[Int16]) -> Float64:
    """Root-mean-square level in [0, 1] (full-scale sine = 0.707). The usual
    VAD input: speech at a laptop mic sits around 0.02-0.2, a quiet room
    below 0.01 — calibrate on the actual room, the numbers move a lot."""
    if len(samples) == 0:
        return 0.0
    var acc = 0.0
    for i in range(len(samples)):
        var v = Float64(samples[i]) / 32768.0
        acc += v * v
    return (acc / Float64(len(samples))) ** 0.5


def _s16(lo: Int, hi: Int) -> Int16:
    """Little-endian bytes -> signed 16-bit, two's complement done by hand."""
    var v = lo | (hi << 8)
    return Int16(v - 65536 if v >= 32768 else v)


struct Pcm16Decoder(Movable):
    """s16le bytes -> samples, across chunks that may end MID-SAMPLE.

    A pipe read returns whatever is there, so a chunk can end on the low
    byte of a sample; that byte is held and paired with the next chunk's
    first. Separate from `MicCapture` so the odd-split path can be gated
    directly — ffmpeg writes whole samples, so a live capture almost never
    exercises it.
    """

    var carry: Int
    """The held low byte, or -1."""

    def __init__(out self):
        self.carry = -1

    def __init__(out self, *, deinit move: Self):
        self.carry = move.carry

    def feed(mut self, ref data: List[UInt8], n: Int, mut out: List[Int16]):
        """Append the samples in `data[:n]` to `out`."""
        var i = 0
        if self.carry >= 0 and n > 0:
            out.append(_s16(self.carry, Int(data[0])))
            self.carry = -1
            i = 1
        while i + 1 < n:
            out.append(_s16(Int(data[i]), Int(data[i + 1])))
            i += 2
        if i < n:
            self.carry = Int(data[i])


struct MicCapture(Movable):
    """A microphone that streams: one long-lived `ffmpeg`, read without
    blocking.

        var mic = MicCapture.start(16000)
        while running:
            var pcm = mic.read()          # new 16-bit mono samples, maybe none
            vad.feed(pcm)
            step_and_draw()
        mic.stop()

    Built for a VAD-gated voice loop: transcribe only the stretches where
    someone speaks, instead of paying a speech-to-text API for silence. A
    fixed-length `record_wav` cannot do that — ffmpeg's ~100 ms startup
    clips the start of every take, which is where a wake word is.

    ⚠ KEEP A PRE-ROLL OF ~500 ms. A level-threshold VAD fires 100-300 ms
    AFTER speech starts — the onset of a word is quieter than its vowel — so
    a segment cut at the trigger loses the first word. With a wake word that
    first word IS the wake word, and the symptom is "the robot ignores me
    when I start with 'Robot'", which sends people debugging the STT or the
    matcher. Keep the last ~500 ms of samples in a ring and prepend them to
    every segment.

    ⚠ WHY IT STOPPED IS IN THE ERROR. ffmpeg's stderr goes to a per-capture
    log, and when the stream ends `read` raises with its last lines — a
    missing device, a bad input, a refused permission each say so there.
    `digital_silence()` covers the case that raises nothing: a stream of
    EXACT zeros, which a live microphone never produces (its noise floor
    alone moves the low bits) — a muted or disabled input, or plausibly a
    denied permission. A quiet room is NOT digital silence.

    ⚠ READ EVERY FRAME, OR AT LEAST OFTEN. ffmpeg blocks once the pipe is full
    (64 KiB on macOS = 2 s at 16 kHz); after that, the capture device drops
    audio. `read` drains everything waiting on each call.

    A process that dies without `stop()` (an unhandled error skips
    destructors) leaves no recorder behind: ffmpeg's next write hits the
    closed pipe and it exits on EPIPE.

    `input_args` replaces the capture device with any ffmpeg input — the
    gate uses `-re -f lavfi -i sine=...` so it can run with no microphone
    and no permission prompt.
    """

    var sample_rate: Int
    var pid: Int
    """ffmpeg's pid: the shell prints `$$` and then `exec`s ffmpeg, which
    keeps it. `stop` needs it — closing the pipe alone would leave `pclose`
    waiting on a child that only notices at its next write, and a stalled
    device never writes."""
    var command: String
    var samples_read: Int
    var ended: Bool
    var _pipe: Pipe
    var _buf: List[UInt8]
    var _pcm: Pcm16Decoder
    var _any_nonzero: Bool
    var log_path: String
    """ffmpeg's stderr for this capture; its tail goes into `read`'s error."""

    def __init__(
        out self,
        sample_rate: Int,
        var command: String,
        var pipe: Pipe,
        pid: Int,
        var log_path: String,
    ):
        self.sample_rate = sample_rate
        self.pid = pid
        self.command = command^
        self.samples_read = 0
        self.ended = False
        self._pipe = pipe^
        self._buf = List[UInt8]()
        self._buf.resize(1 << 16, 0)
        self._pcm = Pcm16Decoder()
        self._any_nonzero = False
        self.log_path = log_path^

    def __init__(out self, *, deinit move: Self):
        self.sample_rate = move.sample_rate
        self.pid = move.pid
        self.command = move.command^
        self.samples_read = move.samples_read
        self.ended = move.ended
        self._pipe = move._pipe^
        self._buf = move._buf^
        self._pcm = move._pcm^
        self._any_nonzero = move._any_nonzero
        self.log_path = move.log_path^

    def __deinit__(deinit self):
        # Kill BEFORE the pipe's own destructor runs its `pclose`, which would
        # otherwise wait for a child that is still recording. SIGKILL, not
        # SIGTERM: on TERM ffmpeg shuts down "gracefully" — it flushes and
        # writes a trailer into the pipe we are closing, and prints a screen
        # of "Broken pipe" errors. A raw capture has nothing to flush.
        if self.pid > 0:
            _ = external_call["kill", Int32](Int32(self.pid), Int32(9))

    @staticmethod
    def start(
        sample_rate: Int = 16000,
        var device: String = String(""),
        var input_args: String = String(""),
    ) raises -> MicCapture:
        """Start capturing. `device` as for `record_wav`; `input_args`, when
        given, is used verbatim as the ffmpeg input instead."""
        var input: String
        if input_args.byte_length() > 0:
            input = input_args^
        else:
            comptime if CompilationTarget.is_macos():
                if device.byte_length() == 0:
                    device = String(":default")
                input = "-f avfoundation -i " + quote_arg(device)
            else:
                if device.byte_length() == 0:
                    device = String("default")
                input = "-f alsa -i " + quote_arg(device)
        # -nostdin: ffmpeg must not read the terminal (it would eat keys).
        # -fflags nobuffer + -flush_packets 1: push each packet down the pipe
        # as it is captured instead of filling a muxer buffer first.
        var ff = (
            "ffmpeg -hide_banner -loglevel error -nostdin -fflags nobuffer "
            + input + " -ac 1 -ar " + String(sample_rate)
            + " -f s16le -flush_packets 1 pipe:1"
        )
        var log_path = "/tmp/noeira_mic_" + String(perf_counter_ns()) + ".log"
        var pipe = Pipe("echo $$; exec " + ff + " 2> " + quote_arg(log_path))
        # The pid line, read RAW, byte by byte: a stdio read would pull PCM
        # into the FILE* buffer, where `poll` cannot see it.
        var fd = pipe.fileno()
        var pid = 0
        var one = SIMD[DType.uint8, 1](0)
        for _ in range(24):
            var n = external_call["read", Int](fd, Pointer(to=one), Int(1))  # Int fd: see Pipe.read_available
            if n != 1:
                raise Error("mic: the capture shell died before starting ffmpeg")
            var c = Int(one[0])
            if c == 0x0A:
                break
            if c < 0x30 or c > 0x39:
                raise Error("mic: unexpected output before the pid: " + ff)
            pid = pid * 10 + (c - 0x30)
        if pid <= 0:
            raise Error("mic: could not read ffmpeg's pid")
        return MicCapture(sample_rate, ff^, pipe^, pid, log_path^)

    def read(mut self) raises -> List[Int16]:
        """Every sample captured since the last call — possibly none. Never
        blocks. Raises once ffmpeg has exited on its own (no device, no
        permission, a bad `input_args`; its message is on stderr)."""
        var out = List[Int16]()
        if self.ended:
            raise Error("mic: capture has ended: " + self.command)
        for _ in range(64):  # bounded: a pipe holds at most a few buffers
            var n = self._pipe.read_available(
                rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](Pointer(to=self._buf[0])),
                len(self._buf),
            )
            if n < 0:
                self.ended = True
                self.pid = 0
                raise Error(
                    "mic: ffmpeg stopped producing audio: " + self.ffmpeg_log()
                    + "\n  (on macOS, also check the terminal's microphone"
                    " permission)\n  command: " + self.command
                )
            if n == 0:
                break
            self._pcm.feed(self._buf, n, out)
        if not self._any_nonzero:
            for i in range(len(out)):
                if out[i] != 0:
                    self._any_nonzero = True
                    break
        self.samples_read += len(out)
        return out^

    def digital_silence(self, min_seconds: Float64 = 0.5) -> Bool:
        """True when at least `min_seconds` have been read and EVERY sample
        was exactly zero — no live microphone does that. Check it once,
        early, and tell the user to look at the input device and the mic
        permission instead of waiting for speech that cannot arrive."""
        return self.seconds_read() >= min_seconds and not self._any_nonzero

    def ffmpeg_log(self) -> String:
        """The last lines ffmpeg wrote to stderr ("" when none)."""
        try:
            var b = read_file_bytes(self.log_path)
            var start = len(b) - 600 if len(b) > 600 else 0
            var t = string_from_byte_span(b, start, len(b))
            return String(t.strip())
        except:
            return String("")

    def seconds_read(self) -> Float64:
        return Float64(self.samples_read) / Float64(self.sample_rate)

    def stop(mut self) raises:
        """Kill ffmpeg (SIGKILL — see `__deinit__`) and reap it. Idempotent."""
        if self.pid > 0:
            _ = external_call["kill", Int32](Int32(self.pid), Int32(9))
            self.pid = 0
        if not self.ended:
            self.ended = True
            try:
                _ = self._pipe.close(allow_broken_pipe=True)
            except:
                pass  # killed on purpose: its exit status says so, not a failure
        try:
            remove_file(self.log_path)
        except:
            pass


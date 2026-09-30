# +--------------------------------------------------------------------------+ #
# | Microphone and speaker for demos — through ffmpeg / afplay
# +--------------------------------------------------------------------------+ #
"""Record a clip, play a clip. Prototype-grade, on purpose.

    record_wav("/tmp/cmd.wav", seconds=4.0)     # blocks while recording
    play_wav("/tmp/reply.wav")                  # blocks while playing
    speak_local("Cube placed.")                 # OS voice, no API

Both shell out: `ffmpeg` (already required on PATH for the video decoder)
records, `afplay` (macOS) / `aplay` (Linux) plays. That is the shortest path
to a push-to-talk demo and nothing more: no streaming, no voice-activity
detection, fixed-length takes.

⚠ THE REAL-TIME PATH IS SDL3, NOT THIS. `noeira/render/sdl/sdl_audio.mojo`
already binds SDL3's audio streams (capture + playback), and the renderer
already runs an SDL event loop; a demo that must keep drawing while it
listens belongs there. This module is for the first version of a demo.

⚠ macOS ASKS FOR MICROPHONE PERMISSION the first time, on behalf of the
TERMINAL app, not of this binary. A denied prompt produces a silent clip,
not an error — check `wav.to_float32()` has energy before blaming the STT.
"""

from std.sys import CompilationTarget

from noeira.io.proc import quote_arg, run_system


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

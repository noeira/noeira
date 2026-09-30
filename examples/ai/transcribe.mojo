# +--------------------------------------------------------------------------+ #
# | Speech to text — a file, or the microphone
# +--------------------------------------------------------------------------+ #
"""Transcribe a WAV file, or record one first.

    pixi run mojo run -I . examples/ai/transcribe.mojo command.wav
    pixi run mojo run -I . examples/ai/transcribe.mojo --record 4
    pixi run mojo run -I . examples/ai/transcribe.mojo command.wav groq

Second argument picks the service: `hf` (default, `HF_TOKEN`), `openai`,
`groq`, or a local OpenAI-compatible base URL (`http://localhost:8000/v1`).

A test clip without a microphone, on macOS:

    say --data-format=LEI16@16000 -o /tmp/cmd.wav "put the red cube in the bowl"
"""

from std.sys import argv

from noeira.ai.audio_io import record_wav
from noeira.ai.speech import SpeechToText


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: transcribe.mojo <file.wav | --record SECONDS> [service]")
    var path = String(args[1])
    var service_arg = 3 if path == "--record" else 2
    if path == "--record":
        var seconds = Float64(String(args[2])) if len(args) > 2 else 4.0
        path = String("/tmp/noeira_record.wav")
        print("recording", seconds, "s ... speak now")
        record_wav(path, seconds)
    var service = String(args[service_arg]) if len(args) > service_arg else String("hf")

    var stt: SpeechToText
    if service == "hf":
        stt = SpeechToText.huggingface()
    elif service == "openai":
        stt = SpeechToText.openai()
    elif service == "groq":
        stt = SpeechToText.groq()
    else:
        stt = SpeechToText.openai_compatible(service, String("whisper-1"))
    var t = stt.transcribe_file(path)
    print(t.text)
    print("--", stt.model, "|", t.latency_ms, "ms")

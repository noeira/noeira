# +--------------------------------------------------------------------------+ #
# | Voice -> Whisper -> Jev -> a robot skill, with calibrated escalation
# +--------------------------------------------------------------------------+ #
"""Speak a command; the robot picks a skill or asks you to repeat.

    pixi run mojo run -I . examples/ai/voice_command.mojo --record 4
    pixi run mojo run -I . examples/ai/voice_command.mojo command.wav

Needs `HF_TOKEN` (Whisper on the HF router) and `JEV_API_KEY`. The reply is
spoken by the OS voice (`speak_local`), so no TTS key is needed.

The pipeline a voice-driven robot demo needs, in ~60 lines:

1. **STT** turns audio into text (~1 s).
2. **Jev** maps the text onto a CLOSED set of skills the robot actually has
   (~0.3 s). It cannot invent a command that does not exist — the property a
   free-form LLM lacks — and it says how sure it is.
3. **Below the threshold, escalate** instead of acting. Here: ask again. In a
   real demo: ask an LLM to clarify, or a human.
"""

from std.sys import argv

from noeira.ai.audio_io import record_wav, speak_local
from noeira.ai.jev import JevClient, JevQuestions
from noeira.ai.speech import SpeechToText
from noeira.io.json import json_quote


comptime THRESHOLD = 0.7


def main() raises:
    var args = argv()
    var path = String(args[1]) if len(args) > 1 else String("--record")
    if path == "--record":
        var seconds = Float64(String(args[2])) if len(args) > 2 else 4.0
        path = String("/tmp/noeira_voice_cmd.wav")
        print("listening for", seconds, "s ...")
        record_wav(path, seconds)

    var stt = SpeechToText.huggingface()
    var heard = stt.transcribe_file(path)
    print("heard:", heard.text, " (", heard.latency_ms, "ms )")

    var q = JevQuestions()
    q.choice(
        "skill", "Which robot skill does the spoken `command` ask for?",
        ["pick_place", "push", "open_gripper", "go_home", "stop", "other"],
        [
            "move an object into or onto something",
            "slide an object along the table",
            "release whatever the gripper holds",
            "return the arm to its rest pose",
            "halt all motion now",
            "none of these, or not a robot command",
        ],
    )
    q.noul("urgent", "Does the speaker sound like they need the robot to stop immediately?")

    var jev = JevClient.from_env()
    var state = String('{"command": ') + json_quote(heard.text) + "}"
    var a = jev.decide(state, q)
    var skill = a.choice("skill")
    var conf = a.confidence("skill")
    print("skill:", skill, " confidence", conf, " P(urgent)", a.noul("urgent"),
          " (", a.latency_ms, "ms )")

    if a.noul("urgent") > 0.5 or skill == "stop":
        speak_local("Stopping.")
    elif conf < THRESHOLD or skill == "other":
        speak_local("Sorry, I did not understand. Please repeat.")
    else:
        speak_local("Okay: " + skill.replace("_", " ") + ".")


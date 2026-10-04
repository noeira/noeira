# +--------------------------------------------------------------------------+ #
# | A 60 Hz loop that never waits for the network
# +--------------------------------------------------------------------------+ #
"""Keep simulating while a Jev decision and an LLM answer are in flight.

    pixi run mojo run -I . examples/ai/background_loop.mojo            # hf + Jev
    pixi run mojo run -I . examples/ai/background_loop.mojo anthropic

The loop below stands in for `env.step(); renderer.draw()`. Each frame it
polls every client once with a ZERO timeout — so a model call costs the
loop nothing — and reacts when an answer lands. Two clients in flight at
once is concurrency for free: each owns its own connection, and one thread
drives both.

⚠ POLL EVERY FRAME. The transfer only advances inside `poll`; a loop that
stops polling stalls its requests.

⚠ WARM UP BEFORE THE LOOP. A client's first call pays the TLS handshake
inside ONE poll — measured 8-42 ms, a dropped frame or two. `warm_up()`
moves it before the loop; after that a poll costs < 0.5 ms.
"""

from std.sys import argv
from std.time import perf_counter_ns, sleep

from noeira.ai.chat import ChatClient, Conversation
from noeira.ai.jev import JevClient, JevQuestions


def main() raises:
    var args = argv()
    var spec = String(args[1]) if len(args) > 1 else String("hf")
    var llm = ChatClient.from_spec(spec)
    llm.max_tokens = 300
    if spec.startswith("hf"):
        llm.extra("chat_template_kwargs", '{"enable_thinking": false}')
    var jev = JevClient.from_env()

    var q = JevQuestions()
    q.choice(
        "next_skill", "Which skill should the arm run next?",
        ["approach", "grasp", "lift", "release"],
        ["move above the cube", "close the jaws on the cube", "raise the held cube",
         "open the jaws over the bowl"],
    )
    var conv = Conversation("One sentence, for a robot operator.")
    conv.user("The gripper just closed on a red cube. What should it be careful about when lifting?")

    llm.warm_up()
    jev.warm_up()
    jev.start('{"jaws": "closed", "cube_between_jaws": true, "cube_height": "on table"}', q)
    conv.start(llm)

    var jev_done = False
    var llm_done = False
    var caption = String("")
    var frame = 0
    var worst_poll_ms = 0.0
    var t0 = perf_counter_ns()
    comptime DT_NS = 16_666_667
    var deadline = t0
    while not (jev_done and llm_done):
        # ── the "simulation" ─────────────────────────────────────────
        frame += 1
        # ── the network, one non-blocking poll per client ───────────
        var tp = perf_counter_ns()
        if not jev_done and jev.poll(0):
            jev_done = True
            var a = jev.result()
            print("frame", frame, ": Jev ->", a.choice("next_skill"),
                  "conf", a.confidence("next_skill"))
        if not llm_done:
            caption += llm.poll(0)
            if llm.done():
                llm_done = True
                var r = conv.finish(llm)
                print("frame", frame, ": LLM ->", r.text)
                print("           first token", r.first_token_ms, "ms, total", r.latency_ms, "ms")
        var poll_ms = Float64(perf_counter_ns() - tp) / 1e6
        if poll_ms > worst_poll_ms:
            worst_poll_ms = poll_ms
        # Pace on absolute deadlines: sleeping "DT minus this frame's work"
        # accumulates the OS's sleep overshoot and settles near 50 fps.
        deadline += DT_NS
        var now = perf_counter_ns()
        if deadline > now:
            sleep(Float64(deadline - now) / 1e9)
    var wall = Float64(perf_counter_ns() - t0) / 1e9
    print(frame, "frames in", wall, "s =", Float64(frame) / wall, "fps while both calls ran")
    print("worst frame spent", worst_poll_ms, "ms inside the polls (incl. parsing answers)")

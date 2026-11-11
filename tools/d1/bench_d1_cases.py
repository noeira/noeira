#!/usr/bin/env python3
"""d1 System One requests at several sizes, and their time on llama.cpp's `/v1/systemone`:
the other column of `benchmarks/bench_d1_system_one_gpu.mojo` (same requests, same board).

    python3 tools/d1/bench_d1_cases.py [--out ~/.cache/noeira/d1/bench] [--url http://127.0.0.1:8030] [--reps 20]

Writes `<out>/<case>.state` and `<case>.q` (the request's state and questions as JSON
text) and `<out>/list.txt`, then times each case against the server if one answers at
`--url` (one warm call, then `--reps` timed calls; p50 / min / max wall time of the
HTTP round trip, and the server's `input_tokens`). Stdlib only.

Cases: two states (`small` = the room_en reference's, 134 tokens in d1's prompt;
`large` = a room state of the size the demo sends, ~300 tokens) times 1, 3, 6 and 12
questions. llama.cpp renders each question as its own prompt (state + question), so
its cost grows with questions x state; the Mojo port packs the state once per row of
up to 16 questions.
"""

import argparse
import json
import os
import statistics
import time
import urllib.request

SMALL = {"robot": {"at": "door", "facing": "kitchen", "doing": "standing"},
         "places": {"fridge": "3.1 m ahead, slightly left", "door": "here", "table": "2.0 m to the right",
                    "sofa": "4.5 m behind"},
         "heard": "Can you go to the fridge and wait there?",
         "recently": ["walked to the door", "heard: hello robot"]}
LARGE = {"robot": {"at": "door", "facing": "kitchen", "doing": "standing", "holding": "nothing"},
         "places": {"fridge": "3.1 m ahead, slightly left, in the kitchen", "door": "here",
                    "table": "2.0 m to the right, four chairs around it", "sofa": "4.5 m behind, facing the window",
                    "shelf": "1.2 m to the left, against the wall", "window": "5.0 m behind, right of the sofa",
                    "starting_point": "2.5 m ahead, where the robot woke up", "plant": "0.8 m to the right"},
         "speaker": {"where": "1.5 m in front of the robot", "facing": "the robot"},
         "heard": "Can you go to the fridge and wait there until I come back?",
         "recently": ["walked to the door", "heard: hello robot", "said: hello, I am ready",
                      "heard: is the kitchen clean", "said: I cannot see the kitchen from here",
                      "turned towards the speaker"]}

QUESTIONS = [
    ("addressed", {"type": "noul", "instructions": "Is the speaker asking the robot to do something?"}),
    ("destination", {"type": "choice", "instructions": "Where should the robot walk?",
                     "criteria": {"fridge": "walk to the fridge", "door": "walk to the door",
                                  "table": "walk to the table", "sofa": "walk to the sofa",
                                  "nowhere": "do not walk anywhere"}}),
    ("urgency", {"type": "score", "instructions": "How urgent is the request?",
                 "criteria": ["not urgent", "somewhat urgent", "very urgent"]}),
    ("other_person", {"type": "noul", "instructions": "Is the speaker talking to someone else?"}),
    ("after", {"type": "choice", "instructions": "What should the robot do once it arrives?",
               "criteria": {"wait": "wait there", "back": "come back", "say": "say something", "none": "nothing"}}),
    ("unsafe", {"type": "noul", "instructions": "Would doing this be unsafe for people nearby?"}),
    ("carry", {"type": "noul", "instructions": "Does the request ask the robot to carry something?"}),
    ("sure", {"type": "score", "instructions": "How sure is it that the speaker addresses the robot?",
              "criteria": ["not sure", "fairly sure", "certain"]}),
    ("reply", {"type": "noul", "instructions": "Should the robot answer out loud?"}),
    ("turn", {"type": "choice", "instructions": "Which way should the robot turn first?",
              "criteria": {"left": "turn left", "right": "turn right", "straight": "go straight",
                           "around": "turn around"}}),
    ("hurry", {"type": "noul", "instructions": "Is the speaker in a hurry?"}),
    ("question", {"type": "noul", "instructions": "Is the speaker asking a question?"}),
]
COUNTS = (1, 3, 6, 12)


def cases():
    for sname, state in (("small", SMALL), ("large", LARGE)):
        for k in COUNTS:
            yield f"{sname}_q{k}", state, dict(QUESTIONS[:k])


def ask(url, body):
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req))
    return (time.perf_counter() - t) * 1000, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/d1/bench"))
    ap.add_argument("--url", default="http://127.0.0.1:8030")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = []
    for name, state, q in cases():
        open(os.path.join(args.out, name + ".state"), "w").write(json.dumps(state))
        open(os.path.join(args.out, name + ".q"), "w").write(json.dumps(q))
        names.append(name)
    open(os.path.join(args.out, "list.txt"), "w").write("\n".join(names) + "\n")
    print(f"{len(names)} cases in {args.out}")
    try:
        urllib.request.urlopen(args.url + "/health", timeout=3)
    except OSError:
        print(f"no server at {args.url}: cases written, nothing timed")
        return
    print(f"llama.cpp {args.url}, 1 warm + {args.reps} timed calls each")
    for name, state, q in cases():
        body = {"state": state, "questions": q}
        _, r = ask(args.url, body)
        ms = [ask(args.url, body)[0] for _ in range(args.reps)]
        print(f"  {name:10s} {len(q):2d} q  {r['usage']['input_tokens']:5d} tokens  p50 {statistics.median(ms):7.1f} ms"
              f"  min {min(ms):7.1f}  max {max(ms):7.1f}")


if __name__ == "__main__":
    main()

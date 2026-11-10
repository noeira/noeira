#!/usr/bin/env python3
"""d1-2B's System One, run by its OWN code in float32, every stage dumped — the
reference the Mojo port of d1 is gated against (docs/NN_MODEL_PORTS_PLAN.md §12).

    ~/.cache/noeira/local-ai/kev-venv/bin/python tools/d1/dump_d1_reference.py \
        [--model ~/.cache/noeira/local-ai/d1/d1-2B] [--out ~/.cache/noeira/d1/ref]

⚠ RUN IT WHERE d1 WAS WRITTEN FOR: transformers 5.19 (the version its config and
README name; the Orin's `kev-venv` has exactly that, with torch 2.8 on the CPU).
Float32 on the CPU: ~7.7 GB of weights, the CPU attention path (chunked SDPA
under the `set` mask), a few minutes per case. Nothing is written to the model's
folder.

Per case (`<out>/<case>/`):

    answers.json              `system_one`'s output, as a caller sees it
    prompt.json               the state prefix and each question's suffix (text),
                              for the tokenizer gate
    row<r>_ids.npy            the packed row's token ids          [T] int32
    row<r>_meta.json          state length, each question's slots
    row<r>_question.npy       each token's question (1..N, -1 on the state) [T] int32
    row<r>_embed.npy          embed_tokens output                 [T, 2048]
    row<r>_layer<l>.npy       decoder layer l's output, l = 0..23 [T, 2048]
    row<r>_final.npy          after the final norm (`embedding_norm`) [T, 2048]
    row<r>_slot_logz.npy      log_softmax of the LM head at each slot [S, 128000]

Three cases: a room state with a yes/no, a choice and a score question (one
row, three groups under `set`); a French utterance with one choice (a LONE row:
fully bidirectional); eighteen yes/no questions (two rows: 16 + 2).
"""

import argparse
import importlib
import json
import os

import numpy as np
import torch
from transformers import AutoModel

ROOM_STATE = {
    "robot": {"at": "door", "facing": "kitchen", "doing": "standing"},
    "places": {"fridge": "3.1 m ahead, slightly left", "door": "here", "table": "2.0 m to the right",
               "sofa": "4.5 m behind"},
    "heard": "Can you go to the fridge and wait there?",
    "recently": ["walked to the door", "heard: hello robot"],
}

CASES = {
    "room_en": (ROOM_STATE, {
        "addressed": {"type": "noul", "instructions": "Is the speaker asking the robot to do something?"},
        "destination": {"type": "choice", "instructions": "Where should the robot walk?",
                        "criteria": {"fridge": "walk to the fridge", "door": "walk to the door",
                                     "table": "walk to the table", "sofa": "walk to the sofa",
                                     "nowhere": "do not walk anywhere"}},
        "urgency": {"type": "score", "instructions": "How urgent is the request?",
                    "criteria": ["not urgent", "somewhat urgent", "very urgent"]},
    }),
    "single_fr": ("Le robot entend : « va vers le canapé, s'il te plaît ».", {
        "move": {"type": "choice", "instructions": "Quelle action le robot doit-il faire ?",
                 "criteria": {"canape": "marcher vers le canapé", "porte": "marcher vers la porte",
                              "accroupi": "s'accroupir", "rien": "ne rien faire"}},
    }),
    "many_noul": ("The user says: please raise your right hand and then turn left slowly.", {
        f"q{i:02d}": {"type": "noul", "instructions": text} for i, text in enumerate([
            "Does the user ask for an arm movement?", "Does the user mention the left hand?",
            "Does the user ask the robot to turn?", "Is a speed mentioned?", "Is the request polite?",
            "Does the user ask the robot to walk?", "Is a place named?", "Does the user ask for two actions?",
            "Is the right hand mentioned?", "Does the user ask the robot to sit?", "Is the user angry?",
            "Does the user ask to stop?", "Is the order of actions given?", "Does the user ask a question?",
            "Is a direction given?", "Does the user greet the robot?", "Does the user ask to squat?",
            "Is the user talking to someone else?",
        ])
    }),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/noeira/local-ai/d1/d1-2B"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/d1/ref"))
    ap.add_argument("--cases", default=",".join(CASES))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    eng = model.engine
    lm = model.model.language_model
    # d1's own `slots` module, from the package transformers loaded its remote code into
    slot_logits = importlib.import_module(type(model).__module__.rsplit(".", 1)[0] + ".slots").slot_logits

    print(f"d1 at {args.model}: attention {eng.attention}, set size {eng.set_size}, "
          f"{len(lm.layers)} layers {[l for l in lm.config.layer_types]}")

    for name in args.cases.split(","):
        state, questions = CASES[name]
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        answers = model.system_one(state, questions)
        with open(os.path.join(out, "answers.json"), "w") as f:
            json.dump(answers, f, ensure_ascii=False, indent=1)

        qs = [eng.ask(q) for q in questions.values()]
        prefix, _ = eng._prefix(state, ())
        suffixes, offsets, _ = eng._suffixes(qs)
        with open(os.path.join(out, "prompt.json"), "w") as f:
            json.dump({"prefix": prefix, "suffixes": suffixes, "slot_offsets": offsets,
                       "names": list(questions)}, f, ensure_ascii=False, indent=1)

        seqs = eng._sequences([(state, qs, [])])
        for r, (_, k, ids, state_len, question, ends, _media) in enumerate(seqs):
            cap = {}
            hooks = [lm.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__("embed", o))]
            for li, layer in enumerate(lm.layers):
                hooks.append(layer.register_forward_hook(
                    lambda m, i, o, li=li: cap.__setitem__(li, o[0] if isinstance(o, tuple) else o)))
            hooks.append(lm.embedding_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("final", o)))
            t = torch.tensor([ids])
            last = [p for own in ends for p in own]
            rows = slot_logits(model, t, torch.ones_like(t), torch.tensor(last), torch.tensor([state_len]),
                               torch.tensor([question]), torch.zeros(len(last), dtype=torch.long))
            for h in hooks:
                h.remove()
            codes = rows.codes.float()
            logz = codes - torch.logsumexp(codes, dim=-1, keepdim=True)

            np.save(os.path.join(out, f"row{r}_ids.npy"), np.array(ids, dtype=np.int32))
            np.save(os.path.join(out, f"row{r}_question.npy"), np.array(question, dtype=np.int32))
            np.save(os.path.join(out, f"row{r}_embed.npy"), cap["embed"][0].float().numpy())
            for li in range(len(lm.layers)):
                np.save(os.path.join(out, f"row{r}_layer{li}.npy"), cap[li][0].float().numpy())
            np.save(os.path.join(out, f"row{r}_final.npy"), cap["final"][0].float().numpy())
            np.save(os.path.join(out, f"row{r}_slot_logz.npy"), logz.numpy())
            with open(os.path.join(out, f"row{r}_meta.json"), "w") as f:
                json.dump({"first_question": k, "length": len(ids), "state_len": state_len, "slots": ends},
                          f, indent=1)
            print(f"  {name} row {r}: {len(ids)} tokens, state {state_len}, questions "
                  f"{sorted(set(q for q in question if q > 0))}, slots {ends}")
        print(f"  {name}: {json.dumps(answers['answers'], ensure_ascii=False)[:300]}")


if __name__ == "__main__":
    main()

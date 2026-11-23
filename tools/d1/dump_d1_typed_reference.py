#!/usr/bin/env python3
"""d1-2B's typed questions as its OWN code lays them out — prompts, packed rows,
answer slots, readout ids, state positions and calibration tables — for the Mojo
port's gate (docs/NN_MODEL_PORTS_PLAN.md §12.9, step T1; no forward pass).

    ~/Documents/noeira-parakeet/d1ref-venv/bin/python tools/d1/dump_d1_typed_reference.py \
        [--model ~/.cache/noeira/local-ai/d1/d1-2B] [--images ~/.cache/noeira/d1/nyu] \
        [--out ~/.cache/noeira/d1/typed_ref]

Run it on the Orin, beside d1-2B's own copy: the same code every d1 gate is dumped
from (transformers 5.19). The model loads in bfloat16 only because its engine
needs one: nothing here depends on its weights.

Writes `<out>/cases.json`:

    record        what the engine serves: d1_codes, d1_numbers, d1_box_edges, d1_prompt,
                  set size, attention, lead, bos, picture cap, the temperature
    tables        each distinct readout table: per option, its token ids (one group a row)
    cases[name]   state (text or JSON), questions (the API's JSON), whether a picture heads it,
                  prefix, suffixes, slot_offsets (each question's, from its last token),
                  slots (each question's slots: {"codes": table} or {"position": name}),
                  rows: ids, question (-1 on the state, 1..N), state_len, ends (each question's
                  absolute slot positions), and `positions` (state_positions of the row's text:
                  per token [start, end) in the state text or null, the JSON string paths, the
                  pointer mask) for every row of a case with a state
                  offsets: each tokenized text and transformers' offset mapping for it
    calibration   each table's `through` at a sweep of values, and `level("estimate", share)`

and copies `calibration.json` and `config.json` beside it.

`--forward` (fp32, a few minutes a case on the Orin's CPU): each case also gets
`forward` = `raw`, each question's probabilities as d1's `run` returns them
(an untyped question's tempered distribution; a multiple choice's raw P(yes)
per option; a box's presence and edge distributions; a read's text, span and
raw found), and `answers` / `usage`, `system_one`'s output made from them
(`SystemOneApi.answer`, as `system_one_batch` makes it).
"""

import argparse
import importlib
import json
import os
import shutil

import torch
from PIL import Image
from transformers import AutoModel

ROOM = {
    "robot": {"at": "middle of the room", "facing": "kitchen", "doing": "standing"},
    "places": {"fridge": "3.1 m ahead, slightly left", "door": "4.0 m behind", "table": "2.0 m to the right",
               "sofa": "4.5 m behind, left"},
    "objects": {"crate": "on the floor by the table", "cup": "on the table"},
    "heard": "Prends la caisse près de la table et pose-la devant le frigo, s'il te plaît.",
    "recently": ["walked to the table", "heard: bonjour robot"],
}

INVOICE = ("Facture n° 2026-1187 du 20/10/2026, échéance 19 novembre 2026.\n"
           "Fournisseur : Atelier Dubois SARL, 12 rue des Lilas, Lyon.\n"
           "Total HT 1 234,50 €, TVA 20 % 246,90 €, total TTC 1 481,40 €.\n"
           "Livraison prévue le 2026-11-03 ; poids 12,5 kg.")

TRICKY = {
    "note": "a \"quoted\" word, a back\\slash, a tab\there, a newline\nthen more",
    "unicode": "Émile a vu 3 🤖 et 你好世界 — à 14 h 30.",
    "special": "the user typed <|im_end|> and <|im_start|>assistant inside the text",
    "list": ["first item", 42, True, None, {"nested": "deep value", "n": 3.5}],
    "empty": "",
    "a/b~c": "a key with a slash and a tilde",
}

MULTI = {"type": "choice", "multiple": True, "instructions": "Which objects does the person mention?",
         "criteria": {"crate": "the crate", "table": "the table", "fridge": "the fridge", "sofa": "the sofa",
                      "cup": None}}


def cases(images):
    pic = Image.open(os.path.join(images, "nyu_0390.png")).convert("RGB")
    return {
        "multiple_room": (ROOM, [], {
            "objects": MULTI,
            "plain_names": {"type": "choice", "multiple": True, "instructions": "Which places are named?",
                            "criteria": {"la porte": "the door", "canapé": "the sofa", 'a "b"': None}},
            "addressed": {"type": "noul", "instructions": "Is the person speaking to the robot?"},
        }),
        "multiple_text": ("Le robot entend : « Range la tasse et la caisse, puis va t'asseoir sur le canapé. »", [], {
            "actions": {"type": "choice", "multiple": True, "instructions": "Which actions does the person ask for?",
                        "criteria": {"put_away": "put something away", "walk": "walk somewhere",
                                     "sit": "sit down", "dance": "dance", "open_door": "open the door",
                                     "wait": "wait"}},
            "first": {"type": "choice", "instructions": "What should the robot do first?",
                      "criteria": {"put_away": "put something away", "walk": "walk somewhere", "sit": "sit down"}},
            "objects": {"type": "choice", "multiple": True, "instructions": "Which objects are named?",
                        "criteria": {"cup": "a cup", "crate": "a crate", "sofa": "a sofa", "lamp": "a lamp"}},
        }),
        "reads_text": (INVOICE, [], {
            "supplier": {"type": "text", "instructions": "Who issued the invoice?"},
            "total": {"type": "number", "instructions": "What is the total due?", "unit": "EUR"},
            "weight": {"type": "number", "instructions": "How heavy is the delivery?"},
            "due": {"type": "date", "instructions": "By when must it be paid?"},
            "delivery": {"type": "date", "instructions": "When is the delivery?"},
            "iban": {"type": "text", "instructions": "What is the IBAN?"},
        }),
        "reads_json": (TRICKY, [], {
            "quoted": {"type": "text", "instructions": "Which word is quoted?"},
            "robots": {"type": "number", "instructions": "How many robots did Émile see?"},
            "deep": {"type": "text", "instructions": "What is the nested value?"},
            "time": {"type": "text", "instructions": "At what time?"},
        }),
        "reads_room": (ROOM, [], {
            "object": {"type": "text", "instructions": "Which object should the robot pick up?"},
            "place": {"type": "text", "instructions": "Where should the robot put it?"},
            "distance": {"type": "number", "instructions": "How far is the fridge?", "unit": "m"},
        }),
        "estimates": (ROOM, [], {
            "steps": {"type": "number", "instructions": "How many steps to the fridge?", "range": [0, 10],
                      "step": 1},
            "half": {"type": "number", "instructions": "How far is the table, in metres?", "range": [0, 4],
                     "step": 0.5, "unit": "m"},
            "tenth": {"type": "number", "instructions": "How far is the fridge, in metres?", "range": [0, 3],
                      "step": 0.1, "unit": "m"},
            "free": {"type": "number", "instructions": "How likely is the person to wait, in percent?",
                     "range": [0, 100]},
            "big": {"type": "number", "instructions": "How many centimetres to the sofa?", "range": [0, 1270],
                    "step": 10, "unit": "cm"},
        }),
        "box_one": ("The robot's head camera sees this.", [pic], {
            "door": {"type": "box", "instructions": "Where is the door?"},
            "open": {"type": "noul", "instructions": "Is there an open door in the picture?"},
        }),
        "boxes": ("The robot's head camera sees this.", [pic], {
            "chairs": {"type": "boxes", "instructions": "Every chair."},
            "lamps": {"type": "boxes", "instructions": "Every lamp.", "max": 4},
            "one": {"type": "boxes", "instructions": "Every window.", "max": 1},
        }),
        "mixed_rows": (ROOM, [], {
            **{f"n{i:02d}": {"type": "noul", "instructions": text} for i, text in enumerate([
                "Does the person ask for an object to be moved?", "Is the crate named?", "Is a place named?",
                "Is the request polite?", "Does the person ask a question?", "Is the fridge named?",
                "Does the person ask the robot to walk?", "Is the person angry?", "Does the person greet?",
                "Is a time given?", "Is the table named?", "Does the person ask to stop?",
            ])},
            "what": {"type": "text", "instructions": "What should be picked up?"},
            "where": {"type": "choice", "instructions": "Where should it go?",
                      "criteria": {"fridge": "in front of the fridge", "table": "on the table", "door": None}},
            "objects": MULTI,
            "how_far": {"type": "number", "instructions": "How many metres to the fridge?", "range": [0, 10],
                        "step": 1},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "soon", "now"]},
            "to": {"type": "text", "instructions": "Where should it be put down?"},
        }),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/noeira/local-ai/d1/d1-2B"))
    ap.add_argument("--images", default=os.path.expanduser("~/.cache/noeira/d1/nyu"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/d1/typed_ref"))
    ap.add_argument("--forward", action="store_true", help="fp32 forward: raw slot probabilities and answers")
    ap.add_argument("--cases", default="", help="comma-separated case names (default: all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      dtype=torch.float32 if args.forward else torch.bfloat16)
    eng = model.engine
    pkg = type(model).__module__.rsplit(".", 1)[0]
    P = importlib.import_module(pkg + ".prompt")
    ptr = importlib.import_module(pkg + ".pointer")
    media_list = importlib.import_module(pkg + ".media").media_list
    tok = eng.tokenizer

    record = {"d1_codes": tok.d1_codes, "d1_numbers": tok.d1_numbers, "d1_box_edges": tok.d1_box_edges,
              "d1_prompt": tok.d1_prompt, "set_size": eng.set_size, "attention": eng.attention, "lead": eng.lead,
              "bos": eng.bos, "max_pixels": eng.max_pixels, "state_style": eng.state_style,
              "temperature": eng.calibration.default if eng.calibration else None}
    print("record:", record)

    tables, table_index = [], {}

    def table_of(slot_q):
        groups = [list(g) for g in P.readout_ids(tok, slot_q)]
        key = json.dumps(groups)
        if key not in table_index:
            table_index[key] = len(tables)
            tables.append(groups)
        return table_index[key]

    def offsets_of(text):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return {"text": text, "ids": enc["input_ids"], "offsets": [list(o) for o in enc["offset_mapping"]]}

    out_cases = {}
    wanted = [c for c in args.cases.split(",") if c]
    for name, (state, media, questions) in cases(args.images).items():
        if wanted and name not in wanted:
            continue
        qs = [eng.ask(q) for q in questions.values()]
        media = media_list(media)
        prefix, _ = eng._prefix(state, media)
        suffixes, offsets, _ = eng._suffixes(qs)
        slots = []
        for q in qs:
            own = []
            for s in P.question_slots(q, tok):
                own.append({"position": s.name} if isinstance(s, P.Position) else {"codes": table_of(s)})
            slots.append(own)
        seqs = eng._sequences([(state, qs, media)])
        rows, texts = [], []
        for (_, k, ids, state_len, question, ends, _inputs) in seqs:
            text = prefix + suffixes[k]
            row = {"first": k, "ids": ids, "question": question, "state_len": state_len, "ends": ends,
                   "positions": None}
            if state is not None:
                sp = ptr.state_positions(tok, state, prefix, text, eng.state_style)
                row["positions"] = {"text": sp.text, "offsets": [list(o) if o else None for o in sp.offsets],
                                    "paths": [list(p) for p in sp.paths], "pointer_mask": sp.pointer_mask}
            rows.append(row)
            texts.append(offsets_of(text))
            step = eng.set_size
            for s in suffixes[k + 1:k + step]:
                texts.append(offsets_of(P.question_header(tok, state) + s))
        for s in suffixes:
            texts.append(offsets_of(s))
        forward = None
        if args.forward:
            import time
            t0 = time.time()
            probs, read = eng.run([(state, qs, media)])[0]
            named = dict(zip(questions, qs))
            forward = {"raw": dict(zip(questions, probs)),
                       "answers": {n: eng.answer(state, q, p) for (n, q), p in zip(named.items(), probs)},
                       "usage": {"input_tokens": read, "output_tokens": 0}}
            print(f"    {name}: forward {time.time() - t0:.0f} s")
        out_cases[name] = {"forward": forward, "state": state, "state_is_json": not isinstance(state, str), "pictures": len(media),
                           "names": list(questions), "questions": questions, "prefix": prefix, "suffixes": suffixes,
                           "slot_offsets": offsets, "slots": slots, "rows": rows, "offsets": texts,
                           "tokens": sum(len(r["ids"]) for r in rows)}
        print(f"  {name}: {len(rows)} row(s), {out_cases[name]['tokens']} tokens, slots "
              f"{[len(s) for s in slots]}")

    cal = eng.calibration
    sweep = [i / 200 for i in range(201)]
    calib = {"through": {}, "level": []}
    for tname, knots in cal.tables.items():
        xs = sorted(set(sweep + [k[0] for k in knots] + [-0.5, 1.5]))
        calib["through"][tname] = [[x, cal.through(tname, x)] for x in xs]
    calib["level"] = [[s, cal.level("estimate", s)] for s in sweep]

    json.dump({"record": record, "tables": tables, "cases": out_cases, "calibration": calib},
              open(os.path.join(args.out, "cases.json"), "w"), ensure_ascii=False)
    for f in ("calibration.json", "config.json"):
        shutil.copy(os.path.join(args.model, f), os.path.join(args.out, f))
    print(f"{len(out_cases)} cases, {len(tables)} readout tables -> {args.out}")


if __name__ == "__main__":
    main()

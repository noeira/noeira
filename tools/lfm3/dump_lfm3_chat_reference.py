#!/usr/bin/env python3
"""LFM3-1.7B CHATTING, run by Liquid's own code in float32: the reference the Mojo
generation path (a cache, a chunk forward, greedy decoding) is gated against
(docs/NN_MODEL_PORTS_PLAN.md §12.7 step 1).

    ~/Documents/noeira-parakeet/d1ref-venv/bin/python tools/lfm3/dump_lfm3_chat_reference.py \
        [--model ~/.cache/noeira/local-ai/lfm3/LFM3-1.7B] [--out ~/.cache/noeira/lfm3/chat_ref] \
        [--steps 48]

⚠ transformers >= 5.19 with `trust_remote_code` (the Orin's `d1ref-venv`: 5.19.0,
torch 2.8 on the CPU). Float32: ~7 GB of weights. Nothing is written to the model's folder.

Per case (`<out>/<case>/`):

    meta.json                 the rendered prompt(s), the generated text, the settings, the checks below
    prompt_ids.npy            turn 1's prompt, as `apply_chat_template` makes it      [P] int32
    prefill_embed.npy         embed_tokens output over the prompt                     [P, 2048]
    prefill_layer<l>.npy      decoder layer l's output over the prompt, l = 0..23     [P, 2048]
    prefill_final.npy         after the final norm (`embedding_norm`)                 [P, 2048]
    gen_ids.npy               turn 1's generated tokens, eos included if reached      [N] int32
    gen_logits.npy            the LM head's logits that chose each of them (raw, before any penalty) [N, 128000]
    turn2_ids.npy             (two-turn cases) the tokens appended for turn 2: the end of turn 1's
                              message, the user's turn, the assistant header          [P2] int32
    turn2_gen_ids.npy / turn2_gen_logits.npy   turn 2's generation, decoded on turn 1's cache

How the tokens are chosen: greedy, `argmax` of the logits after the case's
repetition penalty (transformers' `RepetitionPenaltyLogitsProcessor`: a token
already in the sequence, prompt included, has its logit divided by the penalty
if positive, multiplied if negative). 1.0 = plain greedy. Deterministic, so the
port must reproduce the same tokens. Liquid's chat settings sample
(temperature 0.1, top_k 50); sampling is not gated.

The loop is the cached one (`DynamicCache`, one token a step, as `generate`
runs it). Checks written to meta.json, so the reference is checked before it
gates anything:
- `generate_agrees`: `model.generate(do_sample=False, repetition_penalty=...)`
  gives the same tokens as this loop;
- `recompute_max_abs`: one uncached forward over prompt + generated tokens gives
  the same logits as the cached steps;
- `template_agrees` (two-turn cases): turn 1's prompt + generated tokens + the
  appended tokens equal `apply_chat_template` over the whole conversation. That is
  what lets a chat keep its cache across turns. It holds with thinking OFF only:
  the template drops an earlier turn's thinking from the history.

Cases: a French robot greeting, two turns (thinking off, plain greedy); an
English request with the repetition penalty 1.1 (thinking off); the French
greeting with thinking ON (the `<think>` path; one turn).
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

SYSTEM = (
    "You are Noé, a small humanoid robot in a living room. You speak with the people around you, briefly and "
    "kindly, in the language they use. Room: the fridge is 3.1 m ahead, the door 3.8 m to the left, the table "
    "2.0 m to the right, the sofa 4.5 m behind. You are standing in the middle of the room."
)

CASES = {
    "fr_two_turns": dict(
        turns=["Bonjour ! Qui es-tu, et qu'est-ce que tu sais faire ?",
               "Et qu'est-ce qu'il y a derrière toi ?"],
        thinking=False, penalty=1.0),
    "en_penalty": dict(
        turns=["Tell me a short joke about robots, then tell me where the sofa is."],
        thinking=False, penalty=1.1),
    "fr_thinking": dict(
        turns=["Bonjour ! Qui es-tu, et qu'est-ce que tu sais faire ?"],
        thinking=True, penalty=1.0),
}


def save(out, name, x):
    np.save(os.path.join(out, name), np.ascontiguousarray(x.detach().float().numpy() if torch.is_tensor(x) else x))


def render(tok, messages, thinking):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def ids_of(tok, text):
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids


def decode_loop(model, cache, seq, first_ids, steps, penalty, eos):
    """Feed `first_ids` on `cache`, then one token a step. Returns (tokens, raw logits per step)."""
    proc = RepetitionPenaltyLogitsProcessor(penalty) if penalty != 1.0 else None
    toks, logits = [], []
    x = first_ids
    for _ in range(steps):
        out = model(input_ids=x, past_key_values=cache, use_cache=True)
        raw = out.logits[:, -1, :].float()
        logits.append(raw[0].clone())
        seq = torch.cat([seq, x], dim=1) if seq is not None else x
        scored = proc(seq, raw.clone()) if proc is not None else raw
        t = int(scored.argmax(-1))
        toks.append(t)
        x = torch.tensor([[t]])
        if t == eos:
            break
    return toks, torch.stack(logits), torch.cat([seq, x], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/noeira/local-ai/lfm3/LFM3-1.7B"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/lfm3/chat_ref"))
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--cases", default=",".join(CASES))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)  # else an interactive y/N prompt
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    eos = model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    inner = model.model
    layers = inner.layers

    for name in args.cases.split(","):
        case = CASES[name]
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["turns"][0]}]
        prompt = render(tok, msgs, case["thinking"])
        pids = ids_of(tok, prompt)
        save(out, "prompt_ids.npy", pids[0].numpy().astype(np.int32))
        meta = {"case": name, "system": SYSTEM, "turns": case["turns"], "thinking": case["thinking"],
                "repetition_penalty": case["penalty"], "steps": args.steps, "eos": eos, "prompt": prompt,
                "prompt_tokens": int(pids.shape[1])}

        # the prefill's stages (an uncached pass over the prompt)
        taps = {}
        hooks = [inner.embed_tokens.register_forward_hook(lambda m, i, o: taps.__setitem__("embed", o[0]))]
        for li, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(
                lambda m, i, o, li=li: taps.__setitem__(f"layer{li}", (o[0] if isinstance(o, tuple) else o)[0])))
        hooks.append(inner.embedding_norm.register_forward_hook(lambda m, i, o: taps.__setitem__("final", o[0])))
        model(input_ids=pids, use_cache=False)
        for h in hooks:
            h.remove()
        save(out, "prefill_embed.npy", taps["embed"])
        for li in range(len(layers)):
            save(out, f"prefill_layer{li}.npy", taps[f"layer{li}"])
        save(out, "prefill_final.npy", taps["final"])

        # turn 1, cached
        cache = DynamicCache(config=model.config)
        gen, glog, seq = decode_loop(model, cache, None, pids, args.steps, case["penalty"], eos)
        save(out, "gen_ids.npy", np.array(gen, dtype=np.int32))
        save(out, "gen_logits.npy", glog)
        meta["gen_text"] = tok.decode(gen, skip_special_tokens=False)
        meta["gen_tokens"] = len(gen)
        meta["gen_reached_eos"] = gen[-1] == eos
        print(f"  {name}: {meta['prompt_tokens']} prompt tokens -> {len(gen)}: {meta['gen_text']!r}")

        # check 1: transformers' own generate picks the same tokens
        g = model.generate(pids, do_sample=False, max_new_tokens=args.steps, repetition_penalty=case["penalty"],
                           eos_token_id=eos, temperature=None, top_k=None, top_p=None)
        meta["generate_agrees"] = g[0, pids.shape[1]:].tolist() == gen
        # check 2: an uncached pass over prompt + generated gives the cached steps' logits
        full = torch.cat([pids, torch.tensor([gen[:-1]], dtype=pids.dtype)], dim=1) if len(gen) > 1 else pids
        rl = model(input_ids=full, use_cache=False).logits[0, pids.shape[1] - 1:].float()
        meta["recompute_max_abs"] = float((rl - glog).abs().max())
        meta["logit_max_abs"] = float(glog.abs().max())

        if len(case["turns"]) > 1:
            # turn 2 appended to turn 1's cache: the rest of the message, then the user's turn and the header
            reply = tok.decode(gen[:-1] if gen[-1] == eos else gen, skip_special_tokens=False)
            content = reply.split("</think>")[-1] if "</think>" in reply else reply
            msgs2 = msgs + [{"role": "assistant", "content": content}, {"role": "user", "content": case["turns"][1]}]
            whole = render(tok, msgs2, case["thinking"])
            head = prompt + tok.decode(gen, skip_special_tokens=False)
            if gen[-1] != eos:
                head += "<|im_end|>"
            tail = whole[len(head):] if whole.startswith(head) else None
            meta["turn2_prompt"] = whole
            meta["template_agrees"] = tail is not None
            if tail is None:
                print("  ⚠ the template does not re-render turn 1 as generated: no cache reuse")
            else:
                t2 = ids_of(tok, ("<|im_end|>" if gen[-1] != eos else "") + tail)
                # the conversation's tokens, one way and the other
                ids_whole = ids_of(tok, whole)[0].tolist()
                ids_cat = pids[0].tolist() + gen + t2[0].tolist()
                meta["template_tokens_agree"] = ids_whole == ids_cat
                save(out, "turn2_ids.npy", t2[0].numpy().astype(np.int32))
                # the cache now holds the prompt and gen[:-1]: the last generated token goes in with turn 2
                first = torch.cat([torch.tensor([[gen[-1]]]), t2], dim=1)
                gen2, glog2, seq2 = decode_loop(model, cache, seq[:, :-1], first, args.steps, case["penalty"], eos)
                save(out, "turn2_gen_ids.npy", np.array(gen2, dtype=np.int32))
                save(out, "turn2_gen_logits.npy", glog2)
                meta["turn2_gen_text"] = tok.decode(gen2, skip_special_tokens=False)
                full2 = torch.tensor([ids_cat + gen2[:-1]])
                rl2 = model(input_ids=full2, use_cache=False).logits[0, len(ids_cat) - 1:].float()
                meta["turn2_recompute_max_abs"] = float((rl2 - glog2).abs().max())
                print(f"  {name} turn 2: +{t2.shape[1]} tokens -> {len(gen2)}: {meta['turn2_gen_text']!r}")
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print("  checks:", {k: meta[k] for k in meta if k.endswith(("agrees", "agree", "max_abs"))})


if __name__ == "__main__":
    main()

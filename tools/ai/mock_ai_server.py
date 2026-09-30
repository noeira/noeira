"""A mock AI-provider server for `tests/ai/test_ai_clients.mojo`.

    python3 tools/ai/mock_ai_server.py <port-file> [<seconds>]

Binds port 0, writes the port to `<port-file>` (via `.tmp` + rename), and
serves until `/__shutdown` or the timeout. Stdlib only.

⚠ IT IS A GATE FIXTURE, NOT A MODEL. Each route speaks one provider's WIRE
FORMAT with canned answers, and REFUSES a request whose shape is wrong (400
with the reason), so the Mojo clients are gated on what they send as well as
on what they parse — with no key and no network. `GET /__last` returns the
last request (path, headers, body) for assertions the routes do not make.

Routes
  POST /v1/messages               Anthropic Messages: tool_use first, text
                                  once a tool_result comes back
  POST /v1/chat/completions       OpenAI Chat Completions, same script
  POST /v1/systemone              TypeSafe System One: one answer per question
  POST /v1/audio/transcriptions   multipart WAV -> {"text": "<frames>@<rate>"}
  POST /hf/whisper                raw WAV body -> same
  POST /v1/audio/speech           -> a 0.25 s 16 kHz mono WAV
  POST /slow/v1/systemone         as /v1/systemone, after 0.4 s
  POST /err/v1/messages           a stream that dies with an `error` event
  POST /flaky/v1/systemone        529 twice, then as /v1/systemone
  POST /flaky-bg/v1/systemone     the same, own counter (the background gate)

Streaming: a chat request with `"stream": true` is answered as SSE in that
provider's event grammar, built from the same canned message. ⚠ EACH EVENT
IS WRITTEN IN TWO PIECES split at an arbitrary byte, with a pause between —
so the client sees a line, a JSON document and a UTF-8 character cut in
half, and a parser that assumes whole lines per chunk fails here.
  POST /bad/v1/systemone          422 {"error":"questions: field required"}
  GET  /__last, /__shutdown
"""

import json
import os
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LAST = {}
FLAKY = {"/flaky/v1/systemone": 0, "/flaky-bg/v1/systemone": 0}


def wav_info(b):
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("not a WAV")
    pos, rate, ch = 12, 0, 0
    while pos + 8 <= len(b):
        tag, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        if tag == b"fmt ":
            fmt, ch, rate = struct.unpack("<HHI", b[pos + 8:pos + 16])
            bits = struct.unpack("<H", b[pos + 22:pos + 24])[0]
            if fmt != 1 or bits != 16:
                raise ValueError("not 16-bit PCM")
        if tag == b"data":
            return size // (2 * ch), rate, ch
        pos += 8 + size + (size & 1)
    raise ValueError("no data chunk")


def make_wav(n=4000, rate=16000):
    data = b"".join(struct.pack("<h", (i * 37) % 2000 - 1000) for i in range(n))
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


def multipart_parts(body, ctype):
    boundary = ctype.split("boundary=", 1)[1].encode()
    parts = {}
    for chunk in body.split(b"--" + boundary):
        if not chunk.startswith(b"\r\n"):
            continue
        head, _, val = chunk[2:].partition(b"\r\n\r\n")
        name = head.split(b'name="', 1)[1].split(b'"', 1)[0].decode()
        parts[name] = val[:-2]
    return parts


class Refuse(Exception):
    pass


def need(cond, why):
    if not cond:
        raise Refuse(why)


def anthropic(req, headers):
    need(headers.get("x-api-key") == "test-key", "x-api-key missing")
    need(headers.get("anthropic-version") == "2023-06-01", "anthropic-version missing")
    need("temperature" not in req, "temperature is rejected on this model")
    need(isinstance(req.get("max_tokens"), int), "max_tokens required")
    msgs = req["messages"]
    need(msgs and msgs[0]["role"] == "user", "first message must be user")
    for m in msgs:
        need(m["role"] in ("user", "assistant"), "bad role " + m["role"])
        need(isinstance(m["content"], list), "content must be a block list")
    last = msgs[-1]["content"]
    results = [b for b in last if b["type"] == "tool_result"]
    if results:
        # The previous assistant turn must come back VERBATIM, thinking block
        # included — that is the invariant the client exists to keep.
        prev = msgs[-2]["content"]
        need(prev[0] == {"type": "thinking", "thinking": "", "signature": "sig-1"},
             "thinking block not replayed verbatim")
        need(len(results) == 2, "both tool results must share ONE user message")
        text = "done — " + " | ".join(r["content"] for r in results)
        return {"id": "msg_2", "type": "message", "role": "assistant",
                "model": req["model"], "stop_reason": "end_turn",
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 50, "output_tokens": 5}}
    need(req.get("tools"), "tools expected on the first turn")
    need("input_schema" in req["tools"][0], "anthropic tools use input_schema")
    imgs = [b for b in last if b["type"] == "image"]
    return {"id": "msg_1", "type": "message", "role": "assistant",
            "model": req["model"], "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig-1"},
                {"type": "text", "text": "I see %d image(s)." % len(imgs)},
                {"type": "tool_use", "id": "toolu_a", "name": "get_pose",
                 "input": {"name": "cube", "k": 2}},
                {"type": "tool_use", "id": "toolu_b", "name": "get_pose",
                 "input": {"name": "bowl", "k": 0.5}}],
            "usage": {"input_tokens": 40, "output_tokens": 30}}


def openai(req, headers):
    need(headers.get("authorization") == "Bearer test-key", "bearer missing")
    msgs = req["messages"]
    need(msgs[0]["role"] == "system", "system prompt must be messages[0]")
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    if tool_msgs:
        asst = [m for m in msgs if m["role"] == "assistant"][-1]
        need(len(asst["tool_calls"]) == 2, "assistant tool_calls not replayed")
        need(isinstance(asst["tool_calls"][0]["function"]["arguments"], str),
             "arguments must be a JSON string")
        text = "done — " + " | ".join(m["content"] for m in tool_msgs)
        return {"id": "c2", "model": req["model"], "choices": [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5}}
    need(req["tools"][0]["type"] == "function", "openai tools are type=function")
    user = msgs[-1]["content"]
    n_img = 0
    if isinstance(user, list):
        n_img = sum(1 for p in user if p["type"] == "image_url"
                    and p["image_url"]["url"].startswith("data:image/png;base64,"))
    return {"id": "c1", "model": req["model"], "choices": [{
        "index": 0, "finish_reason": "tool_calls",
        "message": {"role": "assistant", "content": "I see %d image(s)." % n_img,
                    "tool_calls": [
                        {"id": "call_a", "type": "function", "function": {
                            "name": "get_pose", "arguments": '{"name":"cube","k":2}'}},
                        {"id": "call_b", "type": "function", "function": {
                            "name": "get_pose", "arguments": '{"name":"bowl","k":0.5}'}}]}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 30}}


def systemone(req, headers):
    need(headers.get("authorization") == "Bearer test-key", "bearer missing")
    need(req.get("model") == "jev-latest", "model")
    need("state" in req, "state: field required")
    need(isinstance(req.get("questions"), dict), "questions: field required")
    answers = {}
    for qid, q in req["questions"].items():
        t = q["type"]
        need(isinstance(q.get("instructions"), str), qid + ": instructions")
        if t == "noul":
            answers[qid] = {"type": "noul", "noul": 0.93}
        elif t == "choice":
            keys = list(q["criteria"].keys())
            need(len(keys) >= 2, qid + ": >= 2 options")
            probs = {k: 0.0 for k in keys}
            probs[keys[1]] = 0.9
            probs[keys[0]] = 0.1
            answers[qid] = {"type": "choice", "choice": keys[1],
                            "probabilities": probs, "confidence": 0.85}
        elif t == "score":
            lv = q["criteria"]
            need(isinstance(lv, list) and 2 <= len(lv) <= 10, qid + ": levels")
            answers[qid] = {"type": "score", "score": 1.4,
                            "probabilities": {str(i): (0.6 if i == 1 else 0.4 if i == 2 else 0.0)
                                              for i in range(len(lv))},
                            "confidence": 0.35}
        else:
            raise Refuse(qid + ": unknown type " + t)
    return {"model": "jev-1.13.0", "answers": answers,
            "usage": {"input_tokens": 210, "output_tokens": 31}}


def anthropic_events(msg):
    yield "message_start", {"type": "message_start", "message": {
        "id": msg["id"], "type": "message", "role": "assistant",
        "model": msg["model"], "content": [], "stop_reason": None,
        "usage": {"input_tokens": msg["usage"]["input_tokens"], "output_tokens": 1}}}
    yield "ping", {"type": "ping"}
    for i, b in enumerate(msg["content"]):
        t = b["type"]
        if t == "text":
            yield "content_block_start", {"type": "content_block_start", "index": i,
                                          "content_block": {"type": "text", "text": ""}}
            h = len(b["text"]) // 2
            for part in (b["text"][:h], b["text"][h:]):
                yield "content_block_delta", {"type": "content_block_delta", "index": i,
                                              "delta": {"type": "text_delta", "text": part}}
        elif t == "thinking":
            # The real API sends no `signature` key at start: it arrives as a delta.
            yield "content_block_start", {"type": "content_block_start", "index": i,
                                          "content_block": {"type": "thinking", "thinking": ""}}
            yield "content_block_delta", {"type": "content_block_delta", "index": i,
                                          "delta": {"type": "signature_delta",
                                                    "signature": b["signature"]}}
        elif t == "tool_use":
            yield "content_block_start", {"type": "content_block_start", "index": i,
                                          "content_block": {"type": "tool_use", "id": b["id"],
                                                            "name": b["name"], "input": {}}}
            js = json.dumps(b["input"])
            h = len(js) // 2
            for part in (js[:h], js[h:]):
                yield "content_block_delta", {"type": "content_block_delta", "index": i,
                                              "delta": {"type": "input_json_delta",
                                                        "partial_json": part}}
        yield "content_block_stop", {"type": "content_block_stop", "index": i}
    yield "message_delta", {"type": "message_delta",
                            "delta": {"stop_reason": msg["stop_reason"]},
                            "usage": {"output_tokens": msg["usage"]["output_tokens"]}}
    yield "message_stop", {"type": "message_stop"}


def openai_events(resp, include_usage):
    m = resp["choices"][0]["message"]
    base = {"id": resp["id"], "object": "chat.completion.chunk", "model": resp["model"]}

    def chunk(delta, finish=None):
        return dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish}])

    yield None, chunk({"role": "assistant", "content": ""})
    text = m.get("content") or ""
    h = len(text) // 2
    for part in (text[:h], text[h:]):
        yield None, chunk({"content": part})
    for k, tc in enumerate(m.get("tool_calls", [])):
        yield None, chunk({"tool_calls": [{"index": k, "id": tc["id"], "type": "function",
                                           "function": {"name": tc["function"]["name"],
                                                        "arguments": ""}}]})
        a = tc["function"]["arguments"]
        h = len(a) // 2
        for part in (a[:h], a[h:]):
            yield None, chunk({"tool_calls": [{"index": k, "function": {"arguments": part}}]})
    yield None, chunk({}, resp["choices"][0]["finish_reason"])
    if include_usage:
        yield None, dict(base, choices=[], usage=resp["usage"])
    yield None, "[DONE]"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, status, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self, events, pause=0.02):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b": keep-alive\r\n\r\n")
        for name, data in events:
            payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            ev = ("event: %s\n" % name if name else "") + "data: " + payload + "\n\n"
            b = ev.encode("utf-8")
            cut = max(1, (len(b) * 5) // 11)  # arbitrary: lands mid-line / mid-char
            for piece in (b[:cut], b[cut:]):
                self.wfile.write(piece)
                self.wfile.flush()
                time.sleep(pause)

    def do_GET(self):
        if self.path == "/__last":
            return self.reply(200, LAST)
        if self.path == "/__shutdown":
            self.reply(200, {"bye": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self.reply(404, {"error": "no route"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        headers = {k.lower(): v for k, v in self.headers.items()}
        ctype = headers.get("content-type", "")
        LAST.clear()
        LAST.update({"path": self.path, "headers": headers,
                     "body": raw.decode("utf-8", "replace")})
        try:
            if self.path == "/v1/messages":
                req = json.loads(raw)
                msg = anthropic(req, headers)
                if req.get("stream"):
                    return self.stream(anthropic_events(msg))
                return self.reply(200, msg)
            if self.path == "/v1/chat/completions":
                req = json.loads(raw)
                resp = openai(req, headers)
                if req.get("stream"):
                    inc = (req.get("stream_options") or {}).get("include_usage", False)
                    return self.stream(openai_events(resp, inc))
                return self.reply(200, resp)
            if self.path == "/err/v1/messages":
                req = json.loads(raw)
                need(req.get("stream"), "stream expected")
                return self.stream([
                    ("message_start", {"type": "message_start", "message": {
                        "model": req["model"], "usage": {"input_tokens": 3}}}),
                    ("content_block_start", {"type": "content_block_start", "index": 0,
                                             "content_block": {"type": "text", "text": ""}}),
                    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                             "delta": {"type": "text_delta", "text": "Partial é"}}),
                    ("error", {"type": "error", "error": {"type": "overloaded_error",
                                                          "message": "Overloaded"}})])
            if self.path == "/v1/systemone":
                return self.reply(200, systemone(json.loads(raw), headers))
            if self.path == "/slow/v1/systemone":
                time.sleep(0.4)
                return self.reply(200, systemone(json.loads(raw), headers))
            if self.path in FLAKY:
                FLAKY[self.path] += 1
                if FLAKY[self.path] <= 2:
                    return self.reply(529, {"error": "overloaded"})
                return self.reply(200, systemone(json.loads(raw), headers))
            if self.path == "/bad/v1/systemone":
                return self.reply(422, {"error": "questions: field required"})
            if self.path == "/v1/audio/transcriptions":
                need(ctype.startswith("multipart/form-data"), "multipart expected")
                parts = multipart_parts(raw, ctype)
                need(parts.get("model") == b"whisper-1", "model part")
                frames, rate, ch = wav_info(parts["file"])
                need(ch == 1, "mono expected")
                return self.reply(200, {"text": "%d@%d" % (frames, rate)})
            if self.path == "/hf/whisper":
                need(ctype == "audio/wav", "audio/wav body expected")
                frames, rate, ch = wav_info(raw)
                return self.reply(200, {"text": "%d@%d" % (frames, rate)})
            if self.path == "/v1/audio/speech":
                req = json.loads(raw)
                need(req.get("response_format") == "wav", "wav requested")
                need(req.get("input"), "input text")
                return self.reply(200, make_wav(), "audio/wav")
        except Refuse as e:
            return self.reply(400, {"error": str(e)})
        except Exception as e:  # malformed body: a client bug, say so
            return self.reply(400, {"error": "%s: %s" % (type(e).__name__, e)})
        self.reply(404, {"error": "no route " + self.path})


def main():
    port_file = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Timer(seconds, srv.shutdown).start()
    with open(port_file + ".tmp", "w") as f:
        f.write(str(srv.server_address[1]))
    os.replace(port_file + ".tmp", port_file)
    srv.serve_forever()


if __name__ == "__main__":
    main()

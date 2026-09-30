# +--------------------------------------------------------------------------+ #
# | noeira.ai — hosted (and later local) models, called over HTTP
# +--------------------------------------------------------------------------+ #
"""Off-the-shelf AI models as API calls, for agents and interactive demos.

| Module | What |
|---|---|
| ``chat``      | LLMs / VLMs: Claude (Anthropic Messages) and every OpenAI-compatible server — OpenAI, HF router, Groq, OpenRouter, and LOCAL Ollama / llama.cpp / vLLM. Images, tool calls, `Conversation`. |
| ``jev``       | TypeSafe's System One (`Jev`): typed, calibrated decisions over closed option sets, ~100 ms. |
| ``speech``    | Speech-to-text (Whisper via OpenAI / Groq / HF / local) and text-to-speech. |
| ``audio_io``  | Record from the mic / play a WAV, via `ffmpeg` / `afplay`. Demo-grade. |
| ``keys``      | API keys from the environment, then `.env`. |
| ``transport`` | The shared POST — `ApiCall`, blocking or polled: retries on 429/5xx, the server's error body in the message, timing, `warm_up`. |
| ``sse``       | Incremental server-sent-events parser (the streaming framing). |

Everything rides on `noeira/io/http.mojo` — the libcurl shim, built once with
`pixi run build-http` — plus the JSON, base64, PNG and WAV codecs in
`noeira/io/`. No Python, no SDK: a demo binary links nothing new.

⚠ LOCAL LATER IS A FACTORY CHANGE. Chat and speech both speak the OpenAI
wire format as one of their options, and that is the format local servers
(Ollama, llama.cpp, vLLM, speaches, Kokoro-FastAPI) implement. Write demos
against `ChatClient` / `SpeechToText`, not against a provider.

## Blocking, background, streaming

Every client has both forms. `chat()` / `decide()` / `transcribe()` block.
`start(...)` returns at once; `poll(0)` advances the call without waiting
(call it once per frame) and `result()` collects it. `ChatClient.poll`
returns the text that streamed in since the last poll:

    conv.start(llm); jev.start(state, q)
    while running:
        caption += llm.poll()          # "" when nothing new
        if jev.poll(): act(jev.result())
        step_and_draw()

No thread is involved: libcurl's multi interface does the I/O inside `poll`,
so one loop drives any number of clients (one call in flight per client).
Measured: a poll costs < 0.5 ms on a warm connection, and a client's FIRST
call pays its TLS handshake inside one poll (8-42 ms) — `warm_up()` before
the loop.

⚠ NOT IN A CONTROL LOOP. A network round trip is 100 ms – several s with a
long tail; these are System 2 components (planning, labelling, dialogue),
never the policy that drives joints.
"""

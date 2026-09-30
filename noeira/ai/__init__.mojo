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
| ``transport`` | The shared POST: retries on 429/5xx, the server's error body in the message, timing. |

Everything rides on `noeira/io/http.mojo` — the libcurl shim, built once with
`pixi run build-http` — plus the JSON, base64, PNG and WAV codecs in
`noeira/io/`. No Python, no SDK: a demo binary links nothing new.

⚠ LOCAL LATER IS A FACTORY CHANGE. Chat and speech both speak the OpenAI
wire format as one of their options, and that is the format local servers
(Ollama, llama.cpp, vLLM, speaches, Kokoro-FastAPI) implement. Write demos
against `ChatClient` / `SpeechToText`, not against a provider.

⚠ EVERY CALL BLOCKS. There is no streaming and no async yet: a render loop
that calls a model freezes for the call's duration. For a demo that must keep
drawing, call from a worker thread (one `HttpClient` per thread — a libcurl
handle is not thread-safe) or keep model calls between episodes.

⚠ NOT IN A CONTROL LOOP. A network round trip is 100 ms – several s with a
long tail; these are System 2 components (planning, labelling, dialogue),
never the policy that drives joints.
"""

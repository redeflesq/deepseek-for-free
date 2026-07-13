# deepseek4free

> This project is no longer maintained and has been moved to a new repository.
> Please use the updated version here:
> https://github.com/redeflesq/ai-for-free

Unofficial Python client and local HTTP server for `chat.deepseek.com`,
with an automated Cloudflare-challenge bypass built in.

You get three things from this repo:

- **`DeepSeekAPI`** — a Python client for chat, file upload, and streaming
  completions, handling proof-of-work challenges and Cloudflare cookies for you.
- **A FastAPI server** exposing that client over HTTP (sessions, file
  upload, streaming/non-streaming chat) — useful when the consumer of the
  API isn't Python, or you want session state managed for you.
- **A Cloudflare-bypass toolkit** — a headless-browser service plus a
  cookie-refresh CLI, used both standalone and automatically by the client
  whenever a request hits a challenge mid-session.

See [Project layout](#project-layout) for the full module map.

## Requirements

- Python 3.10+
- Google Chrome (for the Cloudflare-bypass browser automation — only
  needed if you're refreshing cookies yourself rather than relying on
  Docker, which installs it in the image)

## Install

```bash
pip install -e ".[dev]"
```

`[dev]` pulls in pytest/ruff/mypy; drop it for a runtime-only install
(`pip install -e .` or `pip install .`).

> **Known install issue:** `curl-cffi==0.8.1b9`'s build step downloads a
> prebuilt `libcurl-impersonate` binary from GitHub Releases at install
> time. On a network that blocks or can't reach
> `github.com`/`objects.githubusercontent.com`, this install step fails
> outright rather than falling back gracefully. If you hit this, install
> from a machine/network with unrestricted GitHub access, or point pip at
> a local wheel cache that already has `curl-cffi` built.

## Configuration

Copy `.env.example` to `.env` in the project root and set at least
`DEEPSEEK_AUTH_TOKEN`. All settings live in `deepseek4free.config.Settings`
— that file is the single source of truth for defaults; every field can be
overridden via an environment variable or a `.env` entry of the same name.

| Variable                | Default   | Used by                                              |
|--------------------------|-----------|-------------------------------------------------------|
| `DEEPSEEK_AUTH_TOKEN`    | *(empty)* | `DeepSeekAPI` — bearer token for `chat.deepseek.com`  |
| `DEEPSEEK_DATA_DIR`      | `data`    | Where `cookies.json` is read from / written to        |
| `FASTAPI_SERVER_PORT`    | `8018`    | The chat HTTP server (`server/app.py`)                |
| `MAX_UPLOAD_BYTES`       | `52428800`| File upload size limit (`routes/files.py`)            |
| `OLLAMA_COMPAT_PORT`     | `11434`   | The Ollama-compatible server (`server/ollama_compat/app.py`) |
| `ENABLE_OLLAMA_API`      | `true`    | Set `false` to skip starting the Ollama-compatible server entirely |
| `SERVER_PORT`            | `8000`    | The Cloudflare-bypass service (`cloudflare/bypass_server.py`) |
| `SERVER_READY_TIMEOUT`   | `30`      | Docker entrypoint's wait for the bypass service        |
| `DOCKERMODE`             | `false`   | Headless-vs-visible browser mode in the bypass service |
| `LOG_LEVEL`              | `INFO`    | Everything                                             |

## Obtaining credentials

The client needs two things, both stored together in
`<DEEPSEEK_DATA_DIR>/cookies.json` (default `./data/cookies.json`): a
`DEEPSEEK_AUTH_TOKEN` (a bearer token tied to your DeepSeek account) and a
Cloudflare `cf_clearance` cookie.

**First-time setup, or an expired account token** — run the interactive
flow. It opens a real, visible browser window for you to log in, then
captures both the token and cookies:

```bash
python -m deepseek4free.cloudflare.cookie_refresher --manual
# or: deepseek4free-refresh-cookies --manual
```

**Routine `cf_clearance` refresh** (no browser window, fully automated) —
this is what `DeepSeekAPI` runs on its own whenever a live request hits a
Cloudflare challenge, and what the Docker entrypoint runs at container
startup:

```bash
python -m deepseek4free.cloudflare.cookie_refresher
# or: deepseek4free-refresh-cookies
```

It spawns `cloudflare.bypass_server` as a short-lived subprocess, asks it
to solve the challenge, saves the result, and tears the subprocess down
again. It does **not** obtain a new account token — only a fresh
`cf_clearance`. You can also run that bypass service standalone (e.g. to
keep it warm instead of paying subprocess-startup cost on every refresh):

```bash
python -m deepseek4free.cloudflare.bypass_server
# or: deepseek4free-bypass-server
```

## Running the chat server

```bash
python -m deepseek4free.server.app
# or: deepseek4free-server
# or, for auto-reload during development:
uvicorn deepseek4free.server.app:app --reload --port 8018
```

Starts uvicorn on `FASTAPI_SERVER_PORT` (default `8018`).

| Method | Path                          | Body / Query                                             | Notes |
|--------|-------------------------------|------------------------------------------------------------|-------|
| GET    | `/health`                     | —                                                            | `status: "ok" \| "degraded"` — checks token + cookies are actually loaded, not just that the process is up |
| POST   | `/sessions`                   | —                                                            | Creates a session, returns `session_id` |
| GET    | `/sessions`                   | —                                                            | Lists sessions currently held in memory |
| GET    | `/sessions/{id}/history`      | —                                                            | Full message history for one session |
| DELETE | `/sessions/{id}`              | —                                                            | Drops a session from memory |
| POST   | `/sessions/{id}/files`        | multipart, field `file`                                     | Returns `file_id` + `status` (`PENDING` until DeepSeek finishes parsing it) |
| GET    | `/files/status`               | `?file_ids=a,b`                                              | Poll until each file's status is `SUCCESS` before referencing it in a message |
| POST   | `/sessions/{id}/messages`     | `{prompt, thinking_enabled?, search_enabled?, file_ids?, stream?}` | `stream: true` → `text/event-stream` (SSE); otherwise one JSON body with the collected `content`/`thinking` |

Sessions live in memory only and are lost on restart — deliberate, given
the current single-process deployment (see `server/session_manager.py`'s
docstring for the reasoning).

## Ollama-compatible API

A second HTTP server, `server/ollama_compat/app.py`, speaks a practical
subset of [Ollama's REST API](https://github.com/ollama/ollama/blob/main/docs/api.md)
on top of the same `DeepSeekAPI`/`SessionManager` used by the chat server
above — so Ollama-speaking tools (Continue.dev, Open WebUI, langchain's
`ChatOllama`, etc.) can talk to DeepSeek without any DeepSeek-specific
integration on their side. It runs as its own process, on its own port
(default `11434`, matching real Ollama's default so most clients work
without reconfiguration).

```bash
python -m deepseek4free.server.ollama_compat.app
# or: deepseek4free-ollama-server
```

| Method | Path              | Notes |
|--------|-------------------|-------|
| POST   | `/api/chat`       | Full chat with message history. Streams NDJSON (`application/x-ndjson`) by default, or one JSON body with `stream: false`. |
| POST   | `/api/generate`   | Single-prompt completion (no history) — always opens a fresh DeepSeek session, matching real Ollama's stateless semantics for this endpoint. |
| GET    | `/api/tags`       | Lists the two available "models": `deepseek-chat:latest`, `deepseek-reasoner:latest`. |
| POST   | `/api/show`       | Model metadata/capabilities. 404 with `{"error": ...}` for any model not in `/api/tags`. |
| GET    | `/api/version`    | Reports this package's version. |
| GET    | `/api/ps`         | Same model list as `/api/tags`, formatted as "currently loaded" (there's no real load/unload — DeepSeek is a remote API, not local weights). |

**Model names.** DeepSeek doesn't have Ollama's per-request model choice —
the only real axis of variation is DeepSeek's own "thinking" (reasoning)
mode. This is mapped onto two model names:

| Ollama model name                              | Maps to                   |
|-------------------------------------------------|----------------------------|
| `deepseek-chat` (`:latest`/`-latest` optional)   | `thinking_enabled=False`  |
| `deepseek-reasoner` / `deepseek-r1`              | `thinking_enabled=True`   |

Any other model name returns `404 {"error": "model \"X\" not found, ..."}` —
not a silent fallback — matching what real Ollama does for an unpulled
model, since some clients (e.g. Continue.dev) key their `ollama pull`
fallback behavior off that exact status code.

**Not supported:** `/api/pull`, `/api/push`, `/api/create`, `/api/copy`,
`/api/delete`, `/api/blobs/*`, `/api/embed`, `/api/embeddings`. These all
either manage local model *files* (DeepSeek has none — it's a remote API)
or need an embeddings endpoint `DeepSeekAPI` doesn't expose. The three most
commonly auto-probed ones (`/api/pull`, `/api/embed`, `/api/embeddings`)
return an explicit `501 {"error": "not supported by ..."}` rather than a
generic, indistinguishable-from-a-typo 404; the rest fall through to
FastAPI's default 404.

**Approximate metrics.** `eval_count`/`prompt_eval_count` in responses are
a crude word-count approximation, not real token counts — `DeepSeekAPI`
exposes no tokenizer or usage accounting. Likewise `prompt_eval_duration`
is always `0` and `eval_duration` equals the full `total_duration`, since
DeepSeek's stream gives no separate timing breakdown between "processing
the prompt" and "generating the reply". Don't rely on these fields for
capacity planning, billing, or benchmarking — they exist only because
Ollama clients expect to find them in the response shape.

**Session reuse.** `/api/chat` is stateless on the wire (Ollama resends the
full message history every call) but DeepSeek threads messages through a
server-side session + `parent_message_id`. This server bridges the two
with an in-memory LRU+TTL cache (`server/ollama_compat/session_cache.py`,
200 entries / 2 hours by default) keyed by a hash of the conversation so
far: the same history prefix reuses the same DeepSeek session and thread,
a new/diverged history starts a fresh one.

### Using with Continue.dev

```json
{
  "models": [
    {
      "title": "DeepSeek Chat",
      "provider": "ollama",
      "model": "deepseek-chat",
      "apiBase": "http://localhost:11434"
    },
    {
      "title": "DeepSeek Reasoner",
      "provider": "ollama",
      "model": "deepseek-reasoner",
      "apiBase": "http://localhost:11434"
    }
  ]
}
```

## Interactive terminal chat

```bash
python -m deepseek4free.cli.chat
# or: deepseek4free-chat
```

A minimal REPL talking to `DeepSeekAPI` directly — no HTTP server
involved. Type `/exit` to quit.

## Using the client as a library

```python
from deepseek4free import DeepSeekAPI

api = DeepSeekAPI(auth_token="...")
session_id = api.create_chat_session()

for chunk in api.chat_completion(session_id, "Hello!"):
    if chunk["type"] == "text":
        print(chunk["content"], end="", flush=True)
```

`chat_completion()` yields dicts with `type` (`"text"` | `"thinking"` |
`"meta"`), `content`, and `finish_reason`. `upload_file(path)` returns the
file record (poll `fetch_file_status([...])` until `status == "SUCCESS"`
before passing its id into `chat_completion(..., file_ids=[...])`).

## Running with Docker

```bash
cp .env.example .env   # fill in DEEPSEEK_AUTH_TOKEN
cd docker
docker compose up --build
```

`docker/entrypoint.sh` starts the Cloudflare-bypass service, actively
waits for its port to come up (failing loudly, not silently, if it dies
first), refreshes cookies, then starts the Ollama-compatible API in the
background (unless `ENABLE_OLLAMA_API=false`) before finally execing the
chat server as PID 1. Both the chat server (`8018`) and the
Ollama-compatible API (`11434`) run in this one container/process group
and are both published on the host by `docker-compose.yml`. Runtime
data (`cookies.json`) persists in `./data` on the host, bind-mounted into
the container — the container image itself stays stateless.

`DEEPSEEK_AUTH_TOKEN` has **no fallback default** in `docker-compose.yml`;
it must come from your `.env` file or the host environment, or Compose
refuses to start the container at all.

## Testing

```bash
pytest tests/unit           # fast, no network
pytest tests/integration    # real chat.deepseek.com calls — needs a real
                             # DEEPSEEK_AUTH_TOKEN + an already-refreshed
                             # cookies.json, otherwise self-skips
ruff check src tests
mypy src
```

`.github/workflows/ci.yml` runs lint + typecheck + `tests/unit` on every
push/PR. `tests/integration` is intentionally excluded from CI, since it
needs real credentials and a browser-driven Cloudflare bypass that CI
runners can't provide.

## Project layout

```
src/deepseek4free/
├── config.py                 # pydantic Settings — single source of runtime config
├── exceptions.py             # DeepSeekError hierarchy shared by client + server
├── pow/solver.py             # DeepSeekHash + DeepSeekPOW (proof-of-work challenge solver)
├── client/
│   ├── transport.py          # curl_cffi HTTP layer: retries, cookies, Cloudflare detection
│   ├── sse.py                # DeepSeek's SSE stream parser (pure function, unit-testable)
│   └── api.py                # DeepSeekAPI — thin façade over transport+pow+sse
├── cloudflare/
│   ├── bypasser.py           # drives a real browser through Cloudflare's Turnstile challenge
│   ├── bypass_server.py      # FastAPI service wrapping bypasser.py
│   └── cookie_refresher.py   # canonical cookie-refresh module: automated + interactive manual login
├── server/
│   ├── schemas.py             # pydantic request/response models
│   ├── session_manager.py     # in-memory ChatSession / SessionManager
│   ├── dependencies.py        # lazy DeepSeekAPI/SessionManager singleton wiring
│   ├── errors.py              # FastAPI exception_handlers (DeepSeekError -> HTTP status)
│   ├── routes/                # one module per resource: health, sessions, files, messages
│   ├── app.py                 # FastAPI app factory (chat server, port 8018)
│   └── ollama_compat/         # Ollama-compatible API (separate process, port 11434)
│       ├── schemas.py         # Ollama-shaped request/response pydantic models
│       ├── mapping.py         # model-name <-> thinking_enabled, history -> prompt
│       ├── session_cache.py   # LRU+TTL cache: history hash -> DeepSeek session
│       ├── dependencies.py    # session_cache singleton wiring
│       ├── routes.py          # /api/chat, /api/generate, /api/tags, ...
│       └── app.py             # FastAPI app factory (this server)
└── cli/chat.py                # terminal REPL client
```

## Roadmap

Things planned but not implemented yet:

- **OpenAI-compatible API** — a third HTTP surface (`server/openai_compat/`,
  alongside the existing native REST API and the Ollama-compatible one)
  speaking `/v1/chat/completions`, `/v1/models`, and `/v1/completions` in
  the actual OpenAI request/response shape (including SSE streaming with
  `data: {...}` chunks and a final `data: [DONE]`, not Ollama's NDJSON) —
  so any OpenAI-SDK-based client (LangChain, LlamaIndex, the official
  `openai` Python/JS packages, etc.) can point `base_url` at this server
  with zero code changes, the same way the Ollama-compatible layer already
  works for Ollama-speaking clients.
- **Tool / function calling** — accept `tools`/`tool_choice` in
  `/v1/chat/completions` (and an equivalent on the native API), translate
  them into a system-prompt-based tool-use convention DeepSeek's web
  client doesn't natively expose over this API, parse the model's reply
  back into `tool_calls`, and support the follow-up turn where the caller
  supplies `role: "tool"` results. This is the single biggest gap for
  agent frameworks that assume Chat Completions-style tool calling.
- **Structured output / JSON mode** — `response_format: {"type": "json_object"}`
  (and possibly a JSON-schema-constrained variant), enforced by
  prompting plus response validation since there's no native grammar
  constraint on DeepSeek's side to hook into.
- **Multi-account / cookie pool** — rotate across more than one
  `cookies.json` account so a single rate-limited or Cloudflare-challenged
  account doesn't stall every in-flight request; today `Transport` only
  ever holds one account's cookies.
- **Usage accounting that isn't a word-count guess** — the Ollama-compatible
  layer's `eval_count`/`prompt_eval_count` are approximations by necessity
  (see that section above); a real tokenizer-based count, if one can be
  sourced or reimplemented for DeepSeek's models, would make those and any
  future OpenAI-compatible `usage` block meaningfully accurate instead of
  a documented approximation.
- **Rate limiting / backoff shared across processes** — currently each of
  the two server processes (`server/app.py`,
  `server/ollama_compat/app.py`) retries independently; a shared
  rate-limit/backoff layer would coordinate them against the same
  upstream account instead of both hammering it after a 429.

None of the above is scheduled to a specific version — this list exists so
contributors/issues can reference concrete, agreed-on next steps instead of
rediscovering the same gaps independently. PRs picking off any one item are
welcome.

## License

MIT

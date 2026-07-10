"""Pydantic request/response models for the Ollama-compatible HTTP layer.

These mirror Ollama's own JSON contract (see docs/api.md in the ollama/ollama
repo) closely enough for common clients (Continue.dev, Open WebUI, langchain's
ChatOllama, etc.) to work against this server unmodified - not a 1:1 port of
every field Ollama's real API accepts, only what actual clients send/read in
practice for /api/chat, /api/generate, /api/tags, /api/show, /api/version,
/api/ps.

Response bodies for /api/chat and /api/generate are intentionally NOT modeled
as pydantic response classes here: the non-final streamed chunks and the
final chunk have different required fields (only the final one carries
done_reason/timing/eval_count), which doesn't map cleanly onto one pydantic
model with sensible required/optional fields. routes.py builds those as plain
dicts instead - the same approach server/routes/messages.py already uses for
its own SSE payloads - since NDJSON streaming bypasses FastAPI's
response_model validation anyway.
"""

from typing import Any

from pydantic import BaseModel, Field


class OllamaMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str
    images: list[str] | None = None
    thinking: str | None = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    stream: bool = True
    # think/format/options are accepted for request-shape compatibility with
    # real Ollama clients (some send them unconditionally) but are not used -
    # DeepSeekAPI has no equivalent knob for any of them. See mapping.py /
    # routes.py docstrings for the explicit "ignored" note.
    think: Any | None = None
    format: Any | None = None
    options: dict[str, Any] | None = None
    keep_alive: Any | None = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    think: Any | None = None
    format: Any | None = None
    options: dict[str, Any] | None = None
    suffix: str | None = None
    images: list[str] | None = None
    keep_alive: Any | None = None


class OllamaModelDetails(BaseModel):
    format: str = "gguf"
    family: str = "deepseek"
    families: list[str] = Field(default_factory=lambda: ["deepseek"])
    parameter_size: str = "N/A"
    quantization_level: str = "N/A"


class OllamaModelEntry(BaseModel):
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: OllamaModelDetails


class OllamaTagsResponse(BaseModel):
    models: list[OllamaModelEntry]


class OllamaShowRequest(BaseModel):
    # Real Ollama has accepted both `model` and the older `name` field across
    # versions - clients still send either. routes.py accepts whichever is
    # present (model takes precedence if both are set) and 400s if neither is.
    model: str | None = None
    name: str | None = None


class OllamaShowResponse(BaseModel):
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: OllamaModelDetails
    model_info: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str]


class OllamaVersionResponse(BaseModel):
    version: str


class OllamaPsModelEntry(OllamaModelEntry):
    expires_at: str
    size_vram: int


class OllamaPsResponse(BaseModel):
    models: list[OllamaPsModelEntry]

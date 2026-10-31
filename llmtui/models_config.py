"""Model/provider configuration, loaded from models.json at call time.

llmtui core knows nothing about specific providers: this file describes
self-provided OpenAI-compatible endpoints (Ollama, llama.cpp server, vLLM,
LM Studio, ...) in the shape the pi harness uses, and `build_chat_model`
turns an entry into a LangChain chat model. Adding a server is a JSON edit,
never a code change. Embeddings work the same way: the top-level `embeddings`
section in models.json is a self-contained provider entry (its own baseUrl,
not a reference into `providers`, so the embedding server can differ from the
chat server) and `build_embeddings` turns it into a LangChain embeddings model.

Read at call time, not import time: the /model picker reloads the file every
time it opens, so edits show up live. The one dialect implemented today is
"openai-completions"; anything else fails loudly naming provider and model.

models.json is gitignored (providers can carry real api keys);
models.example.json is the committed template.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# overridable so tests can point at their own file
MODELS_PATH: Path = Path(
    os.getenv("LLMTUI_MODELS", PROJECT_ROOT / "config" / "models.json")
)

# the one dialect implemented; see build_chat_model
SUPPORTED_APIS = ("openai-completions",)

_ENV_REF = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})$")

# request fields sampling_params is not allowed to set: these are owned by
# the entry itself (model, reasoning) or by the caller (the conversation).
# sampling params should shape generation, never redirect or reshape the
# request; a collision here is far more likely a mistake than an intent
RESERVED_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "reasoning",
        "reasoning_effort",
        "tools",
        "tool_choice",
        "response_format",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "stop",
        "user",
    }
)

# Authorization is a header: control characters (CRLF and friends) would be
# rejected deep inside httpx at first request; fail here instead, where the
# message names the file and the provider
_UNSAFE_KEY_CHARS = re.compile(r"[\x00-\x1f\x7f\s]")

# fields the embeddings entry itself or the caller owns; params must shape
# the embedding call, never redirect it -- mirrors RESERVED_REQUEST_FIELDS
_EMBEDDINGS_RESERVED_FIELDS = frozenset(
    {"model", "api_key", "base_url", "openai_api_key", "openai_api_base"}
)


@dataclass(frozen=True)
class ModelEntry:
    provider: str
    id: str
    # True/False toggles thinking via reasoning_effort (a True maps to
    # "medium", a False to "none"): Ollama's OpenAI endpoint exposes no raw
    # think on/off, only effort levels
    # a string passes through verbatim, for endpoints that name their own
    # effort levels
    reasoning: bool | str | None = None
    # free-form request fields, merged into the request body verbatim;
    # core does not interpret these
    sampling_params: dict | None = None
    # advertised context window in tokens, for the status meter. metadata
    # only -- nothing sends it anywhere
    context_window: int = 0


@dataclass(frozen=True)
class EmbeddingsEntry:
    # picker label: which model this is, shown by the ingest prompt and
    # stored in the db metadata so search can pick the matching entry
    name: str
    # self-contained, not a provider reference: embeddings may run on a
    # different server than chat (e.g. remote API for chat, local ollama for
    # the vector store) and forcing one connection for both would forbid that
    base_url: str
    api: str
    api_key: str
    model: str
    # how many documents one /v1/embeddings call carries. bounded by the
    # model's context window -- batch chars * ~4 tokens must fit it, and the
    # client-side tokenizer only approximates the server's, so leave slack.
    # no default on purpose: required config keys, per-model/server chemistry
    batch_size: int
    # parallel in-flight embedding calls. device-bound for local servers
    # (GPU saturation), rate-limit-bound for remote ones -- a number that
    # was measured for ollama-bge-m3 tells you nothing about a remote model
    concurrency: int
    # free-form constructor kwargs for the embeddings client, merged verbatim;
    # core does not interpret them (e.g. dimensions, check_embedding_ctx_length)
    # optional, unlike the fields above: some servers need none
    params: dict | None = None


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api: str
    api_key: str
    models: tuple[ModelEntry, ...]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. Copy config/models.example.json to"
            " config/models.json and point it at a server."
        )
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{path} is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: top level must be an object")
    return data


def _interpolate(value, path: str, key: str):
    """'$NAME' / '${NAME}' in any string is read from the environment.

    A literal passes through as-is; a referenced variable that is unset or
    empty fails here, naming the file, key, and variable.
    """

    if not (isinstance(value, str) and _ENV_REF.match(value)):
        return value
    var = value[1:].strip("{}")
    resolved = os.environ.get(var)
    if not resolved:
        raise RuntimeError(f"{path}: {key} references {var}, which is unset or empty")
    return resolved


def _require(mapping: dict, key: str, path: str, where: str):
    value = _interpolate(mapping.get(key), path, f"{where}.{key}")
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(f"{path}: {where}.{key} is missing or empty")
    if not isinstance(value, str):
        raise RuntimeError(f"{path}: {where}.{key} must be a string, got {value!r}")
    return value.strip()


def _parse_model(raw, provider: str, base_url: str, path: str) -> ModelEntry:
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"{path}: models entries of provider {provider!r} must be objects"
        )
    where = f"provider {provider!r} model"
    model_id = _require(raw, "id", path, where)

    reasoning = raw.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, (bool, str)):
        raise RuntimeError(
            f"{path}: {where} {model_id!r} 'reasoning' must be a bool or string"
        )
    if isinstance(reasoning, str):
        reasoning = _interpolate(reasoning, path, f"{where} {model_id!r} reasoning")

    sampling = raw.get("samplingParams")
    if sampling is not None:
        if not isinstance(sampling, dict):
            raise RuntimeError(
                f"{path}: {where} {model_id!r} 'samplingParams' must be an object"
            )
        for key in sampling:
            if key.lower() in RESERVED_REQUEST_FIELDS:
                raise RuntimeError(
                    f"{path}: {where} {model_id!r} samplingParams.{key} is a"
                    " reserved request field and cannot be overridden"
                )
        sampling = {
            key: _interpolate(value, path, f"{where} {model_id!r} samplingParams.{key}")
            for key, value in sampling.items()
        }

    context_window = raw.get("contextWindow", 0)
    if not isinstance(context_window, int) or context_window < 0:
        raise RuntimeError(
            f"{path}: {where} {model_id!r} 'contextWindow' must be a non-negative integer"
        )

    return ModelEntry(
        provider=provider,
        id=model_id,
        reasoning=reasoning,
        sampling_params=sampling,
        context_window=context_window,
    )


def load_models(path: Path | None = None) -> dict[str, Provider]:
    """Read and validate models.json. Raises at every call, naming the file.

    Returns providers keyed by name; each holds its model entries. Called at
    app start and again every time the /model picker opens, so live edits
    are picked up.
    """

    path = path or MODELS_PATH
    data = _load_json(path)

    providers_raw = data.get("providers")
    if not isinstance(providers_raw, dict) or not providers_raw:
        raise RuntimeError(f"{path}: 'providers' must be a non-empty object")

    providers: dict[str, Provider] = {}
    for name, prov in providers_raw.items():
        if not isinstance(prov, dict):
            raise RuntimeError(f"{path}: provider {name!r} must be an object")
        where = f"provider {name!r}"
        base_url = _require(prov, "baseUrl", path, f"{where} connection")
        api = _require(prov, "api", path, f"{where} connection")
        api_key = _require(prov, "apiKey", path, f"{where} connection")

        if not base_url.startswith(("http://", "https://")):
            raise RuntimeError(
                f"{path}: {where}.baseUrl must be an http(s) url, got {base_url!r}"
            )
        if _UNSAFE_KEY_CHARS.search(api_key):
            raise RuntimeError(
                f"{path}: {where}.apiKey contains control characters or whitespace"
            )

        if api not in SUPPORTED_APIS:
            raise RuntimeError(
                f"{path}: {where} declares api {api!r}; supported: {SUPPORTED_APIS}"
            )

        models_raw = prov.get("models")
        if not isinstance(models_raw, list) or not models_raw:
            raise RuntimeError(f"{path}: {where}.models must be a non-empty list")

        seen: set[str] = set()
        models = []
        for raw in models_raw:
            entry = _parse_model(raw, name, base_url, path)
            if entry.id in seen:
                raise RuntimeError(
                    f"{path}: duplicate model id {entry.id!r} in {where}"
                )
            seen.add(entry.id)
            models.append(entry)

        providers[name] = Provider(
            name=name,
            base_url=base_url,
            api=api,
            api_key=api_key,
            models=tuple(models),
        )
    return providers


def load_embeddings(path: Path | None = None) -> EmbeddingsEntry:
    """Read and validate the top-level 'embeddings' section of models.json.

    Raises at every call, naming the file. Called at ingest and at tool
    import, so the same live-reload contract as load_models applies. A missing
    or absent section raises: without a configured embedding model the vector
    store cannot load and the run must not start on a guessed default -- no
    server is guaranteed to exist on the device, so the failure comes from
    here, naming the file. Returns the entry.
    """

    path = path or MODELS_PATH
    data = _load_json(path)

    raw = data.get("embeddings")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(
            f"{path}: missing or malformed 'embeddings' section (must be a"
            " non-empty list of entries with name, baseUrl, api, apiKey,"
            " model, batchSize, concurrency)"
        )

    entries: dict[str, EmbeddingsEntry] = {}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeError(f"{path}: embeddings[{i}] must be an object")
        where = f"embeddings[{i}]"
        name = _require(item, "name", path, where)
        if name in entries:
            raise RuntimeError(f"{path}: duplicate embeddings name {name!r}")
        entry = _parse_embeddings_entry(item, path, where, name)
        entries[name] = entry
    return entries


def _parse_embeddings_entry(
    raw: dict, path: str, where: str, name: str
) -> EmbeddingsEntry:
    """Validate one embeddings list entry; raises naming the file and index."""

    base_url = _require(raw, "baseUrl", path, f"{where} connection")
    api = _require(raw, "api", path, f"{where} connection")
    api_key = _require(raw, "apiKey", path, f"{where} connection")
    model = _require(raw, "model", path, where)

    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError(
            f"{path}: {where}.baseUrl must be an http(s) url, got {base_url!r}"
        )
    if _UNSAFE_KEY_CHARS.search(api_key):
        raise RuntimeError(
            f"{path}: {where}.apiKey contains control characters or whitespace"
        )
    if api not in SUPPORTED_APIS:
        raise RuntimeError(
            f"{path}: {where} declares api {api!r}; supported: {SUPPORTED_APIS}"
        )

    params = raw.get("params")
    if params is not None:
        if not isinstance(params, dict):
            raise RuntimeError(f"{path}: {where}.params must be an object")
        for key in params:
            if key.lower() in _EMBEDDINGS_RESERVED_FIELDS:
                raise RuntimeError(
                    f"{path}: {where}.params.{key} is a reserved field and"
                    " cannot be overridden"
                )
        params = {
            key: _interpolate(value, path, f"{where}.params.{key}")
            for key, value in params.items()
        }

    batch_size = raw.get("batchSize")
    concurrency = raw.get("concurrency")
    # required, not defaulted: they are per-model/server chemistry -- one
    # call's document count is bounded by the model's context window, and
    # the safe parallel-call count is device-bound locally (GPU saturation)
    # but rate-limit-bound remotely. there is no number that is right for
    # every model, so the config must state them
    if not isinstance(batch_size, int) or batch_size < 1:
        raise RuntimeError(
            f"{path}: {where}.batchSize is required and must be a positive"
            f" integer, got {batch_size!r}"
        )
    if not isinstance(concurrency, int) or concurrency < 1:
        raise RuntimeError(
            f"{path}: {where}.concurrency is required and must be a positive"
            f" integer, got {concurrency!r}"
        )

    return EmbeddingsEntry(
        name=name,
        base_url=base_url,
        api=api,
        api_key=api_key,
        model=model,
        params=params,
        batch_size=batch_size,
        concurrency=concurrency,
    )


def flatten(providers: dict[str, Provider]) -> list[ModelEntry]:
    """All model entries, in provider-then-file order, for pickers."""

    return [m for p in providers.values() for m in p.models]


# ---------------------------------------------------------------------
# the dialect. one function, one api; a new dialect means a new branch in
# here, not a change anywhere else in the codebase
# ---------------------------------------------------------------------


def _reasoning_effort(entry: ModelEntry):
    if entry.reasoning is None:
        return None
    if isinstance(entry.reasoning, str):
        return entry.reasoning
    return "medium" if entry.reasoning else "none"


def build_chat_model(entry: ModelEntry, provider: Provider):
    """Turn a validated entry into a chat model. Only openai-completions exists.

    sampling_params are merged into the request body verbatim via extra_body --
    the server decides which fields mean what (o. top_k / repeat_penalty on
    Ollama), core does not interpret them. temperature/top_p are also sent
    through extra_body so that what lands on the wire is exactly what the
    json asked for, nothing normalized away.
    """

    if provider.api != "openai-completions":
        raise RuntimeError(
            f"no builder for api {provider.api!r} (provider {provider.name!r})"
        )

    extra_body = dict(entry.sampling_params or {})
    effort = _reasoning_effort(entry)
    if effort is not None:
        extra_body["reasoning_effort"] = effort

    return ChatOpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key,
        model=entry.id,
        extra_body=extra_body or None,
    )


# ---------------------------------------------------------------------
# the embeddings dialect. mirrors build_chat_model: one function, one api,
# a new dialect means a new branch here, not a change anywhere else
# ---------------------------------------------------------------------


def build_embeddings(entry: EmbeddingsEntry):
    """Turn a validated embeddings entry into a LangChain embeddings model.

    Only openai-completions exists: /v1/embeddings is the one convention
    served across the stacks this app targets (Ollama, vLLM, LM Studio,
    remote APIs like TensorX or OpenRouter). params are passed to the
    constructor verbatim -- the server/client decides which fields mean
    what, core does not interpret them.
    """

    if entry.api != "openai-completions":
        raise RuntimeError(f"no embeddings builder for api {entry.api!r}")

    params = dict(entry.params or {})
    # Bound dead response bodies by default while preserving provider overrides.
    params.setdefault("timeout", 120.0)
    params.setdefault("max_retries", 2)

    return OpenAIEmbeddings(
        base_url=entry.base_url,
        api_key=entry.api_key,
        model=entry.model,
        **params,
    )


# ---------------------------------------------------------------------
# context length from the provider
#
# metadata (contextWindow on the entry) wins when declared; this is the
# live fallback for servers whose entries omit it. best-effort only -- a
# failure here leaves the meter at 0 (plain token count), never raises
# into the run
# ---------------------------------------------------------------------


async def fetch_context_window(entry: ModelEntry, provider: Provider) -> int:
    """Ask the server how long the context is. 0 when it cannot say.

    Only ollama has a shaped answer today (/api/show on the native port,
    model_info.<arch>.context_length). The /v1 compat endpoint does not
    expose it, so the baseUrl is rewritten to the native one. Other
    openai-compat servers fall through to 0 promptly.
    """

    if provider.name != "ollama":
        return 0

    import httpx

    # http://localhost:11434/v1 -> http://localhost:11434
    native = provider.base_url.rstrip("/")
    native = native.removesuffix("/v1")

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(
                f"{native}/api/show",
                json={"model": entry.id},
                headers={"Authorization": f"Bearer {provider.api_key}"},
            )
            r.raise_for_status()
            info = r.json().get("model_info") or {}
        lengths = [v for k, v in info.items() if k.endswith(".context_length")]
        return max(lengths) if lengths else 0
    except Exception:
        return 0

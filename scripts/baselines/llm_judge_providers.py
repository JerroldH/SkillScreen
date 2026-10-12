"""Provider adapters for the zero-shot judges: Anthropic Messages and OpenAI-compatible chat completions.

Keys and response bodies are never logged or attached to exceptions. Clients are injected through
``client_factory`` so the whole pipeline is testable offline.
"""
import dataclasses
import json
import os
import time
from types import SimpleNamespace

import importlib

from llm_judge_prompt import SCHEMA, ParseFailure, build_messages, parse_verdict

DEFAULT_PROMPT_MODULE = "llm_judge_prompt"


def load_prompt_module(name=None):
    """Prompt module named by a registry entry (default: the frozen zero-shot prompt). Every module exposes
    build_messages, prompt_sha256 and PROMPT_VERSION and shares the fixed answer schema."""
    module = importlib.import_module(name or DEFAULT_PROMPT_MODULE)
    for attribute in ("build_messages", "prompt_sha256", "PROMPT_VERSION", "SCHEMA"):
        if not hasattr(module, attribute):
            raise ValueError(f"Prompt module {module.__name__} lacks {attribute}")
    if module.SCHEMA != SCHEMA:
        raise ValueError("Prompt modules must share the fixed answer schema")
    return module


def prompt_hash(entry):
    return load_prompt_module(entry.get("prompt_module")).prompt_sha256()


def prompt_version(entry):
    return load_prompt_module(entry.get("prompt_module")).PROMPT_VERSION

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
OUTER_RETRY_DELAY_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 180.0
SDK_MAX_RETRIES = 4
FORBIDDEN_PARAMS = ("temperature", "seed", "top_p", "top_k")
MAX_TOKEN_REASONS = {"max_tokens", "length"}
BATCH_FALLBACK_REASONS = {"batch_errored", "batch_expired", "batch_canceled", "parse_failure"}
ANSWER_KEEP_CHARACTERS = 2000


class JudgeError(RuntimeError):
    """A sanitized provider failure: class name and HTTP status only."""


@dataclasses.dataclass
class Outcome:
    score: object = None
    verdict: object = None
    returned_model: str = ""
    stop_reason: object = None
    refusal_category: object = None
    unavailable_reason: object = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: object = None
    cached_input_tokens: int = 0
    request_id: object = None
    attempts: int = 1
    latency_seconds: float = 0.0
    answer: object = None
    mode: str = "sync"

    def to_dict(self):
        return dataclasses.asdict(self)

    @property
    def available(self):
        return self.unavailable_reason is None


def read_key(provider, root):
    """Key from the provider's environment variable or its git-ignored ``KEY=value`` file."""
    env_name = provider["key_env"]
    value = os.environ.get(env_name, "").strip()
    path = root / provider["key_file"]
    if not value and path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith(env_name + "="):
                value = line.split("=", 1)[1].strip().strip("\"'")
    if not value:
        raise ValueError(f"Set {env_name} or save it in the ignored {provider['key_file']} file")
    return value


def _status_code(exc):
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def retryable(exc):
    status = _status_code(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & {"APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutError"})


def sanitized(provider_name, exc):
    status = _status_code(exc)
    detail = f" (HTTP {status})" if status is not None else ""
    return JudgeError(f"{provider_name} request failed with {type(exc).__name__}{detail}; "
                      "response body and credentials are not logged")


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _int(value):
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _anthropic_client(**kwargs):
    import anthropic
    return anthropic.Anthropic(**kwargs)


def _openai_client(**kwargs):
    import openai
    return openai.OpenAI(**kwargs)


class _Judge:
    provider_name = ""
    supports_batch = False

    def __init__(self, entry, provider, key, client_factory=None, provider_name=None):
        self.entry = entry
        self.provider = provider
        self.provider_name = provider_name or provider.get("name") or self.provider_name
        self.model = entry["id"]
        # Resellers may answer with the upstream model name rather than their own alias.
        self.returned_prefix = entry.get("returned_id_prefix") or entry["id"]
        self.request = dict(entry["request"])
        self.prompt = load_prompt_module(entry.get("prompt_module"))
        self.build_messages = self.prompt.build_messages
        self.supports_batch = bool(provider.get("batch"))
        kwargs = {"api_key": key, "max_retries": SDK_MAX_RETRIES, "timeout": REQUEST_TIMEOUT_SECONDS}
        if provider.get("base_url"):
            kwargs["base_url"] = provider["base_url"]
        self.client = (client_factory or self.default_factory)(**kwargs)
        self.sleep = time.sleep

    @staticmethod
    def default_factory(**kwargs):
        raise NotImplementedError

    def _check_params(self, params):
        for name in FORBIDDEN_PARAMS:
            if name in params:
                raise ValueError(f"Judge requests never set {name}")
        return params

    def _check_returned_model(self, returned):
        if not isinstance(returned, str) or not returned.startswith(self.returned_prefix):
            raise JudgeError(f"{self.provider_name} returned model {returned!r} does not match requested {self.model} "
                             f"(expected prefix {self.returned_prefix!r})")

    def _call(self, params, attempts):
        """One request with a single outer retry after 30 s for rate-limit/5xx/connection errors."""
        for attempt in range(2):
            attempts[0] += 1
            try:
                return self._create(params)
            except JudgeError:
                raise
            except Exception as exc:  # noqa: BLE001 - every SDK error is sanitized here
                if attempt == 0 and retryable(exc):
                    self.sleep(OUTER_RETRY_DELAY_SECONDS)
                    continue
                raise sanitized(self.provider_name, exc) from None
        raise AssertionError("unreachable")

    def judge(self, markdown):
        params = self.request_params(markdown)
        started = time.perf_counter()
        attempts = [0]
        response = self._call(params, attempts)
        try:
            return self._classify(response, attempts=attempts[0], latency=time.perf_counter() - started, mode="sync")
        except ParseFailure:
            pass
        response = self._call(params, attempts)
        return self._classify(response, attempts=attempts[0], latency=time.perf_counter() - started, mode="sync", final=True)

    def check_model(self):
        try:
            returned = self.client.models.retrieve(self.model)
        except Exception as exc:  # noqa: BLE001
            raise sanitized(self.provider_name, exc) from None
        returned_id = getattr(returned, "id", None)
        self._check_returned_model(returned_id)
        return returned_id

    def _finish(self, base, text, final):
        try:
            verdict, probability = parse_verdict(text)
        except ParseFailure:
            if not final:
                raise
            return Outcome(score=None, verdict=None, unavailable_reason="parse_failure",
                           answer=text[:ANSWER_KEEP_CHARACTERS], **base)
        return Outcome(score=probability, verdict=verdict, answer=text[:ANSWER_KEEP_CHARACTERS], **base)


class AnthropicJudge(_Judge):
    provider_name = "anthropic"
    default_factory = staticmethod(_anthropic_client)

    def request_params(self, markdown):
        messages = self.build_messages(markdown)
        request = dict(self.request)
        output_config = dict(request.pop("output_config", None) or {})
        output_config["format"] = {"type": "json_schema", "schema": SCHEMA}
        thinking = request.pop("thinking", None)
        params = {"model": self.model, "max_tokens": int(request.pop("max_tokens")), "system": messages["system"],
                  "messages": [{"role": "user", "content": messages["user"]}], "output_config": output_config}
        if thinking:
            params["thinking"] = thinking
        if request:
            raise ValueError("Unexpected Anthropic request fields: " + ", ".join(sorted(request)))
        return self._check_params(params)

    def _create(self, params):
        return self.client.messages.create(**params)

    def _classify(self, message, attempts, latency, mode, final=False, request_id=None):
        returned = getattr(message, "model", None)
        self._check_returned_model(returned)
        usage = getattr(message, "usage", None)
        stop = getattr(message, "stop_reason", None)
        blocks = getattr(message, "content", None) or []
        text = "".join(getattr(block, "text", "") or "" for block in blocks if getattr(block, "type", None) == "text")
        base = {"returned_model": returned, "stop_reason": stop, "input_tokens": _int(getattr(usage, "input_tokens", 0)),
                "output_tokens": _int(getattr(usage, "output_tokens", 0)), "reasoning_tokens": None,
                "cached_input_tokens": _int(getattr(usage, "cache_read_input_tokens", 0)),
                "request_id": request_id if request_id is not None else getattr(message, "_request_id", None),
                "attempts": attempts, "latency_seconds": latency, "mode": mode}
        if stop == "refusal":
            category = getattr(getattr(message, "stop_details", None), "category", None)
            return Outcome(unavailable_reason="refusal", refusal_category=category, **base)
        if stop == "max_tokens":
            return Outcome(unavailable_reason="max_tokens", answer=text[:ANSWER_KEEP_CHARACTERS] or None, **base)
        if not text.strip():
            return Outcome(unavailable_reason="empty_output", **base)
        if stop not in (None, "end_turn", "stop_sequence"):
            return Outcome(unavailable_reason=str(stop), answer=text[:ANSWER_KEEP_CHARACTERS], **base)
        return self._finish(base, text, final)

    def batch_submit(self, items):
        requests = [{"custom_id": custom_id, "params": self.request_params(markdown)} for custom_id, markdown in items]
        try:
            batch = self.client.messages.batches.create(requests=requests)
        except Exception as exc:  # noqa: BLE001
            raise sanitized(self.provider_name, exc) from None
        return batch.id, {}

    def batch_status(self, batch_id):
        try:
            batch = self.client.messages.batches.retrieve(batch_id)
        except Exception as exc:  # noqa: BLE001
            raise sanitized(self.provider_name, exc) from None
        counts = getattr(batch, "request_counts", None)
        return {"status": batch.processing_status, "finished": batch.processing_status == "ended",
                "counts": {name: _int(getattr(counts, name, 0)) for name in ("processing", "succeeded", "errored", "canceled", "expired")}}

    def batch_results(self, batch_id):
        outcomes = {}
        try:
            for item in self.client.messages.batches.results(batch_id):
                kind = item.result.type
                if kind == "succeeded":
                    outcomes[item.custom_id] = self._classify(item.result.message, attempts=1, latency=0.0, mode="batch", final=True)
                elif kind in ("errored", "expired", "canceled"):
                    outcomes[item.custom_id] = Outcome(unavailable_reason="batch_" + kind, mode="batch")
                else:
                    raise JudgeError(f"{self.provider_name} batch returned unknown result type {kind!r}")
        except JudgeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise sanitized(self.provider_name, exc) from None
        return outcomes


class OpenAICompatibleJudge(_Judge):
    provider_name = "openai"
    default_factory = staticmethod(_openai_client)

    def request_params(self, markdown):
        messages = self.build_messages(markdown)
        params = {"model": self.model,
                  "messages": [{"role": "system", "content": messages["system"]}, {"role": "user", "content": messages["user"]}],
                  "response_format": {"type": "json_schema", "json_schema": {"name": "skill_verdict", "schema": SCHEMA, "strict": True}}}
        params.update(self.request)
        if self.provider.get("store_false", True):
            params["store"] = False
        return self._check_params(params)

    def _create(self, params):
        return self.client.chat.completions.create(**params)

    def _classify(self, completion, attempts, latency, mode, final=False, request_id=None):
        returned = getattr(completion, "model", None)
        self._check_returned_model(returned)
        choices = getattr(completion, "choices", None) or []
        choice = choices[0] if choices else SimpleNamespace(finish_reason=None, message=None)
        message = getattr(choice, "message", None)
        finish = getattr(choice, "finish_reason", None)
        usage = getattr(completion, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None)
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0)
        text = getattr(message, "content", None) or ""
        if not isinstance(text, str):
            text = "".join(getattr(part, "text", "") or "" for part in text if getattr(part, "type", None) == "text")
        base = {"returned_model": returned, "stop_reason": finish, "input_tokens": _int(getattr(usage, "prompt_tokens", 0)),
                "output_tokens": _int(getattr(usage, "completion_tokens", 0)),
                "reasoning_tokens": _int(reasoning) if reasoning is not None else None, "cached_input_tokens": _int(cached),
                "request_id": request_id if request_id is not None else getattr(completion, "_request_id", None),
                "attempts": attempts, "latency_seconds": latency, "mode": mode}
        if getattr(message, "refusal", None):
            return Outcome(unavailable_reason="refusal", **base)
        if finish in ("length", "content_filter"):
            return Outcome(unavailable_reason=finish, answer=text[:ANSWER_KEEP_CHARACTERS] or None, **base)
        if not text.strip():
            return Outcome(unavailable_reason="empty_output", **base)
        return self._finish(base, text, final)

    def batch_submit(self, items):
        if not self.supports_batch:
            raise JudgeError(f"{self.provider_name} has no batch endpoint")
        lines = [{"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": self.request_params(markdown)}
                 for custom_id, markdown in items]
        payload = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines).encode("utf-8") + b"\n"
        try:
            uploaded = self.client.files.create(file=("judge-batch.jsonl", payload), purpose="batch")
            batch = self.client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")
        except Exception as exc:  # noqa: BLE001
            raise sanitized(self.provider_name, exc) from None
        return batch.id, {"input_file_id": uploaded.id}

    def batch_status(self, batch_id):
        try:
            batch = self.client.batches.retrieve(batch_id)
        except Exception as exc:  # noqa: BLE001
            raise sanitized(self.provider_name, exc) from None
        counts = getattr(batch, "request_counts", None)
        return {"status": batch.status, "finished": batch.status in ("completed", "failed", "expired", "cancelled"),
                "counts": {name: _int(getattr(counts, name, 0)) for name in ("total", "completed", "failed")},
                "output_file_id": getattr(batch, "output_file_id", None), "error_file_id": getattr(batch, "error_file_id", None)}

    @staticmethod
    def _file_text(content):
        text = getattr(content, "text", None)
        if isinstance(text, str):
            return text
        raw = getattr(content, "content", None)
        if raw is None and hasattr(content, "read"):
            raw = content.read()
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw or "")

    def batch_results(self, batch_id):
        outcomes = {}
        try:
            batch = self.client.batches.retrieve(batch_id)
            if getattr(batch, "output_file_id", None):
                for line in self._file_text(self.client.files.content(batch.output_file_id)).splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    custom_id = record["custom_id"]
                    response = record.get("response") or {}
                    if record.get("error") or response.get("status_code") != 200 or not response.get("body"):
                        outcomes[custom_id] = Outcome(unavailable_reason="batch_errored", mode="batch")
                        continue
                    outcomes[custom_id] = self._classify(_namespace(response["body"]), attempts=1, latency=0.0, mode="batch",
                                                         final=True, request_id=response.get("request_id"))
            if getattr(batch, "error_file_id", None):
                for line in self._file_text(self.client.files.content(batch.error_file_id)).splitlines():
                    if line.strip():
                        outcomes.setdefault(json.loads(line)["custom_id"], Outcome(unavailable_reason="batch_errored", mode="batch"))
        except JudgeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise sanitized(self.provider_name, exc) from None
        return outcomes


JUDGES = {"anthropic": AnthropicJudge, "openai": OpenAICompatibleJudge}


def make_judge(entry, provider, provider_name, key, client_factory=None):
    try:
        cls = JUDGES[provider["sdk"]]
    except KeyError:
        raise ValueError(f"Unknown provider SDK for {provider_name}") from None
    return cls(entry, provider, key, client_factory=client_factory, provider_name=provider_name)

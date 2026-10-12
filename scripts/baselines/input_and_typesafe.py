"""Input and HTTP functions extracted from the frozen experiment implementation."""
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def tokenizer_length_fingerprint(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # Encoding sets these runtime flags; vocabulary, normalization and BPE remain pinned.
    backend["padding"] = backend["truncation"] = None
    return hashlib.sha256(json.dumps(backend, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def encode_skill(tokenizer, text, hypothesis, max_length):
    if len(text) > 262144:
        profile_path = ROOT / "configs/baselines/large-input-profile.json"
        if not profile_path.exists():
            raise ValueError("Very large inputs require a verified text/tokenizer length profile")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        reference = profile.get(hashlib.sha256(text.encode()).hexdigest())
        tokenizer_sha = tokenizer_length_fingerprint(tokenizer)
        if not reference or reference["hypothesis"] != hypothesis or reference["tokenizer_sha256"] != tokenizer_sha:
            raise ValueError("Large-input text, hypothesis or tokenizer differs from its profile")
        # Reuse the checked ASCII prefix encoder; never allocate the full giant token list.
        tokens, original_length, _ = bounded_feature(tokenizer, text, hypothesis, max_length, reference["original_tokens"])
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}, original_length
    suffix = "\nHypothesis: " + hypothesis
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if max_length <= len(suffix_ids) + 8:
        raise ValueError("Input budget is too small for the premise and fixed hypothesis")
    tokens = tokenizer.encode("Premise: " + text + suffix, add_special_tokens=False)
    original_length = len(tokens)
    if original_length > max_length:
        # Trim only the premise; generic right truncation would remove the hypothesis.
        tokens = tokens[:max_length - len(suffix_ids)] + suffix_ids
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}, original_length


def bounded_feature(tokenizer, text, hypothesis, max_length, original_length):
    if len(text) <= 262144:
        feature, actual = encode_skill(tokenizer, text, hypothesis, max_length)
        if original_length is not None and actual != original_length:
            raise ValueError("Original token count disagrees with the pinned training profile")
        return feature["input_ids"], actual, False
    if original_length is None or original_length <= max_length:
        raise ValueError("Very large inputs require an existing exact token-count profile")
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # the one giant record is ASCII. Reject other giant inputs rather
    # than introducing a general Unicode streaming-normalization implementation.
    if not text.isascii() or backend["normalizer"] not in (None, {"type": "NFC"}) or backend["model"].get("dropout"):
        raise ValueError("Bounded encoding expects ASCII with identity/NFC normalization and deterministic BPE")
    suffix = tokenizer.encode("\nHypothesis: " + hypothesis, add_special_tokens=False)
    keep = max_length - len(suffix)
    if keep <= 8:
        raise ValueError("Input budget cannot preserve the fixed hypothesis")
    premise = "Premise: " + text
    size = 32768
    while size <= 1048576:
        prefix = premise[:size]
        spans = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(prefix)
        encoded = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
        # All retained tokens must end before the final two pre-tokenizer spans.
        # This excludes a trailing word/whitespace span that could extend later.
        if len(spans) >= 3 and len(encoded["input_ids"]) >= keep and (
                encoded["offset_mapping"][keep - 1][1] <= spans[-2][1][0]):
            stable = encoded["input_ids"][:keep]
            extended = tokenizer.encode(premise[:size * 2], add_special_tokens=False)
            if stable != extended[:keep]:
                raise ValueError("Bounded prefix is not stable under extension")
            return stable + suffix, original_length, True
        size *= 2
    raise ValueError("No safe bounded token prefix found; do not silently change the input policy")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_score(key, model, state, hypothesis, attempts=3):
    payload = {"model": model, "state": state, "questions": {
        "malicious": {"type": "noul", "instructions": hypothesis}}}
    started = time.perf_counter()
    opener = urllib.request.build_opener(NoRedirect())
    for attempt in range(attempts):
        request = urllib.request.Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
        try:
            with opener.open(request, timeout=45) as response:
                data = json.load(response)
                request_id = response.headers.get("x-typesafe-request-id", "")
            break
        except urllib.error.HTTPError as exc:
            retryable = exc.code in (408, 429, 500, 502, 503, 504, 529)
            if not retryable or attempt + 1 == attempts:
                raise RuntimeError(f"TypeSafe HTTP {exc.code}; response text and authorization are not logged") from None
            delay = exc.headers.get("Retry-After", "")
            time.sleep(min(10, float(delay)) if delay.replace(".", "", 1).isdigit() else 2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise RuntimeError("TypeSafe connection failed after bounded retries") from None
            time.sleep(2 ** attempt)
    answer = data.get("answers", {}).get("malicious", {})
    score = answer.get("noul")
    if data.get("model") != model or answer.get("type") != "noul":
        raise ValueError("Unexpected model version or answer type; evaluation stopped")
    if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Expected a finite Noul probability in [0, 1]")
    usage = data.get("usage", {})
    for field in ("input_tokens", "output_tokens"):
        if not isinstance(usage.get(field), int) or usage[field] < 0:
            raise ValueError("Expected recorded token usage")
    return {"score": score, "model": data["model"], "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"], "request_id": request_id,
        "attempts": attempt + 1, "latency_seconds": time.perf_counter() - started}


def premise(tokenizer, text, hypothesis, limit):
    feature, original = encode_skill(tokenizer, text, hypothesis, limit)
    suffix_ids = tokenizer.encode("\nHypothesis: " + hypothesis, add_special_tokens=False)
    if original <= limit:
        state = "Premise: " + text
    else:
        state = tokenizer.decode(feature["input_ids"][:-len(suffix_ids)], skip_special_tokens=False)
    return state, original, len(feature["input_ids"])

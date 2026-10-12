"""Original experiment helpers, limited to functions used by this package."""

import argparse

from collections import Counter

import csv

import hashlib

import importlib.metadata

import json

import os

from pathlib import Path

import random

import time

ROOT = Path(__file__).resolve().parents[1]

def configure_gpu_memory(torch):
    total = torch.cuda.get_device_properties(0).total_memory
    full = os.environ.get("SKILLSCREEN_FULL_GPU") == "1"
    budget = total / 1024**3 if full else min(8., total / 1024**3)
    if not full:
        torch.cuda.set_per_process_memory_fraction(min(1., 8 * 1024**3 / total))
    return total, budget

def tokenizer_length_fingerprint(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # Encoding sets these runtime flags; vocabulary, normalization and BPE remain pinned.
    backend["padding"] = backend["truncation"] = None
    return hashlib.sha256(json.dumps(backend, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def encode_skill(tokenizer, text, hypothesis, max_length):
    if len(text) > 262144:
        profile_path = ROOT / "outputs/large-input-profile.json"
        if not profile_path.exists():
            raise ValueError("Very large inputs require a verified text/tokenizer length profile")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        reference = profile.get(hashlib.sha256(text.encode()).hexdigest())
        tokenizer_sha = tokenizer_length_fingerprint(tokenizer)
        if not reference or reference["hypothesis"] != hypothesis or reference["tokenizer_sha256"] != tokenizer_sha:
            raise ValueError("Large-input text, hypothesis or tokenizer differs from its profile")
        # Reuse the checked ASCII prefix encoder; never allocate the full giant token list.
        from cache_direct_inputs import bounded_feature
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

def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


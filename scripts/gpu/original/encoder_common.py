"""Shared helpers for the encoder baselines trained on raw SKILL.md text (ModernBERT-large and similar).

The encoder path mirrors the Direct route stage for stage (training-only preflight, export verification, one epoch with
validation at the midpoint and the end, validation-only freeze, frozen test scoring, report) but fine-tunes every parameter
of a bidirectional encoder with a fresh two-way head on the raw skill text, without the NLI premise/hypothesis template.
The hash-pinned Direct scripts are imported, never edited. Every decision variable is the two-way log-odds
malicious_logit - benign_logit computed in float64 from the float32 logits written to the score files, so checkpoint
selection, the freeze and the test report agree bit for bit.
"""
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct

import numpy as np

from calibrate_direct import code_digest, probabilities, save_json
from direct_smoke import ROOT
from probe_train_memory import digest

CONFIG = "configs/encoder.json"
CACHE = "outputs/encoder-inputs-8k"
CACHE_FILES = ("manifest.json", "train.bin", "train.json", "validation.bin", "validation.json")
LABELS = {"benign": 0, "malicious": 1}
SCORE_COLUMNS = ("benchmark_id", "label", "score", "benign_logit", "malicious_logit", "original_tokens", "used_tokens", "truncated")
TEST_COLUMNS = SCORE_COLUMNS + ("calibrated_probability", "calibrated_threshold", "review_decision")
CHECKPOINT_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
EVALUATION_SCRIPTS = ("encoder_common.py", "train_encoder.py", "freeze_encoder_run.py", "score_encoder.py", "train_direct.py",
                      "freeze_direct_run.py", "calibrate_direct.py", "word_svm.py", "direct_smoke.py", "probe_train_memory.py")
HEAD_KEYS = {"classifier.weight", "classifier.bias"}
DISCARDED_KEYS = {"decoder.weight", "decoder.bias"}
DEFAULT_TRAINING = {
    "epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 16, "optimizer": "AdamW", "learning_rate": 2e-5,
    "weight_decay": 0.01, "weight_decay_scope": "parameters with at least two dimensions", "warmup_fraction": 0.05,
    "schedule": "linear", "max_grad_norm": 1.0, "trainable": "all parameters", "head_init": "torch seed = run seed",
    "precision": "bfloat16 autocast over float32 master weights", "gradient_checkpointing": "nonreentrant on CUDA"}


def load_config(root=ROOT):
    return json.loads((root / CONFIG).read_text(encoding="utf-8"))


def check_seed(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    return seed


def encoder_config(base_preparation, dataset, run_id, seed, max_length=8192, training=None, variant=None):
    """Workspace configs/encoder.json for one (base, protocol, run, seed)."""
    check_seed(seed)
    if not isinstance(max_length, int) or max_length < 4:
        raise ValueError("max_length must leave room for CLS, content and SEP tokens")
    display = base_preparation["display_name"]
    return {"kind": "encoder", "run_id": run_id, "seed": seed, "variant": variant or base_preparation["name"],
            "model_name": display, "report_model": f"Encoder-{display}",
            "base_model": dict(base_preparation["base_model"]), "dataset": dict(dataset),
            "input": {"scope": "SKILL.md only", "max_length": max_length, "template": "[CLS] {skill_text} [SEP]",
                      "truncation": f"Keep the first {max_length - 2} content tokens between the tokenizer's CLS and SEP tokens"},
            "training": dict(training or DEFAULT_TRAINING),
            "selection": "Lowest validation FPR at recall >= 99% on the two-way log-odds; ties prefer AP, then loss; "
                         "validation at the midpoint and the end of the single epoch.",
            "decision": {"score": "sigmoid(malicious_logit - benign_logit), the softmax probability of the malicious class",
                         "fixed_rule": "score >= 0.5",
                         "gate": "validation 99%-recall log-odds threshold with a validation-fitted sigmoid calibration"}}


def validation_points(updates_per_epoch, epochs):
    """Updates at which validation (and checkpointing) happens: the midpoint and the end of every epoch."""
    if updates_per_epoch < 1 or epochs < 1:
        raise ValueError("Need at least one update per epoch and one epoch")
    points = set()
    for epoch in range(epochs):
        points.add(epoch * updates_per_epoch + math.ceil(updates_per_epoch / 2))
        points.add((epoch + 1) * updates_per_epoch)
    return sorted(points)


def derive_run_config(shared_config, run_id, seed):
    """A seed workspace config: the shared workspace recipe with only run ID and seed replaced."""
    check_seed(seed)
    config = json.loads(json.dumps(shared_config))
    config.update(run_id=run_id, seed=seed)
    return config


def check_tokenizer(tokenizer):
    if not tokenizer.is_fast or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Expected a fast tokenizer with CLS and SEP tokens")
    if tokenizer.num_special_tokens_to_add(False) != 2:
        raise ValueError("Expected exactly CLS and SEP to be added around a single sequence")
    probe = tokenizer.encode("probe", add_special_tokens=True)
    if probe[0] != tokenizer.cls_token_id or probe[-1] != tokenizer.sep_token_id:
        raise ValueError("Tokenizer does not wrap a sequence as [CLS] ... [SEP]")


def tokenizer_sha256(tokenizer):
    return hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()


def encode(tokenizer, text, max_length):
    """Token IDs of [CLS] text [SEP] cut to max_length by keeping the first content tokens; returns (ids, original_count)."""
    ids = list(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    original = len(ids)
    if original > max_length:
        ids = ids[:max_length - 1] + [ids[-1]]
    return ids, original


def two_way_log_odds(benign, malicious):
    """Decision variable: float64 difference of the two Python floats written to the score files."""
    value = float(malicious) - float(benign)
    if not np.isfinite(value):
        raise ValueError("Non-finite two-way log-odds")
    return value


def log_odds(rows):
    values = np.array([two_way_log_odds(r["benign_logit"], r["malicious_logit"]) for r in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Expected finite two-way logits")
    return values


def score_of(value):
    return float(probabilities(value))


def versions():
    found = {}
    for name in ("torch", "transformers", "tokenizers", "numpy", "safetensors"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def official_records(root):
    with (root / "data/split_manifest.csv").open(encoding="utf-8", newline="") as handle:
        return {r["benchmark_id"]: r for r in csv.DictReader(handle)}


def build_cache(root, tokenizer, config, log=print):
    """Cache train/validation token IDs (little-endian uint32) with per-record metadata; test is never cached."""
    from array import array
    summary = json.loads((root / "data/summary.json").read_text(encoding="utf-8"))
    if summary["dataset"] != config["dataset"]:
        raise ValueError("Prepared data and configured split protocol differ")
    max_length = config["input"]["max_length"]
    out = root / CACHE
    out.mkdir(parents=True, exist_ok=False)
    official = official_records(root)
    manifest = {"kind": "encoder_inputs", "base_model": config["base_model"], "dataset": config["dataset"],
                "max_length": max_length, "dtype": "little-endian uint32", "test_cached": False, "input": config["input"],
                "tokenizer_sha256": tokenizer_sha256(tokenizer), "versions": versions(), "partitions": {}}
    for partition in ("train", "validation"):
        info = summary["files"][partition]
        path = root / info["path"]
        if digest(path) != info["sha256"]:
            raise ValueError("Prepared text checksum differs from the summary")
        metadata, offset = [], 0
        with path.open(encoding="utf-8") as source, (out / f"{partition}.bin").open("wb") as output:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                reference = official[row["benchmark_id"]]
                text_sha = hashlib.sha256(row["skill_text"].encode()).hexdigest()
                if (reference["split"], int(reference["label"]), reference["skill_text_sha256"]) != (partition, row["label"], text_sha):
                    raise ValueError("Prepared record differs from the official split manifest")
                ids, original = encode(tokenizer, row["skill_text"], max_length)
                array("I", ids).tofile(output)
                metadata.append({"benchmark_id": row["benchmark_id"], "label": row["label"], "offset": offset,
                                 "used_tokens": len(ids), "original_tokens": original, "truncated": original > len(ids),
                                 "skill_text_sha256": text_sha})
                offset += len(ids)
                if len(metadata) % 1000 == 0:
                    log(f"Cached {partition}: {len(metadata)} records")
        if len(metadata) != summary["prepared_counts"][partition]["total"]:
            raise ValueError("Cache count differs from the prepared partition")
        save_json(out / f"{partition}.json", metadata)
        manifest["partitions"][partition] = {
            "text_sha256": info["sha256"], "tokens_sha256": digest(out / f"{partition}.bin"),
            "metadata_sha256": digest(out / f"{partition}.json"), "records": len(metadata), "tokens": offset,
            "truncated": sum(r["truncated"] for r in metadata),
            "longest_used_tokens": max(r["used_tokens"] for r in metadata)}
    save_json(out / "manifest.json", manifest)
    return manifest


def load_cache(root, config):
    """Verified train/validation token cache: (metadata by partition, memmapped tokens by partition, manifest)."""
    cache_dir = root / CACHE
    cache = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "data/summary.json").read_text(encoding="utf-8"))
    if (cache["base_model"], cache["dataset"], cache["max_length"]) != (config["base_model"], config["dataset"], config["input"]["max_length"]):
        raise ValueError("Token cache was built for another base, protocol or input budget")
    if cache["input"] != config["input"] or summary["dataset"] != config["dataset"]:
        raise ValueError("Token cache input policy or prepared data differ from the configuration")
    official = official_records(root)
    metadata, tokens = {}, {}
    for partition in ("train", "validation"):
        info = cache["partitions"][partition]
        if info["text_sha256"] != summary["files"][partition]["sha256"] or digest(root / summary["files"][partition]["path"]) != info["text_sha256"]:
            raise ValueError("Cache and prepared partition checksums differ")
        for extension, key in (("bin", "tokens_sha256"), ("json", "metadata_sha256")):
            if digest(cache_dir / f"{partition}.{extension}") != info[key]:
                raise ValueError("Token cache checksum differs from its manifest")
        rows = json.loads((cache_dir / f"{partition}.json").read_text(encoding="utf-8"))
        if len(rows) != summary["prepared_counts"][partition]["total"] or len({r["benchmark_id"] for r in rows}) != len(rows):
            raise ValueError("Cache must cover the complete prepared partition exactly once")
        offset = 0
        for row in rows:
            reference = official[row["benchmark_id"]]
            if (reference["split"], int(reference["label"]), reference["skill_text_sha256"]) != (partition, row["label"], row["skill_text_sha256"]):
                raise ValueError("Token cache IDs/labels/text differ from the official prepared records")
            if row["offset"] != offset or not 0 < row["used_tokens"] <= config["input"]["max_length"]:
                raise ValueError("Invalid token-cache offsets or input lengths")
            offset += row["used_tokens"]
        tokens[partition] = np.memmap(cache_dir / f"{partition}.bin", dtype="<u4", mode="r")
        if offset != len(tokens[partition]):
            raise ValueError("Token-cache storage and metadata lengths differ")
        metadata[partition] = rows
    return metadata, tokens, cache


def select_device(torch):
    """CUDA with bfloat16 autocast by default; SKILLSCREEN_DEVICE=cpu allows CPU (bfloat16 autocast only with SKILLSCREEN_CPU_DTYPE=bfloat16)."""
    wanted = os.environ.get("SKILLSCREEN_DEVICE", "cuda")
    if wanted == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Run on a scheduled CUDA GPU or set SKILLSCREEN_DEVICE=cpu explicitly")
        return "cuda", torch.bfloat16
    if wanted != "cpu":
        raise ValueError("SKILLSCREEN_DEVICE must be cuda or cpu")
    return "cpu", torch.bfloat16 if os.environ.get("SKILLSCREEN_CPU_DTYPE", "float32") == "bfloat16" else None


def autocast(torch, device, dtype):
    return torch.autocast(device_type=device, dtype=dtype) if dtype is not None else contextlib.nullcontext()


def device_name(torch, device):
    if device == "cuda":
        return torch.cuda.get_device_name()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "cpu"


def load_model(torch, directory, fresh_head=False, seed=None):
    """Load a sequence-classification encoder in float32 with SDPA attention; a fresh two-way head is seeded when requested."""
    from transformers import AutoModelForSequenceClassification
    kwargs = {"dtype": torch.float32, "attn_implementation": "sdpa", "output_loading_info": True,
              "local_files_only": True, "trust_remote_code": False}
    if fresh_head:
        torch.manual_seed(check_seed(seed))
        kwargs.update(num_labels=2, id2label={0: "benign", 1: "malicious"}, label2id=dict(LABELS))
    model, loading = AutoModelForSequenceClassification.from_pretrained(str(directory), **kwargs)
    missing, unexpected = set(loading.get("missing_keys") or []), set(loading.get("unexpected_keys") or [])
    if loading.get("mismatched_keys"):
        raise ValueError("Mismatched weight shapes while loading the encoder")
    if fresh_head:
        if not missing or not missing <= HEAD_KEYS or not unexpected <= DISCARDED_KEYS:
            raise ValueError(f"Unexpected weight loading for a fresh head: missing {sorted(missing)}, unexpected {sorted(unexpected)}")
    elif missing or unexpected:
        raise ValueError(f"Checkpoint did not load completely: missing {sorted(missing)}, unexpected {sorted(unexpected)}")
    if model.config.num_labels != 2 or dict(model.config.label2id) != LABELS:
        raise ValueError("Expected a two-way benign/malicious head")
    model.config.use_cache = False
    return model


def expected_parameters(model):
    return sum(p.numel() for p in model.parameters())


def inspect_checkpoint(directory, expected, root=ROOT):
    """Hash every exported file and verify the float32 safetensors weights are finite with the expected parameter count."""
    path = directory / "model.safetensors"
    raw = path.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_size])
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    parameters = 0
    for name, info in tensors.items():
        if info["dtype"] != "F32":
            raise ValueError(f"Unexpected checkpoint dtype {info['dtype']} for {name}")
        start, end = info["data_offsets"]
        values = np.frombuffer(raw[8 + header_size + start:8 + header_size + end], dtype="<f4")
        if not np.isfinite(values).all() or len(values) != int(np.prod(info["shape"])):
            raise ValueError(f"Checkpoint tensor {name} is malformed or non-finite")
        parameters += len(values)
    if parameters != expected:
        raise ValueError(f"Checkpoint holds {parameters} parameters, expected {expected}")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(directory.iterdir()) if p.is_file()}
    if not set(CHECKPOINT_FILES) <= files.keys():
        raise ValueError("Checkpoint export is missing weights, config or tokenizer files")
    return {"directory": directory.resolve().relative_to(root.resolve()).as_posix(), "files": files,
            "parameters": parameters, "tensors": len(tensors), "dtype": "F32"}


def check_frozen(frozen, root=ROOT):
    config = load_config(root)
    if frozen["run_id"] != config["run_id"] or frozen["base_model"] != config["base_model"] or frozen["dataset"] != config["dataset"]:
        raise ValueError("Frozen experiment belongs to another run, base or split protocol")
    if frozen["input"] != config["input"]:
        raise ValueError("Frozen input policy differs from the workspace configuration")
    for name, checksum in frozen["evaluation_scripts_sha256"].items():
        if code_digest(root / "scripts" / name) != checksum:
            raise ValueError("Frozen evaluation code changed")
    if frozen["calibration"]["slope"] <= 0:
        raise ValueError("Unexpected frozen calibration")


def evaluation_scripts_sha256(root=ROOT):
    return {name: code_digest(root / "scripts" / name) for name in EVALUATION_SCRIPTS}


def write_scores(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

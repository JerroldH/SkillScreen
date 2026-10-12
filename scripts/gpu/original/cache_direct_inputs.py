"""Cache bounded Direct train/validation tokens without touching held-out test."""

import argparse
from array import array
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from direct_smoke import encode_skill
from probe_train_memory import ROOT, digest, write_json


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/direct-inputs-8k")
    args = parser.parse_args()
    if sys.byteorder != "little" or array("I").itemsize != 4:
        raise RuntimeError("This cache writes little-endian uint32 token IDs")
    config = json.loads((ROOT / "configs/direct.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "data/summary.json").read_text(encoding="utf-8"))
    profile = ROOT / "outputs/train_token_lengths.csv"
    with profile.open(encoding="utf-8", newline="") as handle:
        lengths = {r["benchmark_id"]: (int(r["label"]), int(r["original_token_count"]))
                   for r in csv.DictReader(handle)}
    if len(lengths) != summary["prepared_counts"]["train"]["total"]:
        raise ValueError("Training profile must cover every prepared training identity")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_dir, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.num_special_tokens_to_add(False) != 0:
        raise ValueError("Expected the pinned fast tokenizer without added BOS/EOS")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"base_model": config["base_model"], "dataset": config["dataset"],
        "hypothesis": config["hypothesis"], "max_length": args.max_length,
        "dtype": "little-endian uint32", "test_cached": False,
        "input_policy": "Exact native prompt when untruncated; premise prefix plus complete fixed hypothesis when truncated. Very large inputs use a safe complete pre-token prefix checked under extension.",
        "profile_sha256": digest(profile), "script_sha256": digest(Path(__file__)),
        "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
        "partitions": {}}
    for partition in ("train", "validation"):
        path = ROOT / summary["files"][partition]["path"]
        if digest(path) != summary["files"][partition]["sha256"]:
            raise ValueError("Prepared text checksum differs from the official export")
        metadata, offset = [], 0
        with path.open(encoding="utf-8") as source, (args.output / f"{partition}.bin").open("wb") as output:
            for line in source:
                row = json.loads(line)
                reference = lengths[row["benchmark_id"]] if partition == "train" else None
                if reference and reference[0] != row["label"]:
                    raise ValueError("Training profile label differs from the prepared text")
                tokens, original, bounded = bounded_feature(tokenizer, row["skill_text"],
                    config["hypothesis"], args.max_length, reference[1] if reference else None)
                array("I", tokens).tofile(output)
                metadata.append({"benchmark_id": row["benchmark_id"], "label": row["label"],
                    "offset": offset, "used_tokens": len(tokens), "original_tokens": original,
                    "truncated": original > len(tokens), "bounded_encoding": bounded,
                    "skill_text_sha256": hashlib.sha256(row["skill_text"].encode()).hexdigest()})
                offset += len(tokens)
                if len(metadata) % 1000 == 0:
                    print(f"Cached {partition}: {len(metadata)} records", flush=True)
        if len(metadata) != summary["prepared_counts"][partition]["total"]:
            raise ValueError("Cache count differs from the prepared partition")
        write_json(args.output / f"{partition}.json", metadata)
        result["partitions"][partition] = {"text_sha256": summary["files"][partition]["sha256"],
            "tokens_sha256": digest(args.output / f"{partition}.bin"),
            "metadata_sha256": digest(args.output / f"{partition}.json"),
            "records": len(metadata), "tokens": offset,
            "truncated": sum(r["truncated"] for r in metadata),
            "bounded_encoding": sum(r["bounded_encoding"] for r in metadata)}
    write_json(args.output / "manifest.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

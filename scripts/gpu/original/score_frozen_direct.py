"""Reload a frozen Direct export, verify it, then score all held-out test text."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import numpy as np

from calibrate_direct import check_frozen, load_scores, log_odds, probabilities, read_partition, save_json
from direct_smoke import ROOT, configure_gpu_memory, encode_skill
from probe_train_memory import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--frozen-sha256", required=True)
    parser.add_argument("--model", choices=("direct", "untuned"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.frozen) != args.frozen_sha256:
        raise ValueError("Frozen configuration changed after submission")
    frozen = json.loads(args.frozen.read_text())
    check_frozen(frozen)
    adapter = ROOT / frozen["adapter"]["directory"]
    for name, info in frozen["adapter"]["files"].items():
        if digest(adapter / name) != info["sha256"]:
            raise ValueError("Frozen adapter/head/tokenizer files changed")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("Run inference on a scheduled CUDA GPU")
    torch.manual_seed(42)
    total_memory, allocator_budget = configure_gpu_memory(torch)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, trust_remote_code=False)
    if tokenizer.num_special_tokens_to_add(False) != 0:
        raise ValueError("Unexpected tokenizer special-token policy")
    model, loading = AutoModelForSequenceClassification.from_pretrained(args.base_dir,
        dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True,
        local_files_only=True, trust_remote_code=False)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or model.config.label2id != {
            "contradiction": 0, "entailment": 1, "neutral": 2}:
        raise ValueError("Pinned base/head did not load correctly")
    model.config.use_cache = model.config.text_config.use_cache = False
    tokenizer.pad_token_id = model.config.pad_token_id
    model = model.cuda()
    if args.model == "direct":
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    if any(p.requires_grad for p in model.parameters()) and args.model == "direct":
        # The native original head can retain requires_grad; inference disables autograd.
        for p in model.parameters():
            p.requires_grad_(False)

    def infer(ids):
        data = {"input_ids": torch.tensor([ids], device="cuda"),
                "attention_mask": torch.ones((1, len(ids)), device="cuda", dtype=torch.long)}
        logits = model(**data).logits.float()
        if logits.shape != (1, 3) or not torch.isfinite(logits).all():
            raise ValueError("Invalid native NLI logits")
        return logits

    reload_check = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if args.model == "direct":
            cache_dir = ROOT / "outputs/direct-inputs-8k"
            if digest(cache_dir / "manifest.json") != frozen["cache_manifest_sha256"]:
                raise ValueError("Validation replay cache changed")
            cache = json.loads((cache_dir / "manifest.json").read_text())
            for ext, key in (("json", "metadata_sha256"), ("bin", "tokens_sha256")):
                if digest(cache_dir / f"validation.{ext}") != cache["partitions"]["validation"][key]:
                    raise ValueError("Validation replay cache storage changed")
            metadata = json.loads((cache_dir / "validation.json").read_text())
            source = ROOT / frozen.get("validation_scores_file", "outputs/direct-r001/validation-step-470.csv")
            if digest(source) != frozen["validation_scores_sha256"]:
                raise ValueError("Validation replay score file changed")
            reference = load_scores(source, metadata)
            values = log_odds(reference)
            indices = sorted({0, max(range(len(metadata)), key=lambda i: metadata[i]["used_tokens"]),
                int(np.argmin(abs(values - frozen["operating_point"]["threshold"])))})
            tokens = np.memmap(cache_dir / "validation.bin", dtype="<u4", mode="r")
            differences = []
            for index in indices:
                row = metadata[index]
                after = infer(tokens[row["offset"]:row["offset"] + row["used_tokens"]].tolist()).cpu()[0]
                before = torch.tensor([float(reference[index][n + "_logit"]) for n in
                                       ("contradiction", "entailment", "neutral")])
                if not torch.allclose(after, before, atol=.02, rtol=.01):
                    raise ValueError("Selected adapter/head no longer reproduces validation logits")
                differences.append(float((after - before).abs().max()))
            reload_check = {"passed": True, "validation_indices": indices,
                            "max_logit_difference": max(differences), "test_read_after_check": True}
            print("Selected adapter/head reload verification passed", flush=True)

        # No test text or labels are read before the frozen export has been checked.
        rows, checksum = read_partition("test")
        if checksum != frozen["test_file_sha256"]:
            raise ValueError("Frozen test text changed")
        args.output.mkdir(parents=True, exist_ok=False)
        loaded_seconds = time.perf_counter() - started
        scoring_started = time.perf_counter()
        truncated, used_tokens = 0, 0
        path = args.output / "scores.csv"
        a, b = frozen["calibration"]["slope"], frozen["calibration"]["intercept"]
        threshold = frozen["operating_point"]["threshold"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["benchmark_id", "label", "score", "contradiction_logit", "entailment_logit",
                "neutral_logit", "original_tokens", "used_tokens", "truncated", "calibrated_probability",
                "calibrated_threshold", "review_decision"])
            for index, row in enumerate(rows):
                feature, original = encode_skill(tokenizer, row["skill_text"], frozen["input"]["hypothesis"], 8192)
                logits = infer(feature["input_ids"])
                native = logits.cpu()[0].tolist()
                value = native[1] - float(np.logaddexp(native[0], native[2]))
                raw = float(logits.softmax(-1)[0, 1])
                p = float(probabilities(a * value + b)) if args.model == "direct" else ""
                cutoff = frozen["operating_point"]["calibrated_probability_threshold"] if args.model == "direct" else ""
                review = value >= threshold if args.model == "direct" else raw >= frozen["baselines"]["Untuned OpenJev-2B"]["threshold"]
                used = len(feature["input_ids"])
                truncated += original > used
                used_tokens += used
                writer.writerow([row["benchmark_id"], row["label"], raw, *native, original, used, original > used,
                                 p, cutoff, "review" if review else "pass"])
                if (index + 1) % 100 == 0:
                    handle.flush()
                    print(f"Scored {index + 1}/{len(rows)} {args.model} test records", flush=True)
    run = {"model": args.model, "partition": "test", "records": len(rows),
        "frozen_sha256": digest(args.frozen), "test_file_sha256": checksum,
        "scores_sha256": digest(path), "evaluation_code_commit": frozen["evaluation_code_commit"],
        "base_model": frozen["base_model"], "selected_update": frozen["selected_update"] if args.model == "direct" else None,
        "max_length": 8192, "batch_size": 1, "precision": "bfloat16", "truncated_records": truncated,
        "used_tokens": used_tokens, "model_load_and_verification_seconds": loaded_seconds,
        "scoring_seconds": time.perf_counter() - scoring_started, "reload_verification": reload_check,
        "gpu": torch.cuda.get_device_name(), "allocator_budget_gib": allocator_budget,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "numpy")}}
    save_json(args.output / "run.json", run)
    print("COMPLETED " + json.dumps(run), flush=True)


if __name__ == "__main__":
    main()

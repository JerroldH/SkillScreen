"""Frozen encoder test scoring and reporting.

    score  : verify the committed freeze and the selected checkpoint, replay three validation records, then score every
             held-out test record (test text is read only after the checks) into outputs/<run_id>-test/scores.csv
    report : results/<run_id>-test.json in the shape of the Direct reports (models[<report_model>])
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import numpy as np

from calibrate_direct import load_scores, probabilities, probability_metrics, read_partition, save_json
from direct_smoke import ROOT, configure_gpu_memory
from encoder_common import (CACHE, TEST_COLUMNS, autocast, check_frozen, device_name, encode, load_config, load_model, log_odds,
                            score_of, select_device, two_way_log_odds)
from probe_train_memory import digest


def score(args):
    if digest(args.frozen) != args.frozen_sha256:
        raise ValueError("Frozen configuration changed after submission")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    check_frozen(frozen)
    checkpoint = ROOT / frozen["checkpoint"]["directory"]
    for name, info in frozen["checkpoint"]["files"].items():
        if digest(checkpoint / name) != info["sha256"]:
            raise ValueError("Frozen checkpoint files changed")
    import torch
    from transformers import AutoTokenizer
    device, dtype = select_device(torch)
    allocator_budget = configure_gpu_memory(torch)[1] if device == "cuda" else None
    torch.manual_seed(42)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    model = load_model(torch, checkpoint).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    max_length = frozen["input"]["max_length"]

    def infer(ids):
        logits = model(input_ids=torch.tensor([ids], device=device)).logits.float()
        if logits.shape != (1, 2) or not torch.isfinite(logits).all():
            raise ValueError("Invalid two-way logits")
        return logits

    with torch.inference_mode(), autocast(torch, device, dtype):
        cache_dir = ROOT / CACHE
        if digest(cache_dir / "manifest.json") != frozen["cache_manifest_sha256"]:
            raise ValueError("Validation replay cache changed")
        cache = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        for ext, key in (("json", "metadata_sha256"), ("bin", "tokens_sha256")):
            if digest(cache_dir / f"validation.{ext}") != cache["partitions"]["validation"][key]:
                raise ValueError("Validation replay cache storage changed")
        metadata = json.loads((cache_dir / "validation.json").read_text(encoding="utf-8"))
        source = ROOT / frozen["validation_scores_file"]
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
            before = torch.tensor([float(reference[index]["benign_logit"]), float(reference[index]["malicious_logit"])])
            if not torch.allclose(after, before, atol=0.02, rtol=0.01):
                raise ValueError("Selected checkpoint no longer reproduces validation logits")
            differences.append(float((after - before).abs().max()))
        reload_check = {"passed": True, "validation_indices": indices, "max_logit_difference": max(differences),
                        "test_read_after_check": True}
        print("Selected checkpoint reload verification passed", flush=True)

        # No test text or labels are read before the frozen export has been checked.
        rows, checksum = read_partition("test")
        if checksum != frozen["test_file_sha256"]:
            raise ValueError("Frozen test text changed")
        args.output.mkdir(parents=True, exist_ok=False)
        loaded_seconds = time.perf_counter() - started
        scoring_started = time.perf_counter()
        truncated, used_tokens, tokenize_seconds = 0, 0, 0.0
        a, b = frozen["calibration"]["slope"], frozen["calibration"]["intercept"]
        threshold = frozen["operating_point"]["threshold"]
        cutoff = frozen["operating_point"]["calibrated_probability_threshold"]
        path = args.output / "scores.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            import csv
            writer = csv.writer(handle)
            writer.writerow(TEST_COLUMNS)
            for index, row in enumerate(rows):
                tokenize_started = time.perf_counter()
                ids, original = encode(tokenizer, row["skill_text"], max_length)
                tokenize_seconds += time.perf_counter() - tokenize_started
                benign, malicious = infer(ids).cpu()[0].tolist()
                value = two_way_log_odds(benign, malicious)
                used = len(ids)
                truncated += original > used
                used_tokens += used
                writer.writerow([row["benchmark_id"], row["label"], score_of(value), benign, malicious, original, used,
                                 original > used, float(probabilities(a * value + b)), cutoff,
                                 "review" if value >= threshold else "pass"])
                if (index + 1) % 100 == 0:
                    handle.flush()
                    print(f"Scored {index + 1}/{len(rows)} encoder test records", flush=True)
    run = {"model": "encoder", "report_model": frozen["report_model"], "partition": "test", "records": len(rows),
        "frozen_sha256": digest(args.frozen), "test_file_sha256": checksum, "scores_sha256": digest(path),
        "evaluation_code_commit": frozen["evaluation_code_commit"], "base_model": frozen["base_model"],
        "selected_update": frozen["selected_update"], "max_length": max_length, "batch_size": 1,
        "precision": f"{'bfloat16' if dtype is not None else 'float32'} autocast over float32 weights",
        "truncated_records": truncated, "used_tokens": used_tokens, "tokenize_seconds": tokenize_seconds,
        "model_load_and_verification_seconds": loaded_seconds, "scoring_seconds": time.perf_counter() - scoring_started,
        "reload_verification": reload_check, "device": device, "device_name": device_name(torch, device),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None, "allocator_budget_gib": allocator_budget,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
        "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy")}}
    save_json(args.output / "run.json", run)
    print("COMPLETED " + json.dumps(run), flush=True)


def report(args):
    from sklearn.metrics import average_precision_score
    from word_svm import detection_metrics
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    check_frozen(frozen)
    config = load_config()
    expected, checksum = read_partition("test")
    if checksum != frozen["test_file_sha256"]:
        raise ValueError("Frozen test text changed")
    labels = np.array([r["label"] for r in expected])
    path = args.scores / "scores.csv"
    run = json.loads((args.scores / "run.json").read_text(encoding="utf-8"))
    if (run["partition"] != "test" or run["records"] != len(expected) or run["test_file_sha256"] != checksum
            or run["frozen_sha256"] != digest(args.frozen) or run["scores_sha256"] != digest(path)):
        raise ValueError("Test scores changed or used different frozen settings")
    rows = load_scores(path, expected)
    values = log_odds(rows)
    scores = np.array([float(r["score"]) for r in rows])
    if not np.allclose(probabilities(values), scores, atol=1e-12, rtol=0):
        raise ValueError("Saved scores are not the sigmoid of the saved two-way logits")
    a, b = frozen["calibration"]["slope"], frozen["calibration"]["intercept"]
    if not np.allclose(probabilities(a * values + b), [float(r["calibrated_probability"]) for r in rows], atol=1e-12, rtol=1e-12):
        raise ValueError("Saved probabilities differ from frozen calibration")
    threshold = frozen["operating_point"]["threshold"]
    predictions = values >= threshold
    if predictions.tolist() != [r["review_decision"] == "review" for r in rows]:
        raise ValueError("Saved review decisions differ from the frozen threshold")
    result = {"partition": "test", "records": len(expected), "benign": int((labels == 0).sum()), "malicious": int((labels == 1).sum()),
        "frozen_sha256": digest(args.frozen), "frozen_configuration": frozen, "test_file_sha256": checksum,
        "models": {config["report_model"]: {
            "average_precision": float(average_precision_score(labels, scores)),
            "fixed_operating_point": {"threshold": threshold, "comparator": ">=", **detection_metrics(labels, predictions)},
            "native_rule": {"field": "score", "threshold": 0.5, "comparator": ">=", **detection_metrics(labels, scores >= 0.5)},
            "probability_metrics": {"raw": probability_metrics(labels, values), "calibrated": probability_metrics(labels, a * values + b)},
            "run": run}},
        "policy": "All decision thresholds fixed on validation before test scoring; no test-based tuning."}
    save_json(args.output, result)
    print(json.dumps({n: {"ap": m["average_precision"], "gate": m["fixed_operating_point"], "native_0_5": m["native_rule"]}
                      for n, m in result["models"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("score")
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--frozen-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"score": score, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()

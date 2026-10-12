"""First full Direct LoRA epoch, training-only preflight, and export verification."""

import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

from probe_train_memory import ROOT, TARGETS, digest, write_json
from direct_smoke import configure_gpu_memory


def update_groups(order, accumulation):
    if accumulation <= 0:
        raise ValueError("Accumulation must be positive")
    return [order[start:start + accumulation] for start in range(0, len(order), accumulation)]


def validation_metrics(metadata, scores):
    if len(metadata) != len(scores) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Validation scores must align and be finite")
    positives = sum(row["label"] for row in metadata)
    negatives = len(metadata) - positives
    if not positives or not negatives:
        raise ValueError("Validation needs both official classes")
    ranked = sorted(zip(scores, (r["label"] for r in metadata)), reverse=True)
    ap, tp, seen, index = 0.0, 0, 0, 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        added = sum(label for _, label in ranked[index:end])
        tp += added
        seen += end - index
        ap += added / positives * tp / seen
        index = end
    positive_scores = sorted((s for r, s in zip(metadata, scores) if r["label"]), reverse=True)
    gates = []
    for target in (0.98, 0.99, 0.995):
        threshold = positive_scores[math.ceil(target * positives - 1e-12) - 1]
        tp = sum(r["label"] == 1 and s >= threshold for r, s in zip(metadata, scores))
        fp = sum(r["label"] == 0 and s >= threshold for r, s in zip(metadata, scores))
        gates.append({"target_recall": target, "threshold": threshold, "comparator": ">=",
            "tp": tp, "fp": fp, "tn": negatives - fp, "fn": positives - tp,
            "recall": tp / positives, "fpr": fp / negatives,
            "precision": tp / (tp + fp), "flagged_fraction": (tp + fp) / len(metadata)})
    return {"average_precision": ap, "validation_recall_gates": gates}


def expected_trainable_parameters(model, targets=TARGETS, rank=8):
    """LoRA parameters for every nn.Linear whose name ends in a target (rank x (in + out)) plus the classification head.
    Equals the historical constant 2,193,408 for OpenJev-2B and follows the architecture for other bases."""
    import torch.nn as nn
    total = sum(p.numel() for p in model.score.parameters())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            total += rank * (module.in_features + module.out_features)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/direct.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/direct-inputs-8k")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/direct-r001")
    parser.add_argument("--stage", choices=("preflight", "verify", "train"), default="train")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "data/summary.json").read_text(encoding="utf-8"))
    cache = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    if (cache["base_model"], cache["hypothesis"], cache["max_length"]) != (
            config["base_model"], config["hypothesis"], 8192):
        raise ValueError("Expected pinned OpenJev and the selected 8,192-token input policy")
    if cache["dataset"] != config["dataset"] or summary["dataset"] != config["dataset"]:
        raise ValueError("Cache, prepared data and configured split protocol differ")
    if cache.get("route", "direct") != config.get("route", "direct") or cache.get("input") != config.get("input"):
        raise ValueError("Token cache and configured input route differ")
    if cache.get("formatter_sha256"):
        from calibrate_direct import code_digest
        if cache["formatter_sha256"] != code_digest(ROOT / "scripts/spector_inputs.py"):
            raise ValueError("Scanner context formatter changed after token caching")
    import numpy as np
    metadata, tokens = {}, {}
    with (ROOT / "data/split_manifest.csv").open(encoding="utf-8", newline="") as handle:
        official = {r["benchmark_id"]: r for r in csv.DictReader(handle)}
    for partition in ("train", "validation"):
        info = cache["partitions"][partition]
        if info["text_sha256"] != summary["files"][partition]["sha256"] or (
                digest(ROOT / summary["files"][partition]["path"]) != info["text_sha256"]):
            raise ValueError("Cache and prepared partition checksums differ")
        for extension, key in (("bin", "tokens_sha256"), ("json", "metadata_sha256")):
            if digest(args.cache / f"{partition}.{extension}") != info[key]:
                raise ValueError("Token cache checksum differs from its manifest")
        metadata[partition] = json.loads((args.cache / f"{partition}.json").read_text(encoding="utf-8"))
        tokens[partition] = np.memmap(args.cache / f"{partition}.bin", dtype="<u4", mode="r")
        rows = metadata[partition]
        if len(rows) != summary["prepared_counts"][partition]["total"] or len({r["benchmark_id"] for r in rows}) != len(rows):
            raise ValueError("Cache must cover the complete prepared partition exactly once")
        offset = 0
        for row in rows:
            reference = official[row["benchmark_id"]]
            if (reference["split"], int(reference["label"]), reference["skill_text_sha256"]) != (
                    partition, row["label"], row["skill_text_sha256"]):
                raise ValueError("Token cache IDs/labels/text differ from the official prepared records")
            if row["offset"] != offset or not 0 < row["used_tokens"] <= 8192:
                raise ValueError("Invalid token-cache offsets or input lengths")
            offset += row["used_tokens"]
        if offset != len(tokens[partition]):
            raise ValueError("Token-cache storage and metadata lengths differ")

    import torch
    import resource
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup
    if not torch.cuda.is_available():
        raise RuntimeError("Run training on a scheduled CUDA GPU")
    torch.manual_seed(config["seed"])
    total_memory, allocator_budget = configure_gpu_memory(torch)
    tokenizer = AutoTokenizer.from_pretrained(args.base_dir, local_files_only=True, trust_remote_code=False)
    if args.stage != "verify" and not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def load_base():
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            args.base_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
            output_loading_info=True, local_files_only=True, trust_remote_code=False)
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError("Base or classification-head weights did not load completely")
        if model.config.label2id != {"contradiction": 0, "entailment": 1, "neutral": 2}:
            raise ValueError("Unexpected NLI labels")
        model.config.use_cache = model.config.text_config.use_cache = False
        tokenizer.pad_token_id = model.config.pad_token_id
        return model.cuda()

    def batch(partition, index):
        row = metadata[partition][index]
        ids = tokens[partition][row["offset"]:row["offset"] + row["used_tokens"]].tolist()
        return {"input_ids": torch.tensor([ids], device="cuda"),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device="cuda")}

    def reference_logits(model, indices):
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.cat([model(**batch("train", index)).logits.float().cpu() for index in indices])

    if args.stage == "verify":
        check = json.loads((args.output / "reference.json").read_text(encoding="utf-8"))
        model = PeftModel.from_pretrained(load_base(), args.output / "adapter")
        after = reference_logits(model, check["indices"])
        before = torch.tensor(check["logits"])
        error = float((before - after).abs().max())
        if not torch.allclose(before, after, atol=0.02, rtol=0.01):
            raise ValueError("Saved LoRA/head logits changed after reloading the export")
        state = torch.load(args.output / "training_state.pt", map_location="cpu", weights_only=True)
        if state["completed_updates"] != 2 or not state["optimizer"]["state"]:
            raise ValueError("Preflight optimizer state was not saved")
        write_json(args.output / "verify.json", {"passed": True, "reload_max_logit_difference": error,
            "script_sha256": digest(Path(__file__)), "code_commit": args.code_commit})
        print(f"Preflight export and optimizer-state reload passed: logit difference={error}", flush=True)
        return

    model = load_base()
    expected_parameters = expected_trainable_parameters(model)
    if args.resume:
        model = PeftModel.from_pretrained(model, args.resume / "adapter", is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS,
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=TARGETS, modules_to_save=["score"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in trainable.values()) != expected_parameters or any(
            "lora_" not in n and "score.modules_to_save" not in n for n in trainable):
        raise ValueError("Expected only the checked LoRA and classification head to train")
    head_name = next(n for n in trainable if "score.modules_to_save" in n and n.endswith("weight"))
    lora_name = next(n for n in trainable if "lora_B" in n)
    head_before, lora_before = trainable[head_name].detach().clone(), trainable[lora_name].detach().clone()
    optimizer = torch.optim.AdamW(trainable.values(), lr=2e-4)
    order = list(range(len(metadata["train"])))
    random.Random(config["seed"]).shuffle(order)
    if args.stage == "preflight":
        ranked = sorted(order, key=lambda i: metadata["train"][i]["used_tokens"])
        outlier = max(order, key=lambda i: metadata["train"][i]["original_tokens"])
        longest_benign = max((i for i in order if metadata["train"][i]["label"] == 0),
                             key=lambda i: metadata["train"][i]["used_tokens"])
        choices = [ranked[0], ranked[len(ranked) // 2], ranked[int(len(ranked) * 0.95)], longest_benign, outlier]
        order = [choices[i % len(choices)] for i in range(32)]
    groups = update_groups(order, 16)
    warmup = math.ceil(len(groups) * 0.05) if args.stage == "train" else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, len(groups)) if args.stage == "train" else None
    run = {"kind": "direct_lora_training" if args.stage == "train" else "mixed_input_preflight",
        "run_id": config.get("run_id", "direct-r001"),
        "route": config.get("route", "direct"), "input": config.get("input"),
        "base_model": config["base_model"], "dataset": config["dataset"], "hypothesis": config["hypothesis"],
        "code_commit": args.code_commit, "script_sha256": digest(Path(__file__)),
        "cache_manifest_sha256": digest(args.cache / "manifest.json"), "max_length": 8192,
        "epochs": 1, "batch_size": 1, "gradient_accumulation_steps": 16,
        "optimizer": "AdamW", "learning_rate": 2e-4, "warmup_steps": warmup, "schedule": "linear" if scheduler else None,
        "lora": {"r": 8, "alpha": 16, "target_modules": TARGETS, "modules_to_save": ["score"]},
        "trainable_parameters": expected_parameters, "precision": "bfloat16", "gradient_checkpointing": "nonreentrant",
        "train_records": len(metadata["train"]), "validation_records": len(metadata["validation"]),
        "class_counts": {str(label): sum(r["label"] == label for r in metadata["train"]) for label in (0, 1)},
        "label_policy": "Unchanged official labels; CE contradiction=0, entailment=1; no neutral targets, class weighting or oversampling.",
        "test_evaluated": False, "completed_updates": 0, "seen_training_records": 0, "validation": [],
        "gpu": torch.cuda.get_device_name(), "gpu_total_gib": total_memory / 1024**3,
        "allocator_budget_gib": allocator_budget, "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "numpy")}}
    start_update = 0
    if args.resume:
        if not (args.resume / "READY").is_file():
            raise ValueError("Resume requires a fully written checkpoint")
        previous = json.loads((args.resume / "run.json").read_text(encoding="utf-8"))
        for key in ("code_commit", "script_sha256", "cache_manifest_sha256", "max_length", "warmup_steps"):
            if previous[key] != run[key]:
                raise ValueError("Resume checkpoint configuration or code changed")
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_update = state["completed_updates"]
        run = previous
        del state

    def save_checkpoint(update, folder):
        if (folder / "READY").exists():
            raise ValueError("Do not overwrite an existing complete checkpoint")
        folder.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(folder / "adapter")
        torch.save({"optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "completed_updates": update, "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()}, folder / "training_state.pt")
        write_json(folder / "run.json", run)
        (folder / "READY").write_text("complete\n", encoding="utf-8")

    def validate(update):
        model.eval()
        scores, losses = [], []
        path = args.output / f"validation-step-{update}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            writer = csv.writer(handle)
            writer.writerow(["benchmark_id", "label", "score", "contradiction_logit", "entailment_logit", "neutral_logit", "original_tokens", "used_tokens", "truncated"])
            for index, row in enumerate(metadata["validation"]):
                logits = model(**batch("validation", index)).logits.float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite validation logits")
                score = float(logits.softmax(-1)[0, 1])
                scores.append(score)
                losses.append(float(F.cross_entropy(logits, torch.tensor([row["label"]], device="cuda"))))
                writer.writerow([row["benchmark_id"], row["label"], score, *logits[0].tolist(), row["original_tokens"], row["used_tokens"], row["truncated"]])
        metrics = {"update": update, "loss": sum(losses) / len(losses),
                   "scores_sha256": digest(path), **validation_metrics(metadata["validation"], scores)}
        run["validation"].append(metrics)
        print("VALIDATION " + json.dumps(metrics), flush=True)
        model.train()

    log_path = args.output / "training.jsonl"
    run["status"] = "running"
    write_json(args.output / "run.json", run)
    with log_path.open("a" if args.resume else "w", encoding="utf-8") as log:
        for update_index in range(start_update, len(groups)):
            indices = groups[update_index]
            optimizer.zero_grad(set_to_none=True)
            model.train()
            losses = []
            update_started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            for index in indices:
                row = metadata["train"][index]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**batch("train", index)).logits
                    loss = F.cross_entropy(logits.float(), torch.tensor([row["label"]], device="cuda"))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                (loss / len(indices)).backward()
                losses.append(float(loss.detach()))
                del logits, loss
            norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if scheduler:
                scheduler.step()
            update = update_index + 1
            run["completed_updates"] = update
            run["seen_training_records"] += len(indices)
            event = {"update": update, "total_updates": len(groups), "records": len(indices),
                "seen_training_records": run["seen_training_records"], "loss": sum(losses) / len(losses),
                "grad_norm": float(norm), "learning_rate": learning_rate,
                "seconds": time.perf_counter() - update_started,
                "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
            log.write(json.dumps(event) + "\n")
            log.flush()
            if update <= 3 or update % 10 == 0 or update == len(groups):
                print("TRAIN " + json.dumps(event), flush=True)
            if args.stage == "train" and update in (math.ceil(len(groups) / 2), len(groups)):
                validate(update)
            if args.stage == "train" and (update % 50 == 0 or update in (math.ceil(len(groups) / 2), len(groups))):
                save_checkpoint(update, args.output / f"checkpoint-{update}")
            write_json(args.output / "run.json", run)
    run.update({"status": "completed", "elapsed_seconds_this_process": time.perf_counter() - started,
        "head_max_update_this_process": float((trainable[head_name].detach() - head_before).abs().max()),
        "lora_max_update_this_process": float((trainable[lora_name].detach() - lora_before).abs().max()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
    run["host_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.stage == "preflight":
        if not run["head_max_update_this_process"] or not run["lora_max_update_this_process"]:
            raise ValueError("Preflight must update both LoRA and the classification head")
        indices = [order[0], order[4]]
        write_json(args.output / "reference.json", {"indices": indices, "logits": reference_logits(model, indices).tolist()})
        model.save_pretrained(args.output / "adapter")
        torch.save({"optimizer": optimizer.state_dict(), "completed_updates": 2}, args.output / "training_state.pt")
    else:
        best = min(run["validation"], key=lambda m: (m["validation_recall_gates"][1]["fpr"], -m["average_precision"], m["loss"]))
        run["selected_update"] = best["update"]
        run["selection_policy"] = "Lowest validation FPR at recall>=99%; ties prefer AP, then loss. Scores remain uncalibrated."
        run["selected_adapter"] = f"checkpoint-{best['update']}/adapter"
        tokenizer.save_pretrained(args.output / run["selected_adapter"])
    write_json(args.output / "run.json", run)
    print("COMPLETED " + json.dumps(run), flush=True)


if __name__ == "__main__":
    main()

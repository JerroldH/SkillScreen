"""Full fine-tuning of a bidirectional encoder on raw SKILL.md text: training-only preflight, export verification, training.

Stages mirror train_direct.py: preflight runs two updates on the five hardest/representative training records and exports
the model; verify reloads that export and compares logits; train runs the configured number of epochs with validation at the midpoint and the end of every
epoch, saving a float32 checkpoint at each validation point and selecting by the predefined rule (lowest validation FPR at recall >= 99% on
the two-way log-odds, then AP, then loss). Test text is never read here.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import time

from calibrate_direct import save_json
from direct_smoke import ROOT, configure_gpu_memory
from encoder_common import (SCORE_COLUMNS, autocast, device_name, expected_parameters, load_cache, load_config, load_model,
                            score_of, select_device, two_way_log_odds, validation_points, write_scores)
from probe_train_memory import digest
from train_direct import update_groups, validation_metrics

PREFLIGHT_RECORDS = 32


def preflight_order(metadata, order):
    """Shortest, median, 95th-percentile, longest benign and largest-original records, repeated to 32 inputs."""
    ranked = sorted(order, key=lambda i: metadata[i]["used_tokens"])
    outlier = max(order, key=lambda i: metadata[i]["original_tokens"])
    longest_benign = max((i for i in order if metadata[i]["label"] == 0), key=lambda i: metadata[i]["used_tokens"])
    choices = [ranked[0], ranked[len(ranked) // 2], ranked[int(len(ranked) * 0.95)], longest_benign, outlier]
    return [choices[i % len(choices)] for i in range(PREFLIGHT_RECORDS)]


def parameter_groups(model, weight_decay):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": other, "weight_decay": 0.0}]


def selection_key(metrics):
    return (metrics["validation_recall_gates"][1]["fpr"], -metrics["average_precision"], metrics["loss"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, help="Default: the configured base under outputs/base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "verify", "train"), default="train")
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    config = load_config()
    training, max_length = config["training"], config["input"]["max_length"]
    metadata, tokens, cache = load_cache(ROOT, config)
    base_dir = args.base_dir or ROOT / config["base_model"]["local_dir"]

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    device, dtype = select_device(torch)
    if device == "cuda":
        total_memory, allocator_budget = configure_gpu_memory(torch)
    else:
        total_memory, allocator_budget = 0, None
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True, trust_remote_code=False)
    if args.stage != "verify":
        args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def batch(partition, index):
        row = metadata[partition][index]
        ids = tokens[partition][row["offset"]:row["offset"] + row["used_tokens"]].tolist()
        return {"input_ids": torch.tensor([ids], device=device)}

    def logits_of(model, partition, index):
        logits = model(**batch(partition, index)).logits.float()
        if logits.shape != (1, 2) or not torch.isfinite(logits).all():
            raise ValueError("Invalid two-way logits")
        return logits

    def reference_logits(model, indices):
        model.eval()
        with torch.inference_mode(), autocast(torch, device, dtype):
            return torch.cat([logits_of(model, "train", index).cpu() for index in indices])

    if args.stage == "verify":
        check = json.loads((args.output / "reference.json").read_text(encoding="utf-8"))
        model = load_model(torch, args.output / "model").to(device)
        after = reference_logits(model, check["indices"])
        before = torch.tensor(check["logits"])
        error = float((before - after).abs().max())
        if not torch.allclose(before, after, atol=0.02, rtol=0.01):
            raise ValueError("Exported encoder logits changed after reloading the export")
        state = json.loads((args.output / "training_state.json").read_text(encoding="utf-8"))
        if state["completed_updates"] != 2 or not state["optimizer_state_tensors"]:
            raise ValueError("Preflight optimizer state was not recorded")
        save_json(args.output / "verify.json", {"passed": True, "reload_max_logit_difference": error, "device": device,
            "script_sha256": digest(Path(__file__)), "code_commit": args.code_commit})
        print(f"Preflight export reload passed: logit difference={error}", flush=True)
        return

    model = load_model(torch, base_dir, fresh_head=True, seed=config["seed"]).to(device)
    trainable_parameters = expected_parameters(model)
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("Every encoder parameter must train")
    checkpointing = device == "cuda" or os.environ.get("SKILLSCREEN_CPU_CHECKPOINTING") == "1"
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    head = model.classifier.weight
    body_name, body = next((n, p) for n, p in model.named_parameters() if ".layers.0." in n and p.ndim == 2)
    head_before, body_before = head.detach().clone(), body.detach().clone()
    optimizer = torch.optim.AdamW(parameter_groups(model, training["weight_decay"]), lr=training["learning_rate"])
    rng = random.Random(config["seed"])
    order = list(range(len(metadata["train"])))
    rng.shuffle(order)
    epochs = int(training["epochs"]) if args.stage == "train" else 1
    if args.stage == "preflight":
        order = preflight_order(metadata["train"], order)
    groups = update_groups(order, training["gradient_accumulation_steps"])
    updates_per_epoch = len(groups)
    for _ in range(1, epochs):  # every further epoch reshuffles with the same seeded generator
        order = list(order)
        rng.shuffle(order)
        groups += update_groups(order, training["gradient_accumulation_steps"])
    points = validation_points(updates_per_epoch, epochs)
    warmup = math.ceil(len(groups) * training["warmup_fraction"]) if args.stage == "train" else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, len(groups)) if args.stage == "train" else None
    run = {"kind": "encoder_full_finetune" if args.stage == "train" else "encoder_preflight",
        "run_id": config["run_id"], "seed": config["seed"], "variant": config["variant"], "model_name": config["model_name"],
        "base_model": config["base_model"], "dataset": config["dataset"], "input": config["input"],
        "code_commit": args.code_commit, "script_sha256": digest(Path(__file__)),
        "cache_manifest_sha256": digest(ROOT / "outputs/encoder-inputs-8k/manifest.json"), "max_length": max_length,
        **{k: training[k] for k in ("epochs", "batch_size", "gradient_accumulation_steps", "optimizer", "learning_rate", "weight_decay", "max_grad_norm")},
        "warmup_steps": warmup, "schedule": training["schedule"] if scheduler else None,
        "updates_per_epoch": updates_per_epoch, "total_updates": len(groups), "validation_updates": points,
        "trainable_parameters": trainable_parameters, "head_init_seed": config["seed"],
        "precision": f"{'bfloat16' if dtype is not None else 'float32'} autocast over float32 master weights",
        "gradient_checkpointing": "nonreentrant" if checkpointing else None,
        "device": device, "device_name": device_name(torch, device), "threads": torch.get_num_threads() if device == "cpu" else None,
        "train_records": len(metadata["train"]), "validation_records": len(metadata["validation"]),
        "class_counts": {str(label): sum(r["label"] == label for r in metadata["train"]) for label in (0, 1)},
        "label_policy": "Unchanged official labels; CE benign=0, malicious=1; no class weighting or oversampling.",
        "test_evaluated": False, "completed_updates": 0, "seen_training_records": 0, "validation": [],
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None, "gpu_total_gib": total_memory / 1024**3,
        "allocator_budget_gib": allocator_budget, "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy", "safetensors")}}

    def peak_gib():
        return torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None

    def save_checkpoint(update, folder):
        if (folder / "READY").exists():
            raise ValueError("Do not overwrite an existing complete checkpoint")
        folder.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(folder / "model", safe_serialization=True)
        tokenizer.save_pretrained(folder / "model")
        save_json(folder / "run.json", run)
        (folder / "READY").write_text("complete\n", encoding="utf-8")

    def validate(update):
        model.eval()
        values, losses, rows = [], [], []
        with torch.inference_mode(), autocast(torch, device, dtype):
            for index, row in enumerate(metadata["validation"]):
                logits = logits_of(model, "validation", index)
                benign, malicious = logits[0].tolist()
                value = two_way_log_odds(benign, malicious)
                values.append(value)
                losses.append(float(F.cross_entropy(logits, torch.tensor([row["label"]], device=device))))
                rows.append([row["benchmark_id"], row["label"], score_of(value), benign, malicious,
                             row["original_tokens"], row["used_tokens"], row["truncated"]])
        path = args.output / f"validation-step-{update}.csv"
        write_scores(path, SCORE_COLUMNS, rows)
        metrics = {"update": update, "loss": sum(losses) / len(losses), "scores_sha256": digest(path),
                   "decision_variable": "two_way_log_odds", **validation_metrics(metadata["validation"], values)}
        run["validation"].append(metrics)
        print("VALIDATION " + json.dumps(metrics), flush=True)
        model.train()

    run["status"] = "running"
    save_json(args.output / "run.json", run)
    with (args.output / "training.jsonl").open("w", encoding="utf-8") as log:
        for update_index, indices in enumerate(groups):
            optimizer.zero_grad(set_to_none=True)
            model.train()
            losses = []
            update_started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            for index in indices:
                row = metadata["train"][index]
                with autocast(torch, device, dtype):
                    logits = model(**batch("train", index)).logits
                    loss = F.cross_entropy(logits.float(), torch.tensor([row["label"]], device=device))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                (loss / len(indices)).backward()
                losses.append(float(loss.detach()))
                del logits, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            if scheduler:
                scheduler.step()
            update = update_index + 1
            run["completed_updates"] = update
            run["seen_training_records"] += len(indices)
            event = {"update": update, "total_updates": len(groups), "records": len(indices),
                "seen_training_records": run["seen_training_records"], "loss": sum(losses) / len(losses),
                "grad_norm": float(norm), "learning_rate": learning_rate, "seconds": time.perf_counter() - update_started,
                "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "peak_allocated_gib": peak_gib()}
            log.write(json.dumps(event) + "\n")
            log.flush()
            if update <= 3 or update % 10 == 0 or update == len(groups):
                print("TRAIN " + json.dumps(event), flush=True)
            if args.stage == "train" and update in points:
                validate(update)
                save_checkpoint(update, args.output / f"checkpoint-{update}")
            save_json(args.output / "run.json", run)
    run.update({"status": "completed", "elapsed_seconds_this_process": time.perf_counter() - started,
        "head_max_update_this_process": float((head.detach() - head_before).abs().max()),
        "body_max_update_this_process": float((body.detach() - body_before).abs().max()), "body_parameter": body_name,
        "peak_allocated_gib": peak_gib(), "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    if args.stage == "preflight":
        if not run["head_max_update_this_process"] or not run["body_max_update_this_process"]:
            raise ValueError("Preflight must update both the head and the encoder body")
        indices = [order[0], order[4]]
        save_json(args.output / "reference.json", {"indices": indices, "logits": reference_logits(model, indices).tolist()})
        model.save_pretrained(args.output / "model", safe_serialization=True)
        tokenizer.save_pretrained(args.output / "model")
        save_json(args.output / "training_state.json", {"completed_updates": 2, "optimizer_state_tensors": len(optimizer.state),
            "learning_rate": optimizer.param_groups[0]["lr"]})
    else:
        best = min(run["validation"], key=selection_key)
        run["selected_update"] = best["update"]
        run["selection_policy"] = ("Lowest validation FPR at recall>=99% on the two-way log-odds; ties prefer AP, then loss. "
                                   "Scores remain uncalibrated.")
        run["selected_checkpoint"] = f"checkpoint-{best['update']}/model"
    save_json(args.output / "run.json", run)
    print("COMPLETED " + json.dumps(run), flush=True)


if __name__ == "__main__":
    main()

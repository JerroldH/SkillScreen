"""Original experiment helpers, limited to functions used by this package."""

import argparse

import csv

from datetime import datetime, timezone

import hashlib

import json

from pathlib import Path

import time

import numpy as np

from direct_smoke import ROOT, read_rows

from probe_train_memory import digest

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())

def code_digest(path):
    # Linux checkouts use LF; Windows checkouts may use CRLF for the same Python source.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

def check_experiment(record, config):
    if record["base_model"] != config["base_model"] or record["dataset"] != config["dataset"]:
        raise ValueError("Report/frozen experiment uses a different pinned base or split protocol")
    if record.get("run_id", "direct-r001") != config.get("run_id", "direct-r001"):
        raise ValueError("Report/frozen experiment belongs to a different run")
    hypothesis = record.get("hypothesis", (record.get("input") or {}).get("hypothesis"))
    if hypothesis != config["hypothesis"]:
        raise ValueError("Fixed hypothesis changed")

def check_frozen(frozen):
    config = json.loads((ROOT / "configs/direct.json").read_text())
    check_experiment(frozen, config)
    for name, checksum in frozen["evaluation_scripts_sha256"].items():
        if code_digest(ROOT / "scripts" / name) != checksum:
            raise ValueError("Frozen evaluation code changed")
    if frozen["calibration"]["slope"] <= 0 or frozen["input"]["max_length"] != 8192:
        raise ValueError("Unexpected frozen input/calibration policy")

def read_partition(name):
    summary = json.loads((ROOT / "data/summary.json").read_text())
    info = summary["files"][name]
    path = ROOT / info["path"]
    if digest(path) != info["sha256"]:
        raise ValueError(f"Prepared {name} checksum changed")
    rows = read_rows(path)
    if len(rows) != summary["prepared_counts"][name]["total"]:
        raise ValueError(f"Prepared {name} count changed")
    return rows, info["sha256"]

def load_scores(path, expected):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [(r["benchmark_id"], int(r["label"])) for r in rows] != [
            (r["benchmark_id"], r["label"]) for r in expected]:
        raise ValueError("Score IDs/labels differ from the complete prepared partition")
    return rows

def log_odds(rows):
    logits = np.array([[float(r[f"{n}_logit"]) for n in
                       ("contradiction", "entailment", "neutral")] for r in rows])
    if logits.shape != (len(rows), 3) or not np.isfinite(logits).all():
        raise ValueError("Expected finite native NLI logits")
    # Neutral contributes to non-malicious mass, not an uncertainty estimate.
    return logits[:, 1] - np.logaddexp(logits[:, 0], logits[:, 2])

def probabilities(values):
    return np.exp(-np.logaddexp(0.0, -np.asarray(values, dtype=np.float64)))

def probability_metrics(labels, values):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    if values.shape != labels.shape or not np.isfinite(values).all():
        raise ValueError("Invalid probability inputs")
    p = probabilities(values)
    bins = []
    indices = np.minimum((p * 10).astype(int), 9)
    for index in range(10):
        mask = indices == index
        if mask.any():
            bins.append({"bin": index, "count": int(mask.sum()),
                         "mean_probability": float(p[mask].mean()),
                         "malicious_fraction": float(labels[mask].mean())})
    return {"binary_nll": float(np.mean(np.logaddexp(0, values) - labels * values)),
            "brier_score": float(np.mean((p - labels) ** 2)),
            "ece_10_uniform_bins": sum(r["count"] / len(labels) * abs(
                r["mean_probability"] - r["malicious_fraction"]) for r in bins),
            "reliability_bins": bins}

def fit_sigmoid(values, labels):
    from scipy.optimize import minimize
    values, labels = np.asarray(values), np.asarray(labels)
    if values.shape != labels.shape or set(labels.tolist()) != {0, 1} or not np.isfinite(values).all():
        raise ValueError("Calibration needs both classes and finite aligned log odds")
    regularization = 1e-6

    def objective(parameters):
        a, b = parameters
        z = a * values + b
        error = probabilities(z) - labels
        penalty = np.array([a - 1, b])
        loss = np.mean(np.logaddexp(0, z) - labels * z) + regularization / 2 * (penalty @ penalty)
        gradient = np.array([np.mean(error * values), error.mean()]) + regularization * penalty
        return float(loss), gradient

    result = minimize(objective, [1.0, 0.0], jac=True, method="L-BFGS-B",
                      bounds=[(1e-6, None), (None, None)],
                      options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Calibration optimization failed: {result.message}")
    return {"method": "positive_sigmoid_log_odds", "slope": float(result.x[0]),
            "intercept": float(result.x[1]), "regularization": regularization,
            "fit_partition": "validation", "iterations": int(result.nit)}


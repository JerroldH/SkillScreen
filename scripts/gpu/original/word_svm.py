"""Original experiment helpers, limited to functions used by this package."""

import argparse

import csv

import hashlib

import importlib.metadata

import json

from pathlib import Path

import platform

import time

import joblib

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from sklearn.metrics import precision_recall_curve, roc_curve

from sklearn.pipeline import make_pipeline

from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]

UPSTREAM = "https://github.com/protectskills/MaliciousSkillBench/blob/893afd609be22a2dfac5a2b94bbebb4eeb99515e/baselines/run_baselines.py"

def detection_metrics(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "flagged_fraction": float(np.mean(predictions)),
    }

def recall_gate(labels, scores, target):
    """Pick the highest validation threshold meeting recall, including score ties."""
    if not 0 < target <= 1 or set(np.asarray(labels).tolist()) != {0, 1}:
        raise ValueError("Recall gates require both classes and a target in (0, 1]")
    _, recall, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    threshold = float(thresholds[np.flatnonzero(recall >= target)[0]])
    return {
        "target_recall": target, "threshold": threshold, "comparator": ">=",
        **detection_metrics(labels, np.asarray(scores) >= threshold),
    }


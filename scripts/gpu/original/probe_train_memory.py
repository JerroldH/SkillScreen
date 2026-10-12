"""Original experiment helpers, limited to functions used by this package."""

import argparse

import csv

import hashlib

import json

from pathlib import Path

import subprocess

import sys

import time

ROOT = Path(__file__).resolve().parents[1]

TARGETS = ["q_proj", "v_proj", "in_proj_qkv", "out_proj"]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


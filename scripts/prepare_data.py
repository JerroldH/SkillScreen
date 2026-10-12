"""Restore pinned MSD/SD exact-text inputs without executing any skill content.

Optional dependency: requirements/data.txt. The default Table 1 replay does not
need this script, network access, or raw text.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("train", "validation", "test")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_file(path, expected):
    if not Path(path).is_file():
        raise ValueError(f"Missing required file: {path}")
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path.name}: expected {expected}, got {actual}")
    return actual


def source_file(info, config, source_dir, download):
    path = source_dir / info["path"]
    downloaded = False
    if not path.exists() and download:
        url = (f"https://huggingface.co/datasets/{config['repo_id']}/resolve/"
               f"{config['revision']}/{info['path']}?download=true")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SkillScreen-repro/1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, 1024 * 1024)
            verify_file(temporary, info["sha256"])
            temporary.replace(path)
            downloaded = True
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    verify_file(path, info["sha256"])
    return path, {"file": info["path"], "sha256": info["sha256"], "downloaded_this_run": downloaded}


def counts(rows):
    labels = Counter(int(row["label"]) for row in rows)
    return {"total": len(rows), "benign": labels[0], "malicious": labels[1]}


def verify_metadata(primary, config):
    path = ROOT / config["metadata"]["path"]
    verify_file(path, config["metadata"]["sha256"])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reference = {row["benchmark_id"]: row for row in rows}
    if len(reference) != len(rows) or set(reference) != {row["benchmark_id"] for row in primary}:
        raise ValueError("Benchmark metadata identities do not match the pinned primary table")
    for row in primary:
        expected = reference[row["benchmark_id"]]
        for key in ("source_id", "source_name", "structural_family_id"):
            if (row[key] or "") != expected[key]:
                raise ValueError(f"Metadata {key} mismatch: {row['benchmark_id']}")
        if row["source_ids"] != json.loads(expected["source_ids"]):
            raise ValueError(f"Metadata source_ids mismatch: {row['benchmark_id']}")


def prepare_records(primary, official):
    """Preserve the original experiment's identity, exclusion, and sorting rules."""
    membership = {}
    for row in official:
        key = row["benchmark_id"]
        if key in membership or row["split"] not in (*PARTITIONS, "excluded"):
            raise ValueError(f"Invalid or duplicate official membership: {key}")
        membership[key] = row
    parts = {name: [] for name in PARTITIONS}
    manifest, seen, text_partitions = [], set(), {}
    for row in primary:
        key = row["benchmark_id"]
        if key in seen or key not in membership:
            raise ValueError(f"Duplicate or unassigned primary identity: {key}")
        seen.add(key)
        split = membership[key]
        if str(row["label"]) not in ("0", "1") or str(row["label"]) != str(split["label"]):
            raise ValueError(f"Invalid or inconsistent label: {key}")
        if "source_id" in split and row["source_id"] != split["source_id"]:
            raise ValueError(f"Inconsistent source: {key}")
        text = row["skill_text"]
        if text is not None and not isinstance(text, str):
            raise ValueError(f"Invalid skill_text type: {key}")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
        reason = ""
        if split["split"] == "excluded":
            reason = "official_excluded"
        elif text is None or not row.get("text_available", True):
            reason = "missing_exact_text"
        elif not text.strip():
            reason = "empty_skill_text"
        manifest.append({
            "benchmark_id": key, "split": split["split"], "label": int(row["label"]),
            "source_id": row.get("source_id", ""), "status": "excluded" if reason else "included",
            "exclusion_reason": reason, "skill_text_sha256": text_hash,
        })
        if not reason:
            previous = text_partitions.setdefault(text_hash, split["split"])
            if previous != split["split"]:
                raise ValueError(f"Exact text overlaps partitions: {key}")
            parts[split["split"]].append({"benchmark_id": key, "label": int(row["label"]), "skill_text": text})
    if seen != set(membership):
        raise ValueError("Official manifest has identities absent from the primary table")
    for rows in parts.values():
        rows.sort(key=lambda row: row["benchmark_id"])
    manifest.sort(key=lambda row: row["benchmark_id"])
    return parts, manifest


def restore_protocol(name, primary, official, config, output_root):
    info = config["protocols"][name]
    manifest_path = ROOT / info["manifest"]["path"]
    summary_path = ROOT / info["summary"]["path"]
    verify_file(manifest_path, info["manifest"]["sha256"])
    verify_file(summary_path, info["summary"]["sha256"])
    expected = json.loads(summary_path.read_text(encoding="utf-8"))
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        expected_manifest = list(csv.DictReader(handle))
    parts, manifest = prepare_records(primary, official)
    # Compare every identity, split, label, source, exclusion and exact-text hash.
    actual_manifest = [{key: str(value) for key, value in row.items()} for row in manifest]
    if actual_manifest != expected_manifest:
        raise ValueError(f"{name}: restored per-record metadata differs from the frozen experiment")
    prepared_counts = {key: counts(rows) for key, rows in parts.items()}
    if prepared_counts != expected["prepared_counts"]:
        raise ValueError(f"{name}: prepared counts differ from the frozen experiment")
    official_counts = {key: counts([row for row in manifest if row["split"] == key])
                       for key in (*PARTITIONS, "excluded")}
    exclusions = dict(Counter(row["exclusion_reason"] for row in manifest if row["status"] == "excluded"))
    if official_counts != expected["official_counts"] or exclusions != expected["excluded_counts"]:
        raise ValueError(f"{name}: official counts or exclusions differ")
    # Publish a complete protocol workspace only after all output hashes pass.
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=output_root) as directory:
        stage = Path(directory) / name
        (stage / "data/local").mkdir(parents=True)
        for partition, rows in parts.items():
            path = stage / expected["files"][partition]["path"]
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            verify_file(path, expected["files"][partition]["sha256"])
        shutil.copyfile(manifest_path, stage / "data/split_manifest.csv")
        shutil.copyfile(summary_path, stage / "data/summary.json")
        stage.rename(output_root / name)
    return {"protocol": info["protocol"], "prepared_counts": prepared_counts,
            "excluded_counts": exclusions, "files": expected["files"],
            "all_record_metadata_matches": True, "all_output_sha256_matches": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("all", "msd", "sd"), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/data",
                        help="Creates separate msd/ and sd/ workspaces here; refuses to overwrite them")
    parser.add_argument("--snapshot-dir", type=Path,
                        help="Read a pre-downloaded pinned Parquet snapshot, without network access")
    parser.add_argument("--download-dir", type=Path,
                        help="Verified download cache, default OUTPUT_ROOT/_downloads; no global HF cache is read")
    args = parser.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install optional data dependencies: python -m pip install -r requirements/data.txt") from exc
    config = json.loads((ROOT / "configs/data.json").read_text(encoding="utf-8"))
    names = list(config["protocols"]) if args.protocol == "all" else [args.protocol]
    for name in names:
        if (args.output_root / name).exists():
            parser.error(f"Refusing to overwrite existing workspace: {args.output_root / name}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_dir = args.snapshot_dir or args.download_dir or args.output_root / "_downloads"
    primary_path, primary_source = source_file(config["primary_file"], config, source_dir, args.snapshot_dir is None)
    primary = pq.read_table(primary_path, columns=["benchmark_id", "label", "source_id", "skill_text", "text_available",
                                                  "source_name", "source_ids", "structural_family_id"]).to_pylist()
    verify_metadata(primary, config)
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": {key: config[key] for key in ("repo_id", "revision", "config")},
              "mode": "existing_snapshot" if args.snapshot_dir else "download_or_verified_local_cache",
              "all_source_family_metadata_matches": True,
              "input_files": [primary_source], "protocols": {}}
    for name in names:
        path, provenance = source_file(config["protocols"][name]["official_split_file"], config, source_dir, args.snapshot_dir is None)
        report["input_files"].append(provenance)
        report["protocols"][name] = restore_protocol(name, primary, pq.read_table(path).to_pylist(), config, args.output_root)
        print(f"{name}: all original text hashes and train/validation/test file hashes match")
    (args.output_root / "restoration-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Restored exact inputs to {args.output_root}")


if __name__ == "__main__":
    main()

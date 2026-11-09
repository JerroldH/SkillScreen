---
pretty_name: SkillScreen Data
license: other
license_name: mixed-source-terms
license_link: https://huggingface.co/datasets/SkillScreen/SkillScreen-Data#sources-and-terms
task_categories:
  - text-classification
configs:
  - config_name: markdown_random
    default: true
    data_files:
      - split: train
        path: data/markdown_random/train.parquet
      - split: validation
        path: data/markdown_random/validation.parquet
      - split: test
        path: data/markdown_random/test.parquet
  - config_name: markdown_sd
    data_files:
      - split: train
        path: data/markdown_sd/train.parquet
      - split: validation
        path: data/markdown_sd/validation.parquet
      - split: test
        path: data/markdown_sd/test.parquet
  - config_name: package_random
    data_files:
      - split: train
        path: data/package_random/train.parquet
      - split: validation
        path: data/package_random/validation.parquet
      - split: test
        path: data/package_random/test.parquet
  - config_name: package_sd
    data_files:
      - split: train
        path: data/package_sd/train.parquet
      - split: validation
        path: data/package_sd/validation.parquet
      - split: test
        path: data/package_sd/test.parquet
---

# SkillScreen Data

Agent-skill text, labels and source records for training decision models.
Choose Markdown alone or Markdown with companion files, with the same case IDs.

**v0.1** provides four full-text configurations in
[SkillScreen/SkillScreen-Data](https://huggingface.co/datasets/SkillScreen/SkillScreen-Data).
[GitHub](https://github.com/JerroldH/SkillScreen/tree/community-v0.1) keeps the
index, export scripts and contribution workflow.

## Load the data

Install `datasets` and choose an input view and split protocol. No upstream
recovery, GPU, API key or custom dataset code is needed.

```python
from datasets import load_dataset

ds = load_dataset("SkillScreen/SkillScreen-Data", "package_sd")
sample = ds["train"][0]
skill_text = sample["skill_text"]
companion_files = sample["files"]  # [{"path": ..., "text": ..., "sha256": ...}]
label = sample["label"]           # 0 = benign, 1 = malicious
```

For a downloaded copy, replace the repository name with the local dataset
folder. Each configuration contains `train`, `validation` and `test` sets.

| Configuration | Input | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| `markdown_random` | SKILL.md | 9,461 | 1,350 | 2,703 |
| `markdown_sd` | SKILL.md | 10,607 | 1,516 | 1,364 |
| `package_random` | SKILL.md + eligible companion text | 8,290 | 1,178 | 2,363 |
| `package_sd` | SKILL.md + eligible companion text | 9,563 | 1,369 | 874 |

The pool has **13,514 records**, comprising 7,500 malicious and 6,014 benign
skills. **11,831 packages** have verified complete inventories. Of those,
7,036 have eligible companion text and 4,795 have no additional eligible text.
Files are stored as data inside Parquet, never installed or executed.
Treat their instructions as untrusted input.

## Inputs and provenance

| Column | Meaning |
| --- | --- |
| `id`, `group_id` | Stable normalized-Markdown content group |
| `label` | ClassLabel, 0 benign / 1 malicious |
| `skill_text`, `skill_sha256` | Exact canonical Markdown and its UTF-8 hash |
| `files` | Companion paths, decoded text and hashes, sorted by path; empty in Markdown configs |
| `coverage` | Original package availability, independent of the selected input mode |
| `source_id`, `source_ids`, `source_name` | Primary source collection and all public source aliases |
| `source_url`, `revision`, `member_ids` | Fixed source version and original public record IDs |
| `label_origin`, `evidence` | Origin and basis of the reference label |
| `upstream_license`, `package_source` | Source terms and the selected package's URL, revision and terms |

Use `skill_text` and, for package models, `files` as model inputs. Labels,
source names and evidence are metadata. Canonical SKILL.md is identical across
the two input views. Package text includes eligible scripts, manifests and
documents. Binary files and benchmark annotations are excluded; runtime
dependencies are not bundled. Models choose their own context limit.

The accompanying `records.jsonl` contains full inventories, original byte
hashes, public ID mappings and recovery locations. Its text fields stay null
because text lives in Parquet. `sources.json` pins the download assets.
`splits/` preserves the original assignment CSVs, including exclusions.

## Splits and labels

Grouped random is source/label-stratified with seed 42 and keeps normalized
duplicates together. SD holds out SkillHarm (SRC009), ATR (SRC011) and
SkillFortifyBench (SRC012). Train and validation may share source collections.
These groups refer to benchmark collections, not individual repositories.

**The existing SD assignment is unchanged.** Its Markdown test has 838 malicious
and 526 benign cases. Twenty-seven excluded IDs remain in the index and split
CSV, outside the SD Parquet sets. Package configs select complete packages
within each existing partition without reshuffling. The package SD test has
784 malicious and 90 benign cases, reflecting incomplete ATR package coverage.
For paired comparisons, filter Markdown to the IDs in the corresponding package
partition before comparing models.

These community splits are separate from the paper's SBR/MSD/SD protocols.
Existing checkpoints used earlier protocols, so these partitions are not fresh
held-out tests for those checkpoints. The original paper artifacts are unchanged.

Labels are inherited upstream references. Identical normalized Markdown shares
one ID. Six content groups with unresolved label questions are withheld.
The pool contains no private labels or weak positives. Additional coverage is
1,634 Markdown-only, 25 partial and 24 ambiguous records. Ambiguous package
versions are not combined.

## Build or audit a local copy

From the code repository with Python 3.12+, restore the fixed source data once,
then build and verify all four configs. Use a new output directory.

```sh
python -m pip install -r requirements/data.txt
python scripts/restore_community.py --mode markdown --output outputs/community-markdown.jsonl
python scripts/restore_community.py --mode package --output outputs/community-packages.jsonl
python scripts/export_community.py build --markdown-text outputs/community-markdown.jsonl --package-text outputs/community-packages.jsonl --output outputs/hf-fulltext
```

`build` also runs `load_dataset` for every config and verifies every ID, label,
eligible text hash and split membership. It writes `verification.json` in the
export folder and never uploads. Verify an existing export with:

```sh
python scripts/export_community.py verify --output outputs/hf-fulltext
```

Recovery is an optional provenance check for maintainers, not a step required
by users loading a generated release. Downloads are cached as inert hash blobs.
No source file is executed. The index-only export remains available through
`python scripts/community_data.py export --output outputs/hf-index`.

## Contribute

Start with a public source URL, fixed revision and evidence through the
[case form](https://github.com/JerroldH/SkillScreen/issues/new?template=case.yml).
Maintainers check labels, duplicates, coverage and terms. Unknown cases stay out
of supervised sets. New cases remain excluded from frozen protocols until the
next release. Markdown and package views are generated from one shared index.

## Sources and terms

- [MaliciousSkillBench](https://huggingface.co/datasets/ProtectSkills/MaliciousSkillBench/tree/d4b42ce5766a6e0359c987cf59c1007cb3795a90) supplies benchmark labels and Markdown. Its metadata and splits use CC BY 4.0; third-party skill contents retain their original terms.
- [MalSkillBench](https://github.com/lxyeternal/MalSkillBench/tree/3548e2837adf58b96497f2c688f96beff26485da) supplies additional benign packages and SRC001 cases. Its notice states academic research use only.
- [SkillTrustBench](https://huggingface.co/datasets/cuhk-zhuque/SkillTrustBench/tree/f90517b7058fdcfea89af114c069fbf973f42bc7) supplies SRC010 packages under its CC BY-NC-SA 4.0 notice and retained upstream terms.
- [SkillHarm](https://huggingface.co/datasets/osunlp/SkillHarm/tree/cc21f90982ea49de43bc30a08f338788a92916f4) supplies SRC009 cases; its dataset card lists CC BY 4.0.
- [SkillFortifyBench](https://github.com/qualixar/skillfortifybench/tree/eb9d5a9cbfa13b82ec3392d26dbae82fc6454b39) supplies native malicious and benign cases under MIT.
- [Official package archives](https://github.com/protectskills/MaliciousSkillBench/tree/893afd609be22a2dfac5a2b94bbebb4eeb99515e/packages) provide fixed package snapshots without relicensing their contents.

Our original documentation and annotations use CC BY 4.0. Imported bodies
retain their source terms; the project code's Apache-2.0 license does not apply
to them. `unknown` means a record's original content license has not been
resolved. The dataset does not grant additional rights over imported text.

Please cite the source datasets you use and the [SkillScreen study](https://arxiv.org/abs/2610.17727).

# Optional model reruns

These commands train new models. They are separate from the default CPU Table 1 recomputation and never run implicitly.

The package preserves the original training loops, tokenization, checkpoint selection, calibration functions, and test scorers. `run.py` places every run in its own output workspace and replaces the original cluster-specific orchestration. Historical settings and frozen operating points are under `configs/skillscreen` and `configs/modernbert`.

## Historical records and packaged code

The historical JSON files are reference records, not checksum manifests for this package. Their recorded hashes refer to original experiment artifacts. Input configurations were reduced for packaging and line endings normalized. Platform-specific version suffixes are anonymized as `+localbuild`, and the historical evaluation launcher is named `test_direct.sh`. Numerical training settings, selected checkpoints, calibration and thresholds are retained.

| Status relative to the recorded experiment | Files under `scripts/gpu/original/` |
|---|---|
| Byte-identical to recorded script hashes | `train_direct.py` (hash recorded in ModernBERT's historical freezes), `score_frozen_direct.py`, `score_encoder.py` |
| Modified for packaging | `calibrate_direct.py`, `direct_smoke.py`, `encoder_common.py`, `probe_train_memory.py`, `train_encoder.py`, `word_svm.py` |
| Packaged helper with no hash entry in the historical freeze records | `cache_direct_inputs.py` |
| Historical scripts not included | `freeze_direct_run.py`, `freeze_encoder_run.py`, `test_direct.sh` |

Packaging changes remove unused entry points and anonymize environment-variable names as `SKILLSCREEN_*`. Current distributed file hashes are in the top-level `checksums.json`. New runs generate and verify their own preparation and freeze records; the runner does not validate new files against historical config hashes.

## Environment

Use Linux or WSL2, Python 3.11+, and a CUDA GPU with bfloat16 support. Install `requirements/gpu.txt` in a separate environment. The original runs used H100 10 GB MIG slices and cluster builds of PyTorch 2.11.0 and Transformers 5.17.0. The requirements specify public package equivalents. They have not been validated by training in this release, and they need a compatible CUDA driver. Different kernels or library builds can change new training results.

Prepare stages download the fixed model revision and tokenize training and validation inputs. Skill text is never executed. ModernBERT tokenizes full text before truncation, including the very large training record, so CPU tokenization can need more host memory than training. SkillScreen reuses checked token counts and bounded prefix encoding for that record. The input limit is always 8,192 tokens.

In `sd-train-token-lengths.csv`, the historical `truncated` flag uses a 4,096-token cutoff (571 records). The runner reads only `original_token_count` and applies the current 8,192-token limit, above which there are 60 records; it ignores that historical flag.

## One explicit stage at a time

From the package root, first restore the official data using `scripts/prepare_data.py`. For example, to rerun SkillScreen MSD seed 42:

```sh
python scripts/prepare_data.py --protocol msd --output-root outputs/data
python scripts/gpu/run.py --model skillscreen --protocol msd --seed 42 --stage prepare --dry-run
python scripts/gpu/run.py --model skillscreen --protocol msd --seed 42 --stage prepare
python scripts/gpu/run.py --model skillscreen --protocol msd --seed 42 --stage train
python scripts/gpu/run.py --model skillscreen --protocol msd --seed 42 --stage calibrate
python scripts/gpu/run.py --model skillscreen --protocol msd --seed 42 --stage evaluate
```

Use `--model modernbert` for the fully fine-tuned encoder and `--protocol sd` for Source-Disjoint. `--data-dir` points to a restored protocol workspace containing `data/summary.json`. `--output-root` changes the independent output location. Preparing an existing workspace fails rather than overwriting it.

The Table 1 seed lists are fixed:

| Model | MSD | SD |
|---|---|---|
| SkillScreen | 42, 1, 2, 3, 4 | 42, 1, 2, 3, 4 |
| ModernBERT-large | 42, 1, 2, 3, 4 | 42, 1, 3, 4, 5 |

MSD ModernBERT uses the complete GPU rerun. SD seed 5 replaced the collapsed seed 2 in the reported table. The runner never chooses seeds based on newly obtained scores. Unexpected divergence in a new run should be reported, not silently replaced.

Training uses one epoch, with validation at its midpoint and end. Checkpoint selection minimizes validation FPR at at least 99% recall, then maximizes AP, then minimizes loss. Calibration fits a positive-slope sigmoid to the selected checkpoint's validation log odds. A fresh run fits its own validation gate, rather than reusing a historical checkpoint's threshold. The historical freeze files are reference records only.

Evaluation verifies the frozen checkpoint and replays three validation examples before reading test text. Test predictions are written to `outputs/gpu/<model>/<protocol>/seed-<seed>/outputs/test/scores.csv`. They contain the native probability, calibrated probability, and frozen validation-gate decision. The native probability is the Table 1 score. **Table 1 FPR@99 is a separate test-score sweep, not the frozen validation-gate FPR.**

For the untuned baseline, run only the two stages below. It uses the same OpenJev revision, template, truncation, and raw entailment probability, with a native threshold of 0.5.

```sh
python scripts/gpu/run.py --model untuned --protocol msd --stage prepare
python scripts/gpu/run.py --model untuned --protocol msd --stage evaluate
```

## Release checks

Syntax, all allowed model/protocol/seed dry-run combinations, and CPU calibration replay are checked locally. **No training, model download, GPU probe, or GPU inference was performed while packaging.** The original training and scoring programs already include numerical checks and export-reload checks. Runtime execution of those checks remains part of a future explicit GPU rerun.

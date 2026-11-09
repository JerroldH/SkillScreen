# Choosing and contributing models

Input scope and training scope are separate choices. A package model reads
supporting files. An all-data model uses all eligible examples in a declared
training pool. Either can use LoRA or full parameter fine-tuning.

| Training scope | Weight updates | Evaluation |
| --- | --- | --- |
| `train_only` | Only the named training split. Validation may select the checkpoint, calibration, and threshold. | Report the corresponding untouched test split. |
| `all_data` | The declared full pool, potentially including former validation and test examples. | Report independent new data. Label cross-validation results as OOF results, not tests of the final refit. |

Both types can be released and used for inference. A train-only model is not
automatically a fair baseline on every dataset. Its entire training history,
including the starting checkpoint, determines evaluation overlap.

## Models used by the current entry point

| Model | Input | Training scope | Training data |
| --- | --- | --- | --- |
| [Package-Experimental](https://huggingface.co/SkillScreen/SkillScreen-2B-Package-Experimental) | SKILL.md and supporting files, 32K | Train-only | Recovered SRC010 packages in the original SBR training split, seed 42 |
| [SBR](https://huggingface.co/SkillScreen/SkillScreen-2B-SBR) | SKILL.md, 8K | Train-only | Original MaliciousSkillBench SBR training split, seed 42 |

Exact model and dataset revisions are in [community-models.json](configs/community-models.json).
The CLI uses each released model's native score and a 0.5 review threshold.
It does not substitute a validation-selected or OOF threshold.

The community entry point has passed GPU inference checks on an NVIDIA L4 with
the pinned Linux runtime in `requirements/inference.txt`. Short fixtures and
the full 8K/32K context limits were checked. These are execution checks, not
new detection results. To repeat them, run
`python scripts/smoke_inference.py --output outputs/gpu-smoke.json`.

The published MSD and SD checkpoints retain their original evaluation protocols.
They are available on [HF](https://huggingface.co/SkillScreen), but are not extra
CLI backends in this first community version.

## All-data deployment candidate

A separate local Markdown-only candidate has been trained on 10,079 deduplicated
records, including the public benchmark and additional labeled material. It is
an all-data refit. Its three-fold calibration results belong to the fold models,
and the final refit does not have an independent test result from that pool.

This candidate is not published or selected by `scan.py`. Its training pool is
not the public metadata index in this repository. A later release can use the
name `SkillScreen-2B-Deploy`, with its own training-data description and threshold
settings. The existing model repositories will keep their experiment identities.

## Contribute another model

Use the [model form](https://github.com/JerroldH/SkillScreen/issues/new?template=model.yml).
Any architecture is welcome. Include these items in its model card:

- Input scope and inference example.
- Training scope, dataset revision, protocol, and partitions used for updates.
- Base checkpoint and earlier task-specific training.
- Score direction, threshold, and its selection data.
- License and evaluation results, if available.

Declaring `train_only` describes training history; it is not a badge granted
automatically by the repository. A later fine-tune cannot erase test exposure
from an earlier checkpoint.

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
| [Package v0.1.0](https://huggingface.co/SkillScreen/SkillScreen-2B-Package) | SKILL.md and supporting files, 32K | All-data | 11,831 eligible public packages, seed 42 |
| [Markdown v0.1.0](https://huggingface.co/SkillScreen/SkillScreen-2B-Markdown) | SKILL.md, 8K | All-data | 13,514 public Markdown records, seed 42 |

Model versions are pinned in [community-models.json](configs/community-models.json).
Each model includes its dataset revision and full-precision thresholds in
`deployment_settings.json`. Package defaults to the frozen 99% OOF recall point.
Markdown defaults to native 0.5. Use `--operating-point high-recall` for the saved
99% OOF recall point, or `--operating-point native` for 0.5.
The optional `balanced` policy maximizes recall minus FPR among the three
predeclared 95/98/99% OOF recall gates. It does not search all thresholds.

The deployment entry point has passed GPU inference checks on an NVIDIA A100 with
the pinned Linux runtime in `requirements/inference.txt`. Short fixtures and
the full 8K/32K context limits were checked. These are execution checks, not
new detection results. To repeat them, run
`python scripts/smoke_inference.py --output outputs/gpu-smoke.json`.

The published MSD and SD checkpoints retain their original evaluation protocols.
They are available on [HF](https://huggingface.co/SkillScreen), but are not extra
CLI backends in this first community version.

## Deployment evaluation

The two v0.1.0 releases use the public SkillScreen-Data pool. Three-fold OOF
predictions keep normalized Markdown groups together and stratify by source and
label. Each final model is a separate one-epoch refit on its full eligible pool.
The cards report OOF metrics and include the predictions and frozen thresholds.
The final refits have no independent held-out result from that same training pool.

The earlier private 10,079-record candidate is separate from these releases.
SBR, MSD and SD retain their existing weights and experiment identities.
[Package-SBR](https://huggingface.co/SkillScreen/SkillScreen-2B-Package-SBR), formerly
Package-Experimental, is the original 32K model trained on recovered packages
from the SBR training split. Its weights and existing version tags are unchanged.

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

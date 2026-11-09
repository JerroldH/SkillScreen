# SkillScreen

Screen agent skills with local decision models. Inspect a complete package, score
its instructions and supporting files, or contribute a case for review.

[Paper](https://arxiv.org/abs/2610.17727) · [Project page](https://jerroldh.github.io/SkillScreen-webpage/) · [Models](https://huggingface.co/SkillScreen) · [Dataset](https://huggingface.co/datasets/SkillScreen/SkillScreen-Data)

## Start here

| I want to… | Start with |
| --- | --- |
| Try a complete skill package | [Run the examples](examples/README.md) |
| Screen my own skills | `python scan.py path/to/skill` |
| Share a benign example, missed attack, or dependency concern | [Contribute a case](CONTRIBUTING.md) |
| Add a model or reproduce the paper | [Model contributions](CONTRIBUTING.md#contribute-a-model) · [Reproduction](REPRODUCTION.md) |
| Load training data, including companion files | [Four dataset configurations](data/community/README.md#load-the-data) |

### Try it without a GPU

Use Python 3.12+. This first command needs no extra packages or network access.

```sh
git clone --branch community-v0.1 https://github.com/JerroldH/SkillScreen.git
cd SkillScreen
python scan.py examples/local-inventory --inspect --output outputs/inventory.json
```

It reads `SKILL.md`, a helper script, a reference document, and a requirements
file as text. Nothing inside the package is executed. The JSON report lists the
files, hashes, and skipped entries. Inspection does not run security checks.

### Score with a released model

On Linux with a CUDA GPU supporting BF16:

```sh
python -m pip install -r requirements/inference.txt
python scan.py examples/local-inventory --output outputs/score.json
```

| Input | Released model | Context |
| --- | --- | --- |
| Package directory, the default | [Package-Experimental](https://huggingface.co/SkillScreen/SkillScreen-2B-Package-Experimental) | 32K |
| Single `SKILL.md`, or `--mode markdown` | [SBR](https://huggingface.co/SkillScreen/SkillScreen-2B-SBR) | 8K |

Both use the existing v1.1 weights and original input recipes, pinned in
[configs/community-models.json](configs/community-models.json). First use downloads
the checkpoint. The report includes the native model score, the default 0.5 review
threshold, and input coverage. Package mode records which files fit in the context.
The threshold is not a guarantee of 99% recall on a new collection.

These two releases are **train-only evaluation models**. A future **all-data
deployment release** will have its own name and training record. Package input
does not imply all-data training. See [model scopes](MODELS.md).

### Check dependency versions separately

```sh
python scan.py examples/dependency-case --checks dependencies --output outputs/dependencies.json
```

This CPU-only check queries [OSV](https://osv.dev/) for exact PyPI versions in
`requirements*.txt`. Only package names and versions leave the machine. The
example deliberately contains an old pin for inspection. Do not install it.
Ranges and recognized unsupported manifests are reported as unresolved.

Use `--checks all` to combine the two reports. Model scores and dependency
advisories stay separate. An affected version is evidence of dependency risk,
not proof that the skill author intended harm.

## Contribute

A useful contribution can be just **a source link and an explanation**. Benign
skills that trigger false alarms are particularly useful. You can also submit
a missed attack, a package with relevant supporting files, a dependency advisory,
or another decision model. See [CONTRIBUTING.md](CONTRIBUTING.md).

The [data index](data/community/README.md) contains **13,514 public-source
records**, with inherited labels, provenance and file hashes. It supports
11,831 complete packages and new grouped-random and source-disjoint splits.
The original paper protocols remain frozen separately.
The [HF dataset](https://huggingface.co/datasets/SkillScreen/SkillScreen-Data)
includes the actual text in four Parquet configurations,
`markdown_random`, `markdown_sd`, `package_random`, and `package_sd`.
Each loads directly with `load_dataset` and includes train/validation/test.
Verified source recovery remains available for auditing.
Community submissions remain candidates until reviewed. Upstream skill files
retain their own licenses.

```python
from datasets import load_dataset

ds = load_dataset("SkillScreen/SkillScreen-Data", "package_sd")
```

## Reproduce the paper

Recompute the MSD and SD main results from the included predictions. Runs offline with Python 3.10+ and no extra dependencies.

```sh
python scripts/reproduce.py
```

Outputs `table1.csv`, `table1.tex` and `verification.json` in `outputs/table1/`.

Run selections and metric rules are in [configs/runs.json](configs/runs.json). See [REPRODUCTION.md](REPRODUCTION.md) for details.

For retraining and baseline scoring, see [REPRODUCTION.md](REPRODUCTION.md),
[GPU instructions](scripts/gpu/README.md), and [baseline instructions](scripts/baselines/README.md).
The paper's MSD/SD predictions, split manifests, and metric rules are unchanged.

## Citation

If you use SkillScreen's code, models, or predictions, or build models based on SkillScreen in your research, please cite our paper. If you release a derived model, please identify SkillScreen as its source and link to the original checkpoint.

```bibtex
@article{huang2026skillscreen,
  title   = {SkillScreen: High-Recall Screening of Malicious Agent Skills with Decision Models},
  author  = {Huang, Jerrold and Quan, Junwei and Lyu, Saiyue and Guo, Zihan and Ge, Yingqiang},
  journal = {arXiv preprint arXiv:2610.17727},
  year    = {2026},
  doi     = {10.48550/arXiv.2610.17727},
  url     = {https://arxiv.org/abs/2610.17727}
}
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff). These citation requests do not add conditions to the licenses below.

## License

| Material | License |
| --- | --- |
| Project-authored code, configuration, and code examples | [Apache-2.0](LICENSE) |
| Project-authored documentation, per-skill predictions (`data/predictions/`), reference results (`expected/`), and citation metadata | [CC BY 4.0](LICENSE-CC-BY-4.0) |

These licenses allow research and commercial reuse, modification, and redistribution under their terms, without individual permission requests. Preserve the applicable attribution and license notices, and identify modifications as required by each license.

The licenses cover only rights held by the SkillScreen contributors. Benchmark metadata, split manifests, and other third-party materials retain their original licenses and attribution requirements. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are released separately on [Hugging Face](https://huggingface.co/SkillScreen); each model repository states its license.

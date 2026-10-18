# SkillScreen

High-recall screening of malicious agent skills with decision models.

[Paper](https://arxiv.org/abs/2610.17727) · [Project page](https://jerroldh.github.io/SkillScreen-webpage/) · [Models](https://huggingface.co/SkillScreen)

## Reproduce Table 1

Recompute the MSD and SD main results from the included predictions. Runs offline with Python 3.10+ and no extra dependencies.

```sh
git clone https://github.com/JerroldH/SkillScreen.git
cd SkillScreen
python scripts/reproduce.py
```

Outputs `table1.csv`, `table1.tex` and `verification.json` in `outputs/table1/`.

Run selections and metric rules are in [configs/runs.json](configs/runs.json). See [REPRODUCTION.md](REPRODUCTION.md) for details.

## Rerun experiments

Restore the benchmark text before training or scoring:

```sh
python -m pip install -r requirements/data.txt
python scripts/prepare_data.py --protocol all
```

Follow the [model training and inference](scripts/gpu/README.md) or [baseline evaluation](scripts/baselines/README.md) instructions.

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

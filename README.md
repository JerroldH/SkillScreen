# SkillScreen

High-recall screening of malicious agent skills with decision models.

[Project page](https://jerroldh.github.io/SkillScreen-webpage/) · [Models](https://huggingface.co/SkillScreen)

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

## License

[Limited reproduction permission](LICENSE). Third-party terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

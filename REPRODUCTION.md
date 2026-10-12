# Reproducing the main results

[Project page](https://jerroldh.github.io/SkillScreen-webpage/) · [Models](https://huggingface.co/SkillScreen)

Recompute the MSD and SD main table from the included per-skill predictions. The default command runs offline on a CPU using only Python's standard library. Optional commands restore benchmark text and rerun models.

## Quick start

With Python 3.10 or later:

```sh
git clone https://github.com/JerroldH/SkillScreen.git
cd SkillScreen
python scripts/reproduce.py
```

No installation, GPU, API key, model download, or network connection is needed. `requirements/cpu.txt` is intentionally empty of dependencies.

The command writes three files to `outputs/table1/`:

- `table1.csv` — the full table, including seed standard deviations.
- `table1.tex` — a LaTeX table snippet for inspection, requiring `booktabs`.
- `verification.json` — the comparison with `expected/table1.csv`, per-run metrics, and operating-point replay.

```sh
python scripts/reproduce.py --metric fpr99
python scripts/reproduce.py --output outputs/my-table
```

`--metric fpr99` limits the console display to MSD and SD test FPR@99. It still writes the complete table. A mismatch or invalid prediction file exits with a nonzero status.

## What is included

Only Table 1 methods on MSD and SD are included. `configs/runs.json` lists the selected prediction files, seeds and decision rules. `expected/table1.csv` contains the paper's displayed values. These are ordinary editable files, with no dependency on a particular PDF version.

SkillScreen uses seeds 42, 1, 2, 3, 4 on both splits. ModernBERT uses the five GPU MSD runs with these seeds and SD seeds 42, 1, 3, 4, 5. SD seed 5 replaced the collapsed seed 2, as described in the paper. Historical aliases and the earlier CPU MSD encoder set are not averaged into the table.

Each run is evaluated separately before averaging its metrics. Standard deviations use `ddof=1`. No prediction averaging or ensemble is introduced.

Default decisions use native probability `>= 0.5`, SVM margin `>= 0`, or SkillSpector risk `> 50`. AUROC uses finite native scores with average ranks for ties. Unavailable scanner/judge scores count as flagged in confusion metrics and are excluded from AUROC. Other families reject missing scores. The report records unavailable-score counts.

**Table 1 FPR@99 is measured on the test score sweep.** For each run, take the `ceil(0.99 × malicious_count)`-th highest malicious score as the threshold and flag scores at or above it. Ties remain together, with no interpolation. Unavailable review-policy scores count as positive infinity. Achieved recall can exceed 99% because of ties. This descriptive test statistic is separate from the validation-frozen operating point, which is replayed only as an audit in `verification.json`. Judge operating points are their fixed native threshold, not fitted 99%-recall gates.

SBR, low-data experiments, package-context experiments, model weights, token caches, API request/reply caches and raw skill text are outside this package.

This is the Table 1 subset of the full artifact release described in Appendix O. Here, `direct.csv` and `direct-*` prediction files denote SkillScreen.

| Path cited in the paper | Location in this package |
|---|---|
| `scripts/review_metrics.py` (Appendix K) | Not included. `scripts/reproduce.py` computes the Table 1 point estimates. Appendix K bootstrap intervals are outside this package. |
| `scripts/llm_judge_prompt.py` (Appendix N) | `scripts/baselines/llm_judge_prompt.py` |
| `configs/llm-judge-fewshot-examples.json` | `configs/baselines/fewshot-msd.json` |
| `configs/llm-judge-fewshot-examples-sd.json` | `configs/baselines/fewshot-sd.json` |

## Restore benchmark text

This optional step requires internet access and downloads only the pinned official dataset revision.

```sh
python -m pip install -r requirements/data.txt
python scripts/prepare_data.py --protocol all
```

It creates `outputs/data/msd/data/local/` and `outputs/data/sd/data/local/`, each containing `train.jsonl`, `validation.jsonl` and `test.jsonl`. Use `--protocol msd` or `--protocol sd` for one split, or `--output-root` for another location. Existing restored workspaces are not overwritten.

The source is `ProtectSkills/MaliciousSkillBench` at revision `d4b42ce5766a6e0359c987cf59c1007cb3795a90`. Recovery preserves the original exclusions, ordering, text and serialization. It verifies source files, IDs, labels and text hashes, then checks the six original JSONL hashes. Missing original text is not replaced with sanitized text. Data is treated as inert text throughout.

| Split | Train | Validation | Test |
|---|---:|---:|---:|
| MSD | 6,814 | 974 | 1,947 |
| SD | 7,510 | 833 | 1,384 |

## Optional reruns

These entry points are independent of the offline table command. Install their dependencies in separate environments.

| Path | Instructions |
|---|---|
| SkillScreen, ModernBERT, untuned OpenJev | [GPU instructions](scripts/gpu/README.md) |
| Word-SVM, SkillSpector, TypeSafe, LLM judges | [Baseline instructions](scripts/baselines/README.md) |

For example, inspect the SkillScreen run without downloading or training anything:

```sh
python scripts/gpu/run.py --model skillscreen --protocol msd --seed 42 --stage prepare --dry-run
python scripts/baselines/api.py plan --protocol sd --model gpt-5.5-fewshot
```

GPU reruns start from the fixed upstream base and train a new checkpoint. Fresh training can vary with kernels and hardware. The historical CPU table recomputation is exact, while a fresh training run is not promised to reproduce identical predictions.

API configurations preserve model identifiers, prompts, per-split few-shot IDs, request parameters and parsing rules. Few-shot text is reconstructed from the benchmark. Keys must be supplied through environment variables. Paid operations require an explicit independent command. Historical API models may change or become unavailable, but the supplied predictions remain usable offline.

Grok requests specified `reasoning_effort: none`, but historical returned usage reported 737,572 reasoning tokens on MSD and 463,143 on SD. The configuration preserves the actual request and records this observed behavior. The reseller's handling of the field upstream is unverified.

## Local validation

The supplied 40 test prediction files reproduce all 96 displayed Table 1 values and eight displayed standard deviations. Test IDs and labels are checked against the supplied cohorts. Numeric outputs were also checked against the original metric implementation.

Data restoration passed a fresh network download into an empty cache, including all six JSONL hashes. Model entry points passed syntax and dry-run checks, and the 20 historical trained-model calibration/gate records replayed on CPU. API prompt reconstruction and response parsing were checked without paid requests.

No model was trained or scored on a GPU during packaging. No paid API was called. The new GPU environments and complete training paths have not been runtime-tested in this packaging pass.

`checksums.json` lists the distributed files. See `LICENSE` for the limited reproduction permission and `THIRD_PARTY_NOTICES.md` for upstream terms. Generated outputs stay under `outputs/` by default.

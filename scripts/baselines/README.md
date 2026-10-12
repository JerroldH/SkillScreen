# Optional baseline reruns

The default `scripts/reproduce.py` uses saved predictions and makes no API calls. These separate commands create new runs. Execute them from the package root after restoring the official data. No training or paid calls were performed to validate this package.

## Local models

Word-SVM uses the original 120,000-feature word TF-IDF and balanced LinearSVC recipe. Fit on train, freeze the validation gate, then score test. Install the CPU dependencies first.

```sh
python -m pip install -r requirements/baselines.txt
python scripts/baselines/word_svm.py plan --protocol msd
python scripts/baselines/word_svm.py fit --protocol msd --data-dir outputs/data/msd --output outputs/svm-msd
python scripts/baselines/word_svm.py evaluate --protocol msd --data-dir outputs/data/msd --output outputs/svm-msd
```

Use `sd` and its prepared workspace for SD. New margins are saved as `score`. Table 1 uses margin >= 0. The original estimator's native class decision uses margin > 0; this distinction is recorded in the configuration.

SkillSpector uses the pinned 2.12.0 revision in `requirements/baselines.txt`. A separate environment is recommended. It receives one neutral-named `SKILL.md` containing the full original text. Its worker disables DNS and network connections, uses static mode without an LLM, and never runs package commands. A 90-second timeout produces an unavailable record, which is reviewed at the default operating point. The default rule is strictly **score > 50**.

```sh
python scripts/baselines/static.py plan --protocol sd
python scripts/baselines/static.py score --protocol sd --data-dir outputs/data/sd --output outputs/static-sd
```

Untuned OpenJev inference is provided by `scripts/gpu/run.py --model untuned`; see the adjacent GPU README. It requires only prepare and evaluate stages.

## API models

`configs/baselines/api.json` records each protocol's actual provider, requested and returned model IDs, request parameters, call timestamps, prompt fingerprint, and unavailable-score policy. Grok 4.3 was served through AI/ML API, using `x-ai/grok-4-3`. TypeSafe used `jev-1.13.0`. Providers may change aliases or retire versions, so fresh responses need not match the historical predictions.

Grok's historical requests specified `reasoning_effort: none`, while returned usage reported reasoning tokens. The recorded request is retained; it does not establish that reasoning was disabled or that the reseller forwarded the setting upstream. New score files record returned `reasoning_tokens` when available.

Install `requirements/api.txt`. The following commands are free of paid API calls. The tokenizer command downloads only the pinned upstream tokenizer, while plan is fully offline.

```sh
python scripts/baselines/api.py plan --model gpt-5.5-fewshot --protocol sd
python scripts/baselines/api.py tokenizer --tokenizer-dir outputs/tokenizer
python scripts/baselines/api.py validate-inputs --model gpt-5.5-fewshot --protocol sd --data-dir outputs/data/sd --tokenizer-dir outputs/tokenizer --limit 0
```

Every API model receives the same 8,192-token visible skill prefix as the original experiment. TypeSafe also receives the fixed hypothesis. Few-shot prompts reconstruct eight excerpts from the relevant protocol's **training** records and verify their excerpt hashes. SD uses its own eight examples. Neither the skill texts nor full API replies are bundled or saved by the API entry point.

For a new paid run, set the relevant environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AIMLAPI_API_KEY`, or `TYPESAFE_API_KEY`). Do not write keys into package files. The independent score/submit/collect commands require `--allow-paid`. They default to one record. **`--limit 0` explicitly selects the full test set.** Costs depend on the current provider and include retries. The entry point does not impose a monetary budget.

Historical GPT and Haiku test requests used batch processing. For example, a complete MSD GPT-5.5 rerun is submitted and collected separately.

```sh
python scripts/baselines/api.py submit --model gpt-5.5 --protocol msd --data-dir outputs/data/msd --tokenizer-dir outputs/tokenizer --output outputs/gpt-msd-submit --limit 0 --allow-paid
python scripts/baselines/api.py collect --model gpt-5.5 --protocol msd --data-dir outputs/data/msd --tokenizer-dir outputs/tokenizer --batch-state outputs/gpt-msd-submit/batch.json --output outputs/gpt-msd-results --limit 0 --allow-paid
```

Collect returns immediately if the batch is still running. Failed/expired/cancelled batch records and parse failures receive the original synchronous fallback. Refusals, exhausted token limits, and unparseable final answers remain unavailable. Transport errors stop the run. Completed score rows are flushed to disk, and a partial file must not be treated as a full result. New output directories prevent accidental overwriting. This compact entry point does not resume interrupted paid runs automatically.

Grok and TypeSafe used synchronous requests. For example, `score --model typesafe --protocol sd --data-dir outputs/data/sd --tokenizer-dir outputs/tokenizer --output outputs/typesafe-sd --limit 0 --allow-paid` runs TypeSafe. `score` also supports synchronous reruns of the batch models, but that changes the transport from the historical run. All original provider parameters and parsers are retained.

Offline validation checked all entry-point help/plan commands, the 12 provider configurations using synthetic response replay, TypeSafe's response parser, both full test-set visible-prefix token totals, and byte-identical recovered few-shot system messages. No new model predictions, scanner run, training, or paid API requests were generated.

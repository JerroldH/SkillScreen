# Contribute to SkillScreen

Start with one useful case or one model. No GPU, training run, or large dataset
is required to contribute.

## Share a case

Open a [case issue](https://github.com/JerroldH/SkillScreen/issues/new?template=case.yml)
with the source link and a short explanation. Useful cases include:

- A legitimate skill that is flagged, especially from a new source.
- A missed attack, with the relevant file and behavior identified.
- A skill whose scripts or configuration change the interpretation of SKILL.md.
- A dependency recommendation linked to a published advisory.

Use **unknown** when intent is unclear. A vulnerable dependency alone does not
make a skill malicious. Include the exact version and advisory when reporting
dependency risk. Please link public evidence rather than pasting credentials,
private data, or a whole third-party package into an issue.

If you already have the package locally, you can attach a small inventory:

```sh
python scripts/contribute.py path/to/skill --label unknown --evidence "Explain the behavior and relevant file here." --source-url https://github.com/owner/repo --revision COMMIT_SHA --license unknown --output outputs/case.json
python scripts/contribute.py --check outputs/case.json
```

Replace `COMMIT_SHA` with the full upstream Git commit. The draft contains paths,
hashes, evidence, and a proposed label. It does not copy file contents. You can
omit the URL and revision while preparing a local draft, as the
[example](examples/README.md#prepare-a-contribution) does. The checker reports
missing provenance. It does not verify labels or source ownership.

## How a case becomes data

A maintainer checks the fixed source revision, file coverage, label evidence,
duplicates, and upstream terms. A case remains a candidate until those checks
are complete. We record whether the evidence covers only SKILL.md or includes
supporting files. Unresolved cases stay out of labeled evaluation sets.

The HF dataset is [SkillScreen/SkillScreen-Data](https://huggingface.co/datasets/SkillScreen/SkillScreen-Data). Its
[local index and release instructions](data/community/README.md) now contain
the public index, package recovery and community random/SD splits. The
full-text dataset provides four directly loadable Markdown/package configurations. GitHub
reviews additions and corrections; HF hosts generated releases.
Source text retains its upstream terms.
Our annotations cannot relicense upstream files.

We will freeze evaluation releases instead of changing their test sets whenever
a case arrives. Related versions and duplicate packages must stay in the same
split. Markdown and package evaluations use explicit case IDs and input coverage.
The index contains Markdown-only and complete-package records. Generated HF
files include the text, so users need not recover it separately. New cases initially stay excluded
from frozen evaluation protocols until a new release assigns them.

## Contribute a model

Open a [model issue](https://github.com/JerroldH/SkillScreen/issues/new?template=model.yml)
with its checkpoint or code, license, input scope, training-data description,
and example output. Decision models with other architectures are welcome.
You do not need to use OpenJev or LoRA.

Declare **train-only** or **all-data** training independently of Markdown/package
input scope. Include the dataset revision, protocol, training partitions, and
earlier checkpoint history. See [model scopes](MODELS.md).

For results, include the evaluation dataset and revision, the score direction,
threshold selection, and any known training overlap. Keep raw scores available
so that methods can be compared at the same recall. The existing paper
[reproduction format](REPRODUCTION.md) is available as a reference.

## Small code changes

```sh
python -m unittest discover -s tests -v
python scripts/community_data.py check
python scripts/reproduce.py --output outputs/table1-check
```

These checks run offline on CPU with Python 3.12+. Keep model inference and
dependency findings independent. When adding a file reader, parse its contents
as data and report unsupported inputs. Never install or execute a scanned package.

Code and code examples use Apache-2.0. Project-authored documentation and
contributed annotations use CC BY 4.0. Only contribute material you are entitled
to share under those terms. Third-party source files keep their own licenses.
Credit and evidence links stay with each accepted contribution.

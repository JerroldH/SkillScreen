# Examples

Use Python 3.12 or newer. All commands below run from the repository root.
These project-authored example files are code examples under Apache-2.0.

## Read a complete package

    python scan.py examples/local-inventory --inspect --output outputs/inventory.json

Expected summary:

    Read 4 text files; skipped 0 entries.
    Model: not run.
    Dependencies: not run.
    Inspection only. No security checks were run.

The package contains SKILL.md, a Python helper, a reference document, and a
requirements file. The scanner reads them as text and never runs the helper.
The report includes relative paths, byte hashes, and any skipped entries. It does
not include raw file contents or absolute local paths. Choose a new output path
for each run; existing reports are not overwritten.

The reader skips links, common environment/vendor directories, private `.env`
and key files, binary or non-UTF-8 content, and files above 1 MiB. Each skip is
recorded. It stops at 1,000 entries or a 16 MiB read budget. Input coverage is
always part of the result, including any later token truncation.

## Check dependencies on CPU

    python scan.py examples/dependency-case --checks dependencies --output outputs/dependencies.json

This sends the package name and pinned version to the public OSV API. No skill
text is uploaded. The fixture deliberately pins requests 2.31.0; do not install
its requirements. Advisory counts can change as OSV updates. Check the report's
advisory IDs, evidence locations, and query timestamp.

The initial checker supports unconditional exact PyPI pins in requirements*.txt.
It reports version ranges, conditional requirements, and recognized unsupported
manifests as unresolved. It does not resolve transitive dependencies, container
images, or installation instructions inside Markdown. No matches is not a
certificate of safety. Dependency evidence remains separate from maliciousness.

## Use an existing model

    python -m pip install -r requirements/inference.txt
    python scan.py examples/local-inventory --checks model --output outputs/model.json

This uses the released Package-Experimental model, its original 32K input
formatter, and its native 0.5 threshold. It requires a CUDA GPU with BF16 support
and downloads the pinned HF checkpoint on first use. For SKILL.md-only scoring:

    python scan.py examples/local-inventory --checks model --mode markdown

That uses the released SBR model and its original 8K input recipe. Model outputs
are native scores, not newly calibrated probabilities or a promise of 99% recall.
The report identifies the model revision and records token truncation. Package
mode includes per-file token coverage. Use --checks all to run both components;
an unavailable component is reported separately and the command exits nonzero.

## Prepare a contribution

    python scripts/contribute.py examples/local-inventory --label benign --evidence "Project-authored example that inventories file extensions without reading contents." --license Apache-2.0 --output outputs/contribution.json
    python scripts/contribute.py --check outputs/contribution.json

This example produces a structurally valid candidate draft with missing upstream
provenance. For a real submission, add --source-url and --revision with the
upstream HTTPS URL and immutable Git commit. Unknown licenses stay unknown.
Validation checks structure and inventory hashes, not label correctness or rights.
See [CONTRIBUTING.md](../CONTRIBUTING.md) to submit or review a case.

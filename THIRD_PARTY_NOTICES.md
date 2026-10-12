# Third-party materials

The package license applies only to project-authored material. Upstream rights and attribution remain in force.

## MaliciousSkillBench

Yue Wang, Yi Liu, Gelei Deng, Ying Zhang, Yuekang Li, Zhenyu Chen, and Leo Zhang. *MaliciousSkillBench: A Comprehensive Benchmark for Malicious Agent Skill Detection* (2026).

- Dataset and fixed revision: https://huggingface.co/datasets/ProtectSkills/MaliciousSkillBench/tree/d4b42ce5766a6e0359c987cf59c1007cb3795a90
- Repository and license overview: https://github.com/protectskills/MaliciousSkillBench#license
- Metadata, derived annotations and split manifests: CC BY 4.0, unless otherwise noted. https://creativecommons.org/licenses/by/4.0/
- Upstream benchmark code: Apache License 2.0. https://www.apache.org/licenses/LICENSE-2.0

This package copies the two experiment split manifests and redistributes reduced source/family metadata with only the fields needed for restoration and checking. Per-skill predictions are produced by the project. The Word-SVM implementation follows the benchmark's published baseline recipe; its upstream reference is retained in the source.

Raw skill text and source packages are not distributed here. They retain their respective upstream terms and are not relicensed by MaliciousSkillBench or this package. The optional restoration script downloads the fixed official dataset and does not execute its contents.

## Models and scanner

Model weights and third-party scanner source are not included. Optional entry points download or load the following dependencies under their upstream terms:

- OpenJev: https://huggingface.co/AlexWortega/openjev
- Qwen3.5 upstream base family: https://huggingface.co/Qwen
- ModernBERT-large: https://huggingface.co/answerdotai/ModernBERT-large
- SkillSpector: the exact source/version is recorded in `configs/baselines/skillspector.json`.

Use the model cards and pinned revisions in the configuration files to inspect the applicable model and upstream-data terms. This package grants no additional rights to those materials.

## API services and libraries

TypeSafe, OpenAI, Anthropic and the Grok API reseller are external services. Their names identify evaluated systems and imply no endorsement. Fresh API use is subject to the account holder's provider terms and charges. No credentials or full service responses are included.

Python packages installed through the requirements files retain their own licenses. Their source and binary distributions are not bundled here.

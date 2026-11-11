# Third-party materials

Project-authored code, configuration, and code examples are licensed under Apache-2.0. Project-authored documentation, predictions, reference results, and citation metadata are licensed under CC BY 4.0. See README.md for scope and LICENSE / LICENSE-CC-BY-4.0 for the terms. Upstream rights and attribution remain in force.

## MaliciousSkillBench

Yue Wang, Yi Liu, Gelei Deng, Ying Zhang, Yuekang Li, Zhenyu Chen, and Leo Zhang. *MaliciousSkillBench: A Comprehensive Benchmark for Malicious Agent Skill Detection* (2026).

- Dataset and fixed revision: https://huggingface.co/datasets/ProtectSkills/MaliciousSkillBench/tree/d4b42ce5766a6e0359c987cf59c1007cb3795a90
- Repository and license overview: https://github.com/protectskills/MaliciousSkillBench#license
- Metadata, derived annotations and split manifests: CC BY 4.0, unless otherwise noted. https://creativecommons.org/licenses/by/4.0/
- Upstream benchmark code: Apache License 2.0. https://www.apache.org/licenses/LICENSE-2.0

This package copies the two experiment split manifests and redistributes reduced source/family metadata with only the fields needed for restoration and checking. Per-skill predictions are produced by the project. The Word-SVM implementation follows the benchmark's published baseline recipe; its upstream reference is retained in the source.

The community index in `data/community/` derives from this fixed public dataset.
It preserves upstream reference labels, groups normalized duplicate content,
records provenance and package inventories, and supplies new community random/SD
splits. It embeds no source text. The original paper manifests remain separate
and unchanged. The dataset card records exclusions and attribution.

Raw skill text and source packages are not stored in this Git repository.
The [HF dataset](https://huggingface.co/datasets/SkillScreen/SkillScreen-Data)
contains Markdown and eligible companion text with their provenance and source
terms. They are not relicensed by MaliciousSkillBench or this project.
The optional restoration script downloads the fixed source data and does not
execute its contents.

## MalSkillBench benign additions

Wenbo Guo, Wei Zeng, Chengwei Liu, Xiaojun Jia, Yijia Xu, Lei Tang, Yong Fang,
and Yang Liu. *MalSkillBench: A Runtime-Verified Benchmark of Malicious Agent Skills* (2026).

- Repository: https://github.com/lxyeternal/MalSkillBench
- Fixed revision: `3548e2837adf58b96497f2c688f96beff26485da`
- Paper: https://arxiv.org/abs/2606.07131
- Source notice: **For academic research use only.**

The community index adds 3,781 distinct benign source pointers, file hashes,
and inherited labels. The generated HF dataset includes recovered text. These
records do not grant additional redistribution or commercial-use rights to the upstream
benchmark or its skill packages. Original source terms remain applicable.

Complete-package recovery also references SkillTrustBench (CC BY-NC-SA 4.0),
SkillFortifyBench (MIT), and the source-specific official benchmark archives.
Their fixed revisions and notices are listed in `data/community/README.md`.
File inventories and restoration code do not relicense the recovered contents.

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

## Community tools

The package formatter in `scripts/package_inputs.py` is a reduced copy of our
[released Package-SBR input code](https://huggingface.co/SkillScreen/SkillScreen-2B-Package-SBR/blob/025dae85d31deb79fe0050425f6b6d60f636813d/package_inputs.py),
released under Apache-2.0. The example packages in `examples/` are project-authored
code examples under Apache-2.0, not redistributed benchmark samples.

The optional dependency checker queries the public [OSV API](https://google.github.io/osv.dev/post-v1-query/).
Returned advisory identifiers and links preserve their source attribution.
Third-party advisory content retains its source terms and is not relicensed by
this repository. No vulnerability database is bundled here.

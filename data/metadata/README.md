# Benchmark provenance

These metadata and the split manifests are derived from MaliciousSkillBench by
Yue Wang, Yi Liu, Gelei Deng, Ying Zhang, Yuekang Li, Zhenyu Chen, and Leo Zhang.

- Dataset: https://huggingface.co/datasets/ProtectSkills/MaliciousSkillBench
- Frozen revision: `d4b42ce5766a6e0359c987cf59c1007cb3795a90`
- Paper: https://arxiv.org/abs/2608.19901
- Metadata and official split membership: CC BY 4.0,
  https://creativecommons.org/licenses/by/4.0/
- Source terms: https://github.com/protectskills/MaliciousSkillBench#license

`benchmark_metadata.csv` retains only the benchmark ID, source identifiers, public
source name, and structural-family ID. The split CSVs add the experiment's inclusion
status, exclusion reason, and SHA256 of the exact public `skill_text` when available.
They account for every benchmark identity, including excluded identities. These
columns record the filtering used by this study; they do not alter official split
membership or labels.

No skill text is bundled. Third-party skill artifacts retain their upstream terms.
`scripts/prepare_data.py` downloads only the pinned primary table and selected
official split tables, reads exact text as inert data, and verifies all identities,
labels, source/family metadata, exclusions, per-record text hashes, and output file
hashes. It does not substitute sanitized `public_skill_text` for withheld originals.

MSD includes 6,814 training, 974 validation, and 1,947 test records after five
missing-text exclusions. SD includes 7,510 training, 833 validation, and 1,384 test
records after five missing-text exclusions and eight official exclusions.

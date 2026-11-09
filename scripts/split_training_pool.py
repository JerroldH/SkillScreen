"""Create two local split manifests for a deduplicated labeled Markdown pool."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

try:
    from .community_data import read_jsonl
    from .import_malskillbench import normalized_hash
except ImportError:
    from community_data import read_jsonl
    from import_malskillbench import normalized_hash

HOLDOUT = frozenset(('msb:SRC009', 'msb:SRC011', 'msb:SRC012'))
PARTITIONS = ('train', 'validation', 'test', 'excluded')


def describe(rows):
    """Use all source aliases for isolation; one primary source for counting."""
    records, seen_groups, known_versions = {}, {}, {}
    for row in rows:
        key = row['id']
        if key in records or row['label'] not in (0, 1):
            raise ValueError('Duplicate ID or invalid label: ' + key)
        if normalized_hash(row['skill_text']) != row['normalized_sha256']:
            raise ValueError('Normalized text hash mismatch: ' + key)
        if hashlib.sha256(row['skill_text'].encode()).hexdigest() != row['text_sha256']:
            raise ValueError('Exact text hash mismatch: ' + key)
        if row['label_origin'] not in ('upstream', 'weak', 'reviewed'):
            raise ValueError('Unknown label origin: ' + key)
        members = row['members']
        benchmark = [m for m in members if m.get('collection') == 'benchmark']
        domain = [m for m in members if m.get('collection') == 'domain']
        if benchmark:
            primary = benchmark[0]['source_name']
        elif domain:
            primary = 'Domain / ' + next((m for m in domain if m.get('role') == 'verified'), domain[0])['source_name']
        else:
            primary = members[0]['source_name']
        sources = set()
        groups = {row['normalized_sha256']}
        versions = set()
        for member in members:
            sources.update(member.get('source_ids', []))
            if member.get('group_id'):
                groups.add(member['group_id'])
            if member.get('collection') == 'domain':
                sources.add('domain:' + member['source_name'])
                versions.add(('archive_path', member['zip_member']))
            elif member.get('source_path'):
                versions.add((member['source_url'], member['source_path']))
        if not sources:
            raise ValueError('Missing source identity: ' + key)
        # ponytail: the input is already deduplicated. Stop if new version links
        # require multi-row groups instead of silently splitting those versions.
        for tokens, owners in ((groups, seen_groups), (versions, known_versions)):
            for token in tokens:
                if token in owners and owners[token] != key:
                    raise ValueError('Related versions need grouping before splitting: ' + key)
                owners[token] = key
        evidence = 'weak' if row['label_origin'] == 'weak' else 'verified_reference' if any(m.get('role') == 'verified' for m in domain) else row['label_origin']
        records[key] = {'id': key, 'group_id': row['normalized_sha256'],
                        'label': 'malicious' if row['label'] else 'benign',
                        'label_origin': row['label_origin'], 'label_evidence': evidence,
                        'primary_source': primary, 'source_ids': sorted(sources)}
    if not records:
        raise ValueError('Empty input pool')
    return records


def stratified(records, seed, validation_fraction, test_fraction):
    """Shuffle content groups within source/label strata; weak labels train only."""
    strata, assignments = defaultdict(list), {}
    for key, row in records.items():
        if row['label_origin'] == 'weak':
            assignments[key] = ('train', 'weak_label_train_only')
        else:
            strata[(row['primary_source'], row['label'])].append(key)
    for stratum, keys in sorted(strata.items()):
        keys.sort()
        rng = random.Random(f'{seed}:{stratum[0]}:{stratum[1]}')
        rng.shuffle(keys)
        n_val = int(len(keys) * validation_fraction + 0.5)
        n_test = int(len(keys) * test_fraction + 0.5)
        # Small strata can have no validation/test members; never duplicate rows.
        for position, key in enumerate(keys):
            split = 'validation' if position < n_val else 'test' if position < n_val+n_test else 'train'
            assignments[key] = (split, '')
    return assignments


def make_splits(records, seed=42):
    random_split = stratified(records, seed, 0.1, 0.2)
    sd, development = {}, {}
    for key, row in records.items():
        sources = set(row['source_ids'])
        if sources & HOLDOUT:
            if sources - HOLDOUT:
                sd[key] = ('excluded', 'source_aliases_cross_holdout_boundary')
            elif row['label_origin'] == 'weak':
                sd[key] = ('excluded', 'weak_label_in_heldout_source')
            else:
                sd[key] = ('test', 'heldout_source')
        else:
            development[key] = row
    sd.update(stratified(development, seed, 1/8, 0))
    protocols = {'grouped_random': random_split, 'source_disjoint': sd}
    validate(records, protocols)
    return protocols


def validate(records, protocols):
    for name, assignment in protocols.items():
        if set(assignment) != set(records):
            raise ValueError(name + ': assignments do not cover the pool exactly')
        source_splits, group_splits, labels = defaultdict(set), defaultdict(set), defaultdict(set)
        for key, (split, reason) in assignment.items():
            if split not in PARTITIONS:
                raise ValueError(name + ': invalid split')
            if split == 'excluded':
                if not reason:
                    raise ValueError(name + ': exclusion needs a reason')
                continue
            row = records[key]
            if row['label_origin'] == 'weak' and split != 'train':
                raise ValueError(name + ': weak labels cannot enter evaluation')
            group_splits[row['group_id']].add(split)
            labels[split].add(row['label'])
            for source in row['source_ids']:
                source_splits[source].add(split)
        if any(len(parts) > 1 for parts in group_splits.values()):
            raise ValueError(name + ': a content group crosses partitions')
        if any(labels[s] != {'benign', 'malicious'} for s in ('train', 'validation', 'test')):
            raise ValueError(name + ': each split must have both classes')
        if name == 'source_disjoint':
            if any('test' in parts and len(parts) > 1 for parts in source_splits.values()):
                raise ValueError('source_disjoint: a source alias crosses the test boundary')
            for key, (split, _) in assignment.items():
                sources = set(records[key]['source_ids'])
                if split == 'test' and not sources <= HOLDOUT or split in ('train', 'validation') and sources & HOLDOUT:
                    raise ValueError('source_disjoint: heldout source assignment changed')


def summaries(records, protocols):
    summary, distribution = {}, []
    for protocol, assignment in protocols.items():
        summary[protocol] = {}
        for split in PARTITIONS:
            selected = [records[k] for k, (part, _) in assignment.items() if part == split]
            count = Counter(r['label'] for r in selected)
            evidence = Counter(r['label_evidence'] for r in selected)
            summary[protocol][split] = {'total': len(selected), 'malicious': count['malicious'],
                                       'benign': count['benign'], 'label_evidence': dict(evidence)}
            for source in sorted({r['primary_source'] for r in selected}):
                source_rows = [r for r in selected if r['primary_source'] == source]
                counts = Counter(r['label'] for r in source_rows)
                evidence = Counter(r['label_evidence'] for r in source_rows)
                distribution.append({'protocol': protocol, 'split': split, 'source': source,
                                     'total': len(source_rows), 'malicious': counts['malicious'], 'benign': counts['benign'],
                                     'weak': evidence['weak'], 'verified_reference': evidence['verified_reference']})
    return summary, distribution


def write_csv(path, rows, fields=None):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New local output directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    records = describe(read_jsonl(args.input))
    protocols = make_splits(records, args.seed)
    summary, distribution = summaries(records, protocols)
    report = {'seed': args.seed, 'input_records': len(records),
              'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'grouping': 'Normalized-text groups from the merged pool; known group/version links checked. No semantic or structural-family clustering.',
              'random_rule': 'Source/label-stratified 70/10/20 on non-weak records, rounded per stratum. All weak records added to train.',
              'sd_rule': 'Test uses the three original heldout source collections. Cross-boundary aliases excluded. Remaining non-weak records split train/validation 7:1; weak records train only.',
              'sd_heldout_sources': sorted(HOLDOUT), 'protocols': summary,
              'checks': {'complete_id_accounting': True, 'no_content_group_overlap': True,
                         'no_weak_evaluation_labels': True, 'sd_test_source_aliases_disjoint': True,
                         'both_labels_in_each_split': True}}
    args.output.mkdir(parents=True)
    for name, assignment in protocols.items():
        rows = [{**records[key], 'source_ids': json.dumps(records[key]['source_ids']),
                 'split': assignment[key][0], 'reason': assignment[key][1]} for key in sorted(records)]
        write_csv(args.output/(name+'.csv'), rows)
    write_csv(args.output/'source_distribution.csv', distribution)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    lines = ['# 扩充数据池的新版划分，本地预览', '',
             f'输入 {len(records):,} 条，seed {args.seed}。未训练，未上传，原 SBR/SD 文件未修改。', '',
             '## 集合数量与标签', '', '| 协议 | 集合 | 总数 | Malicious | Benign | 弱标签 | Verified-reference |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for protocol, splits in summary.items():
        for split, counts in splits.items():
            e = counts['label_evidence']
            lines.append(f"| {protocol} | {split} | {counts['total']:,} | {counts['malicious']:,} | {counts['benign']:,} | {e.get('weak',0):,} | {e.get('verified_reference',0):,} |")
    lines += ['', '## 划分规则', '',
              '- Grouped random 以去重后的内容组为单位，按主要来源和标签分层。非弱标签约 70/10/20，弱标签只进入 train。',
              '- Source-disjoint 留出 SkillHarm、ATR Skill Security Benchmark 和 SkillFortifyBench，沿用原 SD 的来源选择。其余非弱标签按 7:1 分为 train/validation。',
              '- SD 中同时出现在留出来源与训练侧来源的内容组列为 excluded，不把它们硬塞进任一集合。',
              '- 来源指 benchmark collection 或域内采集平台，不宣称是 repository-disjoint。SD 的 validation 与 train 可以共享来源。',
              '- 当前分组覆盖规范化相同文本和已提供的版本关系，没有额外做结构家族或语义近似去重。',
              '- 标签继承上游或参考名单。Verified-reference 不是这轮新增的人工复核，weak 也不进入 validation/test。',
              '- 新划分供下一轮从上游基座重新训练使用。GCP 全训版已经见过池内数据，不能作为这两套划分的独立 test 模型。', '']
    for protocol in protocols:
        lines += [f'## {protocol} 来源分布', '', '每个单元格为 malicious / benign。使用主要来源计数，不重复累计多来源样本。', '',
                  '| 来源 | Train | Validation | Test | Excluded |', '| --- | ---: | ---: | ---: | ---: |']
        for source in sorted({r['source'] for r in distribution if r['protocol']==protocol}):
            cells = []
            for split in PARTITIONS:
                r = next((r for r in distribution if r['protocol']==protocol and r['split']==split and r['source']==source), None)
                cells.append(f"{r['malicious']:,} / {r['benign']:,}" if r else '0 / 0')
            lines.append('| '+source+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## 文件', '', '- [Grouped random IDs](grouped_random.csv)', '- [Source-disjoint IDs](source_disjoint.csv)',
              '- [按来源和标签的完整统计](source_distribution.csv)', '- [规则及核验摘要](summary.json)', '',
              '两个 manifest 只保存 ID、来源、标签与划分，通过 ID 连接同一份母池，不复制原始 skill 文本。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

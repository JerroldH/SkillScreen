"""Attach recovered package views by existing member ID; never change pool IDs or splits."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

try:
    from .community_data import read_jsonl
    from .import_malskillbench import normalized_hash
    from .package_inputs import ordered_files, safe_relative_path
    from .split_training_pool import describe
except ImportError:
    from community_data import read_jsonl
    from import_malskillbench import normalized_hash
    from package_inputs import ordered_files, safe_relative_path
    from split_training_pool import describe


def view_fingerprint(markdown, files):
    values = [('SKILL.md', hashlib.sha256(markdown.encode()).hexdigest())]
    values += sorted((f['path'], hashlib.sha256(f['text'].encode()).hexdigest()) for f in files)
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def attach(pool, package_sources):
    by_id = {r['id']: r for r in pool}
    if len(by_id)!=len(pool):
        raise ValueError('Duplicate pool ID')
    by_member = {m['id']: r['id'] for r in pool for m in r['members']}
    candidates, outside = defaultdict(list), []
    for priority, source in enumerate(package_sources):
        for candidate in source:
            member = candidate['source_member_id']
            if member not in by_member:
                outside.append(member)
                continue
            key = by_member[member]
            row = by_id[key]
            if candidate['label']!=row['label'] or normalized_hash(candidate['skill_text'])!=row['normalized_sha256']:
                raise ValueError('Package/label identity mismatch: '+member)
            ordered_files(candidate)
            inventory = candidate['inventory']
            if len({f['path'] for f in inventory}) != len(inventory):
                raise ValueError('Duplicate inventory path: '+member)
            for item in inventory:
                safe_relative_path(item['path'])
            expected = {f['path'] for f in inventory if f['input_eligible'] and f['path']!='SKILL.md'}
            if expected != {f['path'] for f in candidate['files']}:
                raise ValueError('Companion input does not match inventory: '+member)
            if not any(f['path']=='SKILL.md' for f in inventory):
                raise ValueError('Missing source Markdown: '+member)
            hashes = {f['path']:f['sha256'] for f in inventory}
            for companion in candidate['files']:
                raw = companion['text'].encode('utf-8')
                if hashes[companion['path']] not in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(b'\xef\xbb\xbf'+raw).hexdigest()}:
                    raise ValueError('Companion text/hash mismatch: '+member)
            if type(candidate['package_complete']) is not bool:
                raise ValueError('Package completeness must be explicit: '+member)
            candidates[key].append((priority, candidate, view_fingerprint(row['skill_text'], candidate['files'])))
    coverage, views = [], []
    for key, row in sorted(by_id.items()):
        options = candidates[key]
        complete = [item for item in options if item[1]['package_complete']]
        state = 'package_partial' if options else 'markdown_only'
        selected = None
        if complete:
            first_priority = min(p for p, _, _ in complete)
            preferred = [item for item in complete if item[0]==first_priority]
            if len({fingerprint for _, _, fingerprint in preferred}) == 1:
                selected = min(preferred, key=lambda item: item[1]['source_member_id'])
                state = 'package_complete'
            else:
                state = 'package_ambiguous'
        item = {'id': key, 'label': row['label'], 'coverage': state,
                'candidate_member_ids': sorted(c['source_member_id'] for _,c,_ in options)}
        if selected:
            _, source, fingerprint = selected
            item.update({'source_member_id': source['source_member_id'], 'source_url': source['source_url'],
                         'revision': source['revision'], 'source_package_path': source['source_package_path'],
                         'upstream_license': source['upstream_license'], 'input_fingerprint': fingerprint,
                         'companion_text_files': len(source['files']), 'inventory': source['inventory']})
            views.append({'id':key, 'label':row['label'], 'skill_text':row['skill_text'], 'files':source['files']})
        coverage.append(item)
    return coverage, views, outside


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--packages', type=Path, action='append', required=True,
                        help='Recovered source JSONL in fixed preference order; do not order by model results')
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New local output directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    pool = list(read_jsonl(args.pool))
    described = describe(pool)
    coverage, views, outside = attach(pool, [read_jsonl(p) for p in args.packages])
    by_id = {r['id']:r for r in coverage}
    states = Counter(r['coverage'] for r in coverage)
    labels = Counter(r['label'] for r in views)
    with_companions = sum(bool(r['files']) for r in views)
    source_counts = []
    for source in sorted({r['primary_source'] for r in described.values()}):
        members = [r for r in coverage if described[r['id']]['primary_source']==source]
        source_counts.append({'source':source,'total':len(members),
                              'complete_malicious':sum(r['coverage']=='package_complete' and r['label']==1 for r in members),
                              'complete_benign':sum(r['coverage']=='package_complete' and r['label']==0 for r in members),
                              'with_companions':sum(r.get('companion_text_files',0)>0 for r in members)})
    split_counts = []
    split_hashes = {}
    for name in ('grouped_random','source_disjoint'):
        path = args.splits/(name+'.csv')
        split_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        assignment = list(csv.DictReader(path.open(encoding='utf-8',newline='')))
        if len(assignment)!=len(by_id) or {r['id'] for r in assignment}!=set(by_id):
            raise ValueError('Split/pool ID mismatch')
        for split in ('train','validation','test','excluded'):
            ids = [r['id'] for r in assignment if r['split']==split]
            selected = [by_id[k] for k in ids if by_id[k]['coverage']=='package_complete']
            split_counts.append({'protocol':name,'split':split,'all_records':len(ids),'complete_packages':len(selected),
                                 'malicious':sum(r['label']==1 for r in selected), 'benign':sum(r['label']==0 for r in selected),
                                 'with_companions':sum(r['companion_text_files']>0 for r in selected)})
    report = {'pool_records':len(pool),'coverage':dict(states),'complete_labels':dict(labels),
              'with_companion_text':with_companions, 'without_companion_text':len(views)-with_companions,
              'source_priority':[p.name for p in args.packages], 'outside_pool_member_ids':outside,
              'ambiguous_ids':[r['id'] for r in coverage if r['coverage']=='package_ambiguous'],
              'source_counts':source_counts,'split_counts':split_counts,
              'pool_sha256':hashlib.sha256(args.pool.read_bytes()).hexdigest(), 'split_sha256':split_hashes,
              'policy':'One verified package variant per existing ID; never merge files across packages. Ambiguous same-priority variants remain unselected.',
              'input_scope':'Canonical unchanged Markdown plus eligible companion paths/text. Registry and benchmark annotations excluded.',
              'completeness':'All files in the pinned published package recovered. No runtime dependencies fetched. Input exclusions remain in the inventory.'}
    args.output.mkdir(parents=True)
    for filename, rows in [('coverage.jsonl',coverage),('package_records.jsonl',views)]:
        with (args.output/filename).open('x',encoding='utf-8',newline='\n') as handle:
            for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines = ['# 同一母池的 Package 视图', '',
             f"母池 {len(pool):,} 条。已接入 {len(views):,} 个完整发布包，其中 malicious {labels[1]:,}、benign {labels[0]:,}。",
             f"其中 {with_companions:,} 个有额外可输入文本，{len(views)-with_companions:,} 个没有额外可输入文本。", '',
             'ID、标签、原 Markdown 和两套划分均保持不变，没有训练或上传。', '',
             '## 覆盖状态', '', '| 状态 | 数量 |','| --- | ---: |']
    lines += [f'| {state} | {count:,} |' for state,count in sorted(states.items())]
    lines += ['', '## 原划分下的完整包子集', '',
              '| 协议 | 集合 | 原记录数 | 完整包 | Malicious | Benign | 有配套文本 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['protocol']} | {r['split']} | {r['all_records']:,} | {r['complete_packages']:,} | {r['malicious']:,} | {r['benign']:,} | {r['with_companions']:,} |" for r in split_counts]
    lines += ['', '## 来源覆盖', '', '| 来源 | 母池记录 | 完整包 Malicious | 完整包 Benign | 有配套文本 |',
              '| --- | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['source']} | {r['total']:,} | {r['complete_malicious']:,} | {r['complete_benign']:,} | {r['with_companions']:,} |" for r in source_counts]
    lines += ['', '## 使用和范围', '',
              '- [package_records.jsonl](package_records.jsonl) 直接提供同一 ID 的 Markdown 和配套文件文本。没有完整包的 ID 不静默降级为 Markdown。',
              '- [coverage.jsonl](coverage.jsonl) 覆盖全部母池 ID，记录包来源、文件哈希和输入排除原因。',
              '- 优先沿用已经联合验证的原 SRC010 包。新增来源按已声明的顺序处理。同一优先级出现不同包内容时单列 ambiguous，不拼接文件。',
              '- 完整指固定发布目录内的全部文件，不包含运行时下载的依赖。模型输入会排除元数据、非文本及超限文件。',
              '- 原生只有 Markdown 的包可以是完整包；只恢复到 Markdown、无法证明完整目录的样本仍是 markdown_only。',
              '- 新恢复文件尚未做结构、代码克隆或 SkillsBench 外部重叠复核；本次不宣称代码隔离或新的外部评测独立性。',
              '- ATR 的公开数据以 Markdown 表格为主，缺少可逐例核对的原包。SD 完整包子集因此不能代表全部 SD 测试分布。',
              '- 完整包已准备好不表示已训练 package 模型。当前发布模型不变。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:report[k] for k in ('pool_records','coverage','complete_labels','with_companion_text','without_companion_text','split_counts')},indent=2))


if __name__ == '__main__':
    main()

"""Merge local labeled Markdown JSONL files; keep provenance and exclude conflicts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

try:
    from .community_data import read_jsonl
    from .import_malskillbench import normalized_hash
except ImportError:
    from community_data import read_jsonl
    from import_malskillbench import normalized_hash


def merge_rows(rows):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        row = dict(row)
        if row['id'] in seen:
            raise ValueError('Repeated input ID: ' + row['id'])
        seen.add(row['id'])
        text = row['skill_text']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Missing skill_text for ' + row['id'])
        if row['label'] not in (0, 1, 'benign', 'malicious'):
            raise ValueError('Invalid binary label for ' + row['id'])
        row['label'] = int(row['label'] in (1, 'malicious'))
        group = normalized_hash(text)
        if row.get('normalized_sha256', group) != group:
            raise ValueError('Normalized text hash mismatch for ' + row['id'])
        groups[group].append(row)
    retained, conflicts, merged = [], [], []
    for group, members in sorted(groups.items()):
        provenance = [{k: v for k, v in r.items() if k not in ('skill_text', 'files')} for r in members]
        if len({r['label'] for r in members}) > 1:
            conflicts.append({'normalized_sha256': group, 'members': provenance})
            continue
        # Keep the first source's exact text; normalization is only for grouping.
        representative = members[0]
        weak_only = all(r.get('label_origin') == 'weak' for r in members)
        retained.append({'id': group, 'label': representative['label'],
                         'skill_text': representative['skill_text'],
                         'text_sha256': hashlib.sha256(representative['skill_text'].encode('utf-8')).hexdigest(),
                         'normalized_sha256': group,
                         'label_origin': 'weak' if weak_only else 'upstream',
                         'members': provenance})
        if len(members) > 1:
            merged.append({'normalized_sha256': group, 'members': [r['id'] for r in members]})
    report = {'input_records': len(seen), 'unique_text_groups': len(groups),
              'retained': len(retained), 'labels': dict(Counter('malicious' if r['label'] else 'benign' for r in retained)),
              'weak_only': sum(r['label_origin'] == 'weak' for r in retained),
              'conflicting_groups': len(conflicts), 'merged_groups': len(merged),
              'normalization': 'Remove leading BOM, normalize line endings, strip outer whitespace.',
              'scope': 'Local all-data candidate pool. Not a held-out evaluation split.',
              'label_note': 'Labels and reference evidence are inherited; merging is not independent review.'}
    return retained, conflicts, merged, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New local output directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    rows = (row for path in args.input for row in read_jsonl(path))
    retained, conflicts, merged, report = merge_rows(rows)
    args.output.mkdir(parents=True)
    for name, records in [('records.jsonl', retained), ('conflicts.jsonl', conflicts), ('duplicates.jsonl', merged)]:
        with (args.output / name).open('x', encoding='utf-8', newline='\n') as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    report['inputs'] = [{'name': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.input]
    report['records_sha256'] = hashlib.sha256((args.output / 'records.jsonl').read_bytes()).hexdigest()
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

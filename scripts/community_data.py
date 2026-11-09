"""Check, select, or export the community data index. No network or training."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlsplit

try:
    from .package_inputs import safe_relative_path
except ImportError:
    from package_inputs import safe_relative_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data/community'
PROTOCOLS = ('grouped_random', 'source_disjoint')
PARTITIONS = ('train', 'validation', 'test', 'excluded')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{number}: expected an object')
            yield row


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def check_record(row):
    for field in ('id', 'source_url', 'revision', 'source_id', 'group_id', 'evidence', 'upstream_license'):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f'Missing {field} in {row.get("id", "record")}')
    url = urlsplit(row['source_url'])
    if url.scheme != 'https' or not url.hostname or url.username or url.password:
        raise ValueError(f'{row["id"]}: invalid public source URL')
    if not re.fullmatch('[0-9a-f]{40}', row['revision']):
        raise ValueError(f'{row["id"]}: revision must be a fixed Git/dataset commit')
    if row.get('label') not in ('benign', 'malicious', 'unknown'):
        raise ValueError(f'{row["id"]}: invalid label')
    if row.get('label_origin') not in ('upstream', 'reviewed', 'weak'):
        raise ValueError(f'{row["id"]}: state the label origin')
    sources = row.get('source_ids')
    if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or not s for s in sources) or row['source_id'] not in sources:
        raise ValueError(f'{row["id"]}: invalid source groups')
    if row.get('coverage') not in ('markdown_only', 'package_partial', 'package_ambiguous', 'package_complete'):
        raise ValueError(f'{row["id"]}: state package coverage')
    files = row.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError(f'{row["id"]}: a file inventory is required')
    paths = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError('Each file must be an object')
        path = safe_relative_path(item.get('path'))
        if path in paths or not re.fullmatch('[0-9a-f]{64}', str(item.get('sha256'))):
            raise ValueError(f'{row["id"]}: duplicate path or invalid hash')
        paths.add(path)
        text = item.get('text')
        if text is not None:
            if row['upstream_license'].strip().lower() == 'unknown':
                raise ValueError(f'{row["id"]}: source terms required before embedding text')
            if not isinstance(text, str) or sha256(text) != item['sha256']:
                raise ValueError(f'{row["id"]}: file text hash mismatch')
            if path == 'SKILL.md' and not text.strip():
                raise ValueError(f'{row["id"]}: SKILL.md must be nonempty')
    if 'SKILL.md' not in paths:
        raise ValueError(f'{row["id"]}: missing SKILL.md inventory')
    if row['coverage'] == 'markdown_only' and len(paths) != 1:
        raise ValueError(f'{row["id"]}: coverage conflicts with file inventory')


def read_index(directory):
    rows = list(read_jsonl(Path(directory) / 'records.jsonl'))
    if not rows:
        raise ValueError('The data index is empty')
    records = {}
    for row in rows:
        check_record(row)
        if row['id'] in records:
            raise ValueError(f'Duplicate record ID: {row["id"]}')
        records[row['id']] = row
    return records


def read_membership(directory, protocol, records):
    if protocol not in PROTOCOLS:
        raise ValueError('Choose a known protocol')
    with (Path(directory) / 'splits' / (protocol + '.csv')).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    membership = {}
    for row in rows:
        key, split = row['id'], row['split']
        if key not in records or key in membership or split not in PARTITIONS:
            raise ValueError(f'{protocol}: invalid or duplicate split assignment: {key}')
        membership[key] = split
    if set(membership) != set(records):
        raise ValueError(f'{protocol}: every record needs an assignment, including excluded')
    groups, hashes, sources = defaultdict(set), defaultdict(set), defaultdict(set)
    counts = {split: Counter() for split in PARTITIONS}
    for key, split in membership.items():
        row = records[key]
        counts[split][row['label']] += 1
        if split == 'excluded':
            continue
        if row['label'] == 'unknown' or row['label_origin'] == 'weak' and split != 'train':
            raise ValueError(f'{protocol}: unresolved or weak evaluation label: {key}')
        groups[row['group_id']].add(split)
        markdown = next(f for f in row['files'] if f['path'] == 'SKILL.md')
        hashes[markdown['sha256']].add(split)
        for source in row['source_ids']:
            sources[source].add(split)
    if any(len(parts) > 1 for parts in (*groups.values(), *hashes.values())):
        raise ValueError(f'{protocol}: related records or identical Markdown cross partitions')
    # The official SD protocol holds test sources out; train/validation share sources.
    if protocol == 'source_disjoint' and any('test' in parts and len(parts) > 1 for parts in sources.values()):
        raise ValueError('source_disjoint: a test source also occurs in train/validation')
    if any(not counts[split] for split in ('train', 'validation', 'test')):
        raise ValueError(f'{protocol}: train, validation, and test must all be populated')
    return membership, {split: dict(count) for split, count in counts.items()}


def check_dataset(directory):
    records = read_index(directory)
    protocols = {p: read_membership(directory, p, records)[1] for p in PROTOCOLS}
    return {'records': len(records), 'labels': dict(Counter(r['label'] for r in records.values())),
            'coverage': dict(Counter(r['coverage'] for r in records.values())),
            'embedded_files': sum(f.get('text') is not None for r in records.values() for f in r['files']),
            'protocols': protocols, 'checks': 'IDs, declared groups, exact Markdown hashes, source-disjoint test isolation, and file hashes passed'}


def select_records(directory, protocol, split):
    records = read_index(directory)
    if split == 'all':
        if protocol is not None:
            raise ValueError('Use --split all without --protocol for the full labeled pool')
        return [r for r in records.values() if r['label'] in ('benign', 'malicious')]
    if split not in PARTITIONS or protocol is None:
        raise ValueError('Choose --protocol and --split, or --split all')
    membership, _ = read_membership(directory, protocol, records)
    return [r for r in records.values() if membership[r['id']] == split]


def materialize_inputs(rows, mode, text_dir=None, text_files=()):
    """Resolve permitted embedded text or existing locally restored benchmark text."""
    if mode not in ('markdown', 'package'):
        raise ValueError('Choose markdown or package input mode')
    restored = {}
    wanted = {r['id'] for r in rows}
    paths = [Path(text_dir) / (split + '.jsonl') for split in ('train', 'validation', 'test')] if text_dir else []
    for path in [*paths, *text_files]:
        for row in read_jsonl(path):
            key = row.get('id') or 'msb:' + row['benchmark_id']
            if key not in wanted:
                continue
            payload={'SKILL.md':row['skill_text'],**{f['path']:f['text'] for f in row.get('files',[])}}
            if key in restored and restored[key] != payload:
                raise ValueError(f'Conflicting restored text: {key}')
            restored[key] = payload
    inputs = []
    for row in rows:
        if row['label'] == 'unknown':
            raise ValueError(f'{row["id"]}: cannot materialize an unresolved training label')
        if mode == 'package' and row['coverage'] != 'package_complete':
            raise ValueError(f'{row["id"]}: complete package content is unavailable; do not silently substitute Markdown')
        files = [f for f in row['files'] if f.get('input_eligible',True)] if mode == 'package' else [f for f in row['files'] if f['path'] == 'SKILL.md']
        texts = {}
        for item in files:
            text = item.get('text')
            if text is None:
                text = restored.get(row['id'],{}).get(item['path'])
            if not isinstance(text, str) or sha256(text) != item['sha256']:
                raise ValueError(f'{row["id"]}/{item["path"]}: text missing or hash mismatch; restore it locally')
            if item['path'] == 'SKILL.md' and not text.strip():
                raise ValueError(f'{row["id"]}: SKILL.md must be nonempty')
            texts[item['path']] = text
        inputs.append({'id': row['id'], 'label': int(row['label'] == 'malicious'),
                       'skill_text': texts.pop('SKILL.md'),
                       'files': [{'path': path, 'text': text} for path, text in texts.items()]})
    return inputs


def export_dataset(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(directory):
        raise ValueError('Choose a new export directory outside the data index')
    report = check_dataset(directory)
    # Copy only the release files, never nearby raw caches or private drafts.
    files = ['README.md', 'records.jsonl', *['splits/' + p + '.csv' for p in PROTOCOLS]]
    if (directory/'sources.json').exists():
        files.append('sources.json')
    for name in files:
        if not (directory / name).is_file() or (directory / name).is_symlink() or not (directory / name).resolve().is_relative_to(directory):
            raise ValueError(f'Missing or linked release file: {name}')
    output.mkdir(parents=True)
    for name in files:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / name, target)
    # The index-only export must not advertise absent full-text Parquet configs.
    (output / 'README.md').write_text(
        '---\npretty_name: SkillScreen Data Index\nlicense: other\n'
        'configs:\n  - config_name: records\n    default: true\n    data_files:\n'
        '      - split: records\n        path: records.jsonl\n---\n\n'
        '# SkillScreen Data Index\n\nLocal metadata-only preview. No skill bodies are included.\n'
        'Use scripts/export_community.py for the four full-text HF configurations.\n'
        'Imported metadata and source contents retain upstream terms.\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'select', 'export'))
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--protocol', choices=PROTOCOLS)
    parser.add_argument('--split', choices=(*PARTITIONS, 'all'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inputs', choices=('markdown', 'package'), help='Write model inputs rather than metadata')
    parser.add_argument('--coverage',choices=('package_complete',),help='Explicitly select the complete-package cohort for either input mode')
    parser.add_argument('--text-dir', type=Path, help='Existing restored train/validation/test JSONL directory')
    parser.add_argument('--text-file', type=Path, action='append', default=[], help='Additional locally restored JSONL, matched by ID and text hash')
    args = parser.parse_args(argv)
    if args.action == 'check':
        print(json.dumps(check_dataset(args.data), indent=2))
    elif args.action == 'export':
        if not args.output:
            parser.error('Choose a new --output directory')
        print(json.dumps(export_dataset(args.data, args.output), indent=2))
        print('Local HF preview written. Nothing uploaded.')
    else:
        if not args.output or not args.split:
            parser.error('select requires --split and a new --output file')
        if args.output.exists() or args.output.resolve().is_relative_to(args.data.resolve()):
            parser.error('Choose a new output outside the data index')
        if args.text_dir and args.output.resolve().is_relative_to(args.text_dir.resolve()):
            parser.error('Write outputs outside restored inputs')
        if (args.text_dir or args.text_file) and not args.inputs:
            parser.error('--text-dir and --text-file require --inputs')
        if args.inputs and args.split == 'excluded':
            parser.error('Excluded records are metadata only for this protocol')
        rows = select_records(args.data, args.protocol, args.split)
        if args.coverage:
            rows = [r for r in rows if r['coverage']==args.coverage]
        if args.inputs:
            rows = materialize_inputs(rows, args.inputs, args.text_dir, args.text_file)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(f'Wrote {len(rows)} {"model inputs" if args.inputs else "metadata records"}.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

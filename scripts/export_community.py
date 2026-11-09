"""Build or verify the local full-text HF dataset. No uploads or training."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

from datasets import ClassLabel, Features, List, Value, get_dataset_config_names, load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from . import community_data as data
except ImportError:
    import community_data as data

CONFIGS = {
    'markdown_random': ('markdown', 'grouped_random'),
    'markdown_sd': ('markdown', 'source_disjoint'),
    'package_random': ('package', 'grouped_random'),
    'package_sd': ('package', 'source_disjoint'),
}
SPLITS = ('train', 'validation', 'test')
METADATA = ('group_id', 'source_id', 'source_name', 'source_url', 'revision',
            'label_origin', 'evidence', 'coverage', 'upstream_license')
FEATURES = Features({
    'id': Value('string'), 'label': ClassLabel(names=['benign', 'malicious']),
    'skill_text': Value('string'), 'skill_sha256': Value('string'),
    'files': List({'path': Value('string'), 'text': Value('string'), 'sha256': Value('string')}),
    **{key: Value('string') for key in METADATA},
    'source_ids': List(Value('string')), 'member_ids': List(Value('string')),
    'package_source': {key: Value('string') for key in ('source_url', 'revision', 'upstream_license')},
})


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def checked_row(record, body, mode):
    """Verify canonical Markdown and the exact eligible companion inventory."""
    key = record['id']
    if body['id'] != key or body['label'] != int(record['label'] == 'malicious'):
        raise ValueError(f'{key}: input ID or label mismatch')
    if mode == 'package' and record['coverage'] != 'package_complete':
        raise ValueError(f'{key}: incomplete package')
    inventory = {f['path']: f for f in record['files']
                 if f['path'] == 'SKILL.md' or mode == 'package' and f.get('input_eligible', True)}
    companions = body['files']
    paths = [f['path'] for f in companions]
    if len(paths) != len(set(paths)) or set(paths) != set(inventory) - {'SKILL.md'}:
        raise ValueError(f'{key}: companion inventory mismatch')
    texts = {'SKILL.md': body['skill_text'], **{f['path']: f['text'] for f in companions}}
    if not isinstance(body['skill_text'], str) or not body['skill_text'].strip():
        raise ValueError(f'{key}: empty Markdown')
    for path, text in texts.items():
        if not isinstance(text, str) or data.sha256(text) != inventory[path]['sha256']:
            raise ValueError(f'{key}/{path}: text hash mismatch')
    package = record.get('recovery', {}).get('package', {}) if mode == 'package' else {}
    return {
        'id': key, 'label': body['label'], 'skill_text': body['skill_text'],
        'skill_sha256': inventory['SKILL.md']['sha256'],
        'files': [{'path': path, 'text': texts[path], 'sha256': inventory[path]['sha256']}
                  for path in sorted(paths)],
        **{field: record.get(field, '') for field in METADATA},
        'source_ids': record['source_ids'], 'member_ids': record.get('member_ids', []),
        'package_source': {field: package.get(field, '') for field in
                           ('source_url', 'revision', 'upstream_license')},
    }


def build(directory, output, text_files):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(directory):
        raise ValueError('Choose a new output directory outside the data index')
    records = data.read_index(directory)
    memberships = {p: data.read_membership(directory, p, records)[0] for p in data.PROTOCOLS}
    output.mkdir(parents=True)
    for mode, text_file in text_files.items():
        configs = {name: protocol for name, (m, protocol) in CONFIGS.items() if m == mode}
        wanted = {key for key, row in records.items() if mode == 'markdown' or row['coverage'] == 'package_complete'}
        seen, batches = set(), {(name, split): [] for name in configs for split in SPLITS}
        with ExitStack() as stack:
            writers = {}
            for name, split in batches:
                target = output / 'data' / name / (split + '.parquet')
                target.parent.mkdir(parents=True, exist_ok=True)
                writers[name, split] = stack.enter_context(pq.ParquetWriter(target, FEATURES.arrow_schema, compression='zstd'))
            for body in data.read_jsonl(text_file):
                key = body['id']
                if key not in wanted or key in seen:
                    raise ValueError(f'{mode}: unexpected or duplicate restored ID: {key}')
                row = checked_row(records[key], body, mode)
                seen.add(key)
                for name, protocol in configs.items():
                    split = memberships[protocol][key]
                    if split == 'excluded':
                        continue
                    batch = batches[name, split]
                    batch.append(row)
                    if len(batch) == 256:
                        writers[name, split].write_table(pa.Table.from_pylist(batch, schema=FEATURES.arrow_schema))
                        batch.clear()
            if seen != wanted:
                raise ValueError(f'{mode}: {len(wanted - seen)} restored records are missing')
            for key, batch in batches.items():
                if batch:
                    writers[key].write_table(pa.Table.from_pylist(batch, schema=FEATURES.arrow_schema))
        print(f'{mode}: exported {len(seen):,} verified records into two configurations.', flush=True)
    for name in ['records.jsonl', 'sources.json', *['splits/' + p + '.csv' for p in data.PROTOCOLS]]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / name, target)
    # The card is written last so an interrupted build is not presented as ready.
    shutil.copyfile(directory / 'README.md', output / 'README.md')


def verify(directory, output, cache_dir):
    """Actually load all four named configs and check every returned row."""
    directory, output = Path(directory), Path(output)
    records = data.read_index(directory)
    memberships = {p: data.read_membership(directory, p, records)[0] for p in data.PROTOCOLS}
    if set(get_dataset_config_names(str(output.resolve()))) != set(CONFIGS):
        raise ValueError('The dataset card must expose exactly the four named configurations')
    for name in ['records.jsonl', 'sources.json', *['splits/' + p + '.csv' for p in data.PROTOCOLS]]:
        if file_hash(directory / name) != file_hash(output / name):
            raise ValueError(f'Export changed the original index or splits: {name}')
    report = {'passed': False, 'checked_at': datetime.now(timezone.utc).isoformat(), 'configs': {},
              'versions': {p: importlib.metadata.version(p) for p in ('datasets', 'pyarrow')},
              'split_sha256': {p: file_hash(directory / 'splits' / (p + '.csv')) for p in data.PROTOCOLS}}
    for config, (mode, protocol) in CONFIGS.items():
        dataset = load_dataset(str(output.resolve()), config, cache_dir=str(Path(cache_dir).resolve()))
        if set(dataset) != set(SPLITS):
            raise ValueError(f'{config}: expected train, validation, test')
        report['configs'][config] = {}
        for split in SPLITS:
            expected = {key for key, part in memberships[protocol].items()
                        if part == split and (mode == 'markdown' or records[key]['coverage'] == 'package_complete')}
            if dataset[split].features != FEATURES:
                raise ValueError(f'{config}/{split}: feature schema changed')
            seen, labels, sources = set(), Counter(), Counter()
            companions = 0
            for row in dataset[split]:
                key = row['id']
                if key not in expected or key in seen:
                    raise ValueError(f'{config}/{split}: unexpected or duplicate ID: {key}')
                if row != checked_row(records[key], row, mode):
                    raise ValueError(f'{key}: exported metadata or content differs')
                seen.add(key)
                labels[records[key]['label']] += 1
                sources[row['source_id']] += 1
                companions += len(row['files'])
            if seen != expected:
                raise ValueError(f'{config}/{split}: missing records')
            report['configs'][config][split] = {
                'rows': len(seen), 'labels': dict(labels), 'primary_sources': dict(sources),
                'companion_files': companions,
                'ids_sha256': data.sha256('\n'.join(sorted(seen)) + '\n'),
            }
        print(f'{config}: load_dataset and every row verified.', flush=True)
    report['passed'] = True
    report['checks'] = [
        'All four configs loaded by name with train/validation/test',
        'Every ID, label, source field and eligible text hash matches the index',
        'Original split files copied byte-for-byte, including SD exclusions',
        'Package cohorts preserve Markdown memberships and canonical SKILL.md',
        'Groups and exact Markdown hashes do not cross partitions',
        'SD test sources do not appear in train or validation',
    ]
    report['files_sha256'] = {p.relative_to(output).as_posix(): file_hash(p)
                              for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'verification.json'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'verify'))
    parser.add_argument('--data', type=Path, default=data.DEFAULT_DATA)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown-text', type=Path)
    parser.add_argument('--package-text', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=data.ROOT / 'outputs/hf-load-cache')
    args = parser.parse_args()
    if args.action == 'build':
        if not args.markdown_text or not args.package_text:
            parser.error('build requires both --markdown-text and --package-text')
        build(args.data, args.output, {'markdown': args.markdown_text, 'package': args.package_text})
    verify(args.data, args.output, args.cache_dir)
    print('Local HF dataset verified. Nothing uploaded.')


if __name__ == '__main__':
    main()

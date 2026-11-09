"""Restore pinned MalSkillBench benign packages into inert JSON caches."""
import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

try:
    from .community_data import read_jsonl
    from .import_malskillbench import REPO, REVISION, BENIGN_TREE, fetch
    from .package_inputs import safe_relative_path
    from .skill_package import IGNORE_DIRS, MAX_FILE_BYTES
except ImportError:
    from community_data import read_jsonl
    from import_malskillbench import REPO, REVISION, BENIGN_TREE, fetch
    from package_inputs import safe_relative_path
    from skill_package import IGNORE_DIRS, MAX_FILE_BYTES


def decode_input(path, raw):
    safe_relative_path(path)
    parts = Path(path).parts
    name = parts[-1].lower()
    if name in {'_meta.json', '_expected.json', '_evidence.json', 'benchmark_injection.md',
                'ground_truth.json', 'case_metadata.json', 'metadata.json', 'evaluation.json',
                'expected.json', 'evidence.json', 'runtime.json', 'annotations.json', 'labels.json',
                'attacker_goal.json', 'attack_design.json'} or '_runtime' in parts or '.clawhub' in parts:
        return None, 'registry_or_benchmark_metadata'
    if any(p.lower() in IGNORE_DIRS for p in parts[:-1]) or name == '.env' or name.startswith('.env.') or Path(name).suffix in {'.pem', '.key'}:
        return None, 'excluded_configuration_or_directory'
    if len(raw) > MAX_FILE_BYTES:
        return None, 'file_size_limit'
    try:
        text = raw.decode('utf-8-sig')
        if '\x00' in text:
            raise UnicodeError('NUL')
        return text, ''
    except UnicodeError:
        return None, 'not_utf8_text'


def read_blob(item, imported, output):
    sha = item['sha']
    md_cache = imported/'cache'/(sha+'.json')
    cache = output/'cache'/(sha+'.json')
    if cache.exists():
        saved = json.loads(cache.read_text(encoding='utf-8'))
        raw = base64.b64decode(saved['base64'])
    elif md_cache.exists():
        raw = json.loads(md_cache.read_text(encoding='utf-8'))['text'].encode('utf-8')
    else:
        raw = fetch(f'https://raw.githubusercontent.com/{REPO}/{REVISION}/Dataset/Skills/benign/' + quote(item['path'], safe='/'))
    actual = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if actual != sha or len(raw) != item['size']:
        raise ValueError('Pinned Git blob mismatch: ' + item['path'])
    if not cache.exists():
        cache.write_text(json.dumps({'base64': base64.b64encode(raw).decode('ascii')})+'\n', encoding='utf-8', newline='\n')
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imported', type=Path, required=True, help='Existing benign Markdown import directory')
    parser.add_argument('--output', type=Path, required=True, help='Local package cache directory; resumable')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('Use 1 to 16 workers')
    tree = json.loads((args.imported/'tree.json').read_text(encoding='utf-8'))
    if tree['sha'] != BENIGN_TREE or tree.get('truncated'):
        raise ValueError('Expected the complete pinned benign tree')
    markdown = {r['id'].split(':', 1)[1]: r for r in read_jsonl(args.imported/'records-with-text.jsonl')}
    by_package, unique = defaultdict(list), {}
    for item in tree['tree']:
        if item['type'] != 'blob' or item['path'].split('/')[0] not in markdown:
            continue
        if item['mode'] not in ('100644', '100755'):
            raise ValueError('Unsupported linked file in source tree')
        safe_relative_path(item['path'])
        by_package[item['path'].split('/')[0]].append(item)
        unique.setdefault(item['sha'], item)
    (args.output/'cache').mkdir(parents=True, exist_ok=True)
    raw_blobs, errors = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(read_blob, item, args.imported, args.output): item for item in unique.values()}
        for i, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                raw_blobs[item['sha']] = future.result()
            except Exception as exc:
                errors.append({'path': item['path'], 'error': type(exc).__name__+': '+str(exc)})
            if i % 500 == 0:
                print(f'{i}/{len(unique)} blobs verified; {len(errors)} errors', flush=True)
    report = {'repository': REPO, 'revision': REVISION, 'packages_requested': len(markdown),
              'unique_blobs_requested': len(unique), 'verified_blobs': len(raw_blobs), 'errors': errors}
    (args.output/'recovery.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    if errors:
        raise SystemExit('Incomplete recovery; rerun to resume cached downloads.')
    rows, excluded = [], Counter()
    for name, items in sorted(by_package.items()):
        inventory, companions = [], []
        for item in sorted(items, key=lambda r: r['path']):
            path = item['path'].split('/', 1)[1]
            raw = raw_blobs[item['sha']]
            text, reason = decode_input(path, raw)
            inventory.append({'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                              'git_blob_sha1': item['sha'], 'input_eligible': text is not None, 'exclusion_reason': reason})
            if reason:
                excluded[reason] += 1
            if text is not None and path != 'SKILL.md':
                companions.append({'path': path, 'text': text})
        md = markdown[name]
        assert any(i['path']=='SKILL.md' and i['sha256']==md['files'][0]['sha256'] for i in inventory)
        rows.append({'source_member_id': md['id'], 'label': 0, 'skill_text': md['skill_text'],
                     'files': companions, 'inventory': inventory, 'package_complete': True,
                     'source_url': md['source_url'], 'revision': REVISION,
                     'source_package_path': 'Dataset/Skills/benign/'+name,
                     'upstream_license': md['upstream_license']})
    with (args.output/'packages.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    report.update({'packages_recovered': len(rows), 'with_companion_text': sum(bool(r['files']) for r in rows),
                   'all_source_files': sum(len(r['inventory']) for r in rows), 'input_file_exclusions': dict(excluded)})
    (args.output/'recovery.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

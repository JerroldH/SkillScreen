"""Download pinned MalSkillBench benign Markdown as inert local data."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = 'lxyeternal/MalSkillBench'
REVISION = '3548e2837adf58b96497f2c688f96beff26485da'
BENIGN_TREE = '364047c9fb6b0536bc10d527ddd31202064ab22a'


def normalized_hash(text):
    text = text.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n').strip()
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'SkillScreen-data-import'}), timeout=45) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def read_skill(item, output):
    # Content stays inside JSON, never loose executable or agent-discovered files.
    cache = output / 'cache' / (item['sha'] + '.json')
    if cache.exists():
        saved = json.loads(cache.read_text(encoding='utf-8'))
        text = saved['text']
        raw = text.encode('utf-8')
    else:
        path = 'Dataset/Skills/benign/' + item['path']
        raw = fetch(f'https://raw.githubusercontent.com/{REPO}/{REVISION}/' + quote(path, safe='/'))
    blob_sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if blob_sha != item['sha'] or len(raw) != item['size']:
        raise ValueError('Git blob verification failed for ' + item['path'])
    text = raw.decode('utf-8')
    if not text.strip():
        raise ValueError('Empty Markdown in ' + item['path'])
    if not cache.exists():
        cache.write_text(json.dumps({'text': text}, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    package = item['path'].split('/')[0]
    sha = hashlib.sha256(raw).hexdigest()
    return {
        'id': 'malskillbench-benign:' + package,
        'source_url': f'https://github.com/{REPO}', 'revision': REVISION,
        'source_path': 'Dataset/Skills/benign/' + item['path'],
        'source_id': 'msb:SRC001', 'source_ids': ['msb:SRC001'],
        'source_name': 'MalSkillBench', 'group_id': 'text:' + sha,
        'label': 'benign', 'label_origin': 'upstream',
        'evidence': 'MalSkillBench benign reference set of most-downloaded skills. Upstream label, not independently reviewed.',
        'coverage': 'markdown_only', 'upstream_license': 'Academic research use only; underlying skills retain their own terms.',
        'files': [{'path': 'SKILL.md', 'sha256': sha, 'text': None}],
        'normalized_sha256': normalized_hash(text), 'git_blob_sha1': blob_sha,
        'skill_text': text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/malskillbench-benign'))
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    output = args.output.resolve()
    (output / 'cache').mkdir(parents=True, exist_ok=True)
    tree_path = output / 'tree.json'
    if not tree_path.exists():
        tree_path.write_bytes(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{BENIGN_TREE}?recursive=1'))
    tree = json.loads(tree_path.read_text(encoding='utf-8'))
    if tree.get('truncated') or tree['sha'] != BENIGN_TREE:
        raise ValueError('Expected a complete pinned benign tree')
    items = [i for i in tree['tree'] if i['type'] == 'blob' and i['path'].count('/') == 1 and i['path'].endswith('/SKILL.md')]
    if len(items) != 4000 or any(i['mode'] not in ('100644', '100755') for i in items):
        raise ValueError('Unexpected pinned Markdown inventory')
    rows, errors, exclusions = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(read_skill, item, output): item for item in items}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                rows.append(future.result())
            except UnicodeDecodeError:
                exclusions.append({'path': futures[future]['path'], 'reason': 'Not valid UTF-8. No lossy decoding applied.', 'git_blob_sha1': futures[future]['sha']})
            except Exception as exc:
                errors.append({'path': futures[future]['path'], 'error': type(exc).__name__ + ': ' + str(exc)})
            if index % 250 == 0:
                print(f'{index}/{len(items)} completed; {len(errors)} errors', flush=True)
    report = {'repository': REPO, 'revision': REVISION, 'requested': len(items),
              'restored': len(rows), 'errors': errors, 'exclusions': exclusions,
              'unique_normalized_texts': len({r['normalized_sha256'] for r in rows}),
              'labels': dict(Counter(r['label'] for r in rows))}
    (output / 'import.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if errors:
        raise SystemExit('Incomplete import. Cache retained; rerun to resume.')
    with (output / 'records-with-text.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
        for row in sorted(rows, key=lambda r: r['id']):
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

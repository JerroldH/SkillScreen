"""Optional CPU check for exact PyPI pins in requirements*.txt, using OSV."""
from datetime import datetime, timezone
import json
from pathlib import PurePosixPath
import re
from urllib.request import Request, urlopen


OSV_URL = 'https://api.osv.dev/v1/query'
PIN = re.compile(r'([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[A-Za-z0-9_,.-]+\])?==([A-Za-z0-9][A-Za-z0-9.!+_-]*)')
OTHER_MANIFESTS = {'package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
                   'composer.json', 'composer.lock', 'pyproject.toml', 'poetry.lock',
                   'uv.lock', 'pipfile', 'pipfile.lock', 'dockerfile', 'go.mod', 'cargo.lock'}


def query_osv(name, version):
    payload = json.dumps({'package': {'name': name, 'ecosystem': 'PyPI'}, 'version': version}).encode()
    request = Request(OSV_URL, data=payload, headers={'Content-Type': 'application/json',
                                                  'User-Agent': 'SkillScreen/0.1'})
    with urlopen(request, timeout=15) as response:
        raw = response.read(4_194_305)
    if len(raw) > 4_194_304:
        raise ValueError('OSV response exceeded 4 MiB.')
    result = json.loads(raw)
    if not isinstance(result, dict) or result.get('error') or not isinstance(result.get('vulns', []), list):
        raise ValueError('Unexpected OSV response.')
    if result.get('next_page_token'):
        raise ValueError('OSV response is paginated; this check is incomplete.')
    return result.get('vulns', [])


def check_dependencies(row, lookup=None):
    lookup = query_osv if lookup is None else lookup
    checked, unresolved, findings, errors = [], [], [], []
    cache = {}
    manifests = []
    for item in row['files']:
        name = PurePosixPath(item['path']).name.lower()
        if not (name.startswith('requirements') and name.endswith('.txt')):
            if name in OTHER_MANIFESTS:
                unresolved.append({'path': item['path'], 'reason': 'unsupported_manifest'})
            continue
        manifests.append(item['path'])
        for line_number, raw in enumerate(item['text'].splitlines(), 1):
            line = re.split(r'\s+#', raw, maxsplit=1)[0].strip()
            if not line or line.startswith('#'):
                continue
            location = {'path': item['path'], 'line': line_number}
            match = PIN.fullmatch(line)
            if not match:
                unresolved.append({**location, 'reason': 'not_an_unconditional_exact_pin'})
                continue
            package, version = match.groups()
            package = re.sub(r'[-_.]+', '-', package).lower()
            key = package, version
            if key not in cache:
                try:
                    advisories = lookup(*key)
                    if not isinstance(advisories, list) or any(not isinstance(a, dict) or not a.get('id') for a in advisories):
                        raise ValueError('Invalid advisory response.')
                    cache[key] = [a for a in advisories if not a.get('withdrawn')]
                except (OSError, ValueError, TimeoutError) as exc:
                    cache[key] = str(exc)
            result = cache[key]
            dependency = {**location, 'ecosystem': 'PyPI', 'package': package, 'version': version}
            if isinstance(result, str):
                errors.append({**dependency, 'reason': result})
                continue
            checked.append({**dependency, 'advisory_count': len(result)})
            for advisory in result:
                findings.append({**dependency, 'type': 'affected_dependency_version',
                                 'advisory_id': advisory['id'], 'aliases': advisory.get('aliases', []),
                                 'summary': advisory.get('summary', ''),
                                 'url': 'https://osv.dev/vulnerability/' + advisory['id'],
                                 'advisory_modified': advisory.get('modified')})
    status = ('partial' if errors and checked else 'unavailable' if errors else
              'partial' if unresolved else 'complete' if manifests else 'not_applicable')
    return {'status': status, 'checked_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'Exact, unconditional PyPI pins in requirements*.txt; other files and transitive dependencies are not resolved.',
            'manifests': manifests, 'checked': checked, 'findings': findings,
            'unresolved': unresolved, 'errors': errors}

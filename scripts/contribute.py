"""Prepare a metadata-only contribution draft, or check an existing draft."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from skill_package import read_package, write_json
from package_inputs import safe_relative_path


def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError('Expected a contribution object.')
    allowed = {'source_url', 'revision', 'text_manifest_sha256', 'files', 'skipped',
               'proposed_label', 'review_status', 'evidence', 'upstream_license'}
    if set(record) != allowed:
        raise ValueError('Use the fields produced by contribute.py; do not attach raw files.')
    if record.get('review_status') != 'candidate':
        raise ValueError('Contribution drafts must have review_status=candidate.')
    if record.get('proposed_label') not in ('benign', 'malicious', 'unknown'):
        raise ValueError('Choose proposed_label benign, malicious, or unknown.')
    if not isinstance(record.get('evidence'), str) or not record['evidence'].strip():
        raise ValueError('Explain the proposed label or finding in evidence.')
    files = record.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('A nonempty file inventory is required.')
    paths = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {'path', 'sha256', 'bytes'}:
            raise ValueError('Inventory entries contain only path, sha256, and bytes; no raw text.')
        path = safe_relative_path(item['path'])
        if path in paths or not re.fullmatch('[0-9a-f]{64}', str(item['sha256'])):
            raise ValueError('Duplicate path or invalid file hash.')
        if type(item['bytes']) is not int or item['bytes'] < 0:
            raise ValueError('File size must be a nonnegative integer.')
        paths.add(path)
    if 'SKILL.md' not in paths:
        raise ValueError('Inventory must contain SKILL.md.')
    ordered = sorted(files, key=lambda item: item['path'])
    fingerprint = hashlib.sha256(json.dumps(ordered, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if record.get('text_manifest_sha256') != fingerprint:
        raise ValueError('File inventory does not match text_manifest_sha256.')
    source = record.get('source_url')
    revision = record.get('revision')
    if source:
        if not isinstance(source, str):
            raise ValueError('Source URL must be text.')
        url = urlsplit(source)
        if url.scheme != 'https' or not url.hostname or url.username or url.password:
            raise ValueError('Use a public HTTPS source URL without credentials.')
    if revision and (not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision)):
        raise ValueError('Use the immutable 40-character Git commit SHA as revision.')
    if not isinstance(record.get('upstream_license'), str) or not record['upstream_license'].strip():
        raise ValueError('Provide upstream_license, or unknown if it needs checking.')
    missing = [key for key in ('source_url', 'revision') if not record.get(key)]
    return {'valid_draft': True, 'missing_provenance': missing,
            'source_verified': False, 'label_verified': False,
            'note': 'This checks record structure and inventory consistency. Maintainer review must verify the source, label, and redistribution terms.'}


def make_record(path, label, evidence, source_url=None, revision=None, license_name='unknown'):
    _, coverage = read_package(path)
    record = {'source_url': source_url, 'revision': revision,
              'text_manifest_sha256': coverage['text_manifest_sha256'],
              'files': coverage['files'], 'skipped': coverage['skipped'],
              'proposed_label': label, 'review_status': 'candidate',
              'evidence': evidence, 'upstream_license': license_name}
    validate_record(record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path, nargs='?')
    parser.add_argument('--label', choices=('benign', 'malicious', 'unknown'), default='unknown')
    parser.add_argument('--evidence', help='Why this sample or finding is useful; include advisory links if relevant')
    parser.add_argument('--source-url', help='Public upstream repository/package URL')
    parser.add_argument('--revision', help='Immutable upstream Git commit SHA')
    parser.add_argument('--license', default='unknown', dest='license_name')
    parser.add_argument('--output', type=Path, help='New draft JSON file outside the package')
    parser.add_argument('--check', type=Path, help='Validate an existing draft without network access')
    args = parser.parse_args(argv)
    if args.check:
        if args.package or args.output:
            parser.error('Use --check by itself.')
        record = json.loads(args.check.read_text(encoding='utf-8'))
    else:
        if not args.package or not args.output or not args.evidence:
            parser.error('Supply a package, --evidence, and --output, or use --check.')
        root = args.package.resolve() if args.package.is_dir() else args.package.parent.resolve()
        if args.output.resolve().is_relative_to(root):
            parser.error('Write the contribution outside the input package.')
        record = make_record(args.package, args.label, args.evidence,
                             args.source_url, args.revision, args.license_name)
        write_json(args.output, record)
        print('Created a candidate draft without raw file contents: ' + str(args.output))
    print(json.dumps(validate_record(record), indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

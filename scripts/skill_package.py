"""Read a skill as inert text; never import, install, or execute its files."""
import hashlib
import json
import os
from pathlib import Path
import stat


IGNORE_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.ssh', '.aws'}
MAX_FILE_BYTES = 1_048_576
MAX_PACKAGE_BYTES = 16_777_216
MAX_ENTRIES = 1000


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_package(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.is_junction():
        raise ValueError('Choose a real package directory or SKILL.md, not a link.')
    single = path.is_file()
    if single and path.name != 'SKILL.md':
        raise ValueError('A file input must be named SKILL.md.')
    root = path.parent if single else path
    if not root.is_dir():
        raise ValueError('Package directory does not exist.')
    root = root.resolve()
    files, skipped, total, entries = [], [], 0, 0
    if single:
        candidates = [(root, [], ['SKILL.md'])]
    else:
        candidates = os.walk(root, followlinks=False, onerror=lambda exc: (_ for _ in ()).throw(exc))
    for directory, dirs, names in candidates:
        directory = Path(directory)
        entries += len(dirs) + len(names)
        if entries > MAX_ENTRIES:
            raise ValueError(f'Package exceeds {MAX_ENTRIES} entries; select a smaller skill directory.')
        for name in sorted(dirs[:]):
            item = directory / name
            reason = ('excluded_directory' if name.lower() in IGNORE_DIRS else
                      'link' if item.is_symlink() or item.is_junction() else None)
            if reason:
                dirs.remove(name)
                skipped.append({'path': item.relative_to(root).as_posix() + '/', 'reason': reason})
        dirs.sort()
        for name in sorted(names):
            item = directory / name
            relative = item.relative_to(root).as_posix()
            reason = None
            if item.is_symlink() or item.is_junction():
                reason = 'link'
            elif name.lower() == '.env' or name.lower().startswith('.env.') or item.suffix.lower() in {'.pem', '.key'}:
                reason = 'private_configuration'
            elif not stat.S_ISREG(item.stat().st_mode):
                reason = 'not_regular_file'
            elif any(ord(char) < 32 or ord(char) == 127 for char in relative) or '\\' in relative:
                reason = 'unsupported_path'
            elif not item.resolve().is_relative_to(root):
                raise ValueError('A file resolved outside the package.')
            if reason:
                skipped.append({'path': relative, 'reason': reason})
                continue
            with item.open('rb') as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                skipped.append({'path': relative, 'reason': 'file_size_limit'})
                continue
            total += len(raw)
            if total > MAX_PACKAGE_BYTES:
                raise ValueError('Package exceeds the 16 MiB text-read budget.')
            try:
                text = raw.decode('utf-8')
                if '\x00' in text:
                    raise UnicodeError('NUL byte')
            except UnicodeError:
                skipped.append({'path': relative, 'reason': 'not_utf8_text'})
                continue
            files.append({'path': relative, 'sha256': digest(raw), 'bytes': len(raw), 'text': text})
    files.sort(key=lambda item: item['path'])
    markdown = next((item for item in files if item['path'] == 'SKILL.md'), None)
    if markdown is None or not markdown['text'].strip():
        raise ValueError('A nonempty UTF-8 SKILL.md is required at the package root.')
    manifest = [{key: item[key] for key in ('path', 'sha256', 'bytes')} for item in files]
    fingerprint = digest(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode())
    row = {'skill_text': markdown['text'],
           'files': [{'path': item['path'], 'text': item['text']} for item in files if item is not markdown]}
    coverage = {'input_kind': 'markdown_file' if single else 'package_directory',
                'text_manifest_sha256': fingerprint, 'text_files': len(files),
                'text_bytes': sum(item['bytes'] for item in files), 'files': manifest, 'skipped': skipped}
    return row, coverage


def write_json(path, value):
    """Create outputs explicitly; do not replace a user's existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

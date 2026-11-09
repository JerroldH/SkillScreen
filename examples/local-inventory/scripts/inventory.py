"""Count immediate file extensions in a directory explicitly selected by the user."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error('Choose an existing directory.')
    counts = Counter(path.suffix.lower() or '(none)' for path in args.directory.iterdir()
                     if path.is_file() and not path.is_symlink())
    print(json.dumps(dict(sorted(counts.items())), indent=2))


if __name__ == '__main__':
    main()

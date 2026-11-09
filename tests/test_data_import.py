"""Check content identity, conflict handling, and upstream byte verification."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.import_malskillbench import normalized_hash, read_skill
from scripts.merge_training_pool import merge_rows


class ImportTests(unittest.TestCase):
    def test_duplicates_conflicts_and_weak_provenance(self):
        rows = [
            {'id': 'a', 'label': 'benign', 'skill_text': 'Example\r\n', 'label_origin': 'upstream'},
            {'id': 'b', 'label': 0, 'skill_text': '\ufeffExample\n', 'label_origin': 'upstream'},
            {'id': 'c', 'label': 1, 'skill_text': 'Conflicted', 'label_origin': 'weak'},
            {'id': 'd', 'label': 0, 'skill_text': 'Conflicted', 'label_origin': 'upstream'},
            {'id': 'e', 'label': 1, 'skill_text': 'Separate', 'label_origin': 'weak'},
        ]
        kept, conflicts, duplicates, report = merge_rows(rows)
        self.assertEqual(report['retained'], 2)
        self.assertEqual(report['labels'], {'benign': 1, 'malicious': 1})
        self.assertEqual(report['weak_only'], 1)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(duplicates[0]['members'], ['a', 'b'])
        row = next(r for r in kept if r['label'] == 0)
        self.assertEqual(row['skill_text'], rows[0]['skill_text'])
        self.assertEqual(len(row['members']), 2)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            merge_rows([{**rows[0], 'normalized_sha256': '0'*64}])

    def test_cached_bytes_must_match_pinned_git_blob(self):
        raw = b'Fixture\r\n'
        sha = hashlib.sha1(b'blob 9\0' + raw).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'cache').mkdir()
            path = root/'cache'/(sha+'.json')
            path.write_text(json.dumps({'text': raw.decode()}), encoding='utf-8')
            item = {'path': 'fixture/SKILL.md', 'size': len(raw), 'sha': sha}
            row = read_skill(item, root)
            self.assertEqual(row['normalized_sha256'], normalized_hash('Fixture'))
            self.assertIsNone(row['files'][0]['text'])
            path.write_text(json.dumps({'text': 'Changed!!'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Git blob verification failed'):
                read_skill(item, root)


if __name__ == '__main__':
    unittest.main()

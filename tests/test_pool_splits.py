"""Check reproducible memberships and the weak-label/source boundaries."""
import copy
import hashlib
import unittest

from scripts.import_malskillbench import normalized_hash
from scripts.split_training_pool import describe, make_splits, validate


def fixture():
    rows = []
    for source in ('msb:SRC001', 'msb:SRC010', 'msb:SRC011'):
        for label in (0, 1):
            for i in range(20):
                text = f'Fixture {source} {label} {i}'
                key = normalized_hash(text)
                rows.append({'id': key, 'skill_text': text, 'label': label,
                             'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                             'normalized_sha256': key, 'label_origin': 'upstream',
                             'members': [{'id': key, 'collection': 'benchmark',
                                          'source_ids': [source], 'source_name': source}]})
    for i in range(3):
        text = f'Weak fixture {i}'
        key = normalized_hash(text)
        rows.append({'id': key, 'skill_text': text, 'label': 1,
                     'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                     'normalized_sha256': key, 'label_origin': 'weak',
                     'members': [{'id': key, 'collection': 'domain', 'source_name': 'fixture',
                                  'role': 'weak_positive', 'zip_member': f'fixture/{i}/SKILL.md'}]})
    rows[0]['members'][0]['source_ids'].append('msb:SRC011')
    return rows


class SplitTests(unittest.TestCase):
    def test_reproducible_complete_and_source_aliases_excluded(self):
        rows = fixture()
        records = describe(rows)
        assignments = make_splits(records)
        self.assertEqual(assignments, make_splits(describe(list(reversed(rows)))))
        self.assertEqual(assignments['source_disjoint'][rows[0]['id']],
                         ('excluded', 'source_aliases_cross_holdout_boundary'))
        for protocol in assignments.values():
            self.assertEqual(set(protocol), set(records))
            for row in rows[-3:]:
                self.assertEqual(protocol[row['id']][0], 'train')
        changed = copy.deepcopy(assignments)
        changed['grouped_random'][rows[-1]['id']] = ('test', '')
        with self.assertRaisesRegex(ValueError, 'weak labels'):
            validate(records, changed)
        changed = copy.deepcopy(assignments)
        changed['source_disjoint'][rows[0]['id']] = ('train', '')
        with self.assertRaisesRegex(ValueError, 'source alias'):
            validate(records, changed)

    def test_bad_content_or_unresolved_version_groups_stop(self):
        rows = fixture()
        rows[0]['skill_text'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            describe(rows)
        rows = fixture()
        rows[0]['members'][0]['group_id'] = 'same-version-family'
        rows[1]['members'][0]['group_id'] = 'same-version-family'
        with self.assertRaisesRegex(ValueError, 'need grouping'):
            describe(rows)


if __name__ == '__main__':
    unittest.main()

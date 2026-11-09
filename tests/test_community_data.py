"""Dataset boundaries and the shared Markdown/package input path."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import community_data as data
from scripts.skill_package import read_package


def fixture_rows():
    # These are test records only, not released skill labels or a benchmark.
    return [{'id': f'test-{i}', 'source_url': 'https://example.org/fixture', 'revision': 'a'*40,
             'source_id': 'heldout' if i >= 4 else 'development',
             'source_ids': ['heldout' if i >= 4 else 'development'], 'group_id': f'group-{i}',
             'label': 'malicious' if i % 2 else 'benign', 'label_origin': 'reviewed',
             'evidence': 'Synthetic fixture for loader tests.', 'coverage': 'markdown_only',
             'upstream_license': 'Apache-2.0',
             'files': [{'path': 'SKILL.md', 'sha256': data.sha256(f'Fixture {i}'), 'text': f'Fixture {i}'}]}
            for i in range(6)]


def write_fixture(folder, rows):
    folder.mkdir()
    (folder / 'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    (folder / 'README.md').write_text('Test fixture only.', encoding='utf-8')
    (folder / 'splits').mkdir()
    for protocol in data.PROTOCOLS:
        with (folder / 'splits' / (protocol+'.csv')).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['id', 'split', 'reason'])
            writer.writeheader()
            for i, row in enumerate(rows):
                writer.writerow({'id': row['id'], 'split': ('train', 'validation', 'test')[i//2], 'reason': ''})


class DataTests(unittest.TestCase):
    def test_public_release_and_unchanged_paper_manifest(self):
        report = data.check_dataset(data.DEFAULT_DATA)
        self.assertEqual(report['records'], 13514)
        self.assertEqual(report['labels'], {'malicious': 7500, 'benign': 6014})
        self.assertEqual(report['embedded_files'], 0)
        self.assertEqual(report['coverage']['package_complete'], 11831)
        self.assertEqual(report['protocols']['source_disjoint']['test'], {'malicious': 838, 'benign': 526})
        import hashlib
        config=json.loads((ROOT/'configs/data.json').read_text(encoding='utf-8'))
        for protocol in config['protocols'].values():
            spec=protocol['manifest']
            self.assertEqual(hashlib.sha256((ROOT/spec['path']).read_bytes()).hexdigest(),spec['sha256'])
        for row in data.read_index(data.DEFAULT_DATA).values():
            self.assertFalse(any(m.startswith('domain-') for m in row['member_ids']))

    def test_both_protocols_export_and_select(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'export'
            write_fixture(source, fixture_rows())
            (source / 'private-cache.txt').write_text('must not be exported')
            self.assertEqual(data.check_dataset(source)['records'], 6)
            self.assertEqual(len(data.select_records(source, None, 'all')), 6)
            for protocol in data.PROTOCOLS:
                self.assertEqual([r['id'] for r in data.select_records(source, protocol, 'test')], ['test-4', 'test-5'])
            data.export_dataset(source, output)
            self.assertFalse((output / 'private-cache.txt').exists())
            self.assertEqual(data.check_dataset(output), data.check_dataset(source))
            with self.assertRaises(ValueError):
                data.export_dataset(source, output)

    def test_duplicate_ids_and_missing_assignments_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            rows = fixture_rows()
            write_fixture(source, rows)
            with (source/'records.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                data.read_index(source)
            (source/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            split = source/'splits/grouped_random.csv'
            split.write_text('\n'.join(split.read_text().splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError, 'every record'):
                data.check_dataset(source)

    def test_duplicate_content_related_versions_and_source_leakage_rejected(self):
        changes = [
            lambda rows: rows[4].update(group_id=rows[0]['group_id']),
            lambda rows: rows[4].update(files=copy.deepcopy(rows[0]['files'])),
            lambda rows: rows[4]['source_ids'].append('development'),
            lambda rows: rows[4].update(label_origin='weak'),
            lambda rows: rows[2].update(label='unknown'),
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                rows = fixture_rows()
                change(rows)
                source = Path(tmp)/'source'
                write_fixture(source, rows)
                with self.assertRaises(ValueError):
                    data.check_dataset(source)

    def test_metadata_is_not_usable_text_and_package_has_no_silent_fallback(self):
        rows = fixture_rows()
        rows[0]['files'][0]['text'] = None
        with self.assertRaisesRegex(ValueError, 'missing'):
            data.materialize_inputs(rows, 'markdown')
        with self.assertRaisesRegex(ValueError, 'complete package'):
            data.materialize_inputs(fixture_rows(), 'package')

    def test_restored_text_hash_checked_before_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = fixture_rows()
            for row in rows:
                row['files'][0]['text'] = None
            for split in ('train', 'validation', 'test'):
                (directory/(split+'.jsonl')).write_text('', encoding='utf-8')
            text_file = directory/'train.jsonl'
            text_file.write_text(''.join(json.dumps({'id':r['id'], 'skill_text':f'Fixture {i}'})+'\n' for i,r in enumerate(rows)), encoding='utf-8')
            self.assertEqual(len(data.materialize_inputs(rows, 'markdown', directory)), 6)
            self.assertEqual(len(data.materialize_inputs(rows, 'markdown', text_files=[text_file])), 6)
            text_file.write_text(text_file.read_text().replace('Fixture 0', 'Changed 0'))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                data.materialize_inputs(rows, 'markdown', directory)

    def test_complete_authored_package_in_both_input_modes(self):
        package, inventory = read_package(ROOT/'examples/local-inventory')
        row = fixture_rows()[0]
        row['coverage'] = 'package_complete'
        texts = {'SKILL.md': package['skill_text'], **{f['path']:f['text'] for f in package['files']}}
        row['files'] = [{**f, 'text': texts[f['path']]} for f in inventory['files']]
        data.check_record(row)
        full = data.materialize_inputs([row], 'package')[0]
        markdown = data.materialize_inputs([row], 'markdown')[0]
        self.assertEqual(full['skill_text'], markdown['skill_text'])
        self.assertEqual(full['files'], package['files'])
        self.assertEqual(markdown['files'], [])
        self.assertEqual(set(full), {'id', 'label', 'skill_text', 'files'})
        row['files'][0]['text'] += 'tampered'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            data.check_record(row)

    def test_unknown_source_terms_cannot_embed_text(self):
        row = fixture_rows()[0]
        row['upstream_license'] = ' UNKNOWN '
        with self.assertRaisesRegex(ValueError, 'terms required'):
            data.check_record(row)
        row['upstream_license'] = 'Apache-2.0'
        row['files'][0].update(text=' ', sha256=data.sha256(' '))
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            data.materialize_inputs([row], 'markdown')


if __name__ == '__main__':
    unittest.main()

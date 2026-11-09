"""Round-trip the four HF configs using inert, authored fixture text."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_community_data import fixture_rows, write_fixture
from scripts import community_data as data

AVAILABLE = importlib.util.find_spec('datasets') is not None
if AVAILABLE:
    from scripts import export_community as export


@unittest.skipUnless(AVAILABLE, 'Optional requirements/data.txt is not installed')
class ExportTests(unittest.TestCase):
    def test_four_configs_round_trip_with_exclusions_and_hash_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'source', root / 'export'
            rows = fixture_rows()
            for row in rows[1:]:
                row['coverage'] = 'package_complete'
                row['files'].append({'path': 'notes.md', 'text': 'Inert companion text. 中文',
                                     'sha256': data.sha256('Inert companion text. 中文')})
            write_fixture(source, rows)
            (source / 'sources.json').write_text('{}', encoding='utf-8')
            (source / 'README.md').write_text((data.DEFAULT_DATA / 'README.md').read_text(encoding='utf-8'), encoding='utf-8')
            split_path = source / 'splits/source_disjoint.csv'
            with split_path.open(encoding='utf-8', newline='') as handle:
                membership = list(csv.DictReader(handle))
            membership[3]['split'] = 'excluded'
            with split_path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['id', 'split', 'reason'])
                writer.writeheader()
                writer.writerows(membership)
            paths = {}
            for mode in ('markdown', 'package'):
                selected = rows if mode == 'markdown' else rows[1:]
                bodies = data.materialize_inputs(selected, mode)
                paths[mode] = root / (mode + '.jsonl')
                paths[mode].write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in bodies), encoding='utf-8')
            export.build(source, output, paths)
            report = export.verify(source, output, root / 'cache')
            self.assertTrue(report['passed'])
            self.assertEqual(report['configs']['markdown_random']['train']['rows'], 2)
            self.assertEqual(report['configs']['package_random']['train']['rows'], 1)
            self.assertEqual(report['configs']['package_sd']['validation']['rows'], 1)
            self.assertEqual(report['configs']['markdown_sd']['validation']['rows'], 1)
            with self.assertRaises(ValueError):
                export.build(source, output, paths)
            good = data.materialize_inputs([rows[1]], 'package')[0]
            bad = copy.deepcopy(good)
            bad['files'][0]['text'] += 'changed'
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                export.checked_row(rows[1], bad, 'package')
            bad = copy.deepcopy(good)
            bad['files'].append({'path': 'extra.md', 'text': 'unexpected'})
            with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
                export.checked_row(rows[1], bad, 'package')
            bad = copy.deepcopy(good)
            bad['label'] = 0
            with self.assertRaisesRegex(ValueError, 'label mismatch'):
                export.checked_row(rows[1], bad, 'package')


if __name__ == '__main__':
    unittest.main()

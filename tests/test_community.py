"""Offline checks for the community entry points. Never execute sample files."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import contribute
import scan
from scripts.check_dependencies import check_dependencies
from scripts.model_score import encode_input, HYPOTHESIS
from scripts.skill_package import read_package, write_json


class CharacterTokenizer:
    """Deterministic token boundaries for truncation tests; not model inference."""
    is_fast = True

    def num_special_tokens_to_add(self, pair):
        return 0

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text),
                'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class PackageTests(unittest.TestCase):
    def test_complete_example_and_single_file(self):
        path = ROOT / 'examples/local-inventory'
        row, coverage = read_package(path)
        self.assertEqual(coverage['text_files'], 4)
        self.assertEqual(coverage['skipped'], [])
        self.assertEqual(len(row['files']), 3)
        for item in coverage['files']:
            self.assertEqual(item['sha256'], hashlib.sha256((path / item['path']).read_bytes()).hexdigest())
        markdown, single = read_package(path / 'SKILL.md')
        self.assertEqual(markdown['files'], [])
        self.assertEqual(markdown['skill_text'], row['skill_text'])
        self.assertEqual(single['text_files'], 1)

    def test_inert_read_private_and_binary_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'SKILL.md').write_text('A skill.', encoding='utf-8')
            marker = root / 'was-executed'
            (root / 'helper.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n', encoding='utf-8')
            (root / '.ENV').write_text('SECRET=not-for-input', encoding='utf-8')
            (root / 'binary.bin').write_bytes(b'\x00\xff')
            (root / 'node_modules').mkdir()
            (root / 'node_modules/ignored.js').write_text('ignored', encoding='utf-8')
            row, coverage = read_package(root)
            self.assertFalse(marker.exists())
            self.assertEqual([x['path'] for x in row['files']], ['helper.py'])
            self.assertEqual({x['reason'] for x in coverage['skipped']},
                             {'private_configuration', 'not_utf8_text', 'excluded_directory'})
            self.assertEqual(coverage['text_bytes'], sum(f['bytes'] for f in coverage['files']))

    def test_external_symlink_not_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'skill'
            root.mkdir()
            (root / 'SKILL.md').write_text('A skill.', encoding='utf-8')
            outside = Path(tmp) / 'outside.txt'
            outside.write_text('must not be read', encoding='utf-8')
            try:
                (root / 'linked.txt').symlink_to(outside)
            except OSError:
                self.skipTest('Host does not permit creating symlinks.')
            row, coverage = read_package(root)
            self.assertEqual(row['files'], [])
            self.assertEqual(coverage['skipped'], [{'path': 'linked.txt', 'reason': 'link'}])

    def test_missing_and_empty_markdown_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for content in (None, '  '):
                if content is not None:
                    (root / 'SKILL.md').write_text(content, encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'nonempty'):
                    read_package(root)

    def test_model_failure_preserves_dependency_result(self):
        with patch('scripts.model_score.score_model', side_effect=ValueError('No GPU')):
            report = scan.scan(ROOT / 'examples/local-inventory', checks='all')
        self.assertEqual(report['model']['status'], 'unavailable')
        self.assertEqual(report['dependencies']['status'], 'complete')

    def test_inspect_runs_no_model_or_network(self):
        with patch('scripts.model_score.score_model', side_effect=AssertionError('model called')), \
                patch('scripts.check_dependencies.query_osv', side_effect=AssertionError('network called')):
            report = scan.scan(ROOT / 'examples/dependency-case', checks='all', inspect_only=True)
        self.assertEqual(report['model']['status'], 'not_run')
        self.assertEqual(report['dependencies']['status'], 'not_run')

    def test_output_does_not_overwrite_or_enter_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'existing.json'
            write_json(output, {'keep': True})
            with self.assertRaises(FileExistsError):
                write_json(output, {'keep': False})
            self.assertEqual(json.loads(output.read_text()), {'keep': True})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            scan.main([str(ROOT / 'examples/local-inventory'), '--inspect', '--output',
                       str(ROOT / 'examples/local-inventory/should-not-exist.json')])


class InputTests(unittest.TestCase):
    def test_short_and_truncated_inputs_keep_fixed_hypothesis(self):
        tokenizer = CharacterTokenizer()
        suffix = '\nHypothesis: ' + HYPOTHESIS
        row = {'skill_text': 'Skill instructions.',
               'files': [{'path': 'helper.py', 'text': 'x' * 200}]}
        for mode in ('package', 'markdown'):
            for limit in (90, 4096):
                test_row = copy.deepcopy(row)
                if mode == 'markdown':
                    test_row['skill_text'] += 'a' * 200
                ids, original, coverage = encode_input(tokenizer, test_row, mode, limit)
                self.assertLessEqual(len(ids), limit)
                self.assertTrue(''.join(map(chr, ids)).endswith(suffix))
                self.assertEqual(coverage['truncated'], original > limit)
        _, _, coverage = encode_input(tokenizer, row, 'package', 90)
        self.assertEqual(coverage['files'][0]['status'], 'full')
        self.assertEqual(coverage['files'][1]['status'], 'omitted')
        with self.assertRaises(ValueError):
            encode_input(tokenizer, row, 'markdown', 10)


class DependencyTests(unittest.TestCase):
    def test_exact_pins_advisory_evidence_and_unsupported_inputs(self):
        row = {'files': [
            {'path': 'requirements.txt', 'text': 'Requests==2.31.0 # comment\nrequests==2.31.0\nnumpy>=2\n-r private.txt\nflask==2.0; python_version>"3"\n'},
            {'path': 'composer.json', 'text': '{}'}]}
        calls = []

        def lookup(name, version):
            calls.append((name, version))
            return [{'id': 'TEST-1'}, {'id': 'TEST-2', 'withdrawn': '2026-01-01'}]

        result = check_dependencies(row, lookup)
        self.assertEqual(calls, [('requests', '2.31.0')])
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(len(result['unresolved']), 4)
        self.assertEqual([f['advisory_id'] for f in result['findings']], ['TEST-1', 'TEST-1'])
        self.assertEqual([f['line'] for f in result['findings']], [1, 2])
        self.assertNotIn('malicious', result)

    def test_network_failure_is_not_a_clean_result(self):
        def failure(name, version):
            raise OSError('offline')
        row = {'files': [{'path': 'requirements.txt', 'text': 'requests==2.31.0'}]}
        result = check_dependencies(row, failure)
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(result['checked'], [])

    def test_no_manifest_is_explicit(self):
        self.assertEqual(check_dependencies({'files': []})['status'], 'not_applicable')


class ContributionTests(unittest.TestCase):
    def record(self):
        return contribute.make_record(ROOT / 'examples/local-inventory', 'benign',
                                      'Project-authored example.', license_name='Apache-2.0')

    def test_draft_is_not_a_verified_label(self):
        record = self.record()
        report = contribute.validate_record(record)
        self.assertTrue(report['valid_draft'])
        self.assertFalse(report['label_verified'])
        self.assertEqual(report['missing_provenance'], ['source_url', 'revision'])
        self.assertNotIn('text', record['files'][0])

    def test_inconsistent_inventory_and_bad_provenance_rejected(self):
        changes = [
            lambda r: r['files'][0].update(sha256='0' * 64),
            lambda r: r['files'][0].update(path='../outside'),
            lambda r: r.update(review_status='verified'),
            lambda r: r.update(revision='main'),
            lambda r: r.update(source_url='https://user:password@example.com/repo'),
            lambda r: r.update(raw_text='unexpected source contents'),
        ]
        for change in changes:
            record = self.record()
            change(record)
            with self.assertRaises(ValueError):
                contribute.validate_record(record)

    def test_documented_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'candidate.json'
            commands = [
                [str(ROOT / 'examples/local-inventory'), '--label', 'benign', '--evidence',
                 'Project-authored example.', '--license', 'Apache-2.0', '--output', str(output)],
                ['--check', str(output)],
            ]
            for args in commands:
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/contribute.py'), *args],
                                        cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(output.read_text())['review_status'], 'candidate')


if __name__ == '__main__':
    unittest.main()

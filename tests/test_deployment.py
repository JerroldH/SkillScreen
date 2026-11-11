"""Threshold loading and release identity checks, without GPU or network."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.model_score import HYPOTHESIS, load_release, select_threshold


class DeploymentTests(unittest.TestCase):
    def settings(self):
        raw = -4.441408183360328
        return {'repo_id': 'SkillScreen/SkillScreen-2B-Markdown', 'release': 'v0.1.0',
                'input_mode': 'markdown8k', 'hypothesis': HYPOTHESIS, 'default_policy': 'native',
                'thresholds': {'default': {'raw_threshold': 0., 'native_threshold': .5},
                               'high_recall': {'raw_threshold': raw, 'native_threshold': 1/(1+math.exp(-raw))}}}

    def test_threshold_policy_and_precision(self):
        settings = self.settings()
        self.assertEqual(select_threshold(settings)[1]['raw_threshold'], 0.)
        policy, gate = select_threshold(settings, 'high-recall')
        self.assertEqual(policy, 'high-recall')
        self.assertEqual(gate['raw_threshold'], -4.441408183360328)
        settings['default_policy'] = 'high-recall'
        self.assertEqual(select_threshold(settings)[1], gate)
        broken = copy.deepcopy(settings)
        broken['thresholds']['high_recall']['native_threshold'] = .5
        with self.assertRaisesRegex(ValueError, 'disagree'):
            select_threshold(broken)

    def test_local_release_rejects_wrong_input_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'deployment_settings.json').write_text(json.dumps(self.settings()), encoding='utf-8')
            source, options, settings, revision = load_release('markdown', tmp)
            self.assertTrue(options['local_files_only'])
            self.assertFalse(options['trust_remote_code'])
            self.assertEqual(revision, 'v0.1.0')
            with self.assertRaisesRegex(ValueError, 'different input mode'):
                load_release('package', tmp)


if __name__ == '__main__':
    unittest.main()

"""Regress the ambiguous defaults that previously selected a second scorer."""
import ast
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from score_worker import parse_args, beat

ROOT = Path(__file__).resolve().parent


class CurrentProtocolTests(unittest.TestCase):
    def args(self, dataset='BEAT2', *extra):
        return ['--dataset', dataset, '--manifest', 'unused', '--output', 'unused',
                '--split', 'test', '--arm', 'native_gt', *extra]

    def test_beat_default_is_accepted_speaker2_2024(self):
        args = parse_args(self.args())
        self.assertEqual(args.beat_profile, 'emage2024_bc_l1')
        self.assertEqual(args.membership_profile, 'emage_author_speaker2')
        rows = [dict(source_id='sample')]
        with patch('beat2024.run', return_value={'sentinel': 1}) as run:
            self.assertEqual(beat(args, rows), {'sentinel': 1})
            run.assert_called_once_with(args, rows)

    def test_previous_beat_selectors_and_aist_smpl_entry_are_rejected(self):
        for argv in (self.args('BEAT2','--beat-profile','public2025'),
                     self.args('BEAT2','--membership-profile','all_official_sources'),
                     self.args('AIST++')):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error: parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_other_datasets_keep_their_own_membership(self):
        for dataset in ('HumanML3D', 'FineDance'):
            args = parse_args(self.args(dataset))
            self.assertIsNone(args.beat_profile)
            self.assertEqual(args.membership_profile, 'all_official_sources')

    def test_aist_numeric_functions_unchanged_from_measured_source(self):
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        current = functions(ROOT/'aist_smplx.py')
        previous = functions(ROOT/'legacy/aist_smpl_control/aist_smplx.py.txt')
        for name in ('full_smplx_positions', 'smplx_positions', 'save_features'):
            self.assertEqual(current[name], previous[name])
        source = (ROOT/'aist_smplx.py').read_text()
        self.assertNotIn('native_aist_joints', source)
        self.assertIn("for domain in ('smplx',)", source)


if __name__ == '__main__':
    unittest.main()

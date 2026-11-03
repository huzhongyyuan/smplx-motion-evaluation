"""Audit archived repetitions, source cohorts and profile-to-result linkage."""
import json
import unittest
from pathlib import Path
import numpy as np
from gt import command
from source_lock import sha

ROOT = Path(__file__).resolve().parent


class GTResultsTests(unittest.TestCase):
    def test_all_four_recipes_link_exact_archives(self):
        index = json.loads((ROOT/'results/gt_index.json').read_text())
        self.assertEqual(len(index['recipes']), 4)
        self.assertEqual({r['dataset'] for r in index['recipes']}, {'HumanML3D','AIST++','FineDance','BEAT2'})
        for recipe in index['recipes']:
            self.assertEqual(sha(ROOT/recipe['result']), recipe['result_sha256'])
            self.assertTrue((ROOT/recipe['script']).is_file())

    def test_hml_final20_statistics_recompute(self):
        result = json.loads((ROOT/'results/hml_gt20.json').read_text())
        self.assertEqual(len(result['repetitions']), 20)
        self.assertEqual(len(result['inputs']), 4198)
        self.assertEqual(result['dataset_entries'], 4648)
        self.assertEqual(result['evaluated_per_replication'], 4640)
        for name, branches in result['metric_summary'].items():
            for branch, summary in branches.items():
                # Author Diversity operates on float32 activations; JSON numbers
                # alone do not preserve that dtype. Other summary arrays are64.
                values = np.asarray([rep[name][branch] for rep in result['repetitions']],
                    dtype=np.float32 if name == 'Diversity' else np.float64)
                np.testing.assert_allclose(values.mean(axis=0), summary['mean'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(1.96*values.std(axis=0)/np.sqrt(20),
                    summary['confidence_interval_95'], atol=1e-12, rtol=0)

    def test_lodge_gt202_is_not_loader_test19(self):
        members = json.loads((ROOT/'metadata/gt_eval_members.json').read_text())['FineDance']
        result = json.loads((ROOT/'results/finedance_lodge202_gt.json').read_text())
        self.assertEqual(len(members['published_gt202']), 202)
        self.assertEqual(len(members['test']), 19)
        self.assertEqual(result['reference_metadata_sha256'], sha(ROOT/'metadata/lodge_reference.json'))
        self.assertIsNone(result['paper_gt_fid'])
        self.assertEqual(round(result['metrics']['Divk'],2), 9.73)
        self.assertEqual(round(result['metrics']['Divg'],2), 7.44)
        self.assertIn('lodge-gt', command('FineDance', 'unused'))

    def test_beat_legacy_frames_and_display_recompute(self):
        result = json.loads((ROOT/'results/beat2024_speaker2_gt.json').read_text())
        members = json.loads((ROOT/'metadata/beat_speaker2.json').read_text())
        self.assertEqual([r['source_id'] for r in result['inputs']], members['test'])
        self.assertEqual(sum(r['frames'] for r in result['inputs']), 28680)
        count = sum(r['bc_frames'] for r in result['inputs'])
        self.assertEqual(count, 26880)
        bc = sum(r['BC']*r['bc_frames'] for r in result['inputs'])/count
        self.assertAlmostEqual(bc, result['metrics']['BC'], places=14)
        self.assertAlmostEqual(bc*10, result['display']['BC_times_10'], places=13)
        self.assertIsNone(result['FGD'])
        self.assertNotIn('--beat-profile', command('BEAT2','unused'))
        self.assertIn('emage_author_speaker2', command('BEAT2','unused'))

    def test_aist_test_and_val_roles_remain_separate(self):
        result = json.loads((ROOT/'results/aist_smplx30.json').read_text())
        self.assertEqual(len(result['sources']), 992)
        members = json.loads((ROOT/'metadata/gt_eval_members.json').read_text())['AIST++']
        self.assertEqual(len(members['test']), 20)
        self.assertEqual(len(members['supplement_eval40']), 40)
        self.assertTrue(set(members['test']).isdisjoint(members['val']))
        self.assertIn('smplx_test20_vs_train952', result['metrics'])
        self.assertIn('smplx_eval40_vs_train952', result['metrics'])

    def test_current_aist_scores_are_unchanged_smplx_branches_only(self):
        current = json.loads((ROOT/'results/aist_smplx30.json').read_text())
        archived = json.loads((ROOT/current['historical_comparison']).read_text())
        self.assertEqual(current['metrics'], {k:v for k,v in archived['metrics'].items() if k.startswith('smplx_')})
        self.assertEqual(current['sources'], archived['sources'])
        self.assertEqual(len(current['metrics']), 2)

    def test_legacy_mapping_preserves_hashes_and_replacements(self):
        manifest = json.loads((ROOT/'legacy/migration.json').read_text())
        for row in manifest['entries']:
            self.assertEqual(sha(ROOT/row['archived_path']), row['archived_sha256'])
            self.assertTrue((ROOT/row['replacement']).is_file())
        for name in ('beat_full_result.json','beat_speaker2_result.json','public_gt_migration.json'):
            self.assertFalse((ROOT/'results'/name).exists())


if __name__ == '__main__':
    unittest.main()

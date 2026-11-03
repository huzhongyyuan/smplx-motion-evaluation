"""Legacy support and observable author in-place behavior; requires fetched legacy assets."""
import unittest
import numpy as np
from beat2024 import ASSETS, old_support, score_positions, verify_profile
from evaluator_common import load_file


class Beat2024Tests(unittest.TestCase):
    def test_author_asset_identity(self):
        receipt = verify_profile()
        self.assertEqual(receipt['files']['mean_vel_smplxflame_30.npy']['bytes'], 348)

    def test_exact_seconds_and_chunk_support(self):
        self.assertEqual(old_support(299, 159999, 301), (270, 244))
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            old_support(120, 64000, 120)

    def test_old_l1_mutation_is_visible_to_bc(self):
        author = load_file('beat2024_test_author', ASSETS/'source/utils/metric.py')
        seen = []
        original = author.alignment.load_pose
        def capture(instance, pose, *args):
            seen.append(pose.copy())
            return original(instance, pose, *args)
        author.alignment.load_pose = capture
        frames = 244
        time = np.arange(frames, dtype=np.float32)/30
        positions = np.sin(time[:, None]*np.linspace(1, 9, 165, dtype=np.float32)[None])
        untouched = positions.copy()
        wave = np.zeros(160000, dtype=np.float32)
        for start in range(20000, len(wave)-1000, 8000):
            wave[start:start+500] = np.sin(np.arange(500)*.8)
        result = score_positions(author, author.L1div(), positions, wave, np.ones(55))
        expected = np.abs(untouched-untouched.mean(axis=0))
        np.testing.assert_array_equal(seen[0], expected)
        np.testing.assert_array_equal(positions, expected)
        self.assertAlmostEqual(result['L1div'], float(expected.sum()/frames), places=5)
        self.assertEqual(result['bc_frames'], 124)
        self.assertTrue(0 <= result['BC'] <= 1)

"""Legacy support and observable author in-place behavior; requires fetched legacy assets."""
import unittest
import numpy as np
from beat2024 import ASSETS, old_support, score_positions, verify_profile, corrected_motion_beats
from evaluator_common import load_file


class Beat2024Tests(unittest.TestCase):
    def test_author_asset_identity(self):
        receipt = verify_profile()
        self.assertEqual(receipt['files']['mean_vel_smplxflame_30.npy']['bytes'], 348)

    def test_exact_seconds_and_chunk_support(self):
        self.assertEqual(old_support(299, 159999, 301), (270, 244))
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            old_support(120, 64000, 120)

    def test_compatibility_bc_is_exact_but_caller_is_not_mutated(self):
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
        np.testing.assert_array_equal(positions, untouched)
        self.assertAlmostEqual(result['L1div'], float(expected.sum()/frames), places=5)
        self.assertEqual(result['bc_frames'], 124)
        self.assertTrue(0 <= result['BC'] <= 1)
        direct = author.alignment(.3, 7, mmae=np.ones(55))
        onset = direct.load_audio(wave[:int(16000/30*frames)], 32000, len(wave)-32000, True)
        old_beats = direct.load_pose(expected, 60, frames-60, 30, True)
        self.assertEqual(result['BC'], direct.calculate_align(onset, old_beats, 30))
        second = score_positions(author, author.L1div(), positions, wave, np.ones(55))
        self.assertEqual(result, second)

    def test_threshold_uses_the_cropped_minimum_frame(self):
        # Below-threshold motion before frame60; a valid trough at frame80.
        # Testing relative index40 against full-frame speeds would reject it.
        t = np.arange(160, dtype=float)
        x = np.zeros((160, 3))
        x[:, 0] = np.where(t <= 60, .1*t,
            6+(t-60)+((t-80)**3+8000)/300)/30
        beats = corrected_motion_beats(x, 40, 120, 30, np.ones(1))[0]
        np.testing.assert_array_equal(beats, [40])
        np.testing.assert_array_equal(corrected_motion_beats(x, 60, 120, 30, np.ones(1))[0], [20])

"""Physical invariants expose the upstream parentheses error without tuning data."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from corrected_kinetic import average_acceleration, features, prepare_joints, load_joints


class PhysicalKineticTests(unittest.TestCase):
    def test_static_nonzero_pose(self):
        x = np.full((20, 24, 3), 1.05)
        for fps in (30, 60):
            for i in (1, 5, 19):
                self.assertAlmostEqual(average_acceleration(x, i, 4, 2, 1/fps), 0)

    def test_constant_velocity_and_quadratic_motion(self):
        for fps in (30, 60):
            t = np.arange(30)/fps
            x = np.zeros((30, 1, 3)); x[:, 0, 0] = 1+2*t
            for i in (1, 10, 29):
                self.assertAlmostEqual(average_acceleration(x, i, 0, 2, 1/fps), 0, places=9)
            x[:, 0, 0] = 1+2*t+3*t*t
            for i in (1, 10, 29):
                self.assertAlmostEqual(average_acceleration(x, i, 0, 2, 1/fps), 6, places=9)
                self.assertAlmostEqual(average_acceleration(x+7, i, 0, 2, 1/fps), 6, places=8)

    def test_real_extractors_static_pose_and_translation_invariance(self):
        for dataset, count in (('FineDance', 22), ('AIST++', 24)):
            x = np.broadcast_to(np.arange(count*3).reshape(1, count, 3)/100, (12, count, 3)).copy()
            np.testing.assert_allclose(features(x, dataset), 0, atol=1e-7)
            x[:, 1, 0] += np.sin(np.arange(12)/5)
            np.testing.assert_allclose(features(x+9, dataset), features(x, dataset), atol=1e-6)

    def test_body_relative_convention_and_invalid_inputs(self):
        x = np.ones((4, 22, 3)); x[:, 0, 0] = np.arange(4)
        y = prepare_joints(x, 'FineDance')
        np.testing.assert_array_equal(y[:, 1, 0], 1-np.arange(4))
        for x in (np.zeros((2, 22, 3)), np.zeros((4, 24, 3)), np.full((4, 22, 3), np.nan)):
            with self.assertRaises(ValueError): features(x, 'FineDance')

    def test_old_features_and_wrong_fps_are_not_motion_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cache = folder/'old_feature.npy'
            np.save(cache, np.zeros(66))
            with self.assertRaisesRegex(ValueError, 'Expected finite'):
                features(load_joints(cache, 'FineDance', 1024), 'FineDance')
            motion = folder/'aist.npz'
            joints = np.zeros((6, 24, 3))
            for metadata in ({}, {'fps': 60}):
                np.savez(motion, joints24=joints, **metadata)
                with self.assertRaisesRegex(ValueError, 'explicit 30fps'):
                    load_joints(motion, 'AIST++', 600)
            np.savez(motion, joints24=joints, fps=30)
            np.testing.assert_array_equal(load_joints(motion, 'AIST++', 4), joints[:4])


if __name__ == '__main__': unittest.main()

"""Boundary checks for clocks, rotations, hands and real target preprocessing."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from contracts import rotations, resample_motion, sample_at, merge_body_hands, smplx_shape

class BoundaryTests(unittest.TestCase):
    def test_shortest_arc_and_linear_world_displacement(self):
        # Midpoint from+179deg to-179deg is180deg, never0deg.
        rr = Rotation.from_euler('y', [179, -179], degrees=True).as_matrix()[:, None]
        tt = np.array([[3., 2., -5.], [5., 4., -1.]])
        r, t, clock = resample_motion(rr, tt, 20, 40)
        np.testing.assert_allclose(Rotation.from_matrix(r[1, 0]).apply([0, 0, 1]), [0, 0, -1], atol=1e-8)
        np.testing.assert_allclose(t[1], [4, 3, -3]); np.testing.assert_allclose(clock, [0, .025, .05])

    def test_no_endpoint_stretch_and_no_extrapolation(self):
        r = np.tile(np.eye(3), (180, 1, 1, 1)); t = np.zeros((180, 3))
        _, _, clock = resample_motion(r, t, 20, 30)
        self.assertEqual(len(clock), 269)
        self.assertLess(clock[-1], 179/20)
        with self.assertRaisesRegex(ValueError, 'Missing real endpoint'):
            sample_at(r, t, 20, 0, np.arange(181)/20)

    def test_reject_reflection(self):
        r = np.tile(np.eye(3), (2, 1, 1, 1)); r[:, :, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'Reflected'): rotations(r, 1)

    def test_absolute_clock(self):
        r = np.tile(np.eye(3), (31, 1, 1, 1)); t = np.arange(31)[:, None]*np.array([[1., 2., -3.]])/30
        _, out = sample_at(r, t, 30, 22, [22, 22.5, 23])
        np.testing.assert_allclose(out[1], [.5, 1, -1.5])

    def test_body_only_fingers_default_wrists_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); poses = np.ones((3, 165), dtype=np.float32)
            np.savez(p/'body.npz', poses=poses, trans=np.zeros((3, 3)), fps=30)
            merge_body_hands(p/'body.npz', None, p/'out.npz', 'FineDance', 'generated', '001')
            with np.load(p/'out.npz') as z:
                np.testing.assert_array_equal(z['poses'][:, :66], poses[:, :66])
                self.assertEqual(np.count_nonzero(z['poses'][:, 66:]), 0)

    def test_reject_wrong_hand_arm_and_source(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); np.savez(p/'body.npz', poses=np.zeros((3, 165)), trans=np.zeros((3, 3)), fps=30)
            common = dict(left_hand_pose=np.zeros((3, 45)), right_hand_pose=np.zeros((3, 45)), fps=30, time_origin_s=0)
            np.savez(p/'hand.npz', **common, arm='native_gt', source_id='a')
            with self.assertRaisesRegex(ValueError, 'arm mismatch'):
                merge_body_hands(p/'body.npz', p/'hand.npz', p/'out.npz', 'BEAT2', 'generated', 'a')
            np.savez(p/'other.npz', **common, arm='generated', source_id='b')
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                merge_body_hands(p/'body.npz', p/'other.npz', p/'out.npz', 'BEAT2', 'generated', 'a')

    def test_packed_converter_shape_is_explicit_and_not30_betas(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'shape.npz'
            np.savez(p, betas=np.arange(30, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'Ambiguous betas'):
                smplx_shape(p)
            beta, expression = smplx_shape(p, 'betas20_expression10_concatenated')
            np.testing.assert_array_equal(beta, np.arange(20))
            np.testing.assert_array_equal(expression, np.arange(20, 30))

    def test_world_fixed_child_not_rotated_by_ancestor_factorization(self):
        # Two moving ancestors with a world-fixed descendant. Independent
        # local geodesics need not compose to the global geodesic.
        r = np.tile(np.eye(3), (2, 3, 1, 1))
        r[1,0] = Rotation.from_euler('x', 120, degrees=True).as_matrix()
        r[1,1] = Rotation.from_euler('y', 120, degrees=True).as_matrix()
        r[1,2] = (r[1,0]@r[1,1]).T
        world, _ = sample_at(r, np.zeros((2,3)), 1, 0, [0,.5,1], parents=[-1,0,1])
        local, _ = sample_at(r, np.zeros((2,3)), 1, 0, [0,.5,1])
        np.testing.assert_allclose(world[1,0]@world[1,1]@world[1,2], np.eye(3), atol=1e-12)
        self.assertGreater(np.linalg.norm(local[1,0]@local[1,1]@local[1,2]-np.eye(3)), .1)
        np.testing.assert_allclose(world[[0,2]], r, atol=1e-12)

if __name__ == '__main__': unittest.main()

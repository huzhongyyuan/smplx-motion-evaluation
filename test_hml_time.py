import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from hml_time import transport,roundtrip20_30_20

class TimeTests(unittest.TestCase):
    def source(self):
        x=np.zeros((81,263),dtype=np.float64)
        m=Rotation.from_euler('y',np.linspace(179,181,81),degrees=True).as_matrix()
        six=np.concatenate([m[...,0],m[...,1]],axis=-1)
        x[:,67:193]=np.tile(six[:,None,:],(1,21,1)).reshape(81,126)
        x[:,1]=.05; x[:,193]=.1; x[:,259:]=1
        return x
    def test_units_rotations_and_common_times(self):
        x=self.source(); up,t=transport(x,20,30)
        self.assertEqual(len(up),121); self.assertAlmostEqual(t[-1],4)
        np.testing.assert_allclose(up[:,1],1/30)
        np.testing.assert_allclose(up[:,193],2/30)
        np.testing.assert_allclose(up[::3,67:193],x[::2,67:193],atol=1e-12)
        self.assertGreater(abs(up[1,67]),.99)
        _,back,original=roundtrip20_30_20(x)
        np.testing.assert_allclose(back,original,atol=1e-12)
    def test_no_tail_fill(self):
        x=self.source()[:-1]
        up,back,original=roundtrip20_30_20(x)
        self.assertEqual(len(up),119)
        self.assertEqual(len(back),79)
        self.assertEqual(len(original),79)

if __name__=='__main__':unittest.main()

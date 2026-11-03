"""Prevent fps mismatches and restoration of interpolation at AIST evaluation."""
import unittest
import numpy as np
from aist30 import extractors, protocol, select_export_frames, validate_reference


class Aist30Tests(unittest.TestCase):
    def test_linear_motion_uses_seconds(self):
        kinetic,_=extractors()
        # Constant1m/s horizontal,2m/s vertical. Kinetic entries are v^2.
        x=np.zeros((61,24,3)); x[:,:,0]=np.arange(61)[:,None]/30
        x[:,:,1]=2*np.arange(61)[:,None]/30
        result=kinetic(x).reshape(24,3)
        np.testing.assert_allclose(result[:,0],1,rtol=0,atol=1e-6)
        np.testing.assert_allclose(result[:,1],4,rtol=0,atol=1e-6)

    def test_manual_all_velocity_functions_use_seconds(self):
        _,extract=extractors()
        cls=extract.__globals__['ManualFeatures']
        u=cls.f_fast.__globals__['feat_utils']
        a=np.zeros(3); x=np.array([1.,0,0]); y=np.array([0.,1,0]); z=np.array([0.,0,1])
        self.assertFalse(u.velocity_above_threshold(x/30,a,1.5))
        self.assertTrue(u.velocity_above_threshold(x/30,a,.5))
        self.assertFalse(u.velocity_direction_above_threshold(a,a,x,x,x/30,a,1.5))
        self.assertTrue(u.velocity_direction_above_threshold(a,a,x,x,x/30,a,.5))
        self.assertFalse(u.velocity_direction_above_threshold_normal(a,a,y,x,z/30,a,1.5))
        self.assertTrue(u.velocity_direction_above_threshold_normal(a,a,y,x,z/30,a,.5))

    def test_upstream_defaults_unchanged(self):
        import importlib
        extractors()
        k=importlib.import_module('_soma_eval_bailando_features.kinetic')
        m=importlib.import_module('_soma_eval_bailando_features.manual_new')
        self.assertEqual(k.KineticFeatures(np.zeros((3,24,3))).frame_time,1/60)
        self.assertEqual(m.feat_utils.velocity_above_threshold.__defaults__,(1/120,))

    def test_clock_selection_is_exact_and_has_no_endpoint_fill(self):
        # Arbitrary values remain byte-identical, including extreme rotations.
        r=np.arange(61*55*9,dtype=np.float64).reshape(61,55,3,3)
        t=np.arange(61*3,dtype=np.float64).reshape(61,3)
        rr,tt=select_export_frames(r,t,-1/30,60)
        np.testing.assert_array_equal(rr,r[1:]); np.testing.assert_array_equal(tt,t[1:])
        for origin,count in ((0,62),(1/30,60),(-.01,60)):
            with self.assertRaises(ValueError): select_export_frames(r,t,origin,count)

    def test_wrong_reference_is_rejected(self):
        validate_reference({'feature_protocol':protocol()})
        for value in ({},{'feature_protocol':{'fps':60}},
                      {'feature_protocol':dict(protocol(),manual_time_per_frame_s=1/120)}):
            with self.assertRaises(ValueError): validate_reference(value)


if __name__=='__main__': unittest.main()

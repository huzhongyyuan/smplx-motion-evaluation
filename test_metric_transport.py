"""Regression for the exact official AIST full-TEST sqrtm scalar that failed JSON."""
import json
import unittest
import numpy as np
from evaluator_common import json_value
from report import numeric


class MetricTransport(unittest.TestCase):
    def test_official_complex_is_lossless(self):
        value=np.complex128(21.18015233987407-1.5233360564733955e-6j)
        encoded=json_value({'fid_k':value})
        recovered=numeric(json.loads(json.dumps(encoded))['fid_k'])
        self.assertEqual(recovered,value)
        self.assertEqual(encoded['fid_k']['imag'],value.imag)

    def test_real_metrics_stay_real(self):
        self.assertEqual(json_value({'score':np.float64(1.25)}),{'score':1.25})

    def test_nonfinite_complex_is_rejected(self):
        for value in (complex(float('nan'),1),complex(1,float('inf'))):
            with self.assertRaises(ValueError):
                json_value(value)


if __name__=='__main__':
    unittest.main()

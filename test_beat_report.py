"""Caller and temporal support are part of a BEAT result identity."""
import copy
import unittest
from report import identity


class BeatReportTests(unittest.TestCase):
    def test_profiles_and_support_must_not_pair(self):
        public = dict(dataset='BEAT2', split='test', beat_profile='public2025',
                      inputs=[dict(source_id='sample', native_sha256='pose', audio_sha256='audio',
                                   frames=300, bc_frames=180)])
        legacy = copy.deepcopy(public)
        legacy['beat_profile'] = 'emage2024_bc_l1'
        self.assertNotEqual(identity(public), identity(legacy))
        shorter = copy.deepcopy(public)
        shorter['inputs'][0].update(frames=244, bc_frames=124)
        self.assertNotEqual(identity(public), identity(shorter))
        reconstructed = copy.deepcopy(public)
        reconstructed['inputs'][0]['prediction_sha256'] = 'different_prediction'
        self.assertEqual(identity(public), identity(reconstructed))

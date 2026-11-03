"""Caller and temporal support are part of a BEAT result identity."""
import copy
import unittest
from report import identity


class BeatReportTests(unittest.TestCase):
    def test_previous_or_missing_profiles_are_rejected(self):
        public = dict(dataset='BEAT2', split='test', beat_profile='public2025',
                      inputs=[dict(source_id='sample', native_sha256='pose', audio_sha256='audio',
                                   frames=300, bc_frames=180)])
        with self.assertRaises(ValueError): identity(public)
        del public['beat_profile']
        with self.assertRaises(ValueError): identity(public)

    def test_support_must_match_and_predictions_can_pair(self):
        current = dict(dataset='BEAT2', split='test', beat_profile='emage2024_bc_l1',
                       inputs=[dict(source_id='sample', native_sha256='pose', audio_sha256='audio',
                                    frames=300, bc_frames=180)])
        shorter = copy.deepcopy(current)
        shorter['inputs'][0].update(frames=244, bc_frames=124)
        self.assertNotEqual(identity(current), identity(shorter))
        reconstructed = copy.deepcopy(current)
        reconstructed['inputs'][0]['prediction_sha256'] = 'different_prediction'
        self.assertEqual(identity(current), identity(reconstructed))

"""Regression for manifest-authoritative native inputs; no score emulation."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from hml_sources import stage_hml_sources
from source_lock import sha


class NativeSources(unittest.TestCase):
    def test_manifest_source_and_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); original = base/'author.npy'
            value = np.arange(40*263, dtype=np.float64).reshape(40,263)
            np.save(original, value)
            row = dict(dataset='HumanML3D', source_id='007975', native_path=str(original), native_sha256=sha(original))
            staged = stage_hml_sources([row], base/'staged')
            self.assertEqual((staged/'007975.npy').resolve(), original)
            self.assertTrue(np.array_equal(np.load(staged/'007975.npy'), value))
            row['native_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                stage_hml_sources([row], base/'stale')

    def test_nonfinite_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); path = base/'bad.npy'
            np.save(path, np.full((84,263), np.nan))
            with self.assertRaisesRegex(ValueError, 'invalid native'):
                stage_hml_sources([dict(dataset='HumanML3D', source_id='007975', native_path=str(path))], base/'staged')


if __name__ == '__main__':
    unittest.main()

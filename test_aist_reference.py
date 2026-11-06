"""Reject model/time/member corruption instead of silently switching pools."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from aist_reference import verify_reference, PROTOCOL_ID
from aist30 import protocol
from source_lock import sha


class ReferenceTests(unittest.TestCase):
    def test_real_cache_and_identity(self):
        root, receipt = verify_reference()
        self.assertEqual(receipt['samples'],1363)
        self.assertEqual(len(receipt['files']),2726)
        self.assertEqual(receipt['domain'],'SMPL-X')
        self.assertEqual(receipt['feature_protocol'],protocol())

    def test_reject_wrong_time_model_and_incomplete_members(self):
        # Small invalid fixtures fail before any large feature scan.
        for change in ({'samples':952},{'domain':'SMPL'},{'feature_protocol':{'fps':60}}):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d); (root/'metadata').mkdir()
                r=dict(protocol_id=PROTOCOL_ID,samples=1363,domain='SMPL-X',feature_protocol=protocol())
                r.update(change)
                receipt=root/'reference_receipt.json'; receipt.write_text(json.dumps(r))
                (root/'metadata/aist_reference1363.json').write_text(json.dumps(
                    dict(receipt_sha256=sha(receipt),source_ids=[])))
                with patch('aist_reference.ROOT',root),self.assertRaises(ValueError): verify_reference(root)

    def test_changed_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'metadata').mkdir()
            (root/'reference_receipt.json').write_text('{}')
            (root/'metadata/aist_reference1363.json').write_text(json.dumps(dict(receipt_sha256='0'*64)))
            with patch('aist_reference.ROOT',root),self.assertRaises(ValueError): verify_reference(root)


if __name__=='__main__': unittest.main()

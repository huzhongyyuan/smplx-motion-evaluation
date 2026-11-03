"""Public membership coverage and profile distinctions, with no data assets."""
from collections import Counter
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent

class PublicMetadataTests(unittest.TestCase):
    def setUp(self):
        self.rows=[json.loads(s) for s in (ROOT/'metadata/members.jsonl').read_text().splitlines()]
    def test_exact_coverage_no_duplicates_or_private_paths(self):
        self.assertEqual(len(self.rows),30582)
        self.assertEqual(len({(r['dataset'],r['source_id']) for r in self.rows}),30582)
        self.assertEqual({ds:dict(Counter(r['split'] for r in self.rows if r['dataset']==ds)) for ds in ('HumanML3D','AIST++','FineDance','BEAT2')},
            dict(HumanML3D=dict(train=22326,val=1400,test=4198),**{'AIST++':dict(train=952,val=20,test=20),'FineDance':dict(train=183,test=19),'BEAT2':dict(train=1093,val=106,test=265)}))
        for r in self.rows:
            self.assertFalse(any('path' in k for k in r))
            if r['dataset']=='HumanML3D':self.assertEqual(len(r['native_sha256']),64)
    def test_speaker2_roles_are_original_roles(self):
        profile=json.loads((ROOT/'metadata/beat_speaker2.json').read_text())
        self.assertEqual({k:len(v) for k,v in profile.items()},dict(train=95,val=8,test=15))
        by_id={r['source_id']:r['split'] for r in self.rows if r['dataset']=='BEAT2'}
        for role,ids in profile.items():
            for sid in ids:self.assertEqual(by_id[sid],role)
    def test_lodge_loader19_and_inference20_are_explicit(self):
        infer=json.loads((ROOT/'metadata/lodge_inference20.json').read_text())
        loader={r['source_id'] for r in self.rows if r['dataset']=='FineDance' and r['split']=='test'}
        self.assertEqual(set(infer['source_ids'])-loader,{'120'})
        self.assertEqual(loader-set(infer['source_ids']),set())
        self.assertFalse(infer['actual_paper_output_membership_verified'])

if __name__=='__main__':unittest.main()

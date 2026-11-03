"""Independent small real-input control against the previously verified AIST exports."""
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluator_common import setup,write_json
from aist_smplx import smplx_positions,smplx_neutral,features

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exports',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    setup();model=smplx_neutral();records=[]
    files=[p for p in sorted(a.exports.glob('*.npz')) if (a.features/'kinetic_features'/(p.stem+'.npy')).is_file()][:3]
    if len(files)!=3:raise ValueError('Need three real frozen AIST exports')
    for path in files:
        with np.load(path,allow_pickle=False) as z:
            actual,error=smplx_positions(model,z['poses'],z['trans'])
            np.testing.assert_allclose(actual,z['joints24'],atol=1e-6,rtol=0)
            for kind,value in zip(('kinetic_features','manual_features_new'),features(actual)):
                expected=np.load(a.features/kind/(path.stem+'.npy'),allow_pickle=False)
                np.testing.assert_array_equal(value,expected)
            records.append(dict(source_id=path.stem,frames=len(actual),max_full_model_error_m=error,features_exactly_equal=True))
    write_json(a.output,dict(status='REAL3_GEOMETRY_AND_FEATURE_CONTROL_PASSED',sources=records,scope='small migration control, not a new992-source evaluation'))

if __name__=='__main__':main()

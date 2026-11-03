"""Physical kinetic diagnostic for dance; never reuse author feature caches.

The author's FID and distance functions are reused, but both distributions are
re-extracted with seconds-based derivatives. This is NOT a paper-compatible FID.
"""
import argparse
import functools
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np

PROFILE = 'physical_kinetic30_v1'
ISSUE = 'author_acceleration_missing_parentheses'


def average_acceleration(positions, i, joint, window, dt):
    """Mean centered second differences, using the author's boundary support."""
    indices = np.arange(max(1, i-window), min(len(positions)-1, i+window+1))
    if not len(indices):
        raise ValueError('Acceleration needs at least three frames')
    acceleration = (positions[indices+1, joint] - 2*positions[indices, joint]
                    + positions[indices-1, joint]) / (dt*dt)
    return np.linalg.norm(acceleration.mean(axis=0))


@functools.lru_cache(maxsize=2)
def extractor(dataset):
    from evaluator_common import UP
    folder = UP/('LODGE/metric/features' if dataset == 'FineDance' else 'Bailando/utils/features')
    package_name = '_physical_kinetic_'+('fd' if dataset == 'FineDance' else 'aist')
    package = types.ModuleType(package_name)
    package.__path__ = [str(folder)]
    sys.modules[package_name] = package
    author = importlib.import_module(package_name+'.kinetic')

    class PhysicalKinetic(author.KineticFeatures):
        def average_energy_expenditure(self, joint):
            return np.mean([average_acceleration(self.positions, i, joint,
                           self.sliding_window, self.frame_time)
                           for i in range(1, len(self.positions))])

    function = author.extract_kinetic_features
    context = dict(function.__globals__)
    context['KineticFeatures'] = functools.partial(PhysicalKinetic, frame_time=1/30)
    return types.FunctionType(function.__code__, context, function.__name__)


def prepare_joints(joints, dataset):
    count = 22 if dataset == 'FineDance' else 24
    x = np.asarray(joints, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (count, 3) or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError(f'Expected finite T>=3,{count},3 joints in metres, Y-up')
    x = x-x[:1, :1]
    if dataset == 'FineDance':
        x[:, 1:] -= x[:, :1]  # Same body-relative convention as LODGE.
    return x


def features(joints, dataset):
    if dataset not in ('FineDance', 'AIST++'):
        raise ValueError('Only FineDance and AIST++ use this kinetic diagnostic')
    value = extractor(dataset)(prepare_joints(joints, dataset))
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite corrected kinetic feature')
    return value


def input_files(folder):
    paths = sorted([*folder.glob('*.npy'), *folder.glob('*.npz')])
    if len(paths) < 2 or len({p.stem for p in paths}) != len(paths):
        raise ValueError('Need at least two unique motion IDs; pass motions, not feature caches')
    return paths


def load_joints(path, dataset, cap, model=None):
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            if 'fps' not in z or float(z['fps']) != 30:
                raise ValueError(f'{path.name}: expected explicit 30fps')
            value = z['joints24'].copy()
    else:
        value = np.load(path, allow_pickle=False)
    if cap is not None:
        value = value[:cap]
    if dataset == 'FineDance' and value.ndim == 2:
        import torch
        from dld.data.render_joints.smplfk import do_smplxfk
        if value.shape[1] not in (135, 139, 315, 319) or not np.isfinite(value).all():
            raise ValueError('FineDance requires already preprocessed LODGE motion139/319 or joints22')
        body = value[:, :139] if value.shape[1] in (139, 319) else np.concatenate(
            [np.zeros((len(value), 4)), value[:, :135]], axis=1)
        with torch.no_grad():
            value = do_smplxfk(torch.as_tensor(body, dtype=torch.float32), model)[:, :22].cpu().numpy()
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=('FineDance', 'AIST++'))
    p.add_argument('--predictions', required=True, type=Path,
                   help='Y-up/metre joints .npy, AIST joints24 .npz, or preprocessed LODGE139/319 .npy')
    p.add_argument('--reference', required=True, type=Path, help='Raw motions/joints; old feature pools are rejected')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    from evaluator_common import UP, setup, load_file, write_json
    from source_lock import sha
    setup()
    predictions, references = input_files(a.predictions), input_files(a.reference)
    out = a.output.resolve()
    if out.exists():
        raise ValueError('Use a new output directory')
    model = None
    if a.dataset == 'FineDance':
        sys.path[:0] = [str(UP/'LODGE/metric'), str(UP/'LODGE')]
        from dld.data.render_joints.smplfk import SMPLX_Skeleton
        model = SMPLX_Skeleton(Jpath=str(UP/'LODGE/data/smplx_neu_J_1.npy'))
        metric_path = UP/'LODGE/metric/metrics_finedance.py'
    else:
        sys.path.insert(0, str(UP/'Bailando/utils'))
        metric_path = UP/'Bailando/utils/metrics_new.py'
    metric = load_file('physical_kinetic_author_fid', metric_path)
    protocol = dict(id=PROFILE, dataset=a.dataset, fps=30, frame_time_s=1/30,
        acceleration='centered second difference / dt**2', sliding_half_window_frames=2,
        reference_reextracted=True, official_paper_comparable=False,
        coordinate_convention='metres, Y-up; LODGE body-relative or AIST first-root-relative',
        time_support='FineDance: already cropped, max1024 both sides; AIST: max600 prediction, full reference',
        corrected_code_sha256=sha(__file__), author_metric_sha256=sha(metric_path))
    out.mkdir(parents=True)
    write_json(out/'run.json', dict(status='RUNNING', protocol=protocol))
    arrays, records = {}, []
    for role, paths in (('reference', references), ('prediction', predictions)):
        target = out/role/'kinetic_features'; target.mkdir(parents=True)
        arrays[role] = []
        cap = 1024 if a.dataset == 'FineDance' else (600 if role == 'prediction' else None)
        for path in paths:
            x = load_joints(path, a.dataset, cap, model)
            value = features(x, a.dataset)
            np.save(target/(path.stem+'.npy'), value)
            arrays[role].append(value)
            records.append(dict(role=role, source_id=path.stem, input_sha256=sha(path),
                frames=len(x), feature_sha256=sha(target/(path.stem+'.npy'))))
    ref, pred = metric.normalize(np.stack(arrays['reference']), np.stack(arrays['prediction']))
    scores = dict(FIDk_corrected=metric.calc_fid(pred, ref),
                  Divk_corrected=metric.calculate_avg_distance(pred),
                  reference_Divk_corrected=metric.calculate_avg_distance(ref))
    result = dict(status='SCORER_RUN_FINISHED', protocol=protocol, metrics=scores,
        prediction_count=len(predictions), reference_count=len(references), inputs=records,
        acceptance='Diagnostic only; no claim that SOMA conversion is repaired or paper GT reproduced')
    write_json(out/'result.json', result)
    write_json(out/'run.json', dict(status='SCORER_RUN_FINISHED', protocol=protocol,
                                   result_sha256=sha(out/'result.json')))
    print(json.dumps(dict(status='SCORER_RUN_FINISHED', result=str(out/'result.json'))))


if __name__ == '__main__':
    main()

"""Pinned EMAGE2024 BC/L1 caller; author metric source stays unmodified."""
import importlib
import json
import sys
import types

import librosa
import numpy as np
import torch

from contracts import load_smplx, require
from evaluator_common import ROOT, UP, cwd, load_file, write_json
from source_lock import sha

PROFILE = 'emage2024_bc_l1'
ASSETS = ROOT/'assets/emage2024_v2'
METRIC_SHA = 'ab2871867ab5f56eacd8e56b3fa39c488ccaff3df9f49ec8d673c03c3b30e8c5'
NORMALIZER_SHA = '53b5e48f2a7bf78c41a6de6395d6bb4f29018465ca5d0ee2820a2be3eebb7137'


def verify_profile():
    receipt = json.loads((ASSETS/'receipt.json').read_text())
    require(receipt['status'] == 'AUTHOR_LEGACY_ASSETS_VERIFIED', 'Unverified legacy assets')
    require(receipt['source_revision'] == '6ca70b9541285b124da2eeedcd80f7c5a54eb111', 'Wrong old caller revision')
    for name, item in receipt['files'].items():
        require(sha(ASSETS/name) == item['sha256'], 'Legacy source/asset changed: '+name)
    require(sha(ASSETS/'source/utils/metric.py') == METRIC_SHA, 'Historical author metric changed')
    require(sha(ASSETS/'mean_vel_smplxflame_30.npy') == NORMALIZER_SHA, 'Unexpected legacy normalizer')
    return receipt


def old_support(pose_frames, audio_samples, face_frames):
    """Original loader seconds intersection and generator64/overlap4 support."""
    require(min(pose_frames, audio_samples, face_frames) > 0, 'Empty old-caller source')
    loader = min(pose_frames//30, audio_samples//16000, face_frames//30)*30
    n8 = loader-loader%8
    generated = 4+((n8-4)//60)*60
    require(120 < generated <= pose_frames, 'Insufficient EMAGE2024 BC time support')
    return loader, generated


def score_positions(metric, l1, position, wave, normalizer):
    """Preserve the old shared-array order, including L1's in-place mutation."""
    n = len(position)
    before_sum, before_count = l1.sum, l1.counter
    l1.run(position)
    bc = metric.alignment(.3, 7, mmae=normalizer)
    onset = bc.load_audio(wave[:int(16000/30*n)], 32000, len(wave)-32000, True)
    require(len(onset) > 0, 'No historical BC speech onsets')
    beats = bc.load_pose(position, 60, n-60, 30, True)
    value = bc.calculate_align(onset, beats, 30)
    require(np.isfinite(value) and 0 <= value <= 1, 'Invalid historical raw BC')
    return dict(BC=float(value), L1div=float((l1.sum-before_sum)/(l1.counter-before_count)),
                frames=n, bc_frames=n-120, audio_onsets=len(onset),
                motion_beats=sum(len(b) for b in beats))


def run(args, rows):
    receipt = verify_profile()
    metric = load_file('emage2024_frozen_metric', ASSETS/'source/utils/metric.py')
    sys.path.insert(0, str(UP))
    package = types.ModuleType('emage2024_official_geometry')
    package.__path__ = [str(UP/'EMAGE_caller')]
    sys.modules[package.__name__] = package
    with cwd(ROOT/'assets'):
        geometry = importlib.import_module(package.__name__+'.motion_rep_transfer')
    model = geometry.smplx_model
    normalizer = np.load(ASSETS/'mean_vel_smplxflame_30.npy', allow_pickle=False)
    l1 = metric.L1div()
    bc_sum, bc_count, cases = 0., 0, []
    for row in rows:
        gt, _, betas = load_smplx(row['native_path'], 30)
        require(betas is not None and np.asarray(betas).shape == (300,), 'Legacy caller requires native300 betas')
        pred = gt.copy() if args.arm == 'native_gt' else load_smplx(row['prediction_path'], 30)[0]
        require(pred.shape == gt.shape, 'Prediction must cover the full native timeline before old-caller cropping')
        wave, sr = librosa.load(row['audio_path'])
        wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
        with np.load(row['native_path'], allow_pickle=False) as raw:
            require('expressions' in raw, 'Missing old-loader face clock; never guess support')
            face_frames = len(raw['expressions'])
        loader, n = old_support(len(gt), len(wave), face_frames)
        full_length = len(pred)
        boundaries = list(range(0, full_length, 64))
        if full_length-boundaries[-1] == 1:
            boundaries.pop()
        boundaries.append(full_length)
        position = []
        with torch.no_grad():
            for a, b in zip(boundaries[:-1], boundaries[1:]):
                pose = torch.as_tensor(pred[a:b], dtype=torch.float32)
                batch = len(pose)
                value = model(betas=torch.as_tensor(betas, dtype=torch.float32)[None].repeat(batch, 1),
                    transl=torch.zeros(batch, 3), expression=torch.zeros(batch, 100),
                    global_orient=pose[:, :3], body_pose=pose[:, 3:66], jaw_pose=pose[:, 66:69],
                    leye_pose=pose[:, 69:72], reye_pose=pose[:, 72:75],
                    left_hand_pose=pose[:, 75:120], right_hand_pose=pose[:, 120:165]).joints[:, :55]
                position.append(value.numpy())
        position = np.concatenate(position).reshape(full_length, 165)[:n].copy()
        scores = score_positions(metric, l1, position, wave, normalizer)
        bc_sum += scores['BC']*scores['bc_frames']
        bc_count += scores['bc_frames']
        cases.append(dict(source_id=row['source_id'], native_frames=len(gt), old_loader_frames=loader,
            native_sha256=sha(row['native_path']), audio_sha256=sha(row['audio_path']),
            prediction_sha256=None if args.arm == 'native_gt' else sha(row['prediction_path']), **scores))
        write_json(args.output/'progress.json', dict(completed=len(cases), total=len(rows), inputs=cases))
        print(json.dumps(dict(dataset='BEAT2', profile=PROFILE, completed=len(cases), total=len(rows))), flush=True)
    raw_bc = bc_sum/bc_count
    return dict(metrics=dict(BC=raw_bc, L1div=l1.avg()), inputs=cases,
        beat_profile=PROFILE, motion_frames=l1.counter, bc_frames=bc_count,
        profile_receipt_sha256=sha(ASSETS/'receipt.json'), asset_provenance=receipt,
        display=dict(BC_times_10=10*raw_bc, convention='10^-1 display: raw BC multiplied by10'),
        FGD=None, FGD_reason='The current frozen protocol implements BC/L1 only; previous FGD results are archived in legacy/.',
        evaluated_joints='55 local SMPL-X joints; body+hands; no separate face metric or GT filling',
        paper_reproduction=False,
        scope='Pinned historical caller compatibility; accepted approximate reproduction, not exact paper identity',
        historical_behavior='L1div mutates shared positions before BC, as in the2024 caller; no score-based tuning')

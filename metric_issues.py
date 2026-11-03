"""Scientific limitations of frozen author scores, not silent formula changes."""


def known_issues(dataset):
    if dataset in ('FineDance', 'AIST++'):
        return [dict(id='author_acceleration_missing_parentheses',
            scope='kinetic energy-expenditure channel',
            consequence='Nonzero static joint positions produce nonzero acceleration features; FIDk is not a pure jitter measure',
            corrected_entry='evaluate.py corrected-kinetic',
            old_reference_features_compatible=False,
            paper_compatibility='Author scores are retained only for explicit paper comparison; corrected scores require both pools re-extracted')]
    if dataset == 'BEAT2':
        return [dict(id='emage2024_l1_mutation_and_threshold_frame_offset',
            consequence='Historical BC uses absolute mean deviations and checks cropped beat indices against full-motion thresholds',
            corrected_diagnostic='BC_unmutated_aligned_threshold',
            paper_compatibility='Historical BC/L1 are computed on a private copy; independent diagnostic is not a paper score')]
    return []

"""Small analytical checks; run in the existing genome environment on P8."""
import itertools
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import run_postprocess as analysis


def main():
    row = dict(method='ASEM', reference='a', target='b', depth=1, replicate=0,
               status='completed', runtime_s=1, theta_len=9, c_theta=8, w_theta=1,
               u_t=1, precision=8/9, recall=8/9, f1=8/9, identity_pct=80)
    analysis.validate_scores(pd.DataFrame([row]), {'b': 10})
    wrong = dict(row, f1=.1)
    try:
        analysis.validate_scores(pd.DataFrame([wrong]), {'b': 10})
    except AssertionError:
        pass
    else:
        raise AssertionError('inconsistent F1 accepted')
    missing = dict(row, status='no_output', **{k: np.nan for k in analysis.METRICS})
    frame = pd.DataFrame([row, missing])
    analysis.validate_scores(frame, {'b': 10})
    summary = analysis.summarise(frame, ['method']).iloc[0]
    assert summary.completed == 1 and summary.total == 2
    assert np.isclose(summary.f1_conditional_mean, 8/9)
    assert np.isclose(summary.failure_zero_utility_f1, 4/9)
    rows = []
    for method in analysis.METHODS:
        for ref, target in itertools.permutations(['a', 'b', 'c'], 2):
            rows.append(dict(method=method, reference=ref, target=target, depth=1,
                replicate=0, divergence='test', status='completed',
                f1=.9 if method in ['ASEM', 'ASEM_Hybrid'] else .8,
                identity_pct=90 if method in ['ASEM', 'ASEM_Hybrid'] else 80))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        analysis.species_sensitivity(pd.DataFrame(rows), path)
        deleted = pd.read_csv(path/'leave_one_species_out.csv')
        assert deleted.remaining_ordered_pairs.eq(2).all(), 'must delete both roles'
        paired = pd.read_csv(path/'paired_species_sensitivity.csv')
        comparison = paired[paired.anchor.eq('ASEM_Hybrid') & paired.comparator.eq('Pilon') & paired.metric.eq('f1')]
        assert np.allclose(comparison.anchor_minus_comparator, .1)
        assert np.allclose(comparison.loso_min, .1) and np.allclose(comparison.loso_max, .1)
    print('PASS: score consistency, missing-output policy, paired signs and deletion of both species roles', flush=True)


if __name__ == '__main__':
    main()

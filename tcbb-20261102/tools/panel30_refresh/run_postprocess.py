"""Audit and analyse the finished panel on P8; never launch method jobs.

Primary accuracy is conditional on output. An explicitly named failure-zero
utility is secondary, not an imputed biological accuracy. Species uncertainty
is delete-one-species sensitivity (both reference and target roles removed),
not a confidence interval or a claim of random sampling of mammals.
"""
from __future__ import annotations

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ['ASEM', 'ASEM_Hybrid', 'SRSC', 'Pilon', 'MIA', 'NOVOPlasty', 'GetOrganelle', 'MITObim']
INTERNAL = {'ASEM', 'ASEM_Hybrid', 'SRSC', 'Reference'}
KEY = ['method', 'reference', 'target', 'depth', 'replicate']
JOB = KEY[1:]
METRICS = ['f1', 'identity_pct', 'recall', 'precision', 'c_theta', 'w_theta', 'u_t']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_scores(frame, lengths):
    assert frame.status.isin(['completed', 'no_output', 'error', 'timeout']).all()
    completed = frame[frame.status.eq('completed')].copy()
    for column in METRICS + ['theta_len']:
        completed[column] = pd.to_numeric(completed[column], errors='raise')
        assert np.isfinite(completed[column]).all(), column
    assert (completed[['c_theta', 'w_theta', 'u_t']] >= 0).all().all()
    c, w, u = (completed[k].to_numpy(float) for k in ['c_theta', 'w_theta', 'u_t'])
    assert np.array_equal(c + w + u, completed.target.map(lengths).to_numpy()), 'target count mismatch'
    precision = np.divide(c, c+w, out=np.zeros_like(c), where=(c+w)>0)
    recall = np.divide(c, c+u, out=np.zeros_like(c), where=(c+u)>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(c), where=(precision+recall)>0)
    for k, expected in [('f1', f1), ('precision', precision), ('recall', recall),
                        ('identity_pct', 100*c/(completed.theta_len.to_numpy(float)+u))]:
        assert np.allclose(completed[k], expected, atol=1e-9, rtol=1e-9), k
    missing = frame[~frame.status.eq('completed')]
    assert missing[METRICS].isna().all().all(), 'non-output rows contain accuracy values'
    assert (pd.to_numeric(frame.runtime_s) >= 0).all()


def verify_sequence(job):
    path, sha, expected_len, expected_ambiguous, whole_file = job
    body = gzip.decompress(Path(path).read_bytes())
    seqs = [''.join(s.splitlines()[1:]).upper() for s in body.decode().split('>')[1:]]
    assert seqs, path
    seq = max(seqs, key=len)
    measured = hashlib.sha256(body if whole_file else seq.encode()).hexdigest()
    assert measured == sha, str(path) + ': SHA mismatch'
    assert len(seq) == int(expected_len), str(path) + ': length mismatch'
    assert sum(c not in 'ACGT' for c in seq) == int(expected_ambiguous), str(path) + ': ambiguity mismatch'
    return 1


def verify_manifest(root, mapping):
    for relative, expected in mapping.items():
        assert digest(root / relative) == expected, 'snapshot mismatch: ' + relative


def audit(args, out, state):
    source = args.root / 'source'
    frozen = source / 'genome-live-handoff-20261030'
    meta = json.loads((frozen / 'snapshot.json').read_text())
    verify_manifest(frozen / 'snapshot', {**meta['source_sha256'], **meta['input_sha256']})
    inputs = source / 'mitobim-inputs'
    manifest = json.loads((inputs / 'manifest.json').read_text())
    assert digest(frozen / 'snapshot.json') == manifest['snapshot_sha256']
    verify_manifest(inputs, manifest['files_sha256'])
    panel = load_module(frozen / 'snapshot/plos-one/analysis/panel30_grid/panel30_species.py', 'panel')
    os.environ['ASEM_PAIR_SET'] = 'phase1'
    old = load_module(frozen / 'snapshot/scripts/phase1_species.py', 'old_panel')
    seqs = {p.stem: ''.join(p.read_text().splitlines()[1:]).upper() for p in (inputs / 'references').glob('*.fasta')}
    assert set(seqs) == set(panel.SPECIES) and len(seqs) == 30
    for name, entry in panel.SPECIES.items():
        original = frozen / 'snapshot/data/ieee2021_mtdna_panel' / entry[0]
        assert seqs[name] == ''.join(original.read_text().splitlines()[1:]).upper(), name
    lengths = {n: len(s) for n, s in seqs.items()}
    hashes = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in seqs.items()}
    grid_path = source / 'panel30-export-20261101/results.csv'
    ready = json.loads((args.root / 'source.ready').read_text())
    assert digest(grid_path) == ready['grid_csv_sha256'], 'i9/P8 grid CSV differs'
    mito_path = args.research / 'tcbb-refresh-20261102/mitobim_scored.csv'
    grid = pd.read_csv(grid_path)
    mito = pd.read_csv(mito_path)
    mito_audit = json.loads(mito_path.with_suffix('.audit.json').read_text())
    assert digest(mito_path) == mito_audit['output_sha256']
    assert digest(args.research / 'mitobim-run/results.jsonl') == mito_audit['source_results_sha256']
    assert mito_audit['snapshot_sha256'] == manifest['snapshot_sha256']
    frame = pd.concat([grid, mito], ignore_index=True)
    frame.depth = pd.to_numeric(frame.depth).astype(int)
    frame.replicate = pd.to_numeric(frame.replicate).astype(int)
    expected = {(m, r, t, d, rep) for m in METHODS for r, t in panel.ordered_pairs()
                for d in range(1, 9) for rep in range(3)}
    actual = set(frame[KEY].itertuples(index=False, name=None))
    assert not frame.duplicated(KEY).any() and actual == expected, 'main key coverage'
    assert all(row.divergence == panel.divergence_level(row.reference, row.target)
               for row in frame.itertuples())
    validate_scores(frame, lengths)
    jobs = []
    for row in frame[frame.status.eq('completed')].to_dict('records'):
        mito_row = row['method'] == 'MITObim'
        root = args.research / 'mitobim-run' if mito_row else source / 'panel30-export-20261101'
        jobs.append((root / row['assembly' if mito_row else 'theta_fasta'],
                     row['assembly_sha256' if mito_row else 'theta_sha256'],
                     row['theta_len'], row['n_ambiguous'], mito_row))
    followups = {}
    follow_meta = json.loads((args.research / 'followup-20261101/followup_snapshot.json').read_text())
    assert follow_meta['base_snapshot_sha256'] == manifest['snapshot_sha256']
    verify_manifest(args.research / 'followup-20261101', follow_meta['files_sha256'])
    verify_manifest(args.research / 'followup-20261101', follow_meta['patches_sha256'])
    for experiment, count in [('mixed_orientation', 1440), ('high_depth', 576)]:
        work = args.research / ('followup-20261101/work-' + experiment)
        data = pd.DataFrame(records(work / 'results.jsonl'))
        planned = pd.DataFrame(records(work / 'jobs.jsonl'))
        key = KEY + ['reads']
        assert len(data) == count and not data.duplicated(key).any()
        assert set(data[key].itertuples(index=False, name=None)) == set(planned[key].itertuples(index=False, name=None))
        plan = json.loads((work / 'plan.json').read_text())
        assert plan['jobs'] == count
        assert plan['snapshot_files_sha256'] == hashlib.sha256(json.dumps(follow_meta['files_sha256'], sort_keys=True).encode()).hexdigest()
        assert all(r.divergence == panel.divergence_level(r.reference, r.target) for r in data.itertuples())
        validate_scores(data, lengths)
        for row in data[data.status.eq('completed')].to_dict('records'):
            jobs.append((work / row['theta_fasta'], row['theta_sha256'], row['theta_len'], row['n_ambiguous'], False))
        followups[experiment] = data
    state('assembly_hashes', total=len(jobs))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, _ in enumerate(pool.map(verify_sequence, jobs), 1):
            if n % 10000 == 0:
                state('assembly_hashes', checked=n, total=len(jobs))
    sources = [grid_path, mito_path, mito_path.with_suffix('.audit.json'), frozen / 'snapshot.json',
               inputs / 'manifest.json', args.research / 'followup-20261101/followup_snapshot.json',
               args.research / 'mitobim-run/results.jsonl']
    for experiment in followups:
        sources.extend((args.research / ('followup-20261101/work-' + experiment) / n)
                       for n in ['plan.json', 'jobs.jsonl', 'results.jsonl'])
    audit_record = {'status': 'passed', 'rows': len(frame), 'main_completed': int(frame.status.eq('completed').sum()),
                    'assemblies_checked': len(jobs), 'source_files_sha256': {str(p): digest(p) for p in sources},
                    'method_counts': {str(k): int(v) for k,v in frame.method.value_counts().items()},
                    'followup_rows': {k: len(v) for k,v in followups.items()},
                    'analysis_scripts_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                    'scope': 'Coverage, status/metric consistency, source/input manifests, retained sequence hashes; not scientific claim approval.'}
    write_json(out / 'acceptance.json', audit_record)
    frame['policy'] = np.where(frame.method.isin(INTERNAL), 'fixed', 'normalized')
    frame['old_pair_split'] = np.where(frame.reference.isin(old.SPECIES) & frame.target.isin(old.SPECIES),
                                        'retained_old_species_pair', 'expanded_panel_pair')
    frame['unchanged_reference'] = frame.status.eq('completed') & frame.theta_sha256.eq(frame.reference.map(hashes))
    # MITObim hashes cover the FASTA file, so compare the longest sequence directly.
    for index, row in frame[frame.method.eq('MITObim') & frame.status.eq('completed')].iterrows():
        body = gzip.decompress((args.research / 'mitobim-run' / row.assembly).read_bytes())
        seq = max((''.join(s.splitlines()[1:]).upper() for s in body.decode().split('>')[1:]), key=len)
        frame.loc[index, 'unchanged_reference'] = hashlib.sha256(seq.encode()).hexdigest() == hashes[row.reference]
    return frame, followups


def summarise(frame, groups):
    rows = []
    for values, group in frame.groupby(groups, observed=True):
        if not isinstance(values, tuple): values = (values,)
        row = dict(zip(groups, values))
        ok = group[group.status.eq('completed')]
        row.update(total=len(group), completed=len(ok), output_rate=len(ok)/len(group),
                   no_output=int(group.status.eq('no_output').sum()), errors=int(group.status.eq('error').sum()),
                   timeouts=int(group.status.eq('timeout').sum()))
        row.update({k + '_conditional_mean': ok[k].mean() for k in METRICS})
        row['failure_zero_utility_f1'] = group.f1.where(group.status.eq('completed'), 0).mean()
        if 'unchanged_reference' in group: row['unchanged_reference_outputs'] = int(ok.unchanged_reference.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def species_sensitivity(frame, out):
    paired, deleted = [], []
    index = JOB + ['divergence']
    for anchor in ['ASEM', 'ASEM_Hybrid']:
        left = frame[frame.method.eq(anchor)][index + ['status', 'f1', 'identity_pct']]
        for method in METHODS:
            if method == anchor: continue
            right = frame[frame.method.eq(method)][index + ['status', 'f1', 'identity_pct']]
            merged = left.merge(right, on=index, validate='one_to_one', suffixes=('_anchor', '_other'))
            for estimand in ['both_completed', 'failure_zero_utility']:
                usable = merged.copy()
                if estimand == 'both_completed':
                    usable = usable[usable.status_anchor.eq('completed') & usable.status_other.eq('completed')]
                for metric in ['f1', 'identity_pct']:
                    if estimand == 'failure_zero_utility' and metric != 'f1': continue
                    a, b = usable[metric+'_anchor'], usable[metric+'_other']
                    if estimand == 'failure_zero_utility':
                        a = a.where(usable.status_anchor.eq('completed'), 0)
                        b = b.where(usable.status_other.eq('completed'), 0)
                    usable = usable.assign(delta=a-b)
                    for tier, tier_frame in usable.groupby('divergence'):
                        selections = [('all_1_8X', tier_frame)] + [(str(d), g) for d,g in tier_frame.groupby('depth')]
                        for depth, group in selections:
                            pair = group.groupby(['reference','target'], as_index=False).delta.mean()
                            names = sorted(set(pair.reference) | set(pair.target))
                            mean = pair.delta.mean()
                            vals = []
                            for name in names:
                                keep = pair[~pair.reference.eq(name) & ~pair.target.eq(name)]
                                value = keep.delta.mean()
                                if len(keep): vals.append(value)
                                deleted.append(dict(anchor=anchor, comparator=method, estimand=estimand, metric=metric,
                                    divergence=tier, depth=depth, removed_species=name, remaining_ordered_pairs=len(keep), delta=value))
                            paired.append(dict(anchor=anchor, comparator=method, estimand=estimand, metric=metric,
                                divergence=tier, depth=depth, ordered_pairs=len(pair), matched_jobs=len(group), species=len(names),
                                anchor_minus_comparator=mean, loso_min=min(vals) if vals else np.nan,
                                loso_max=max(vals) if vals else np.nan,
                                interpretation='delete-one-species sensitivity; not a confidence interval'))
    pd.DataFrame(paired).to_csv(out / 'paired_species_sensitivity.csv', index=False)
    pd.DataFrame(deleted).to_csv(out / 'leave_one_species_out.csv', index=False)


def analyse(frame, followups, baseline_path, out):
    summarise(frame, ['method','divergence','depth']).to_csv(out/'main_by_tier_depth.csv', index=False)
    summarise(frame, ['method','old_pair_split','divergence','depth']).to_csv(out/'old_new_pair_split.csv', index=False)
    summarise(frame, ['method']).to_csv(out/'main_overall.csv', index=False)
    baseline = pd.read_csv(baseline_path)
    assert len(baseline) == 1740 and not baseline.duplicated(['reference','target','policy']).any()
    merged = frame.merge(baseline[['reference','target','policy']+METRICS],
                         on=['reference','target','policy'], validate='many_to_one', suffixes=('', '_reference'))
    assert len(merged) == len(frame)
    completed = merged[merged.status.eq('completed')].copy()
    for metric in METRICS:
        completed['net_delta_'+metric] = completed[metric]-completed[metric+'_reference']
    columns = ['net_delta_'+k for k in METRICS]
    completed.groupby(['method','divergence','depth'])[columns].mean().reset_index().to_csv(out/'reference_net_changes.csv', index=False)
    completed[KEY+['divergence','policy','unchanged_reference']+METRICS+[k+'_reference' for k in METRICS]+columns].to_csv(out/'reference_net_changes_jobs.csv', index=False)
    species_sensitivity(frame, out)
    for experiment, data in followups.items():
        summarise(data, ['method','reads','divergence','depth']).to_csv(out/(experiment+'_summary.csv'), index=False)
    mixed = followups['mixed_orientation']
    internal = mixed[mixed.method.isin(['ASEM','ASEM_Hybrid'])]
    controls = internal[internal.reads.eq('forward')].merge(internal[internal.reads.eq('mixed')],
        on=KEY, validate='one_to_one', suffixes=('_forward','_mixed'))
    controls['identical_sequence'] = controls.theta_sha256_forward.eq(controls.theta_sha256_mixed)
    assert len(controls) == 288, 'mixed orientation controls coverage'
    controls[KEY+['identical_sequence','f1_forward','f1_mixed','identity_pct_forward','identity_pct_mixed']].to_csv(out/'mixed_controls.csv', index=False)
    runtime = frame.groupby(['method','status']).runtime_s.agg(['count','min','median','max']).reset_index()
    runtime['interpretation'] = 'parallel-grid observed duration; timeouts right-censored; not controlled runtime comparison'
    runtime.to_csv(out/'grid_execution_descriptive.csv', index=False)
    write_json(out/'analysis_contract.json', {
        'primary_accuracy':'Completed outputs only; report F1 and identity with output counts.',
        'secondary_utility':'failure_zero_utility_f1 assigns zero utility to no output; raw scores remain missing.',
        'pairing':'Identical reference/target/depth/replicate; mean technical replicates within ordered pair before species sensitivity.',
        'uncertainty':'Leave-one-species-out removes every dyad containing that species in either role. Min/max are sensitivity, not confidence limits.',
        'carry_over':'Same-policy unchanged-reference baseline; exact unchanged output count; net matched/mismatched/missing-base changes. Alignment may change; these are not counts of individually corrected sites or causal mechanisms.',
        'normalization':'fixed for CIRR/Hybrid/SRSC; strand+origin for external tools including MITObim.',
        'old_new':'Both species present in frozen phase1_species.SPECIES vs expanded-panel pair; Saimiri sciureus is absent from new panel.',
        'runtime':'Observed parallel-grid times are descriptive; timeout is censored. Controlled runtime must use the separate experiment.',
        'scope_limits':['No new method runs, no empirical/read-length MITObim rows invented.',
                        'Acceptance does not by itself verify algorithm/figure prose or approve final scientific claims.']})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--research', type=Path, default=Path.home()/'research-work')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    lock = (args.root/'.postprocess.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = args.root/'results'; out.mkdir(exist_ok=True)
    def state(stage, **extra):
        value = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **extra)
        write_json(args.root/'status.json', value)
        print(json.dumps(value), flush=True)
    try:
        state('acceptance')
        frame, followups = audit(args, out, state)
        state('reference_baselines', workers=args.workers)
        baseline = out/'reference_baseline.csv'
        subprocess.run([sys.executable, str(Path(__file__).with_name('reference_baseline.py')),
                        '--source', str(args.root/'source'), '--output', str(baseline),
                        '--workers', str(args.workers)], check=True)
        state('summaries_and_species_sensitivity')
        analyse(frame, followups, baseline, out)
        write_json(out/'output_manifest.json', {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'output_manifest.json'})
        state('complete', result_files=len(list(out.iterdir())))
    except BaseException as exc:
        state('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()

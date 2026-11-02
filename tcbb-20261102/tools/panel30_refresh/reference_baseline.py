"""Unchanged-reference scores with each method family's evaluation policy.

Reuse an accepted external-tool score only when its stored sequence SHA-256
exactly equals the supplied reference sequence. Otherwise use the frozen
evaluator. Differences in matched-base counts are net changes, not a claim
that individual corrected positions have been causally identified.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,itertools,json,multiprocessing,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import score_mitobim_parallel as scoring

def one(job):
    ref,target,policy=job
    scores=scoring.COMMON.evaluate_theta_vs_target(scoring.TRUTH[ref],scoring.TRUTH[target],
        normalize_strand=policy=='normalized',normalize_circular_origin=policy=='normalized')
    return dict(reference=ref,target=target,policy=policy,**scores)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=12)
    a=ap.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.source/'panel30-export-20261101/results.csv')
    seqs={p.stem:''.join(p.read_text().splitlines()[1:]).upper() for p in (a.source/'mitobim-inputs/references').glob('*.fasta')}
    digests={n:hashlib.sha256(s.encode()).hexdigest() for n,s in seqs.items()}
    cache={}
    for row in d[d.method.isin(['Pilon','MIA','NOVOPlasty','GetOrganelle']) & d.status.eq('completed')].to_dict('records'):
        if row['theta_sha256']==digests[row['reference']]:
            key=(row['reference'],row['target'],'normalized')
            scores={k:row[k] for k in ('f1','identity_pct','recall','precision','c_theta','w_theta','u_t')}
            if key in cache:
                assert all(abs(cache[key][k]-v)<1e-12 for k,v in scores.items()),key
            cache[key]=scores
    rows=[dict(reference=r,target=t,policy=p,**v) for (r,t,p),v in cache.items()]
    done=set(cache);journal=a.output.with_suffix('.jsonl')
    if journal.exists():
        for line in journal.read_text().splitlines():
            row=json.loads(line);key=(row['reference'],row['target'],row['policy'])
            if key not in done:rows.append(row);done.add(key)
    jobs=[(r,t,p) for r,t in itertools.permutations(seqs,2) for p in ('fixed','normalized') if (r,t,p) not in done]
    print(json.dumps({'reused_identical_reference_scores':len(cache),'to_evaluate':len(jobs)}),flush=True)
    with journal.open('a') as h,ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('fork'),
         initializer=scoring.initialize,initargs=(str(a.source/'genome-live-handoff-20261030'),str(a.source/'mitobim-final'),str(a.source/'mitobim-inputs'))) as pool:
        for row in pool.map(one,jobs,chunksize=1):
            rows.append(row);h.write(json.dumps(row)+'\n');h.flush()
            if len(rows)%100==0:print('baselines',len(rows),'/ 1740',flush=True)
    result=pd.DataFrame(rows);assert len(result)==1740 and not result.duplicated(['reference','target','policy']).any()
    result.to_csv(a.output,index=False)
    print(json.dumps({'rows':len(result),'output':str(a.output)}),flush=True)

if __name__=='__main__':main()

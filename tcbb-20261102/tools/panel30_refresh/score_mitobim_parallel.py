"""Score the complete MITObim run with the frozen grid evaluator, resumably.

Only scheduling changes relative to mitobim_grid/score_mitobim.py. Missing
outputs remain missing; the longest record is scored with strand/origin
normalization. Input, snapshot and decompressed assembly hashes are checked.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import argparse, csv, gzip, hashlib, json, sys, time, multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

COMMON = None
TRUTH = {}
WORK = None

def initialize(root, work, inputs):
    global COMMON, WORK, TRUTH
    root, WORK, inputs = Path(root), Path(work), Path(inputs)
    snap = root/'snapshot'
    os.environ['BENCH_SCRIPTS'] = str(snap/'scripts')
    sys.path[:0] = [str(snap/'scripts'), str(snap/'plos-one/analysis/evaluator_n_fix')]
    import common, nsafe
    nsafe.install()
    COMMON = common
    TRUTH = {p.stem: common.read_fasta(str(p))[1] for p in (inputs/'references').glob('*.fasta')}

def score(record):
    row = dict(record, method='MITObim')
    if row['status'] != 'completed':
        return row
    body = gzip.decompress((WORK/row['assembly']).read_bytes())
    assert hashlib.sha256(body).hexdigest() == row['assembly_sha256'], row['key']
    seqs = [''.join(s.splitlines()[1:]).upper() for s in body.decode().split('>')[1:]]
    assert seqs and len(seqs) == row['n_contigs']
    theta = max(seqs, key=len)
    row.update(COMMON.evaluate_theta_vs_target(theta, TRUTH[row['target']],
               normalize_strand=True, normalize_circular_origin=True))
    row.update(theta_len=len(theta), total_len=sum(map(len,seqs)),
               n_ambiguous=sum(c not in 'ACGT' for c in theta))
    return row

def main():
    ap=argparse.ArgumentParser()
    for n in ('root','work','inputs','output'): ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--workers',type=int,default=24)
    ap.add_argument('--limit',type=int)
    a=ap.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.root/'snapshot.json').read_text())
    for rel,digest in {**meta['input_sha256'],**meta['source_sha256']}.items():
        assert hashlib.sha256((a.root/'snapshot'/rel).read_bytes()).hexdigest()==digest,rel
    manifest=json.loads((a.inputs/'manifest.json').read_text())
    assert hashlib.sha256((a.root/'snapshot.json').read_bytes()).hexdigest()==manifest['snapshot_sha256']
    for rel,digest in manifest['files_sha256'].items():
        assert hashlib.sha256((a.inputs/rel).read_bytes()).hexdigest()==digest,rel
    records=[json.loads(x) for x in (a.work/'results.jsonl').read_text().splitlines()]
    assert len(records)==len({r['key'] for r in records})==manifest['total_jobs']==20880
    if a.limit: records=records[:a.limit]
    journal=a.output.with_suffix('.jsonl'); done={}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row=json.loads(line); assert row['key'] not in done; done[row['key']]=row
    todo=[r for r in records if r['key'] not in done]
    started=time.time()
    with journal.open('a') as handle, ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('fork'), initializer=initialize,
            initargs=(str(a.root),str(a.work),str(a.inputs))) as pool:
        for row in pool.map(score,todo,chunksize=4):
            done[row['key']]=row;handle.write(json.dumps(row,sort_keys=True)+'\n')
            if len(done)%100==0:
                handle.flush();os.fsync(handle.fileno())
                print(json.dumps({'scored':len(done),'total':len(records),'elapsed_s':round(time.time()-started,1)}),flush=True)
    rows=sorted((done[r['key']] for r in records),key=lambda r:r['key'])
    fields=sorted({k for r in rows for k in r})
    with a.output.open('w',newline='') as h:
        w=csv.DictWriter(h,fields);w.writeheader();w.writerows(rows)
    audit={'rows':len(rows),'completed':sum(r['status']=='completed' for r in rows),
        'snapshot_sha256':manifest['snapshot_sha256'],'source_results_sha256':hashlib.sha256((a.work/'results.jsonl').read_bytes()).hexdigest(),
        'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'scoring_elapsed_s':time.time()-started,
        'evaluator':'frozen common + nsafe; longest record, strand and circular-origin normalization'}
    a.output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)

if __name__=='__main__': main()

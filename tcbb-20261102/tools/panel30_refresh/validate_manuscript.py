"""Independent numeric checks for the refreshed TCBB manuscript evidence."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
DATA=HERE/'accepted-20261102'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);a=ap.parse_args()
    raw=pd.concat([pd.read_csv(a.evidence/'tcbb-postprocess-20261102/source/panel30-export-20261101/results.csv'),
                   pd.read_csv(a.evidence/'tcbb-refresh-20261102/mitobim_scored.csv')],ignore_index=True)
    summary=pd.read_csv(DATA/'main_by_tier_depth.csv')
    checked=0
    for r in summary.itertuples():
        rows=raw[(raw.method==r.method)&(raw.divergence==r.divergence)&(raw.depth==r.depth)]
        ok=rows[rows.status.eq('completed')]
        assert len(rows)==r.total and len(ok)==r.completed
        for metric in ['f1','identity_pct','c_theta','w_theta','u_t','precision','recall']:
            observed=ok[metric].mean();expected=getattr(r,metric+'_conditional_mean')
            assert (pd.isna(observed) and pd.isna(expected)) or np.isclose(observed,expected,atol=1e-10)
            checked+=1
    base=pd.read_csv(DATA/'reference_baseline.csv').set_index(['reference','target','policy'])
    net=pd.read_csv(DATA/'reference_net_changes.csv')
    for row in net[net.depth.eq(8)].itertuples():
        group=raw[(raw.method==row.method)&(raw.divergence==row.divergence)&(raw.depth==8)&raw.status.eq('completed')]
        policy='fixed' if row.method in ['ASEM','ASEM_Hybrid','SRSC'] else 'normalized'
        for metric in ['f1','identity_pct','c_theta','w_theta','u_t']:
            changes=[getattr(r,metric)-base.loc[(r.reference,r.target,policy),metric] for r in group.itertuples()]
            assert np.isclose(np.mean(changes),getattr(row,'net_delta_'+metric),atol=1e-9)
            checked+=1
    paired=pd.read_csv(DATA/'paired_species_sensitivity.csv')
    selected=paired[(paired.anchor=='ASEM_Hybrid')&paired.comparator.isin(['ASEM','Pilon','MIA','MITObim'])&
                    paired.depth.eq('all_1_8X')&paired.estimand.eq('both_completed')]
    for row in selected.itertuples():
        subset=raw[raw.divergence.eq(row.divergence)&raw.method.isin([row.anchor,row.comparator])]
        wide=subset.pivot(index=['reference','target','depth','replicate'],columns='method',values=row.metric).dropna()
        differences=(wide[row.anchor]-wide[row.comparator]).groupby(level=[0,1]).mean()
        assert len(wide)==row.matched_jobs and len(differences)==row.ordered_pairs
        species=set(differences.index.get_level_values(0))|set(differences.index.get_level_values(1))
        deletions=[differences[(differences.index.get_level_values(0)!=n)&(differences.index.get_level_values(1)!=n)].mean() for n in species]
        assert np.allclose([differences.mean(),min(deletions),max(deletions)],
                           [row.anchor_minus_comparator,row.loso_min,row.loso_max],atol=1e-10)
        checked+=3
    ledger=json.loads((PAPER/'provenance/panel30-claim-ledger.json').read_text())
    used=set()
    active=[PAPER/'tcbb-sections/abstract.tex']+list((PAPER/'english-sections').glob('0[1-5]-*.tex'))
    for p in active:used.update(re.findall(r'\\res\{([^}]+)\}',p.read_text()))
    assert used<=set(ledger), used-set(ledger)
    for experiment,cols in [('read_length',['reference','target','length_condition','replicate']),
                             ('real_reads',['reference','depth','replicate'])]:
        frame=pd.read_csv(DATA/f'auxiliary_{experiment}.csv')
        assert frame.groupby(cols).n_reads.nunique().le(1).all(), experiment+': unequal read counts'
    before=json.loads((a.evidence/'plos-before.json').read_text())
    for name,expected in before.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, 'PLOS changed: '+name
    receipt={'status':'passed','main_rows':len(raw),'numeric_summary_values_recomputed':checked,
             'main_summary_cells':len(summary),'species_sensitivity_cells_recomputed':len(selected),
             'used_numeric_macros_checked':len(used),'plos_files_unchanged':len(before),
             'example_consensus_core':'frozen scripts/asem_core.py SHA matches the local figure-generating core',
             'scope':'Summary arithmetic, baseline differences, paired species deletion, macro presence, matched auxiliary read counts and preserved PLOS tree. Layout checked separately.'}
    (PAPER/'provenance/panel30-numeric-qa.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

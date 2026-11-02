"""Build TCBB tables, figures, supplements and an inspectable claim ledger.

Run with the existing genome Python environment. Large input tables stay in
--evidence outside OneDrive; compact accepted summaries live beside this file.
No scientific methods or external comparator jobs are executed here.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tcbb-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
ACCEPTED = HERE/'accepted-20261102'
GEN = PAPER/'generated'
METHODS = ['ASEM','ASEM_Hybrid','SRSC','Pilon','MIA','MITObim','NOVOPlasty','GetOrganelle']
NAMES = dict(zip(METHODS, ['CIRR','CIRR-Hybrid','SRSC','Pilon','MIA','MITObim','NOVOPlasty','GetOrganelle']))
NAMES['Reference'] = 'Reference'
TIERS = ['same-genus','same-family','same-order','different-order']
TITLES = ['Same genus','Same family','Same order','Different order']
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#332288','#777777']
STYLE = {m:dict(color=c,marker=mk,linestyle=ls) for m,c,mk,ls in zip(
    METHODS,COLORS,['o','s','^','D','v','P','x','+'],['-','-','--','-.','--',':','-.',':'])}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt(x, places=4):
    return '--' if pd.isna(x) else f'{x:.{places}f}'



def bold_accuracy(body, columns, across_rows=True):
    """Bold displayed maxima, including rounded ties; preserve missing/superscripts.

    Compare within a divergence/reference block (or each read-length row).
    Counts, baseline-change diagnostics and censored runtimes are not ranked.
    """
    result=list(body)
    groups=[]
    current=[]
    for i,line in enumerate(result):
        if line.startswith(r'\multicolumn') or line.startswith(r'\bottomrule'):
            if current: groups.append(current)
            current=[]
        if ' & ' in line and not line.startswith(('\\','Method','Length',' &')):
            current.append(i)
    if current: groups.append(current)
    for group in groups:
        parsed={i:result[i].split(' & ') for i in group}
        cells=[]
        for i,parts in parsed.items():
            for col in columns:
                # Superscripts remain outside the bold numeric value.
                match=re.match(r'^(\d+(?:\.\d+)?)(.*)$',parts[col])
                if match: cells.append((i,col,float(match[1]),match[1],match[2]))
        keys=columns if across_rows else group
        for key in keys:
            selected=[c for c in cells if (c[1] if across_rows else c[0])==key]
            if not selected: continue
            best=max(c[2] for c in selected)
            for i,col,value,number,suffix in selected:
                if value==best: parsed[i][col]=r'\textbf{'+number+'}'+suffix
        for i,parts in parsed.items(): result[i]=' & '.join(parts)
    return result


def table(path, cols, head, body):
    ranked={"panel30_main.tex":[1,2,4,5], "panel30_highdepth.tex":[1,2,4,5],
            "tbl_real_reads.tex":list(range(1,9))}
    if path.name in ranked: body=bold_accuracy(body,ranked[path.name])
    if path.name in {'panel30_main.tex','panel30_carryover.tex','panel30_highdepth.tex'}:
        # Keep the four tier blocks and all data at the original font size,
        # using two side-by-side panels instead of four full-width blocks.
        groups=[]
        for line in body:
            if line.startswith(r'\multicolumn') and r'\textbf{' in line:
                groups.append([])
            assert groups
            if path.name=='panel30_carryover.tex' and 'Reference F1/identity:' in line:
                fixed,normalized=line.split('Reference F1/identity: ',1)[1].rsplit(r'} \\',1)[0].split('; ')
                groups[-1].extend([r'\multicolumn{6}{l}{Ref. F1/Id.: '+fixed+r'} \\',
                                   r'\multicolumn{6}{l}{\hspace{3em}'+normalized+r'} \\'])
            else:
                groups[-1].append(line)
        assert len(groups)==4
        head=[h.replace(r'Id. (\%)','Id.').replace('Outputs/attempts','Outputs') for h in head]
        content=[r'\begingroup\setlength{\tabcolsep}{1pt}']
        for i in range(2):
            if i:content.append(r'\hfill')
            content.extend([r'\begin{minipage}[t]{0.49\textwidth}\centering',
                r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}'+cols+r'@{}}',
                r'\toprule',*head,r'\midrule',*groups[2*i],*groups[2*i+1],r'\bottomrule',
                r'\end{tabular*}',r'\end{minipage}%'])
        content.append(r'\endgroup')
        path.write_text('% Two-panel layout; every accepted data cell retained.\n'+'\n'.join(content)+'\n')
        return
    text = [r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'+cols+r'@{}}',r'\toprule',*head,r'\midrule',*body,r'\bottomrule',r'\end{tabular*}']
    path.write_text('% Generated from accepted evidence by prepare_manuscript.py\n'+'\n'.join(text)+'\n')


def workbook(path, sheets):
    book = Workbook(write_only=True)
    for name, frame in sheets.items():
        sheet = book.create_sheet(name[:31])
        sheet.append(list(frame.columns))
        for row in frame.itertuples(index=False, name=None):
            cells=[]
            for value in row:
                if pd.isna(value): value=None
                if isinstance(value, np.generic): value=value.item()
                if isinstance(value,str):
                    value=''.join(c for c in value if c in '\t\n\r' or ord(c)>=32)
                    cell=WriteOnlyCell(sheet,value=value);cell.data_type='s';cells.append(cell)
                else: cells.append(value)
            sheet.append(cells)
    book.save(path)


def aux_inputs(evidence):
    internal=evidence/'auxiliary-i9'
    inputs={}
    for experiment,file in [('read_length','results.csv'),('real_reads','results.csv'),('runtime','runtime_results.csv')]:
        path=internal/experiment/file
        audit=json.loads((internal/experiment/'acceptance.json').read_text())
        assert sha(path)==audit['sha256']
        newest=pd.read_csv(path);assert len(newest)==audit['rows']
        if experiment=='runtime':
            base=PAPER/'analysis/runtime_benchmark/results/runtime_results.csv'
        else:base=PAPER/('analysis/'+('read_length_all_methods' if experiment=='read_length' else 'real_reads_all_methods')+'/results/results.csv')
        data=pd.read_csv(base); sources=[base,path]
        if experiment!='runtime':
            fixed=PAPER/('analysis/evaluator_n_fix/results/'+('read_length.csv' if experiment=='read_length' else 'real_reads.csv'))
            replacement=pd.read_csv(fixed); sources.append(fixed)
            data=pd.concat([data[~data.method.isin(replacement.method.unique())],replacement],ignore_index=True)
        data=pd.concat([data[~data.method.isin(newest.method.unique())],newest],ignore_index=True)
        data=data[data.method.isin(METHODS)].copy()
        expected={'read_length':63,'real_reads':48,'runtime':36}[experiment]
        assert data.method.value_counts().eq(expected).all() and data.method.nunique()==7
        keys=['method','reference','replicate', 'length_condition' if experiment=='read_length' else 'depth']
        if experiment!='real_reads':keys.insert(2,'target')
        assert not data.duplicated(keys).any()
        inputs[experiment]=data
        compact=ACCEPTED/('auxiliary_'+experiment+'.csv');data.to_csv(compact,index=False)
        inputs[experiment+'_provenance']={str(p):sha(p) for p in sources}
    return inputs



def verify_single_row_legend(fig):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    boxes=[t.get_window_extent(renderer) for t in fig.legends[0].get_texts()]
    assert max(b.y0 for b in boxes)-min(b.y0 for b in boxes)<.5, 'legend wraps'
    for b in boxes:
        assert fig.bbox.contains(b.x0,b.y0) and fig.bbox.contains(b.x1,b.y1), 'legend clipped'
    for a,b in zip(boxes,boxes[1:]): assert not a.overlaps(b), 'legend text overlaps'
    legend=fig.legends[0].get_window_extent(renderer)
    for ax in fig.axes:
        assert not ax.get_window_extent(renderer).overlaps(legend), 'legend overlaps plot'
        if ax.get_xlabel():
            assert not ax.xaxis.label.get_window_extent(renderer).overlaps(legend), 'legend overlaps x label'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--skip-workbooks',action='store_true');a=ap.parse_args()
    summary=pd.read_csv(ACCEPTED/'main_by_tier_depth.csv')
    overall=pd.read_csv(ACCEPTED/'main_overall.csv')
    net=pd.read_csv(ACCEPTED/'reference_net_changes.csv')
    paired=pd.read_csv(ACCEPTED/'paired_species_sensitivity.csv')
    high=pd.read_csv(ACCEPTED/'high_depth_summary.csv')
    base=pd.read_csv(ACCEPTED/'reference_baseline.csv')
    raw=pd.concat([pd.read_csv(a.evidence/'tcbb-postprocess-20261102/source/panel30-export-20261101/results.csv'),
                   pd.read_csv(a.evidence/'tcbb-refresh-20261102/mitobim_scored.csv')],ignore_index=True)
    assert len(raw)==167040
    tier_map=raw[['reference','target','divergence']].drop_duplicates()
    base=base.merge(tier_map,on=['reference','target'],validate='many_to_one')
    aux=aux_inputs(a.evidence)
    ledger={}
    def value(key,v,places=None,source='accepted summary'):
        if isinstance(v,np.generic):v=v.item()
        rendered=f'{int(v):,}' if places is None else f'{v:.{places}f}'
        ledger[key]=dict(value=v,text=rendered,source=source)
    for row in overall.itertuples():
        m={'ASEM':'Cirr','ASEM_Hybrid':'Hybrid'}.get(row.method,row.method)
        for suffix,v,p in [('MeanF',row.f1_conditional_mean,4),('MeanId',row.identity_pct_conditional_mean,2),
                            ('Outputs',row.completed,None),('Errors',row.errors,None),('NoOutput',row.no_output,None),
                            ('Unchanged',row.unchanged_reference_outputs,None)]:value(m+suffix,v,p,'main_overall.csv')
    wide=raw.pivot(index=['reference','target','depth','replicate'],columns='method',values='f1')
    diff=wide.ASEM_Hybrid-wide.ASEM
    for k,v in [('HybridImproved',(diff>1e-12).sum()),('HybridTied',(diff.abs()<=1e-12).sum()),
                ('HybridRegressed',(diff<-1e-12).sum())]:value(k,v,source='paired raw grid')
    main_body=[]; carry=[]; uncertainty=[]; high_body=[]
    for tier,title in zip(TIERS,TITLES):
        n_pairs=len(tier_map[tier_map.divergence.eq(tier)])
        value(tier+'Pairs',n_pairs,source='unique panel dyads')
        main_body.append(r'\multicolumn{8}{l}{\textbf{'+title+f': {n_pairs} ordered pairs'+r'}} \\')
        for m in METHODS:
            cells=[]
            for depth in [1,8]:
                row=summary[(summary.method==m)&(summary.divergence==tier)&(summary.depth==depth)].iloc[0]
                cells += [fmt(row.f1_conditional_mean),fmt(row.identity_pct_conditional_mean,1),str(int(row.completed))]
                short={'ASEM':'Cirr','ASEM_Hybrid':'Hybrid'}.get(m,m)
                value(f'{short}-{tier}-{depth}-F',row.f1_conditional_mean,4) if pd.notna(row.f1_conditional_mean) else None
                value(f'{short}-{tier}-{depth}-Id',row.identity_pct_conditional_mean,2) if pd.notna(row.identity_pct_conditional_mean) else None
            group=summary[(summary.method==m)&(summary.divergence==tier)]
            cells.append(f'{group.completed.sum()}/{group.total.sum()}')
            main_body.append(NAMES[m]+' & '+' & '.join(cells)+r' \\')
        main_body.append(r'\addlinespace[3pt]')
        carry.append(r'\multicolumn{6}{l}{\textbf{'+title+r'}} \\')
        refs=[]
        for policy in ['fixed','normalized']:
            b=base[(base.divergence==tier)&(base.policy==policy)]
            refs.append(f'{policy}: {b.f1.mean():.4f}/{b.identity_pct.mean():.1f}'+r'\%')
            value(f'Reference-{tier}-{policy}-F',b.f1.mean(),4,'reference_baseline.csv')
            value(f'Reference-{tier}-{policy}-Id',b.identity_pct.mean(),2,'reference_baseline.csv')
        carry.append(r'\multicolumn{6}{l}{Reference F1/identity: '+ '; '.join(refs)+r'} \\')
        for m in METHODS[:6]:
            row=net[(net.method==m)&(net.divergence==tier)&(net.depth==8)].iloc[0]
            vals=[f'{row.net_delta_f1:+.4f}',f'{row.net_delta_identity_pct:+.2f}',f'{row.net_delta_c_theta:+.0f}',f'{row.net_delta_w_theta:+.0f}',f'{row.net_delta_u_t:+.0f}']
            carry.append(NAMES[m]+' & '+' & '.join(vals)+r' \\')
        carry.append(r'\addlinespace[3pt]')
        uncertainty.append(r'\multicolumn{5}{l}{\textbf{'+title+r'}} \\')
        for m in ['ASEM','Pilon','MIA','MITObim']:
            cells=[]
            for metric,p in [('f1',4),('identity_pct',2)]:
                row=paired[(paired.anchor=='ASEM_Hybrid')&(paired.comparator==m)&(paired.divergence==tier)&
                    (paired.depth=='all_1_8X')&(paired.estimand=='both_completed')&(paired.metric==metric)].iloc[0]
                cells.append(f'{row.anchor_minus_comparator:+.{p}f} [{row.loso_min:+.{p}f}, {row.loso_max:+.{p}f}]')
                prefix=f'HybridVs{m}-{tier}-{metric}'
                for suf,col in [('Mean','anchor_minus_comparator'),('Low','loso_min'),('High','loso_max')]:value(prefix+suf,row[col],p,'paired_species_sensitivity.csv')
            uncertainty.append(NAMES[m]+' & '+' & '.join(cells)+f' & {int(row.matched_jobs):,} & {int(row.species)}'+r' \\')
        uncertainty.append(r'\addlinespace[3pt]')
        high_body.append(r'\multicolumn{7}{l}{\textbf{'+title+r': three selected ordered pairs}} \\')
        for m in ['Reference']+[m for m in METHODS if m!='MITObim']:
            cells=[]
            for depth in [15,30]:
                row=high[(high.method==m)&(high.divergence==tier)&(high.depth==depth)].iloc[0]
                cells += [fmt(row.f1_conditional_mean),fmt(row.identity_pct_conditional_mean,1),f'{int(row.completed)}/9']
            high_body.append(NAMES[m]+' & '+' & '.join(cells)+r' \\')
        high_body.append(r'\addlinespace[3pt]')
    table(GEN/'panel30_main.tex','lrrrrrrr',[
        r' & \multicolumn{3}{c}{1X} & \multicolumn{3}{c}{8X} & 1--8X \\',
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r'Method & F1 & Id. (\%) & $n$ & F1 & Id. (\%) & $n$ & Outputs/attempts \\'],main_body)
    table(GEN/'panel30_carryover.tex','lrrrrr',[
        r'Method & $\Delta F1$ & $\Delta$Id. (pp) & $\Delta c_\Theta$ & $\Delta m_\Theta$ & $\Delta u_\Theta$ \\'],carry)
    table(GEN/'panel30_paired.tex','lllrr',[
        r'Comparator & $\Delta F1$ [LOSO min, max] & $\Delta$Id. (pp) [LOSO min, max] & Paired jobs & Species \\'],uncertainty)
    table(GEN/'panel30_highdepth.tex','lrrrrrr',[
        r' & \multicolumn{3}{c}{15X} & \multicolumn{3}{c}{30X} \\',
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r'Method & F1 & Id. (\%) & Outputs & F1 & Id. (\%) & Outputs \\'],high_body)
    # Auxiliaries use latest internal reruns and retained N-safe external rows.
    length=aux['read_length']; real=aux['real_reads']; runtime=aux['runtime']
    length_summary=length[length.status.eq('completed')].groupby(['method','divergence','length_condition'])[['f1','identity_pct']].mean().reset_index()
    length_summary.to_csv(ACCEPTED/'read_length_means.csv',index=False)
    lines=[r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}',r'\toprule',r'Length & CIRR & \shortstack{CIRR-\\Hybrid} & SRSC & Pilon & MIA \\',r'\midrule']
    for tier,title in zip(TIERS[:3],TITLES[:3]):
        lines.append(r'\multicolumn{6}{l}{\textbf{'+title+r'}} \\')
        for condition in ['50','75','100','150','250','300','variable']:
            cells=[]
            for m in METHODS[:5]:
                group=length[(length.method==m)&(length.divergence==tier)&(length.length_condition.astype(str)==condition)]
                ok=group[group.status.eq('completed')]
                cells.append(fmt(ok.f1.mean())+(rf'$^{{{len(ok)}/{len(group)}}}$' if len(ok)<len(group) else ''))
            lines.append(condition+' & '+' & '.join(cells)+r' \\')
        lines.append(r'\addlinespace[2pt]')
    lines=bold_accuracy(lines,list(range(1,6)),across_rows=False)
    (GEN/'tbl_read_length_column.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular*}'])+'\n')
    real_summary=real[real.status.eq('completed')].groupby(['method','reference','depth']).mean(numeric_only=True).reset_index()
    real_summary.to_csv(ACCEPTED/'empirical_means.csv',index=False)
    rows=[]
    initial=pd.read_csv(PAPER/'analysis/real_reads_all_methods/results/initial_states.csv')
    for ref,title in [('Pan_troglodytes_NC_001643','Chimpanzee (same family)'),('Varecia_variegata_NC_012773',r'\textit{Varecia} (same order)')]:
        row=initial[(initial.reference==ref)&(initial.label=='truth_primary')].iloc[0]
        rows.append(r'\multicolumn{9}{l}{\textbf{'+title+f'}}: reference F1 {row.f1:.4f}, identity {row.identity_pct:.2f}'+r'\%} \\')
        for m in METHODS[:5]+['NOVOPlasty','GetOrganelle']:
            cells=[]
            for depth in [1,2,4,8]:
                group=real[(real.method==m)&(real.reference==ref)&(real.depth==depth)]
                ok=group[group.status.eq('completed')]
                cells += [fmt(ok.truth_primary_f1.mean())+(rf'$^{{{len(ok)}/3}}$' if len(ok)<3 else ''),fmt(ok.truth_primary_identity_pct.mean(),2)]
            rows.append(NAMES[m]+' & '+' & '.join(cells)+r' \\')
        rows.append(r'\addlinespace[3pt]')
    table(GEN/'tbl_real_reads.tex','lrrrrrrrr',[
        r' & \multicolumn{2}{c}{1X} & \multicolumn{2}{c}{2X} & \multicolumn{2}{c}{4X} & \multicolumn{2}{c}{8X} \\',
        r'Method & F1 & Id. (\%) & F1 & Id. (\%) & F1 & Id. (\%) & F1 & Id. (\%) \\'],rows)
    rows=[]; runtime_summary=[]
    for m in [x for x in METHODS if x!='MITObim']:
        group=runtime[runtime.method.eq(m)]
        med=group.groupby('depth').worker_method_s.median()
        ok=group[group.status.eq('completed')];rss=ok.method_peak_rss_kb.max()/1024
        rows.append(NAMES[m]+' & '+' & '.join(fmt(med.loc[d],1) for d in [1,2,4,8])+f' & {len(ok)}/{len(group)} & {rss:.0f}'+r' \\')
        runtime_summary.append(dict(method=m,**{f'median_{d}X':med.loc[d] for d in [1,2,4,8]},outputs=len(ok),total=len(group),rss_mb=rss))
    (GEN/'tbl_runtime_column.tex').write_text('\n'.join([r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Method & 1X & 2X & 4X & 8X & Outputs & \shortstack{Peak RSS\\(MB)} \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}'])+'\n')
    pd.DataFrame(runtime_summary).to_csv(ACCEPTED/'runtime_means.csv',index=False)
    # Four divergence tiers at a readable native two-column width.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                         'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42})
    plot_receipt={}
    fig,axes=plt.subplots(2,4,figsize=(7.15,3.7),sharex=True,sharey='row')
    fig.subplots_adjust(left=.07,right=.99,top=.91,bottom=.20,wspace=.14,hspace=.24)
    for i,(tier,title) in enumerate(zip(TIERS,TITLES)):
        for j,(metric,ylim) in enumerate([('f1_conditional_mean',(0,1.02)),('identity_pct_conditional_mean',(0,102))]):
            ax=axes[j,i]
            for m in METHODS[:6]:
                d=summary[(summary.method==m)&(summary.divergence==tier)].sort_values('depth')
                plot_receipt[f'{tier}:{m}:{metric}']=list(zip(d.depth.tolist(),d[metric].tolist()))
                ax.plot(d.depth,d[metric],label=NAMES[m],**STYLE[m],linewidth=1.05,markersize=2.8,markerfacecolor='white' if m in ['MIA','MITObim'] else STYLE[m]['color'])
            b=base[(base.divergence==tier)&(base.policy=='fixed')]
            k='f1' if j==0 else 'identity_pct'
            ax.axhline(b[k].mean(),color='#666666',linestyle=(0,(3,3)),linewidth=.8,label='Reference (fixed)')
            ax.set_ylim(*ylim);ax.set_xlim(.8,8.2);ax.set_xticks([1,2,4,6,8]);ax.grid(axis='y',alpha=.2)
            if j==0:ax.set_title(f'({chr(65+i)}) {title}',loc='left',pad=5,fontsize=9)
            if i==0:ax.set_ylabel('F1' if j==0 else 'Identity (%)')
            ax.spines[['top','right']].set_visible(False)
    xlabel=fig.supxlabel('Mitochondrial depth (X)',y=.09,fontsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=len(labels),frameon=False,fontsize=8,
               handlelength=1.4,handletextpad=.4,columnspacing=.8,borderaxespad=0)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for text in fig.legends[0].get_texts():
        assert not xlabel.get_window_extent(renderer).overlaps(text.get_window_extent(renderer)), 'depth label/legend overlap'
    verify_single_row_legend(fig)
    fig.savefig(PAPER/'assets/fig_panel30_depth.pdf');fig.savefig(PAPER/'assets/fig_panel30_depth.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.85))
    fig.subplots_adjust(left=.08,right=.98,top=.91,bottom=.29,wspace=.22)
    for m in METHODS:
        group=raw[raw.method.eq(m)]
        for j in [0,1]:
            if j==0:
                vals=group.assign(output=group.status.eq('completed')).groupby('depth').output.mean()*100
            else:vals=group[group.status.eq('completed')].groupby('depth').f1.mean()
            plot_receipt[f'output:{j}:{m}']=list(zip(vals.index.tolist(),vals.tolist()))
            axes[j].plot(vals.index,vals,label=NAMES[m],**STYLE[m],linewidth=1.2,markersize=3.5,markerfacecolor='white')
    for ax in axes:
        ax.set_xticks([1,2,4,6,8]);ax.set_xlim(.8,8.2);ax.set_xlabel('Mitochondrial depth (X)');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylim(-2,102);axes[0].set_ylabel('Outputs / attempted jobs (%)');axes[0].set_title('(A) Output rate',loc='left')
    axes[1].set_ylim(-.02,1.02);axes[1].set_ylabel('F1 conditional on output');axes[1].set_title('(B) Conditional accuracy',loc='left')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.005),ncol=len(METHODS),frameon=False,fontsize=8,
               handlelength=1.4,handletextpad=.4,columnspacing=.8,borderaxespad=0)
    verify_single_row_legend(fig)
    fig.savefig(PAPER/'assets/fig_panel30_output.pdf');fig.savefig(PAPER/'assets/fig_panel30_output.png',dpi=170);plt.close(fig)
    (PAPER/'provenance/panel30-figure-data.json').write_text(json.dumps(plot_receipt,indent=2)+'\n')
    # Supplement updates retain the historical diagnostic S3/S4 files separately.
    description=pd.DataFrame([
        ('Dataset','30 species, 870 ordered pairs, depths 1-8X, three technical replicates, eight methods'),
        ('Names','CIRR=ASEM; CIRR-Hybrid=ASEM_Hybrid in machine-readable provenance'),
        ('Accuracy','Scores remain missing for no output/error/timeout; output rates are separate'),
        ('LOSO','Remove all pairs containing a species in either role; min/max are sensitivity, not CI'),
        ('Net changes','Same-policy reference baseline; changes in aligned counts do not identify individual corrected sites'),
        ('Primary source','Frozen benchmark commit 226a2892a6dd481deb1e6539ffc97320e998f690'),
        ('Follow-up source','Strand-aware patch snapshot dated 2026-11-01; separate from main forward-only snapshot')],columns=['Item','Definition'])
    maincols=['method','reference','target','divergence','depth','replicate','status','f1','identity_pct','precision','recall','c_theta','w_theta','u_t','theta_len','n_ambiguous']
    if not a.skip_workbooks:
        workbook(PAPER/'supporting/S2_Table.xlsx',{'Read me':description,'Main summaries':summary,'Main jobs':raw[maincols],
            'Paired species sensitivity':paired,'LOSO deletions':pd.read_csv(a.evidence/'tcbb-postprocess-20261102/results/leave_one_species_out.csv'),
            'Reference baseline':base,'Reference net changes':net,'High depth':high,
            'Mixed orientation':pd.read_csv(ACCEPTED/'mixed_orientation_summary.csv'),
            'Mixed controls':pd.read_csv(ACCEPTED/'mixed_controls.csv'),
            'Old-new pair split':pd.read_csv(ACCEPTED/'old_new_pair_split.csv')})
        workbook(PAPER/'supporting/S5_Table.xlsx',{'Read me':description.iloc[[1,2]],'Read length jobs':length,'Cell means':length_summary})
        workbook(PAPER/'supporting/S6_Table.xlsx',{'Read me':description.iloc[[1,2]],'Empirical jobs':real,'Cell means':real_summary,'Starting references':initial})
    # All manuscript numbers can be traced to exact source rows or summaries.
    lines=[r'\newcommand{\res}[1]{\csname tcbbval#1\endcsname}']
    for k,v in ledger.items():lines.append(r'\expandafter\def\csname tcbbval'+k+r'\endcsname{'+v['text']+'}')
    (GEN/'panel30_values.tex').write_text('\n'.join(lines)+'\n')
    (PAPER/'provenance/panel30-claim-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    receipt={'main_raw_rows':len(raw),'summaries_sha256':{p.name:sha(p) for p in ACCEPTED.glob('*.csv')},
             'auxiliary_sources':{k:v for k,v in aux.items() if k.endswith('_provenance')},
             'auxiliary_counts':{k:len(aux[k]) for k in ['read_length','real_reads','runtime']},
             'generator_sha256':sha(Path(__file__)), 'large_input_cache':str(a.evidence),
             'main_display_names':NAMES,'plots':'Conditional means, output rates with fixed axes; four main tiers; no failed score imputation.',
             'uncertainty':'LOSO sensitivity, no confidence intervals or significance tests.'}
    (PAPER/'provenance/panel30-manuscript-inputs.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Generated main tables, two figures, S2/S5/S6, and provenance; auxiliary counts',receipt['auxiliary_counts'],flush=True)


if __name__=='__main__':main()

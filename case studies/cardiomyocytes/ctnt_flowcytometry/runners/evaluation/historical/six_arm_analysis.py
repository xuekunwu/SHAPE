"""Post-hoc six-arm readouts; preserve frozen evidence and never refit on held-out labels."""

# Internal worker CLI; the supported public entrypoint is run.py.
if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[6]))
    __package__ = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners.evaluation.historical'

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np

ARMS = ['control', 'shape', 'codex', 'shape_new', 'biomni', 'claude']
LABELS = ['CONTROL', 'SHAPE-OLD', 'CODEX', 'SHAPE-NEW', 'Biomni', 'Claude Code']


def regression_readouts(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isfinite([y, p]).all():
        raise ValueError('finite paired 1D observations required')
    e = p-y
    ae = np.abs(e)
    denom = np.var(y)+np.var(p)+(y.mean()-p.mean())**2
    corr = float(np.corrcoef(y,p)[0,1]) if np.std(y)>0 and np.std(p)>0 else None
    return {'n':len(y), 'mae_pp':float(ae.mean()), 'rmse_pp':float(np.sqrt(np.mean(e**2))),
            'bias_pp':float(e.mean()), 'median_ae_pp':float(np.median(ae)), 'p90_ae_pp':float(np.quantile(ae,.9)),
            'max_ae_pp':float(ae.max()), 'pearson_r':corr,
            'ccc':float(2*np.mean((y-y.mean())*(p-p.mean()))/denom) if denom>0 else None,
            'r2':float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if np.var(y)>0 else None,
            'within_pp':{str(t):{'count':int(np.sum(ae<=t)), 'fraction':float(np.mean(ae<=t))} for t in [5,10,15,20]}}


def table(headers, rows):
    def fmt(v):
        if v is None: return 'unavailable'
        if isinstance(v,float): return f'{v:.4f}'
        return str(v)
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def analyze(root):
    consumed={}
    def load(rel):
        path=root/rel
        consumed[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding='utf-8'))
    manifest=load('EXPORT.json')
    assert (root/'EXPORT_COMPLETE').read_text().strip()==consumed['EXPORT.json']
    # Export schema records the source evidence hashes in the comparison.
    expected=load('compare/comparison.json')['input_sha256']
    for rel,sha in expected.items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=sha:
            raise ValueError('Evidence hash mismatch: '+rel)
    data={}
    truth=None
    for a in ARMS:
        rel=f'{a}/heldout_predictions.csv'
        consumed[rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
        with (root/rel).open(newline='',encoding='utf-8') as stream:
            rows=sorted(csv.DictReader(stream),key=lambda r:r['well'])
        actual={r['well']:float(r['ctnt_positive_percent']) for r in rows}
        assert len(actual)==len(rows)==9
        if truth is None: truth=actual
        assert truth==actual
        y=list(actual.values()); p=[float(r['predicted_ctnt_positive_percent']) for r in rows]
        result=load(f'{a}/RESULT.json')
        endpoint=regression_readouts(y,p)
        assert abs(endpoint['mae_pp']-result['primary']['mae_percentage_points'])<1e-9
        readouts=load(f'{a}/STAGE_READOUTS.json')
        reg=load(f'{a}/REGRESSOR.json')
        process={}
        for s in ['segment','crop','embed','batch','predict']:
            rec=load(f'{a}/{s}_COMPLETE.json'); usage=rec.get('usage') or {}
            process[s]={'minutes':rec['seconds']/60,'tokens':usage.get('total_tokens'),
                'llm_calls':usage.get('calls',usage.get('n_rounds')), 'native_calls':usage.get('n_calls'),
                'input_tokens':usage.get('input_tokens'), 'output_tokens':usage.get('output_tokens'),
                'reasoning_tokens':usage.get('reasoning_tokens'), 'reasoning':rec.get('reasoning')}
        decisions={}
        for s in ['segment','crop','embed','batch']:
            rel=f'{a}/agent/{s}/ROUNDS.json'
            if (root/rel).exists():
                rounds=load(rel)
                decisions[s]={'rounds':len(rounds),'actions':dict(Counter(r['action'] for r in rounds)),
                    'outcomes':dict(Counter(r.get('outcome') for r in rounds)),
                    'final_decisions':[{'action':r['action'],'reason':r.get('reason'),'request_json':r.get('request_json')} for r in rounds if r['action']=='finish']}
        data[a]={'endpoint':endpoint,'predictions':rows,'result':result,'readouts':readouts,'process':process,
            'selected_validation_mae_pp':next(t['validation_mae_percentage_points'] for t in reg['trials'] if t['alpha']==reg['alpha']),
            'ridge_alpha':reg['alpha'],'agent_decisions':decisions}
    per_image=[]
    old,new=data['shape']['readouts'],data['shape_new']['readouts']
    for stage in ['segment','crop']:
        left={r['image_id']:r for r in old[stage]['per_image']}; right={r['image_id']:r for r in new[stage]['per_image']}
        assert left.keys()==right.keys() and len(left)==114
        for ident in sorted(left):
            l,r=left[ident],right[ident]
            row={'stage':stage,'image_id':ident,'well':l['well']}
            lv=l if stage=='segment' else {**l['quality'],'retained':l['retained'],'retention':l['retention']}
            rv=r if stage=='segment' else {**r['quality'],'retained':r['retained'],'retention':r['retention']}
            for k,v in lv.items():
                if isinstance(v,(float,int)) and isinstance(rv.get(k),(float,int)):
                    row[k+'_old']=v; row[k+'_new']=rv[k]; row[k+'_delta']=rv[k]-v
            per_image.append(row)
    out=root/'compare'
    fields=list(dict.fromkeys(k for r in per_image for k in r))
    with (out/'per_image_output_changes.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n'); writer.writeheader(); writer.writerows(per_image)
    trajectory=load('compare/trajectory_comparison.json') if (out/'trajectory_comparison.json').exists() else None
    summary={'scope':'posthoc_descriptive_extended_readouts','thresholds_pp':[5,10,15,20],
        'threshold_policy':'exploratory tolerances; not prespecified acceptance criteria',
        'input_sha256':consumed,'arms':data,'per_image_output_changes':per_image,
        'trajectory_counts':trajectory['counts'] if trajectory else None}
    (out/'full_comparison.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    blocks=['<h1>cTnT six-arm pilot: full-stage readouts</h1>',
        '<p>114 images; 18 wells: train 7, validation 2, held out 9. Errors are percentage points (pp). Equal weight per held-out well. Descriptive, single split; no population inference.</p>',
        '<h2>1. Endpoint prediction and tolerance accuracy</h2>',
        '<p>Accuracy within ±t pp = fraction of wells with |predicted CM% − observed cTnT+%| ≤ t. These are exploratory tolerances, not predefined acceptance criteria. This is well-level percentage prediction, not single-cell classification accuracy.</p>',
        table(['Arm','MAE pp','RMSE pp','Bias pp','R²','Pearson r','CCC','Median AE pp','P90 AE pp','≤5 pp','≤10 pp','≤15 pp','≤20 pp'],[
            [label]+[data[a]['endpoint'][k] for k in ['mae_pp','rmse_pp','bias_pp','r2','pearson_r','ccc','median_ae_pp','p90_ae_pp']]+[f"{data[a]['endpoint']['within_pp'][str(t)]['count']}/9 ({data[a]['endpoint']['within_pp'][str(t)]['fraction']:.1%})" for t in [5,10,15,20]] for a,label in zip(ARMS,LABELS)]),
        '<img src="endpoint_predictions.png" alt="Observed versus predicted cTnT percentages in all six arms">',
        '<h2>2. Segmentation and crop quality</h2>',
        '<p>All-image well-macro metrics, including training/validation/held-out images. QC proxies use predicted masks; they are not instance ground-truth AP/F1. Retained cell counts depend on segmentation and filtering and should not be maximized without quality checks.</p>']
    for title,stage,field in [('Segmentation','segment','well_macro'),('Crop','crop','well_macro_quality')]:
        keys=list(data['control']['readouts'][stage][field])
        blocks += ['<h3>'+title+'</h3>',table(['Metric']+LABELS,[[k]+[data[a]['readouts'][stage][field][k] for a in ARMS] for k in keys])]
    blocks += [table(['Arm','Detected cells (all images)','Retained cells','Excluded cells'],[[label,sum(r['cells'] for r in data[a]['readouts']['segment']['per_image']),data[a]['readouts']['crop']['retained_cells'],data[a]['readouts']['crop']['excluded_cells']] for a,label in zip(ARMS,LABELS)]),
        '<h2>3. Embedding and selection</h2>',
        '<p>Validation sibling-site mAP has zero valid query units for every arm. SHAPE therefore skips tuning; native-agent arms can tune on tool-provided SSL loss. SSL eval_loss is not cTnT regression MAE. Different upstream crops prevent causal attribution of endpoint differences to SSL alone.</p>']
    for split in ['development_train','development_val','evaluation_only']:
        rows=[]
        for a,label in zip(ARMS,LABELS):
            e=data[a]['readouts']['embed']['by_split'][split]
            rows.append([label,data[a]['readouts']['embed']['selection']['mode'],e['sampled_cells'],e['health']['effective_rank'],e['health']['dead_dim_frac'],e['health']['mean_cosine'],e['image_leakage'],e['sibling_site_map']['value'],e['sibling_site_map']['n_queries']])
        blocks += ['<h3>'+split+'</h3>',table(['Arm','Selection','Sampled cells','Effective rank','Dead dims','Mean cosine','Image leakage','Sibling-site mAP','Queries'],rows)]
    blocks += ['<h2>4. Batch structure and admission</h2>',
        '<p>All arms leave profiles unchanged. Silhouette reflects batch-associated structure, possibly biology; it is not proof of harmful batch effects or successful correction.</p>',
        table(['Arm','Decision','Unchanged','Date silhouette','Plate silhouette'],[[label,data[a]['readouts']['batch']['admission']['decision'],data[a]['readouts']['batch']['profiles_unchanged']]+[data[a]['readouts']['batch']['structure'][k]['delivered']['value'] for k in ['experiment_date','plate_id']] for a,label in zip(ARMS,LABELS)]),
        '<h2>5. Prediction robustness and model selection</h2>',
        table(['Arm','Ridge alpha','Selected val MAE pp','Donor macro MAE pp','Date macro MAE pp','Plate macro MAE pp'],[[label,data[a]['ridge_alpha'],data[a]['selected_validation_mae_pp'],data[a]['result']['donor_macro_mae_percentage_points']]+[data[a]['result']['batch_metrics'][k]['macro_mae_percentage_points'] for k in ['experiment_date','plate_id']] for a,label in zip(ARMS,LABELS)]),
        '<h2>6. Process and resource use</h2>',
        '<p>Times are recorded stage durations, not matched wall-clock benchmarks. Old SHAPE crop token accounting is a lower bound after recovery. LLM calls and native tool calls are different counts. Provider-reported tokens are not reconciled bills.</p>']
    for metric in ['minutes','tokens','llm_calls','native_calls']:
        blocks += ['<h3>'+metric+'</h3>',table(['Stage']+LABELS,[[s]+[data[a]['process'][s][metric] for a in ARMS] for s in ['segment','crop','embed','batch','predict']])]
    if trajectory:
        old=trajectory['logged_usage']['shape']['crop']['logged_tokens']
        new=trajectory['logged_usage']['shape_new']['crop']['logged_tokens']
        new_total=data['shape_new']['process']['crop']['tokens']
        blocks += ['<h3>Recovered crop accounting supersedes the compact-export lower bound</h3>',
            f'<p>The original old-SHAPE crop aggregate above is incomplete. Search logs recover {old:,} tokens, versus {new:,} in new-SHAPE search logs ({1-new/old:.1%} fewer). New-SHAPE full crop usage is {new_total:,}; hence even this old partial total proves at least {1-new_total/old:.1%} fewer full-stage tokens, subject to backend usage accounting. The exact old full-stage total is still unavailable. Do not add the original 493,695 to the recovered total: they overlap.</p>']
    blocks += ['<h3>Native-agent decision outcomes</h3>',table(['Arm','Stage','Decision rounds','Executed rounds','Rejected rounds','Finished rounds'],[[a,s,d['rounds'],d['outcomes'].get('executed',0),d['outcomes'].get('rejected',0),d['outcomes'].get('finished',0)] for a in ARMS for s,d in data[a]['agent_decisions'].items()])]
    blocks += ['<h2>7. Paired image outputs and decision trajectories</h2>',
        '<p><a href="per_image_output_changes.csv">Every-image output differences</a> compare identical image IDs. Output differences do not reveal which skipped LLM call caused them. <a href="trajectory_report.html">Decision trajectory report</a> uses separately harvested search logs when available.</p>',
        '<h2>8. What remains unmeasured</h2>',
        table(['Readout','Reason unavailable'],list(data['control']['readouts']['unavailable'].items())),
        '<p>SHAPE-new changes both code/policy and image-stage reasoning effort; it is not a clean ablation. Claude uses a different model. Biomni is decision-only. Segmentation calibrates jointly on unlabeled images across splits (transductive preprocessing). Do not tune against these held-out results and then treat the same wells as an untouched confirmation set.</p>',
        '<p>Sources and hashes: <a href="full_comparison.json">full_comparison.json</a>. Reproduce with analyze_full_suite.py EXPORT_ROOT. Frozen observations remain unchanged.</p>']
    document='<!doctype html><meta charset="utf-8"><title>cTnT full-stage comparison</title><style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#172238}table{border-collapse:collapse;font-size:14px;width:100%;margin:16px 0 28px}th,td{padding:9px;border-bottom:1px solid #d7dfe9;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#ecf1f7}h2{margin-top:40px}img{max-width:100%}p{line-height:1.6;max-width:1100px}</style>'+''.join(blocks)
    (out/'full_report.html').write_text(document,encoding='utf-8',newline='\n')
    plot(root, data, consumed)
    print(json.dumps({a:data[a]['endpoint'] for a in ARMS},indent=2))


def plot(root,data,inputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
    colors={'p000':'#0072B2','p004':'#E69F00','p005':'#009E73','p006':'#CC79A7'}
    for ax,a,label in zip(axes.flat,ARMS,LABELS):
        ax.plot([0,100],[0,100],color='#555555',lw=1)
        ax.fill_between([0,100],[-10,90],[10,110],color='#aaaaaa',alpha=.15)
        for r in data[a]['predictions']:
            x=float(r['ctnt_positive_percent']); y=float(r['predicted_ctnt_positive_percent'])
            ax.scatter(x,y,color=colors[r['plate_id']],s=45,edgecolor='white',linewidth=.5)
            if r['well'] in ['w0077','w0085']:
                ax.annotate(r['well'],(x,y),xytext=(5,8 if r['well']=='w0085' else -13),textcoords='offset points',fontsize=8)
        e=data[a]['endpoint']
        ax.set_title(f"{label}\nMAE {e['mae_pp']:.2f} pp; within ±10 pp {e['within_pp']['10']['count']}/9",fontsize=11)
        ax.set(xlim=(0,100),ylim=(0,100));ax.grid(alpha=.15);ax.set_aspect('equal')
    fig.supxlabel('Observed cTnT-positive (%)');fig.supylabel('Predicted cTnT-positive (%)')
    fig.suptitle('Same 9 held-out wells per arm | gray band: ±10 percentage points',fontsize=13)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,label=p) for p,c in colors.items()]
    axes[0,0].legend(handles=handles,loc='lower right',ncol=2,title='Plate',fontsize=8,title_fontsize=8)
    path=root/'compare/endpoint_predictions.png'
    fig.savefig(path,dpi=180);plt.close(fig)
    evidence={'inputs':inputs,'script':{'path':str(Path(__file__).name),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'parameters':{'seed':None,'tolerance_band_pp':10,'wells_per_arm':9,'excluded_wells':[]},
        'software':{'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'output':{'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
        'render_qa':{'inspected':False,'disposition':'pending'}}
    (root/'compare/FIGURE_EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('export_root',type=Path)
    analyze(parser.parse_args().export_root.resolve())

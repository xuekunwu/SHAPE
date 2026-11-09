"""Pair all SHAPE image trajectories from a verified harvest, without replaying inference."""

# Internal worker CLI; the supported public entrypoint is run.py.
if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[6]))
    __package__ = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners.evaluation.historical'

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import random
from pathlib import Path


def load(path): return json.loads(path.read_text(encoding='utf-8'))


def log_summary(path):
    labels=defaultdict(Counter); kinds=Counter(); sequence=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            e=json.loads(line); kinds[e['kind']]+=1
            if e['kind']=='usage':
                usage=json.loads(e['text']); label=e.get('label','unlabelled')
                labels[label]['calls']+=1
                labels[label].update({k:v for k,v in usage.items() if isinstance(v,(float,int))})
                sequence.append(label)
    return {'log_exists':path.exists(),'labels':dict(labels),'kinds':dict(kinds),'usage_sequence':sequence}


def image_record(base, stage, ident):
    folder='logs/segment/search' if stage=='segment' else 'logs/crop'
    detail=base/folder/(ident+'.details.json')
    search=base/stage/ident/'SEARCH.json'
    if not search.exists(): return None
    outcome=load(search)
    d=load(detail)['details'] if detail.exists() else {}
    mask=d.get('mask_check') or {}
    return {'details_present':detail.exists(),'delivered':outcome['delivered'],'levers':outcome.get('levers'),
        'per_cell_overrides':outcome.get('cell_overrides'),
        'boxes':outcome.get('boxes'),'rejected_cells':outcome.get('rejected_tool_cells'),
        'admission':outcome.get('admission'),'n_native_calls':outcome.get('n_calls'),
        'stop_reason':outcome.get('resolution',{}).get('reason'),'seconds':outcome.get('seconds'),
        'rounds':d.get('rounds'),'entries':[e for e in d.get('entries',[]) if e],
        'strategy':(d.get('search_strategy') or {}).get('name','legacy'),
        'search_phases':[h.get('phase') for h in (d.get('search_strategy') or {}).get('history',[])],
        'choice':d.get('choice'),'plan_delivery':d.get('plan_delivery'),
        'efficiency_events':d.get('efficiency_events',[]),'efficiency_policy':d.get('efficiency_policy'),
        'mask_check':{**{k:v for k,v in mask.items() if k not in ['montages','rejected_cells']},'montage_count':len(mask.get('montages',[]))},
        'log':log_summary(base/folder/(ident+'.jsonl'))}


def analyze(evidence, export):
    manifest=load(evidence/'MANIFEST.json')
    for rel,meta in manifest['files'].items():
        if hashlib.sha256((evidence/rel).read_bytes()).hexdigest()!=meta['sha256']:
            raise ValueError('harvest hash mismatch: '+rel)
    for a in ['shape','shape_new','codex','claude','biomni']:
        assert (evidence/a/'DELIVERY.json').read_bytes()==(export/a/'DELIVERY.json').read_bytes()
    masks={a:{r['image_id']:r for r in load(evidence/a/'segmentation.json')} for a in ['shape','shape_new']}
    assert masks['shape'].keys()==masks['shape_new'].keys() and len(masks['shape'])==114
    images={}; totals={}; events={}
    for a in masks:
        totals[a]={};events[a]=Counter()
        for stage in ['segment','crop']:
            labels=defaultdict(Counter);kinds=Counter()
            for path in (evidence/a/'logs'/stage).rglob('*.jsonl'):
                logs=log_summary(path);kinds.update(logs['kinds'])
                for label,counts in logs['labels'].items(): labels[label].update(counts)
            totals[a][stage]={'labels':dict(labels),'kinds':dict(kinds),
                'logged_tokens':sum(v.get('total_tokens',0) for v in labels.values()),'logged_calls':sum(v['calls'] for v in labels.values())}
        for path in (evidence/a/'logs/segment').rglob('*.details.json'):
            for e in load(path)['details'].get('efficiency_events',[]): events[a][e['action']]+=1
    flat=[]
    for ident in sorted(masks['shape']):
        pair={'mask_identical':masks['shape'][ident]['sha256']==masks['shape_new'][ident]['sha256']}
        row={'image_id':ident,'mask_identical':pair['mask_identical']}
        for a in masks:
            record={'final_segmentation':masks[a][ident],
                'segment_search':image_record(evidence/a,'segment',ident),
                'crop':image_record(evidence/a,'crop',ident)}
            cohort=evidence/a/'segment'/ident/'COHORT_SEARCH.json'
            record['segment_cohort']=load(cohort) if cohort.exists() else None
            pair[a]=record
            row[a+'_routing']=masks[a][ident]['parameter_source']
            for stage,key in [('segment','segment_search'),('crop','crop')]:
                r=record[key]
                row[a+'_'+stage+'_calls']=r['n_native_calls'] if r else None
                row[a+'_'+stage+'_choice']=r['delivered'] if r else None
                row[a+'_'+stage+'_logged_tokens']=sum(v.get('total_tokens',0) for v in r['log']['labels'].values()) if r else None
        c0,c1=pair['shape']['crop'],pair['shape_new']['crop']
        pair['same_crop_geometry']=c0['boxes']==c1['boxes'] if pair['mask_identical'] else None
        pair['same_crop_rejections']=set(c0['rejected_cells'] or [])==set(c1['rejected_cells'] or []) if pair['mask_identical'] else None
        pair['same_crop_image_parameters']=c0['levers']==c1['levers']
        row.update({k:pair[k] for k in ['same_crop_geometry','same_crop_rejections','same_crop_image_parameters']})
        flat.append(row);images[ident]=pair
    ssl={}
    for a in ['codex','claude','biomni']:
        rows=[]
        for path in sorted((evidence/a/'agent/embed/calls').glob('*/ssl_result.json')):
            j=load(path);train=j.get('training') or {}; conf=j.get('training_config') or {}
            vals=[v['eval_loss'] for v in j.get('val_history',[]) if v.get('eval_loss') is not None]
            rows.append({'call':path.parent.name,'mode':j.get('mode'),'epochs_trained':j.get('epochs_trained'),
                'best_ssl_eval_loss':min(vals) if vals else None,'checkpoint_best_by':train.get('best_by'),
                'checkpoint_best_epoch_within_call':train.get('best_epoch'),
                'checkpoint_cumulative_epoch':((conf.get('resume') or {}).get('epoch',0)+train['best_epoch']) if train.get('best_epoch') is not None else None,
                'resume_from_epoch':(conf.get('resume') or {}).get('epoch'),
                'last_epoch_reached':(conf.get('resume') or {}).get('epoch',0)+(j.get('epochs_trained') or 0),
                'trainable_blocks':train.get('trainable_blocks'),
                'valid_sibling_site_map_values':sum(v.get('sibling_site_map') is not None for v in j.get('val_history',[])),
                'biological_labels_used_for_ssl_selection':(j.get('ssl_diagnostics') or {}).get('biological_labels_used_for_ssl_selection')})
        ssl[a]=rows
    counts={'images':114,'identical_final_masks':sum(p['mask_identical'] for p in images.values()),
        'same_crop_image_parameters':sum(p['same_crop_image_parameters'] for p in images.values()),
        'same_geometry_given_identical_mask':sum(p['same_crop_geometry'] is True for p in images.values()),
        'same_rejections_given_identical_mask':sum(p['same_crop_rejections'] is True for p in images.values()),
        'changed_geometry_given_identical_mask':sum(p['same_crop_geometry'] is False for p in images.values()),
        'changed_rejections_given_identical_mask':sum(p['same_crop_rejections'] is False for p in images.values())}
    cohort_config={}
    for a in masks:
        c=load(evidence/a/'segment_COHORT.json')
        reproduced=[]
        for g in c['groups']:
            if not g.get('calibration'): continue
            key=[g['group']['specimen_context'],g['group']['modality']]
            seed=int(hashlib.sha256(json.dumps([key,sorted(g['images'])]).encode()).hexdigest()[:16],16)
            drawn=sorted(random.Random(seed).sample(sorted(g['images']),len(g['calibration'])))
            assert drawn==g['calibration'], 'cohort draw could not be reproduced'
            reproduced.append(True)
        cohort_config[a]={'calibration_mode_recorded':c.get('calibration_mode'),'counts':c['counts'],
            'calibration_draw_reproduced':all(reproduced),
            'groups':[{'group':g['group'],'images':len(g['images']),'image_ids':g['images'],'calibration':g.get('calibration'),
                'calibration_policy':g.get('calibration_policy'),
                'consensus_selected_params':(g.get('consensus') or {}).get('selected_params')} for g in c['groups']]}
    summary={'evidence_manifest_sha256':hashlib.sha256((evidence/'MANIFEST.json').read_bytes()).hexdigest(),
        'counts':counts,'logged_usage':totals,'efficiency_events':events,
        'cohort_config':cohort_config,
        'ssl_candidates':ssl,'images':images,
        'scope':'Search-log tokens omit preflight/mask-check calls not emitted through the search logger; no billing reconciliation.',
        'caution':'Log/details files were hashed at harvest; SEARCH.json and final output hashes are independently attested by the original DELIVERY manifest.'}
    out=export/'compare'
    (out/'trajectory_comparison.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    with (out/'per_image_trajectories.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    def table(headers,rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    blocks=['<h1>SHAPE per-image trajectory comparison</h1><p><a href="full_report.html">Full six-arm report</a> | <a href="per_image_trajectories.csv">114-image summary CSV</a> | <a href="trajectory_comparison.json">Structured traces</a></p>',
        '<p>All 114 matched images. Decision paths and output differences are observed; causality requires fixed-input ablations. Original stage records remain unchanged.</p>',
        table(['Output comparison','Count'],counts.items()),
        '<h2>Logged LLM questions and tokens</h2><p>This is the search logger scope, excluding independently invoked preflight/mask checks. Old crop aggregate usage was lost after recovery; per-question logs recover a larger, defensible lower bound.</p>']
    for stage in ['segment','crop']:
        labels=sorted(set(totals['shape'][stage]['labels'])|set(totals['shape_new'][stage]['labels']))
        rows=[]
        for label in labels:
            old=totals['shape'][stage]['labels'].get(label,{})
            new=totals['shape_new'][stage]['labels'].get(label,{})
            rows.append([label,old.get('calls',0),new.get('calls',0),old.get('total_tokens',0),new.get('total_tokens',0)])
        blocks+=['<h3>'+stage+'</h3>',table(['Question','Old calls','New calls','Old tokens','New tokens'],rows)]
    blocks += ['<h2>Mechanistic interpretation</h2>',
        '<p>Segmentation keeps four-stage search and six-image calibration. The old record does not contain a calibration_mode field; flagged is a routing count, not a calibration policy. Preflight group membership changes: old i0532 and new i0594 are classified as microorganism, while the other 113 images are cell_culture. The sampler seeds from the group key and sorted membership. Replaying that sampler exactly reproduces both disjoint six-image calibration sets. Thus upstream LLM classification instability changes calibration exposure before tuning. Both cohort consensus selected_params are empty.</p>',
        '<p>Compact prompts are disabled. New efficiency events: nine valid-stop confirmation skips and one zero-problem early stop/final-selection skip. The early-stop image i0507 was not searched in the old run, so this event is not a matched old/new saving on that image. Native tool calls, search questions and global LLM calls are distinct counts.</p>',
        '<p>Crop changes from four-stage search to legacy search, besides skipping redundant valid-stop confirmation and unique-candidate ranking. Mask checking stays enabled. This is a broader policy change plus high-to-medium effort, not just removal of redundant questions.</p>',
        '<p>Same-mask comparisons isolate observed downstream crop changes from changed segmentation inputs, but still combine effort, search policy and stochastic LLM decisions.</p>',
        '<p>The four images of the worst-error well w0085 (i0592–i0595) have identical final segmentation masks across old/new SHAPE, but all four crop geometries differ. Old/new well predictions are 36.29%/51.41% versus 7.45% observed. This narrows the changed stages to cropping and downstream profiles/regression; it does not isolate the effect of that well’s crops, because training-well profiles also changed.</p>',
        '<h2>SSL candidate evidence</h2><p>The outer agents choose c8 citing SSL evaluation loss. The delivered tool checkpoint is marked best_by=train_loss. Neither criterion is cTnT regression error. All candidate validation sibling-site mAP values are unavailable. Final epoch reached is not necessarily the epoch of the delivered best checkpoint.</p>']
    for a,rows in ssl.items():
        blocks+=['<h3>'+a+'</h3>',table(['Candidate','Epochs this call','Resume epoch','Last epoch reached','Min SSL eval loss','Checkpoint criterion','Best cumulative epoch'],[[r[k] for k in ['call','epochs_trained','resume_from_epoch','last_epoch_reached','best_ssl_eval_loss','checkpoint_best_by','checkpoint_cumulative_epoch']] for r in rows])]
    blocks+=['<h2>Every image</h2><p>Expand an image to inspect candidate parameter changes, admitted choices, question categories and selected outputs. Candidate numbers are local to each run and are not semantically comparable without parameters.</p>']
    for ident,pair in images.items():
        blocks+=['<details><summary>'+ident+' — mask identical: '+str(pair['mask_identical'])+'; crop geometry identical: '+str(pair['same_crop_geometry'])+'</summary>']
        for a in masks:
            blocks+=['<h3>'+a+'</h3>']
            blocks+=['<p>Final routing: '+html.escape(pair[a]['final_segmentation']['parameter_source'])+'</p>']
            if pair[a]['segment_cohort']:
                cohort=pair[a]['segment_cohort']
                blocks+=['<h4>Cohort application</h4><pre>'+html.escape(json.dumps({k:cohort.get(k) for k in ['delivered','levers','n_calls','admission','resolution']},indent=2))+'</pre>']
            for s in ['segment_search','crop']:
                r=pair[a][s]
                if r is None:
                    blocks+=['<p>'+s+': no per-image search record; see cohort route.</p>'];continue
                info={k:r[k] for k in ['delivered','levers','n_native_calls','stop_reason','strategy','search_phases','choice','efficiency_events','mask_check']}
                blocks+=['<h4>'+s+'</h4><pre>'+html.escape(json.dumps(info,indent=2))+'</pre>',
                    table(['Index','Parent','Parameter change','Outcome'],[[e.get('index'),e.get('parent'),json.dumps(e.get('param_delta')),e.get('outcome')] for e in r['entries']]),
                    '<p>Question sequence: '+html.escape(' → '.join(r['log']['usage_sequence']))+'</p>']
        blocks+=['</details>']
    document='<!doctype html><meta charset="utf-8"><title>SHAPE trajectories</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#172238}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}td,th{padding:8px;border-bottom:1px solid #d7dfe9;text-align:left}th{background:#edf2f8}pre{white-space:pre-wrap;background:#f6f8fb;padding:12px}summary{cursor:pointer;padding:14px;background:#edf2f8;margin:8px 0}p{line-height:1.6}</style>'+''.join(blocks)
    (out/'trajectory_report.html').write_text(document,encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['images','ssl_candidates']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('export',type=Path)
    args=p.parse_args();analyze(args.evidence.resolve(),args.export.resolve())

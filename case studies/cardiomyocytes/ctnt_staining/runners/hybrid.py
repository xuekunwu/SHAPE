"""Eligible-plate SSL search plus independently fitted cell heads; development-only routing."""
from pathlib import Path
from collections import defaultdict
import shutil
import numpy as np

from . import data
from ...ctnt_flowcytometry.runners.prediction.plate_ssl_routing import eligible_plates

POLICY='staining-plate-hybrid-validation-brier-v1'


def training_summary(search):
    """Training execution and checkpoint adoption are separate facts."""
    calls=[]
    for path in sorted((Path(search)/'ssl').glob('call_*.params.json')):
        params=data.read_json(path);result_path=path.with_name(path.name.replace('.params.json','.result.json'))
        result=data.read_json(result_path) if result_path.exists() else {}
        calls.append(dict(parameters=str(path),result=str(result_path) if result_path.exists() else None,
            training_requested=params.get('adaptation_depth')!='zero_shot' and not params.get('embed_only',False),
            epochs_trained=result.get('epochs_trained'),steps_trained=result.get('steps_trained'),
            checkpoint_available=bool(result.get('resume_best'))))
    requested=[c for c in calls if c['training_requested']]
    executed=any((c.get('epochs_trained') or 0)>0 or (c.get('steps_trained') or 0)>0 for c in requested)
    unknown=any(c['epochs_trained'] is None and c['steps_trained'] is None for c in requested)
    return dict(training_attempted=bool(requested),training_executed=True if executed else None if unknown else False,
                execution_telemetry_complete=not unknown,calls=calls)


def representation(prep,run,config,rows):
    from benchmarks.full_CPJUMP1.harness import pipeline
    cells=data.read_json(run/'cells.json')
    eligibility=eligible_plates(rows)
    data.write_json(run/'ELIGIBILITY.json',eligibility)
    eligible={p for p,f in eligibility.items() if f['eligible']}
    zero={'mode':'zero_shot','checkpoint':None,'reason':'HYBRID zero-shot expert; eligible-plate training searched separately',
          'hybrid_protocol':POLICY,'eligible_plates':sorted(eligible)}
    pipeline.representation(prep,run,{**config,'ssl_selection':zero},rows)
    search_selection=None
    if eligible:
        search=run/'hybrid/ssl_search';search.mkdir(parents=True)
        development=[r for r in rows if r['plate_id'] in eligible and r['usage']!='evaluation_only']
        development_cells=[c for c in cells if c['plate'] in eligible and c['usage']!='evaluation_only']
        assert development_cells and not any(c['usage']=='evaluation_only' for c in development_cells)
        search_config={**config,'images':[r['image_id'] for r in development],
            'ssl_epoch_budget':config['ssl_epoch_budget'],
            'ssl_budget_scope':'full frozen pilot, identical epoch cap across arms',
            'fit_only':True}
        search_config.pop('ssl_selection',None)
        data.write_json(search/'CONFIG.json',search_config)
        data.write_json(search/'cells.json',development_cells)
        # Only eligible development cells enter the tool. Full image metadata supplies the shared epoch cap.
        pipeline.representation(prep,search,search_config,rows)
        search_selection=data.read_json(search/'SSL_SELECTION.json')
        if search_selection.get('checkpoint'):
            fine=run/'hybrid/finetuned';fine.mkdir(parents=True)
            data.write_json(fine/'cells.json',cells)
            fine_config={**config,'ssl_selection':search_selection}
            data.write_json(fine/'CONFIG.json',fine_config)
            pipeline.representation(prep,fine,fine_config,rows)
    data.write_json(run/'HYBRID_SSL.json',{'policy':POLICY,'eligibility':eligibility,
        'eligible_plates':sorted(eligible),'candidate':search_selection,
        'frozen_finetuned_embedding_available':(run/'hybrid/finetuned/embedding.npy').exists(),
        'training_scope':'eligible plate development_train only; validation selects; held-out inference only',
        'routing':'Per-plate validation well-macro Brier must improve; ties and ineligible plates keep zero-shot',
        'heldout_labels_used':False})
    execution=training_summary(run/'hybrid/ssl_search')
    data.write_json(run/'SSL_SELECTION.json',dict(mode='plate_hybrid',hybrid_protocol=POLICY,
        routing_status='pending_development_validation',checkpoint=None,
        experts=dict(zero_shot=zero,finetuned=search_selection),
        training_execution=execution,training_executed=execution['training_executed'],
        checkpoint_selected=bool(search_selection and search_selection.get('checkpoint')),
        eligible_plates=sorted(eligible),
        fallback_embedding_scope='embedding.npy is the frozen expert; final predictions use HYBRID_ROUTES.json'))


def choose_routes(cells,labels,zero,fine,eligibility):
    """Inputs contain development labels only; equal total weight per validation well."""
    if any(r['usage']=='evaluation_only' for r in labels.values()):
        raise ValueError('held-out labels cannot route experts')
    routes={}
    for plate,info in eligibility.items():
        wells=defaultdict(list)
        for i,c in enumerate(cells):
            if c['plate']==plate and c['usage']=='development_val' and c['cell_id'] in labels:
                wells[c['well']].append(i)
        def loss(prob):
            if prob is None or not wells:return None
            return float(np.mean([np.mean([(float(prob[i])-labels[cells[i]['cell_id']]['ctnt_positive'])**2
                                          for i in ix]) for ix in wells.values()]))
        z,f=loss(zero),loss(fine)
        use=bool(info['eligible'] and len(wells)>=2 and f is not None and z is not None and f<z)
        reason=('validation_brier_improved' if use else 'insufficient_validation_wells' if not info['eligible'] or len(wells)<2
                else 'ssl_candidate_not_adopted' if fine is None else 'no_validation_brier_improvement')
        routes[plate]={'branch':'finetuned' if use else 'zero_shot','reason':reason,
            'matched_validation_wells':len(wells),'zero_validation_brier':z,'fine_validation_brier':f}
    return routes


def predict(prep,run,references,minimum_iou):
    from . import score
    cells=data.read_json(run/'cells.json');config=data.read_json(run/'CONFIG.json')
    labels=score.matched_development(prep,run,references,minimum_iou)
    zero,zparams,zselection=score.fit_classifier(np.load(run/'embedding.npy'),cells,labels,config['seed'])
    fine=None;fselection=None
    path=run/'hybrid/finetuned/embedding.npy'
    if path.exists():
        fine,fparams,fselection=score.fit_classifier(np.load(path),cells,labels,config['seed'])
        np.savez(run/'hybrid/finetuned/classifier.npz',**fparams)
    eligibility=data.read_json(run/'ELIGIBILITY.json')
    routes=choose_routes(cells,labels,zero,fine,eligibility)
    np.savez(run/'classifier.npz',**zparams)
    selection={'mode':'plate_hybrid','C':None,'classifier_probability_threshold':None,
        'hybrid_protocol':POLICY,'routes':routes,'zero_shot':zselection,'finetuned':fselection,
        'classifier_probability_threshold_scope':'Per expert; each prediction records its frozen threshold',
        'weights_sha256':data.digest(run/'classifier.npz'),'input_embedding_sha256':data.digest(run/'embedding.npy'),
        'reference_complete_sha256':data.digest(references/'REFERENCE_COMPLETE.json'),'minimum_matching_iou':minimum_iou,
        'target':'ctnt_positive','prediction_channels':['Brightfield']}
    records=[]
    for i,c in enumerate(cells):
        branch=routes[c['plate']]['branch'];fin=branch=='finetuned'
        p=float(fine[i] if fin else zero[i]);head=fselection if fin else zselection
        threshold=head['classifier_probability_threshold']
        records.append({k:c[k] for k in ('cell_id','image_id','well','plate','usage','mask_label')})
        records[-1].update(ctnt_positive_probability=p,ctnt_positive_prediction=int(p>=threshold),
            prediction_branch=branch,classifier_probability_threshold=threshold)
    validation=[r for r in records if r['usage']=='development_val' and r['cell_id'] in labels]
    selection['validation']=score.metrics([labels[r['cell_id']]['ctnt_positive'] for r in validation],
        [r['ctnt_positive_probability'] for r in validation],None,
        hard_predictions=[r['ctnt_positive_prediction'] for r in validation])
    selection['validation_scope']='actual routed validation predictions; per-expert frozen thresholds'
    selection['thresholds_by_expert']={'zero_shot':zselection['classifier_probability_threshold'],
        'finetuned':fselection['classifier_probability_threshold'] if fselection else None}
    ssl=data.read_json(run/'SSL_SELECTION.json')
    ssl.update(mode='plate_hybrid',routing_status='frozen',routes=routes,
        adopted_finetuned_plates=sorted(p for p,v in routes.items() if v['branch']=='finetuned'))
    data.write_json(run/'SSL_SELECTION.json',ssl)
    data.write_json(run/'HYBRID_ROUTES.json',routes)
    data.write_json(run/'CLASSIFIER.json',selection)
    data.write_json(run/'predictions.json',records)
    frozen={'status':'frozen','predictions_sha256':data.digest(run/'predictions.json'),
        'classifier_sha256':data.digest(run/'CLASSIFIER.json'),'weights_sha256':data.digest(run/'classifier.npz'),
        'embedding_sha256':data.digest(run/'embedding.npy'),'cells_sha256':data.digest(run/'cells.json'),
        'segmentation_sha256':data.digest(run/'segmentation.json'),
        'reference_complete_sha256':data.digest(references/'REFERENCE_COMPLETE.json'),
        'hybrid_routes_sha256':data.digest(run/'HYBRID_ROUTES.json')}
    if path.exists():
        frozen['finetuned_artifacts']={str(p.relative_to(run)):data.digest(p) for p in
            (path,run/'hybrid/finetuned/classifier.npz')}
    data.write_json(run/'PREDICTIONS_COMPLETE.json',frozen)
    return selection

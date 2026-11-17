"""Training-only cell classifier and independent, coverage-aware fluorescence scoring."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import tifffile
from . import data
from .matching import match_objects
from .references import verify as verify_references


def metrics(labels, probabilities, threshold=.5, *, hard_predictions=None):
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
    y=np.asarray(labels);p=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or y.shape!=p.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not set(y.tolist())<={0,1}:
        raise ValueError('invalid binary labels/probabilities')
    if threshold is None:
        if hard_predictions is None:raise ValueError('mixed thresholds require explicit predictions')
    elif not 0<=threshold<=1:raise ValueError('invalid classifier threshold')
    result={k:None for k in ['auroc','auprc','accuracy','balanced_accuracy','sensitivity','specificity','precision','f1','brier','ece_10bins']}
    result.update(cells=len(y),positive_cells=int(y.sum()),negative_cells=int(len(y)-y.sum()))
    if not len(y):return result
    pred=p>=threshold if hard_predictions is None else np.asarray(hard_predictions,dtype=bool)
    if pred.shape!=y.shape:raise ValueError('hard predictions do not match labels')
    tp=int(np.sum(pred&(y==1)));tn=int(np.sum(~pred&(y==0)))
    fp=int(np.sum(pred&(y==0)));fn=int(np.sum(~pred&(y==1)))
    sensitivity=tp/(tp+fn) if tp+fn else None;specificity=tn/(tn+fp) if tn+fp else None
    both=len(np.unique(y))==2
    result.update(auroc=float(roc_auc_score(y,p)) if both else None,auprc=float(average_precision_score(y,p)) if both else None,
        accuracy=(tp+tn)/len(y),balanced_accuracy=(sensitivity+specificity)/2 if both else None,
        sensitivity=sensitivity,specificity=specificity,precision=tp/(tp+fp) if tp+fp else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,brier=float(brier_score_loss(y,p)),
        confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn),classifier_probability_threshold=threshold,
        single_class_note=None if both else 'AUROC/AUPRC/balanced accuracy are not informative for a single-class group')
    bins=np.minimum((p*10).astype(int),9)
    result['ece_10bins']=float(sum(np.mean(bins==i)*abs(np.mean(p[bins==i])-np.mean(y[bins==i])) for i in range(10) if np.any(bins==i)))
    return result


def fit_classifier(x,cells,labels,seed=20261017):
    """Accept development targets only; held-out inputs never enter scaling or selection."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or len(x)!=len(cells) or not np.isfinite(x).all():raise ValueError('invalid cell embeddings')
    if any(r['usage']=='evaluation_only' for r in labels.values()):raise ValueError('held-out target passed to fitting')
    train=[i for i,c in enumerate(cells) if c['usage']=='development_train' and c['cell_id'] in labels]
    val=[i for i,c in enumerate(cells) if c['usage']=='development_val' and c['cell_id'] in labels]
    if not train or not val:raise ValueError('need matched, confident train and validation reference cells')
    if {cells[i]['well'] for i in train}&{cells[i]['well'] for i in val}:raise ValueError('well leakage')
    y=np.array([labels[cells[i]['cell_id']]['ctnt_positive'] for i in train])
    vy=np.array([labels[cells[i]['cell_id']]['ctnt_positive'] for i in val])
    if set(y)!={0,1}:raise ValueError('training references lack both cTnT classes')
    counts=Counter(cells[i]['well'] for i in train)
    weights=np.array([1/counts[cells[i]['well']] for i in train]);weights*=len(weights)/weights.sum()
    scaler=StandardScaler().fit(x[train]);xt=scaler.transform(x[train]);xv=scaler.transform(x[val])
    trials=[];best=None
    # If validation is single-class, do not pretend to select by AUROC or optimize its class threshold.
    candidates=(.01,.1,1.,10.,100.) if set(vy)=={0,1} else (1.,)
    for c in candidates:
        model=LogisticRegression(C=c,max_iter=1000,random_state=seed).fit(xt,y,sample_weight=weights)
        vp=model.predict_proba(xv)[:,1];quality=metrics(vy,vp)
        trials.append(dict(C=c,validation=quality))
        value=quality['auroc'] if quality['auroc'] is not None else 0.
        if best is None or value>best[0]:best=(value,c,model,vp)
    threshold=.5
    if set(vy)=={0,1}:
        thresholds=[.5,*[float(t) for t in np.linspace(.05,.95,19) if abs(t-.5)>1e-8]]
        threshold=max(thresholds,key=lambda t:metrics(vy,best[3],t)['balanced_accuracy'])
    probability=best[2].predict_proba(scaler.transform(x))[:,1]
    params=dict(mean=scaler.mean_,scale=scaler.scale_,coef=best[2].coef_,intercept=best[2].intercept_)
    return probability,params,dict(C=best[1],trials=trials,classifier_probability_threshold=threshold,
        fit_split='development_train',selection_split='development_val',refit_on_validation=False,
        train_cells=len(train),validation_cells=len(val),training_wells=sorted(counts),
        training_weighting='equal total weight per training well',
        threshold_selection='validation balanced accuracy' if set(vy)=={0,1} else 'fixed .5; single-class validation',
        validation=metrics(vy,best[3],threshold))


def matched_development(prep,run,references,minimum_iou=.5):
    _,ref=verify_references(prep,references)
    rows={r['image_id']:r for r in data.load_prepared(prep)}
    labels={(r['image_id'],r['mask_label']):r for r in data.read_json(references/'development_labels.json')}
    cells=data.read_json(run/'cells.json');masks={r['image_id']:r for r in data.read_json(run/'segmentation.json')}
    result={}
    for ident,mask in masks.items():
        if rows[ident]['usage']=='evaluation_only':continue
        selected={c['mask_label']:c for c in cells if c['image_id']==ident}
        predicted=tifffile.imread(mask['mask'])
        predicted=np.where(np.isin(predicted,list(selected)),predicted,0)
        matched=match_objects(predicted,tifffile.imread(ref[ident]['mask']),minimum_iou)
        for pair in matched['matches']:
            r=labels[(ident,pair['reference_label'])]
            if r['ctnt_positive'] is not None:result[selected[pair['predicted_label']]['cell_id']]=r
    return result


def predict(prep,run,references,minimum_iou=.5):
    cells=data.read_json(run/'cells.json');config=data.read_json(run/'CONFIG.json')
    labels=matched_development(prep,run,references,minimum_iou)
    probability,params,selection=fit_classifier(np.load(run/'embedding.npy'),cells,labels,config['seed'])
    np.savez(run/'classifier.npz',**params)
    selection.update(weights_sha256=data.digest(run/'classifier.npz'),input_embedding_sha256=data.digest(run/'embedding.npy'),
        reference_complete_sha256=data.digest(references/'REFERENCE_COMPLETE.json'),minimum_matching_iou=minimum_iou,
        target='ctnt_positive',prediction_channels=['Brightfield'])
    data.write_json(run/'CLASSIFIER.json',selection)
    predictions=[dict(cell_id=c['cell_id'],image_id=c['image_id'],well=c['well'],plate=c['plate'],usage=c['usage'],
        mask_label=c['mask_label'],ctnt_positive_probability=float(p),ctnt_positive_prediction=int(p>=selection['classifier_probability_threshold'])) for c,p in zip(cells,probability)]
    data.write_json(run/'predictions.json',predictions)
    data.write_json(run/'PREDICTIONS_COMPLETE.json',dict(status='frozen',predictions_sha256=data.digest(run/'predictions.json'),
        classifier_sha256=data.digest(run/'CLASSIFIER.json'),weights_sha256=data.digest(run/'classifier.npz'),
        embedding_sha256=data.digest(run/'embedding.npy'),cells_sha256=data.digest(run/'cells.json'),
        segmentation_sha256=data.digest(run/'segmentation.json'),reference_complete_sha256=data.digest(references/'REFERENCE_COMPLETE.json')))
    return selection


def clustered_interval(records,threshold,draws=500,seed=20261017):
    wells=sorted({r['well'] for r in records})
    if len(wells)<2 or draws<1:return dict(value=None,reason='need at least two held-out wells and positive bootstrap draws')
    groups={w:[r for r in records if r['well']==w] for w in wells};rng=np.random.default_rng(seed);auc=[]
    for _ in range(draws):
        sample=[r for w in rng.choice(wells,len(wells),replace=True) for r in groups[w]]
        value=metrics([r['ctnt_positive'] for r in sample],[r['probability'] for r in sample],threshold,
            hard_predictions=[r['predicted_positive'] for r in sample])['auroc']
        if value is not None:auc.append(value)
    return dict(auroc_95_percent_interval=[float(v) for v in np.percentile(auc,[2.5,97.5])] if auc else None,
        unit='well',valid_draws=len(auc),draws=draws,seed=seed,
        caveat='Conditional descriptive interval; shared donor/acquisition dependence is not removed.')


def score(prep,run,references,minimum_coverage=.8,bootstrap_draws=500):
    freeze=data.read_json(run/'PREDICTIONS_COMPLETE.json')
    for filename,key in [('predictions.json','predictions_sha256'),('CLASSIFIER.json','classifier_sha256'),
                         ('classifier.npz','weights_sha256'),('embedding.npy','embedding_sha256'),
                         ('cells.json','cells_sha256'),('segmentation.json','segmentation_sha256')]:
        if data.digest(run/filename)!=freeze[key]:raise ValueError('predictions or upstream delivery changed before scoring')
    for name,sha in (freeze.get('finetuned_artifacts') or {}).items():
        if data.digest(run/name)!=sha:raise ValueError('fine-tuned expert changed after prediction freeze')
    if freeze.get('hybrid_routes_sha256') and data.digest(run/'HYBRID_ROUTES.json')!=freeze['hybrid_routes_sha256']:
        raise ValueError('hybrid routes changed after prediction freeze')
    reference_record,ref=verify_references(prep,references)
    if data.digest(references/'REFERENCE_COMPLETE.json')!=freeze['reference_complete_sha256']:raise ValueError('reference identity changed')
    labels={(r['image_id'],r['mask_label']):r for r in data.read_json(references/'heldout_labels.json')}
    predictions=data.read_json(run/'predictions.json');config=data.read_json(run/'CONFIG.json')
    masks={r['image_id']:r for r in data.read_json(run/'segmentation.json')};classifier=data.read_json(run/'CLASSIFIER.json')
    threshold=classifier['classifier_probability_threshold'];pairs=[];detection=[];fractions=[]
    heldout_ids={r['image_id'] for r in data.load_prepared(prep) if r['usage']=='evaluation_only' and r['image_id'] in config['images']}
    for ident,mask in masks.items():
        if ident not in heldout_ids:continue
        selected={p['mask_label']:p for p in predictions if p['image_id']==ident and p['usage']=='evaluation_only'}
        if data.digest(Path(mask['mask']))!=mask['sha256']:raise ValueError('mask pixels changed')
        image_labels={k[1]:v for k,v in labels.items() if k[0]==ident}
        predicted=tifffile.imread(mask['mask']);predicted=np.where(np.isin(predicted,list(selected)),predicted,0)
        matching=match_objects(predicted,tifffile.imread(ref[ident]['mask']),classifier['minimum_matching_iou'])
        confident_matches=[]
        for pair in matching['matches']:
            reference=image_labels[pair['reference_label']];p=selected[pair['predicted_label']]
            if reference['ctnt_positive'] is not None:
                record=dict(image_id=ident,cell_id=p['cell_id'],reference_cell_id=reference['reference_cell_id'],well=p['well'],
                    plate=p['plate'],ctnt_positive=reference['ctnt_positive'],probability=p['ctnt_positive_probability'],
                    predicted_positive=p['ctnt_positive_prediction'],prediction_branch=p.get('prediction_branch','single_expert'),iou=pair['iou'])
                pairs.append(record);confident_matches.append(record)
        positive_refs=sum(r['ctnt_positive']==1 for r in image_labels.values())
        true_positive=sum(r['ctnt_positive']==1 and r['predicted_positive']==1 for r in confident_matches)
        assessed_pred_positive=sum(r['predicted_positive'] for r in confident_matches)+sum(selected[i]['ctnt_positive_prediction'] for i in matching['unmatched_predicted'])
        detection.append(dict(image_id=ident,well=ref[ident]['well'],reference_cells=len(image_labels),retained_predicted_cells=len(selected),
            status='no_reference_detections' if not image_labels else 'no_predictions' if not selected else 'matched_evaluation',
            matched_cells=len(matching['matches']),unmatched_predicted=len(matching['unmatched_predicted']),
            unmatched_reference=len(matching['unmatched_reference']),confident_reference_cells=sum(r['ctnt_positive'] is not None for r in image_labels.values()),
            ambiguous_or_excluded_reference_cells=sum(r['ctnt_positive'] is None for r in image_labels.values()),
            reference_positive_cells=positive_refs,true_positive_cells=true_positive,assessed_predicted_positive_cells=assessed_pred_positive,
            unmatched_positive_references=sum(image_labels[i]['ctnt_positive']==1 for i in matching['unmatched_reference'])))
        if confident_matches:
            fractions.append(dict(image_id=ident,well=ref[ident]['well'],denominator='matched_confident_reference_cells',cells=len(confident_matches),
                observed_positive_percent=100*np.mean([r['ctnt_positive'] for r in confident_matches]),
                predicted_positive_percent=100*np.mean([r['predicted_positive'] for r in confident_matches])))
    primary=metrics([r['ctnt_positive'] for r in pairs],[r['probability'] for r in pairs],threshold,hard_predictions=[r['predicted_positive'] for r in pairs])
    totals={key:sum(d[key] for d in detection) for key in ['reference_cells','retained_predicted_cells','matched_cells',
        'unmatched_predicted','unmatched_reference','confident_reference_cells','ambiguous_or_excluded_reference_cells',
        'reference_positive_cells','true_positive_cells','assessed_predicted_positive_cells','unmatched_positive_references']}
    coverage=totals['matched_cells']/totals['reference_cells'] if totals['reference_cells'] else None
    group_metrics={field:{value:metrics([r['ctnt_positive'] for r in pairs if r[field]==value],
        [r['probability'] for r in pairs if r[field]==value],threshold,hard_predictions=[r['predicted_positive'] for r in pairs if r[field]==value]) for value in sorted({r[field] for r in pairs})} for field in ['well','plate']}
    fractions_mae=float(np.mean([abs(r['observed_positive_percent']-r['predicted_positive_percent']) for r in fractions])) if fractions else None
    result=dict(status='complete',dataset='ctnt_staining',arm=config['arm'],scope='exploratory_cell_fluorescence_prediction',
        source_sha256=config['source_sha256'],reference_kind=reference_record['reference_kind'],
        target='ctnt_positive',prediction_channels=['Brightfield'],primary=primary,by_group=group_metrics,
        detection={**totals,'reference_matching_coverage':coverage,
            'scheduled_heldout_images':len(heldout_ids),'assessed_heldout_images':len(detection),
            'no_reference_detection_images':[d['image_id'] for d in detection if not d['reference_cells']],
            'no_prediction_images':[d['image_id'] for d in detection if not d['retained_predicted_cells']],
            'detection_precision':totals['matched_cells']/totals['retained_predicted_cells'] if totals['retained_predicted_cells'] else None,
            'detection_recall':coverage,
            'positive_end_to_end_recall':totals['true_positive_cells']/totals['reference_positive_cells'] if totals['reference_positive_cells'] else None,
            'positive_end_to_end_precision':totals['true_positive_cells']/totals['assessed_predicted_positive_cells'] if totals['assessed_predicted_positive_cells'] else None},
        coverage_criterion=dict(minimum=minimum_coverage,met=coverage is not None and coverage>=minimum_coverage),
        label_coverage=totals['confident_reference_cells']/totals['reference_cells'] if totals['reference_cells'] else None,
        well_cluster_bootstrap=clustered_interval(pairs,threshold,bootstrap_draws,config['seed']),
        matched_image_positive_fraction_mae_percentage_points=fractions_mae,
        matching_rule=dict(kind='one_to_one_IoU',minimum_iou=classifier['minimum_matching_iou']),
        fluorescence_threshold_sha256=data.digest(references/'THRESHOLD.json'),classifier_probability_threshold=threshold,
        classifier_selection=classifier,
        limitations=reference_record['limitations']+['Matched-cell accuracy alone omits missed cells; inspect detection and label coverage',
            'DAPI nuclei are not whole-cell ground truth','No validated cell-level batch transform; original cell embeddings retained'])
    data.write_json(run/'RESULT.json',result)
    data.write_json(run/'MATCHED_HELDOUT_CELLS.json',pairs)
    data.write_json(run/'DETECTION_BY_IMAGE.json',detection)
    data.write_json(run/'MATCHED_IMAGE_FRACTIONS.json',fractions)
    return result

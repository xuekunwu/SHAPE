"""Frozen one-to-one whole-cell matching; no duplicate credit for fragmented detections."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .fluorescence import label_mask


def match_objects(predicted,reference,minimum_iou=.5):
    predicted=label_mask(predicted);reference=label_mask(reference)
    if predicted.shape!=reference.shape or not 0<minimum_iou<=1:raise ValueError('invalid grid or IoU rule')
    pids,pi=np.unique(predicted,return_inverse=True);rids,ri=np.unique(reference,return_inverse=True)
    intersection=np.bincount(pi*(len(rids))+ri,minlength=len(pids)*len(rids)).reshape(len(pids),len(rids))
    pa=intersection.sum(axis=1);ra=intersection.sum(axis=0)
    pk=np.flatnonzero(pids>0);rk=np.flatnonzero(rids>0)
    matches=[]
    if len(pk) and len(rk):
        inter=intersection[np.ix_(pk,rk)];union=pa[pk,None]+ra[None,rk]-inter
        iou=inter/np.maximum(union,1);valid=iou>=minimum_iou
        cost=-valid.astype(float)*(max(len(pk),len(rk))+1+iou)
        ii,jj=linear_sum_assignment(cost)
        matches=[dict(predicted_label=int(pids[pk[i]]),reference_label=int(rids[rk[j]]),iou=float(iou[i,j])) for i,j in zip(ii,jj) if valid[i,j]]
    usedp={m['predicted_label'] for m in matches};usedr={m['reference_label'] for m in matches}
    return dict(rule='maximum_cardinality_then_IoU_one_to_one',minimum_iou=minimum_iou,matches=matches,
        unmatched_predicted=[int(i) for i in pids if i>0 and i not in usedp],
        unmatched_reference=[int(i) for i in rids if i>0 and i not in usedr])

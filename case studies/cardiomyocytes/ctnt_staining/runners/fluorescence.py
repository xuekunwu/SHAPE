"""Private fluorescence measurements and development-only exploratory cTnT thresholds."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class MeasurementProtocol:
    compartment: str = 'cytoplasm_excluding_dapi'
    background_ring_px: int = 8
    background_guard_px: int = 2
    minimum_pixels: int = 16
    maximum_saturated_fraction: float = .01
    nucleus_min_area_px: int = 20
    nucleus_peak_distance_px: int = 5
    signal_transform: str = 'asinh((mean_ctnt-background_median)/(1.4826*background_MAD))'

    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


def label_mask(mask):
    x=np.asarray(mask)
    if x.ndim!=2 or not np.issubdtype(x.dtype,np.integer) or np.any(x<0):
        raise ValueError('expected a nonnegative integer instance mask')
    return x


def nuclei_from_dapi(dapi, protocol):
    """Frozen Otsu+watershed diagnostic; nuclei are not whole-cell ground truth."""
    from skimage.filters import threshold_otsu
    from skimage.feature import peak_local_max
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed
    x=np.asarray(dapi,dtype=float)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('invalid DAPI image')
    if np.ptp(x)==0:return np.zeros(x.shape,dtype=np.int32)
    smoothed=ndi.gaussian_filter(x,1.)
    foreground=remove_small_objects(smoothed>threshold_otsu(smoothed),min_size=protocol.nucleus_min_area_px)
    distance=ndi.distance_transform_edt(foreground)
    points=peak_local_max(distance,min_distance=protocol.nucleus_peak_distance_px,labels=foreground,exclude_border=False)
    markers=np.zeros(x.shape,dtype=np.int32)
    for i,(y,z) in enumerate(points,1):markers[y,z]=i
    if not len(points):return markers
    return watershed(-distance,markers,mask=foreground).astype(np.int32)


def nuclear_qc(mask,nuclei):
    """Coverage/splitting diagnostics; multinucleation alone is not a segmentation failure."""
    mask=label_mask(mask);nuclei=label_mask(nuclei)
    if mask.shape!=nuclei.shape:raise ValueError('DAPI/cell grids differ')
    ids=np.unique(mask);ids=ids[ids>0];nids=np.unique(nuclei);nids=nids[nids>0]
    assigned={int(i):[] for i in ids};covered=split=0
    pids,pi=np.unique(mask,return_inverse=True)
    joint=np.bincount(nuclei.ravel().astype(np.int64)*len(pids)+pi,minlength=(int(nuclei.max())+1)*len(pids)).reshape(-1,len(pids))
    for n in nids:
        overlap,counts=pids,joint[n]
        fractions={int(k):v/counts.sum() for k,v in zip(overlap,counts) if k>0}
        if fractions and max(fractions.values())>=.5:
            chosen=max(fractions,key=fractions.get);assigned[chosen].append(int(n));covered+=1
        split+=sum(v>=.2 for v in fractions.values())>=2
    return dict(reference_type='DAPI_nuclear_proxy_not_cell_GT',n_cells=len(ids),n_nuclei=len(nids),
        nucleus_coverage=covered/len(nids) if len(nids) else None,
        split_nucleus_fraction=split/len(nids) if len(nids) else None,
        anucleate_object_fraction=sum(not v for v in assigned.values())/len(ids) if len(ids) else None,
        multinucleated_object_fraction=sum(len(v)>1 for v in assigned.values())/len(ids) if len(ids) else None,
        nuclei_per_object={str(k):len(v) for k,v in assigned.items()},
        interpretation='Multinucleation may be biological; these are diagnostics, not mask AP/IoU.')


def measure(mask,ctnt,dapi,protocol):
    mask=label_mask(mask);ctnt=np.asarray(ctnt);dapi=np.asarray(dapi)
    if mask.shape!=ctnt.shape or mask.shape!=dapi.shape or ctnt.ndim!=2:
        raise ValueError('reference channels are not on the cell-mask grid')
    if not np.issubdtype(ctnt.dtype,np.integer):raise ValueError('raw integer fluorescence required')
    nuclei=nuclei_from_dapi(dapi,protocol)
    guarded=ndi.binary_dilation(mask>0,iterations=protocol.background_guard_px)
    global_background=~guarded & (nuclei==0)
    records=[]
    if mask.max()>1_000_000:raise ValueError('reference mask IDs must be bounded, relabel before import')
    slices=ndi.find_objects(mask)
    for ident in np.unique(mask):
        if ident==0:continue
        sy,sx=slices[int(ident)-1];pad=protocol.background_ring_px
        region=(slice(max(0,sy.start-pad),min(mask.shape[0],sy.stop+pad)),slice(max(0,sx.start-pad),min(mask.shape[1],sx.stop+pad)))
        local_ctnt=ctnt[region];cell=mask[region]==ident
        cytoplasm=cell & (nuclei[region]==0)
        ring=ndi.binary_dilation(cell,iterations=protocol.background_ring_px)&global_background[region]
        background_scope='local_annulus'
        background=local_ctnt[ring]
        if ring.sum()<protocol.minimum_pixels:background=ctnt[global_background];background_scope='field_background_fallback'
        reasons=[]
        if cytoplasm.sum()<protocol.minimum_pixels:reasons.append('insufficient_cytoplasmic_pixels')
        if len(background)<protocol.minimum_pixels:reasons.append('insufficient_background_pixels')
        saturation=float(np.mean(local_ctnt[cell]==np.iinfo(ctnt.dtype).max))
        if saturation>protocol.maximum_saturated_fraction:reasons.append('saturation_above_protocol_limit')
        bg=float(np.median(background)) if len(background) else None
        noise=float(1.4826*np.median(np.abs(background.astype(float)-bg))) if bg is not None else None
        if noise is not None and noise<=0:reasons.append('zero_background_noise')
        mean=float(np.mean(local_ctnt[cytoplasm])) if cytoplasm.any() else None
        signal=float(np.arcsinh((mean-bg)/noise)) if not reasons else None
        records.append(dict(mask_label=int(ident),raw_ctnt_mean=mean,background_median=bg,background_mad_sigma=noise,
            background_scope=background_scope,background_pixels=int(len(background)),cytoplasm_pixels=int(cytoplasm.sum()),
            saturated_fraction=saturation,saturation_flag=saturation>0,signal=signal,qc_status='evaluable' if not reasons else 'excluded',
            qc_reasons=reasons,measurement_protocol_sha256=protocol.identity()))
    return records,nuclei,nuclear_qc(mask,nuclei)


def fit_threshold(records, protocol_hash, seed=20261017, minimum_cells=100, minimum_wells=2):
    """Fit equal-well-subsampled training references only; no held-out threshold adaptation."""
    from sklearn.mixture import GaussianMixture
    train=[r for r in records if r['usage']=='development_train' and r['qc_status']=='evaluable']
    if any(r['measurement_protocol_sha256']!=protocol_hash for r in train):raise ValueError('mixed measurement protocols')
    groups={w:[r for r in train if r['well']==w] for w in sorted({r['well'] for r in train})}
    if len(groups)<minimum_wells:raise ValueError('insufficient training wells for threshold calibration')
    per_well=min(200,min(len(v) for v in groups.values()))
    selected=[]
    for well,items in groups.items():
        items=sorted(items,key=lambda r:hashlib.sha256(f'{seed}|{r["reference_cell_id"]}'.encode()).hexdigest())
        selected.extend(items[:per_well])
    if len(selected)<minimum_cells:raise ValueError('insufficient evaluable training reference cells')
    x=np.array([r['signal'] for r in selected],dtype=float)[:,None]
    model=GaussianMixture(n_components=2,n_init=5,random_state=seed,reg_covar=1e-4).fit(x)
    order=np.argsort(model.means_[:,0]);means=model.means_[order,0];variances=model.covariances_[order,0,0];weights=model.weights_[order]
    separation=float((means[1]-means[0])/np.sqrt(np.mean(variances)))
    if not model.converged_ or min(weights)<.05 or separation<1.5:
        raise ValueError('training fluorescence does not support a stable two-population exploratory threshold')
    result=dict(status='frozen',kind='exploratory_training_mixture_not_assay_validated',
        calibration_split='development_train',calibration_wells=sorted(groups),calibration_cells=len(selected),
        calibration_reference_ids_sha256=hashlib.sha256(json.dumps(sorted(r['reference_cell_id'] for r in selected)).encode()).hexdigest(),
        measurement_protocol_sha256=protocol_hash,means=means.tolist(),variances=variances.tolist(),weights=weights.tolist(),
        standardized_separation=separation,negative_posterior_max=.1,positive_posterior_min=.9,seed=seed,
        uncertainty_policy='middle posterior band is ambiguous; no forced negative labels',
        limitations=['No independent negative/positive assay controls','Mixture populations are fluorescence proxies, not independently established cardiomyocyte identities'])
    grid=np.linspace(means[0],means[1],4097)
    posterior=positive_posterior(grid,result)
    # Both population means must have their intended confidence before applying the gate.
    if posterior[0]>.1 or posterior[-1]<.9:raise ValueError('mixture modes do not support confident positive and negative populations')
    for key,value in [('negative_signal_boundary',.1),('decision_signal',.5),('positive_signal_boundary',.9)]:
        result[key]=float(grid[np.argmin(np.abs(posterior-value))])
    return result


def positive_posterior(values,threshold):
    x=np.asarray(values,dtype=float)[...,None];m=np.array(threshold['means']);v=np.array(threshold['variances']);w=np.array(threshold['weights'])
    logp=np.log(w)-.5*np.log(2*np.pi*v)-.5*(x-m)**2/v
    logp-=np.max(logp,axis=-1,keepdims=True)
    p=np.exp(logp)
    return p[...,1]/p.sum(axis=-1)


def apply_threshold(records,threshold):
    labeled=[]
    for record in records:
        r=dict(record)
        if r['measurement_protocol_sha256']!=threshold['measurement_protocol_sha256']:raise ValueError('measurement/threshold protocol mismatch')
        label=None;probability=None
        if r['qc_status']=='evaluable':
            probability=float(positive_posterior(r['signal'],threshold))
            if probability<=threshold['negative_posterior_max']:label=0
            elif probability>=threshold['positive_posterior_min']:label=1
            else:r['qc_status']='ambiguous';r['qc_reasons']=[*r['qc_reasons'],'fluorescence_threshold_gray_band']
        r.update(ctnt_positive=label,fluorescence_positive_posterior=probability,
            reference_label_kind=threshold['kind'])
        labeled.append(r)
    return labeled

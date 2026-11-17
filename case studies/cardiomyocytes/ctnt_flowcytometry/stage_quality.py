"""Post-delivery stage readouts, using the same tool-side quant functions as full benchmarks.

No targets are read here. QC proxies are descriptive, not ground-truth accuracy.
Unavailable label-based endpoints and replay attribution are explicit, never zeros.
"""

from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import multiprocessing
import os
import time
import numpy as np

from .preparation import USAGES
from .preparation import digest
from .preparation import load_prepared
from .preparation import read_json
from .preparation import write_json
from .preparation import within


def unavailable(reason):
    return {"value": None, "reason": reason}


def averages(records, keys):
    return {k: float(np.mean([r[k] for r in records])) if records else None for k in keys}


def batch_structure(matrix, table, field):
    """Unconditional batch silhouette is a confounding diagnostic, not a correction-quality score."""
    from sklearn.metrics import silhouette_score
    labels = [r[field] for r in table]
    if not 1 < len(set(labels)) < len(labels):
        return unavailable("need at least two batch groups and more wells than groups")
    if np.any(np.linalg.norm(matrix, axis=1) <= 1e-12):
        return unavailable("cosine silhouette undefined for zero-norm well profiles")
    return {"value": float(silhouette_score(matrix, labels, metric="cosine")),
            "interpretation": "batch clustering; may reflect biology, not evidence that correction is needed"}


def _worker_init():
    # Each image is independent. Avoid workers each starting a full BLAS thread pool.
    global _thread_limit
    from threadpoolctl import threadpool_limits
    _thread_limit = threadpool_limits(limits=1)


def image_readout(job):
    """Exact original QC formulas; one image and its verified delivered crops."""
    import tifffile
    from shape.tools.cell_segmenter.quant import quant_score
    from shape.tools.single_cell_cropper.quant import crop_quality_score
    prep, row, mask_record, cells = job
    image = tifffile.imread(within(Path(prep)/'public', row['file']))
    mask = tifffile.imread(mask_record['mask'])
    seg = dict(image_id=row['image_id'],well=row['well'],usage=row['usage'],
               cells=mask_record['n_cells'],coverage=float(np.mean(mask>0)),**quant_score(image,mask))
    per_cell = []
    for cell in cells:
        # Hash and decode the very same bytes, avoiding a second shared-filesystem open.
        raw = Path(cell['crop_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != cell['sha256']:
            raise ValueError('crop changed before stage readouts')
        pixels = np.squeeze(tifffile.imread(io.BytesIO(raw)))
        y0,x0,y1,x1 = map(int,cell['crop_bbox'])
        target = mask[y0:y1,x0:x1] == cell['mask_label']
        if pixels.shape != target.shape:
            raise ValueError('crop geometry differs from its mask window')
        per_cell.append(crop_quality_score(pixels,target))
    crop = dict(image_id=row['image_id'],well=row['well'],retained=len(per_cell),
                retention=len(per_cell)/seg['cells'] if seg['cells'] else None,
                quality=averages(per_cell,per_cell[0].keys()) if per_cell else {})
    return seg,crop


def image_readouts(jobs, run, workers):
    """Parallel image QC, deterministic output order and visible progress."""
    started = time.time()
    progress = Path(run)/'READOUTS_PROGRESS.json'
    processor_sha = digest(Path(__file__))
    results = [None]*len(jobs)
    def report(count, status):
        record = dict(phase='image_qc',status=status,completed=count,total=len(jobs),
                      workers=workers,seconds=time.time()-started,updated_at=time.time(),
                      processor_sha256=processor_sha)
        temporary = progress.with_suffix('.tmp')
        write_json(temporary,record)
        os.replace(temporary,progress)
        if count == len(jobs) or count % 10 == 0:
            print(f'READOUTS image_qc {count}/{len(jobs)} workers={workers}',flush=True)
    report(0,'running')
    if workers == 1:
        for i,job in enumerate(jobs):
            results[i] = image_readout(job)
            report(i+1,'running')
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_init) as pool:
            pending = {pool.submit(image_readout,job):i for i,job in enumerate(jobs)}
            for done,future in enumerate(as_completed(pending),1):
                results[pending[future]] = future.result()
                report(done,'running')
    report(len(jobs),'complete')
    return [r[0] for r in results],[r[1] for r in results]


def collect(prep: Path, run: Path) -> dict:
    import tifffile
    from shape.tools.cell_segmenter.quant import quant_score
    from shape.tools.single_cell_cropper.quant import crop_quality_score
    from shape.tools.cell_state_analyzer import quant as representation

    delivery = read_json(run/"DELIVERY.json")
    for name, sha in delivery["artifacts"].items():
        if digest(run/name) != sha:
            raise ValueError(f"delivery changed before stage readouts: {name}")
    config = read_json(run/"CONFIG.json")
    wanted = set(config["images"])
    rows = {r["image_id"]:r for r in load_prepared(prep) if r["image_id"] in wanted}
    masks = {r["image_id"]:r for r in read_json(run/"segmentation.json")}
    if set(masks) != wanted:
        raise ValueError("segmentation does not cover scheduled images")
    cells = read_json(run/"cells.json")
    by_image = defaultdict(list)
    for cell in cells:
        by_image[cell["image_id"]].append(cell)
    workers = max(1,min(int(config.get('cores',1)),os.cpu_count() or 1,len(rows)))
    jobs = [(str(prep),row,masks[ident],by_image[ident]) for ident,row in rows.items()]
    segment, crop = image_readouts(jobs,run,workers)
    # Equal weight per well, then per view; high-density images do not dominate the summary.
    seg_keys = [k for k,v in segment[0].items() if isinstance(v,(int,float))]
    def well_macro(records, keys):
        groups = defaultdict(list)
        for r in records:
            groups[r["well"]].append(r)
        return averages([averages(v,keys) for v in groups.values()],keys)
    seg_summary = well_macro(segment,seg_keys)
    crop_flat = [{"well":r["well"],**r["quality"]} for r in crop]
    crop_summary = well_macro(crop_flat,list(crop[0]["quality"]))
    x = np.load(run/"embedding.npy",mmap_mode="r")
    if len(x) != len(cells):
        raise ValueError("cell/embedding row mismatch")
    embeddings = {}
    for usage in USAGES:
        write_json(run/'READOUTS_PROGRESS.json',dict(phase='embedding_'+usage,status='running',
                   completed=len(rows),total=len(rows),workers=workers,updated_at=time.time()))
        print('READOUTS embedding',usage,flush=True)
        chosen = [i for i,c in enumerate(cells) if c["usage"]==usage]
        if not chosen:
            raise ValueError("missing cells in a scheduled split")
        # Bound diagnostic work without changing the fitted or delivered embeddings.
        chosen = np.array(sorted(chosen,key=lambda i:cells[i]["cell_id"]))
        chosen = chosen[np.random.default_rng(0).permutation(len(chosen))[:representation.HEALTH_ROWS]]
        meta = [{"site":cells[i]["image_id"],"well":cells[i]["well"],"plate":cells[i]["plate"]} for i in chosen]
        values = np.asarray(x[chosen])
        embeddings[usage] = {"sampled_cells":len(chosen),"sampling":"seed 0, at most HEALTH_ROWS delivered cells per split",
            "health":representation.health(values), "image_leakage":representation.image_leakage(values,meta),
            "sibling_site_map":representation.sibling_site_map(values,meta,seed=representation.FRESH_SEED)}
    with (run/"profiles.csv").open(newline="",encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    before, after = np.load(run/"profiles.npy"), np.load(run/"corrected.npy")
    batch = {"admission":read_json(run/"batch_SEARCH.json")["admission"],
             "profiles_unchanged":bool(np.array_equal(before,after)),
             "structure":{field:{"input":batch_structure(before,table,field),
                                  "delivered":batch_structure(after,table,field)}
                          for field in ("experiment_date","plate_id")},
             "biological_preservation":unavailable("no validated categorical biological/technical reference labels")}
    process = {}
    for stage in ("segment","crop","embed","batch","predict"):
        complete = read_json(run/f"{stage}_COMPLETE.json")
        process[stage] = {"seconds":complete.get("seconds"),"usage":complete.get("usage"),
                          "reasoning":complete.get("reasoning")}
    result = {"source_sha256":delivery["source_sha256"], "prepare_sha256":delivery["prepare_sha256"],
              "readouts_execution":{"workers":workers,"processor_sha256":digest(Path(__file__)),
                  "image_qc":"process pool; original formulas and deterministic input order"},
              "segment":{"well_macro":seg_summary,"per_image":segment},
              "crop":{"well_macro_quality":crop_summary,"per_image":crop,
                      "retained_cells":len(cells),"excluded_cells":len(read_json(run/"excluded_cells.json"))},
              "embed":{"selection":read_json(run/"SSL_SELECTION.json"),"by_split":embeddings},
              "batch":batch,"process":process,
              "unavailable":{
                  "segmentation_AP_F1": "no annotated instance masks",
                  "crop_GT_recall_contamination": "no annotated instance masks; mask-based QC uses predicted masks",
                  "classification_mAP_F1_balanced_accuracy": "target is a well-level continuous percentage; no observed cell classes",
                  "candidate_regret": "hidden-endpoint scoring of all candidates is not implemented for this regression adapter",
                  "factorial_attribution": "requires separate crossed objects/representation replay runs; not part of this pilot",
                  "population_confidence_intervals": "small pilot; wells share donors and batches"},
              "interpretation":"GT-free stage QC and process evidence; not segmentation or cell-classification accuracy"}
    write_json(run/"STAGE_READOUTS.json",result)
    write_json(run/'READOUTS_PROGRESS.json',dict(phase='complete',status='complete',
               completed=len(rows),total=len(rows),workers=workers,updated_at=time.time()))
    return result

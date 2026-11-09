"""STAGE 4 -- cross-vendor batch correction for HCMA Config-A, driven by a REAL VLRR loop.

Distinct from stage4_screening.py (which only DECIDES from metadata whether batch correction
is warranted). This module runs the repo's BatchCorrectionVerifier through StepVerifier:
Harmony corrects the stage-3 embedding for the vendor batch (batch_key='source'); the
verifier reads GT-FREE readouts on the corrected embedding and, while residual batch signal
remains, RETUNES Harmony (theta/sigma/lambda) and reruns -- a bounded Verify/Retune/Rerun loop.

Batch_Corrector_Tool.execute() returns the GT-free readouts the verifier's quantify() reads
(result["gt_free_metrics"], computed on the corrected embedding with every call).

After the loop we compare RAW vs the FINAL corrected embedding on organ_category recovery
(image-grouped grouped_probes) + vendor predictability, and record the VLRR trajectory.

  python -m benchmarks.cell_state_HCMA.stage4_harmony \
      --h5ad benchmarks/cell_state_HCMA/results/hcma_configA_stage3.h5ad \
      --out-dir benchmarks/cell_state_HCMA/results
  python -m benchmarks.cell_state_HCMA.stage4_harmony --selftest
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.verifiers.batch_correction_unified import (
    BatchCorrectionVerifier, adapt_batch_correction)

_INIT_HARMONY = {"theta": 2.0, "lambda_value": None, "sigma": 0.1,
                 "nclust": None, "random_state": 0}


def run_vlrr_batch_correction(raw_X, meta, *, batch_key="source", max_iters=4):
    """Drive BatchCorrectionVerifier through StepVerifier. Returns (StageResolution, trajectory)."""
    from shape.tools.batch_corrector.tool import Batch_Corrector_Tool
    raw_X = np.asarray(raw_X, dtype=float)
    traj = []

    def tool_runner(params):
        # Merge over the full default so retuned params ALWAYS carry all 5 schema keys:
        # the verifier's _next_params drops random_state (and starts from an empty
        # current_params when Harmony was SKIPPED, e.g. no batch effect on a single-vendor
        # subset), which otherwise trips Batch_Corrector_Tool's exact-5-key schema guard.
        hp = {**_INIT_HARMONY, **(params.get("harmony_params") or {})}
        res = Batch_Corrector_Tool().execute(cls_output=raw_X, metadata=meta,
                                             batch_key=batch_key, harmony_params=hp)
        gfm = res["gt_free_metrics"]
        traj.append({"theta": hp.get("theta"), "sigma": hp.get("sigma"),
                     "lambda_value": hp.get("lambda_value"), "status": res.get("status"),
                     **{k: round(v, 4) for k, v in gfm.items() if isinstance(v, float)}})
        return res

    initial_params = {"harmony_params": dict(_INIT_HARMONY)}
    initial_result = tool_runner(initial_params)
    sv = StepVerifier(max_iters=max_iters)
    res = sv.resolve_stage(BatchCorrectionVerifier(), adapt_batch_correction,
                           tool_runner, initial_params, initial_result, upstream_ctx={})
    return res, traj


def vendor_predictability(X, vendor, *, seed=0) -> float:
    """5-fold accuracy predicting vendor from the embedding. Lower = better mixing."""
    y = LabelEncoder().fit_transform(np.asarray(vendor).astype(str))
    counts = np.bincount(y)
    if len(counts) < 2 or counts[counts > 0].min() < 2:
        return float("nan")
    n_splits = int(min(5, counts[counts > 0].min()))
    Xs = StandardScaler().fit_transform(np.asarray(X, float))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(lr, Xs, y, cv=skf)))


def _probe_organ(X, obs):
    from benchmarks.cell_state_HCMA.evaluate import grouped_probes
    y = LabelEncoder().fit_transform(obs["organ_category"].astype(str).to_numpy())
    groups = obs["image_name"].astype(str).to_numpy()
    return grouped_probes(np.asarray(X, float), y, groups)


def _metrics_block(X, obs, vendor):
    p = _probe_organ(X, obs)
    return {"organ_linear_acc": p.get("linear_accuracy"),
            "organ_knn_acc": p.get("knn_accuracy"),
            "organ_linear_nll": p.get("linear_nll"),
            "vendor_predictability": vendor_predictability(X, vendor)}


def run_stage4(h5ad_path, out_dir, *, max_iters=4):
    import anndata as ad
    os.makedirs(out_dir, exist_ok=True)
    adata = ad.read_h5ad(h5ad_path)
    for c in ("organ_category", "image_name", "source"):
        if c not in adata.obs:
            raise KeyError(f"AnnData.obs missing {c!r}")
    raw = np.asarray(adata.X, dtype=float)
    meta = adata.obs.reset_index(drop=True)
    vendor = adata.obs["source"].to_numpy()
    print(f"[stage4] {adata.n_obs} cells | {adata.obs['organ_category'].nunique()} organs | "
          f"{adata.obs['source'].nunique()} vendors", flush=True)

    res, traj = run_vlrr_batch_correction(raw, meta, batch_key="source", max_iters=max_iters)
    final = np.asarray(res.result["corrected_cls_output"], dtype=float)
    n_retunes = max(0, len(traj) - 1)
    print(f"[stage4] VLRR status={res.status} verdict={res.verdict.decision} "
          f"reason={res.reason} | {n_retunes} retune(s) | ledger={res.ledger}", flush=True)
    for i, t in enumerate(traj):
        print(f"   round {i}: theta={t['theta']} sigma={t['sigma']} "
              f"batch_pred={t['batch_predictability']} disp={t['normalized_displacement']} "
              f"rank_ratio={t['effective_rank_ratio']} [{t['status']}]", flush=True)

    report = {
        "vlrr": {"status": res.status, "verdict": res.verdict.decision, "reason": res.reason,
                 "ledger": res.ledger, "n_retunes": n_retunes,
                 "final_harmony_params": res.params.get("harmony_params"),
                 "trajectory": traj},
        "n_cells": int(adata.n_obs), "n_organ_categories": int(adata.obs["organ_category"].nunique()),
        "n_vendors": int(adata.obs["source"].nunique()),
        "RAW": _metrics_block(raw, adata.obs, vendor),
        "HARMONY": _metrics_block(final, adata.obs, vendor),
    }
    report["delta"] = {k: (report["HARMONY"][k] - report["RAW"][k])
                       if (report["HARMONY"][k] is not None and report["RAW"][k] is not None
                           and np.isfinite(report["HARMONY"][k]) and np.isfinite(report["RAW"][k]))
                       else None for k in report["RAW"]}

    out_json = os.path.join(out_dir, "hcma_configA_stage4_batchcorr.json")
    json.dump(report, open(out_json, "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("RAW", "HARMONY", "delta")}, indent=2))
    print(f"[stage4] report -> {out_json}", flush=True)
    _figure(report, os.path.join(out_dir, "hcma_configA_stage4.png"))
    print(f"[stage4] figure -> {os.path.join(out_dir, 'hcma_configA_stage4.png')}", flush=True)
    return report


def _figure(report, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["organ_linear_acc", "organ_knn_acc", "organ_linear_nll", "vendor_predictability"]
    raw = [report["RAW"].get(m, np.nan) for m in metrics]
    har = [report["HARMONY"].get(m, np.nan) for m in metrics]
    traj = report["vlrr"]["trajectory"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(len(metrics)); w = 0.38
    ax.bar(x - w / 2, raw, w, label="RAW", color="#95a5a6")
    ax.bar(x + w / 2, har, w, label="HARMONY", color="#2e86de")
    for i, (r, h) in enumerate(zip(raw, har)):
        for off, v in ((-w / 2, r), (w / 2, h)):
            if v is not None and np.isfinite(v):
                ax.text(i + off, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=15, ha="right"); ax.set_ylabel("score")
    ax.set_title("RAW vs Harmony (organ recovery should hold; vendor_pred should DROP)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    rounds = list(range(len(traj)))
    ax2.plot(rounds, [t["batch_predictability"] for t in traj], "-o", color="#e74c3c", label="batch_predictability")
    ax2.plot(rounds, [t["normalized_displacement"] for t in traj], "-s", color="#27ae60", label="displacement")
    ax2.plot(rounds, [t["effective_rank_ratio"] for t in traj], "-^", color="#8e44ad", label="eff_rank_ratio")
    ax2.axhline(0.55, ls="--", c="#e74c3c", alpha=.5, label="batch gate 0.55")
    ax2.set_xlabel("VLRR round"); ax2.set_xticks(rounds); ax2.set_ylabel("gt-free metric")
    ax2.set_title(f"VLRR trajectory ({report['vlrr']['n_retunes']} retune(s))")
    ax2.legend(fontsize=8); ax2.grid(alpha=.3)
    fig.suptitle("HCMA Config-A stage-4 cross-vendor batch correction (real VLRR loop)", fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _selftest():
    """Synthesize an AnnData with a LARGE vendor offset entangled with organ so the default
    theta leaves residual batch_predictability > 0.55 -> >=1 Harmony retune must fire."""
    import tempfile
    import anndata as ad
    rng = np.random.RandomState(0)
    rows, X = [], []
    organs = [f"organ_{i}" for i in range(5)]
    cid = 0
    for oi, organ in enumerate(organs):
        organ_center = rng.randn(48) * 1.2
        for vi, vendor in enumerate(["ATCC", "ScienCell"]):
            vendor_shift = (rng.randn(48) * 6.0) * (1 + vi)   # large, vendor-specific offset
            for img in range(2):
                image_name = f"img_{oi}_{vi}_{img}"
                for _ in range(24):
                    rows.append({"cell_id": f"c{cid}", "image_name": image_name,
                                 "organ_category": organ, "source": vendor})
                    X.append(organ_center + vendor_shift + rng.randn(48) * 0.6)
                    cid += 1
    obs = pd.DataFrame(rows); obs.index = obs["cell_id"].to_numpy()
    adata = ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=obs)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s3.h5ad"); adata.write_h5ad(p)
        rep = run_stage4(p, d, max_iters=4)
        v = rep["vlrr"]
        assert v["n_retunes"] >= 1, f"expected >=1 Harmony retune, got {v['n_retunes']} (traj={v['trajectory']})"
        assert np.isfinite(rep["RAW"]["vendor_predictability"]), "RAW vendor pred not finite"
        assert np.isfinite(rep["HARMONY"]["vendor_predictability"]), "HARMONY vendor pred not finite"
        print(f"\n[selftest] PASS -- {v['n_retunes']} retune(s), ledger={v['ledger']}, "
              f"vendor_pred RAW={rep['RAW']['vendor_predictability']:.3f} -> "
              f"HARMONY={rep['HARMONY']['vendor_predictability']:.3f}; "
              f"batch_pred traj={[t['batch_predictability'] for t in v['trajectory']]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad")
    ap.add_argument("--out-dir", default="benchmarks/cell_state_HCMA/results")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    if not a.h5ad:
        raise SystemExit("need --h5ad (or --selftest)")
    run_stage4(a.h5ad, a.out_dir, max_iters=a.max_iters)


if __name__ == "__main__":
    main()

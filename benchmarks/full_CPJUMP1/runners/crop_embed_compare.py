"""Does the crop method (per-image VRR vs cohort) change the DOWNSTREAM embedding?

The crop cohort delivers the same cells as the per-image crop but with ~0.77-IoU different crop boxes
(different context margin). This asks whether that box difference survives into the DINOv3 embedding:
embed BOTH crop sets of `results/crop_cohort_probe` (zero-shot, run `cli stage embed` on each dir
first, on the GPU node), pair each cell by cell_id, and report the cosine similarity between its
per-image-crop embedding and its cohort-crop embedding. Cosine ~1 => the margin difference does not
reach the representation (cohort crop is downstream-safe); lower => it does.

  # 1) embed both crop sets on the GPU node (zero-shot; arm=control CONFIG already written):
  for side in perimage cohort; do
    .venv/bin/python -m benchmarks.full_CPJUMP1.harness.cli stage embed \
      --prep data/CPJUMP1/prepared --run benchmarks/full_CPJUMP1/results/crop_cohort_probe/$side
  done
  # 2) compare (anywhere):
  python -m benchmarks.full_CPJUMP1.runners.crop_embed_compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_side(d: Path):
    cells = json.load(open(d / "cells.json"))
    emb = np.load(d / "embedding.npy")
    if len(cells) != len(emb):
        raise SystemExit(f"{d}: embedding rows {len(emb)} != cells {len(cells)} (embed not run / stale?)")
    return {c["cell_id"]: emb[i] for i, c in enumerate(cells)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=Path, default=Path("benchmarks/full_CPJUMP1/results/crop_cohort_probe"))
    a = ap.parse_args(argv)
    for side in ("perimage", "cohort"):
        if not (a.probe / side / "embedding.npy").exists():
            raise SystemExit(f"missing {a.probe/side}/embedding.npy -- run `cli stage embed --run {a.probe/side}` "
                             "on the GPU node first (see this file's header).")
    pi, co = load_side(a.probe / "perimage"), load_side(a.probe / "cohort")
    shared = sorted(set(pi) & set(co))
    if not shared:
        raise SystemExit("no shared cell_id between the two crop sets")
    P = np.stack([pi[c] for c in shared]).astype(np.float64)
    C = np.stack([co[c] for c in shared]).astype(np.float64)
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-12)
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12)
    cos = (Pn * Cn).sum(1)                                   # per-cell cosine, same cell two crop boxes
    # a baseline: cosine to a RANDOM other cell's per-image embedding (how big is "different"?)
    rng = np.arange(len(shared))
    shuf = np.concatenate([rng[1:], rng[:1]])                # deterministic shift (no RNG; scripts ban random)
    base = (Pn * Cn[shuf]).sum(1)
    mean_cos, p5_cos, base_mean = float(cos.mean()), float(np.percentile(cos, 5)), float(base.mean())
    if mean_cos >= 0.98 and p5_cos >= 0.95:
        verdict = (f"same-cell cosine ~1 (mean {mean_cos:.3f}, p5 {p5_cos:.3f}): the crop margin "
                   "difference barely reaches the embedding -- cohort crop looks downstream-safe.")
    elif mean_cos > base_mean + 0.1:
        verdict = (f"same-cell cosine mean {mean_cos:.3f} (p5 {p5_cos:.3f}) is well above the "
                   f"random-pair baseline {base_mean:.3f} -- still the same cell -- but NOT ~1: the "
                   "crop margin difference measurably shifts the embedding. Whether that helps / "
                   "hurts / is neutral on retrieval needs a full-pipeline mAP comparison before "
                   "adopting the crop cohort.")
    else:
        verdict = (f"same-cell cosine mean {mean_cos:.3f} is close to the random-pair baseline "
                   f"{base_mean:.3f}: the crop method changes the embedding substantially.")
    report = {
        "n_cells_paired": len(shared),
        "same_cell_cosine": {"mean": round(mean_cos, 4), "median": round(float(np.median(cos)), 4),
                             "min": round(float(cos.min()), 4), "p5": round(p5_cos, 4),
                             "frac_below_0.99": round(float((cos < 0.99).mean()), 4),
                             "frac_below_0.95": round(float((cos < 0.95).mean()), 4)},
        "random_pair_cosine_baseline": {"mean": round(base_mean, 4),
                                        "median": round(float(np.median(base)), 4)},
        "interpretation": verdict,
    }
    out = a.probe / "embed_compare.json"
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

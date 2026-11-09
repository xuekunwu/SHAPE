"""Glue between STAGE 2 (cropper) and STAGE 3 (embed): turn a cropper run's final
shape-verified crops into the crop manifest stage3_embed.py consumes.

The cropper writes final tight-bbox crops to
    <crop-run>/images/<stem>/shape/tool_cache/<stem>_cell_<NNNN>_crop.{tiff,png}
where <stem> is the source-image filename without extension. This joins each crop to its
Config-A label (organ_category) + vendor (source) from the Config-A manifest, keyed by the
image filename stem, and emits [path, cell_id, image_name, organ_category, source].

  python -m benchmarks.cell_state_HCMA.build_crop_manifest \
      --crop-run  benchmarks/cell_state_HCMA/results/stage2_crops/<ts>_concurrent \
      --configA-manifest benchmarks/cell_state_HCMA/results/hcma_configA_manifest.parquet \
      --out benchmarks/cell_state_HCMA/results/hcma_configA_crops.parquet
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def build(crop_run: str, configA_manifest: str, out: str) -> pd.DataFrame:
    man = pd.read_parquet(configA_manifest)
    # map image filename stem -> (organ_category, source)
    man["stem"] = man["image_path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    label = man.set_index("stem")[["organ_category", "source"]].to_dict("index")

    rows, missing = [], 0
    for stem, lab in label.items():
        # The per-image DIR is keyed by the full stem (reliable); the crop FILE prefix is
        # truncated for long image names, so glob every crop in the dir rather than requiring
        # a `{stem}_` filename prefix (that dropped ~460 long-named, mostly ScienCell, images).
        crops = sorted(glob.glob(os.path.join(
            crop_run, "images", stem, "shape", "tool_cache", "*_crop.*")))
        if not crops:
            missing += 1
            continue
        for cp in crops:
            cell_id = os.path.splitext(os.path.basename(cp))[0]   # <stem>_cell_<NNNN>_crop
            rows.append({"path": cp, "cell_id": cell_id, "image_name": stem,
                         "organ_category": lab["organ_category"], "source": lab["source"]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no crops found under {crop_run}/images/*/shape/tool_cache -- "
                         "did stage 2 (cropper) finish, and is --crop-run the right dir?")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_parquet(out)
    print(f"[manifest] {len(df)} crops over {df['image_name'].nunique()} images, "
          f"{df['organ_category'].nunique()} organ categories, {df['source'].nunique()} vendors")
    print(f"[manifest] {missing} config-A images had no crops (segmentation/crop produced none)")
    print(f"[manifest] -> {out}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop-run", required=True, help="stage-2 cropper run dir (…_concurrent)")
    ap.add_argument("--configA-manifest",
                    default="benchmarks/cell_state_HCMA/results/hcma_configA_manifest.parquet")
    ap.add_argument("--out",
                    default="benchmarks/cell_state_HCMA/results/hcma_configA_crops.parquet")
    a = ap.parse_args()
    build(a.crop_run, a.configA_manifest, a.out)


if __name__ == "__main__":
    main()

"""Pinned model files for every arm (the shared environment), under $ROOT/assets:

  torch/hub/facebookresearch_dinov3_main/   the tracked files of the pinned DINOv3 checkout, loaded
                                            explicitly with torch.hub source="local"
  hf/hub/models--5xuekun--dinov3_vits16/    the pinned ViT-B/16 weights in the Hugging Face cache
                                            layout, so hf_hub_download resolves them offline
  cellpose/cpsam                            the Cellpose-SAM weights
  ASSETS.json                               every path + sha256, the DINOv3 commit

The tools point TORCH_HOME, HF_HOME (+ HF_HUB_OFFLINE=1) and CELLPOSE_LOCAL_MODELS_PATH here
(tools/_config.json: env) before they import SHAPE.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .prepare import sha256

DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
WEIGHTS_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
HF_REPO = "5xuekun/dinov3_vits16"                 # the repo id Cell_State_Analyzer_Tool downloads from
WEIGHTS_NAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
HUB_DIR_NAME = "facebookresearch_dinov3_main"     # torch.hub's cache dir for facebookresearch/dinov3 (main)


def tool_env(root: Path) -> dict[str, str]:
    """The environment every tool sets before importing SHAPE (tools/_config.json: env)."""
    assets = Path(root).absolute() / "assets"
    return {"TORCH_HOME": str(assets / "torch"), "HF_HOME": str(assets / "hf"),
            "SHAPE_DINOV3_REPO": str(assets / "torch" / "hub" / HUB_DIR_NAME), "HF_HUB_OFFLINE": "1",
            "CELLPOSE_LOCAL_MODELS_PATH": str(assets / "cellpose"), "SHAPE_FAST_FIGURES": "1",
            "MPLBACKEND": "Agg"}


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=300)
    if out.returncode:
        raise SystemExit(f"NOT_READY: git {' '.join(args)} failed in {repo}: {out.stderr.strip()}")
    return out.stdout


def tree_digest(directory: Path) -> tuple[str, int]:
    """One sha256 over (relative path, file sha256) of every file except bytecode caches (a
    torch.hub import may write them); and the file count."""
    import hashlib

    digest, count = hashlib.sha256(), 0
    for path in sorted(p for p in directory.rglob("*")
                       if p.is_file() and "__pycache__" not in p.relative_to(directory).parts):
        digest.update(f"{path.relative_to(directory).as_posix()}\0{sha256(path)}\n".encode())
        count += 1
    return digest.hexdigest(), count


def copy_dinov3(repo: Path, commit: str, dest: Path) -> dict:
    """Copy the tracked files of a clean checkout at `commit` (no .git, no untracked files)."""
    head = _git(repo, "rev-parse", "HEAD").strip()
    if head != commit:
        raise SystemExit(f"NOT_READY: DINOv3 source {repo} is at {head}, expected {commit}")
    if _git(repo, "status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit(f"NOT_READY: DINOv3 source {repo} has tracked changes")
    files = [f for f in _git(repo, "ls-files", "-z").split("\0") if f]
    for name in files:
        src, dst = repo / name, dest / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)
    if not (dest / "hubconf.py").is_file():
        raise SystemExit(f"NOT_READY: {repo} has no hubconf.py")
    digest, count = tree_digest(dest)
    return {"source": str(repo), "commit": commit, "path": str(dest), "n_files": count, "tree_sha256": digest}


def place_hf_weights(weights: Path, hf_home: Path, expected_sha256: str | None) -> dict:
    """The weights as snapshot <rev> of HF_REPO in the HF cache under hf_home, refs/main -> <rev>,
    <rev> = the first 40 hex characters of the weights' sha256 (a stable fake commit hash)."""
    digest = sha256(weights)
    if expected_sha256 and digest != expected_sha256:
        raise SystemExit(f"NOT_READY: {weights} sha256 {digest} != expected {expected_sha256}")
    rev = digest[:40]
    repo_dir = hf_home / "hub" / ("models--" + HF_REPO.replace("/", "--"))
    target = repo_dir / "snapshots" / rev / WEIGHTS_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights, target)
    (repo_dir / "refs").mkdir(exist_ok=True)
    (repo_dir / "refs" / "main").write_text(rev, encoding="utf-8")    # no newline: read verbatim
    if sha256(target) != digest:
        raise SystemExit(f"NOT_READY: copy of {weights} differs from its source")
    return {"source": str(weights), "repo_id": HF_REPO, "filename": WEIGHTS_NAME, "revision": rev,
            "path": str(target), "sha256": digest}


def check_hf_offline(hf_home: Path, expected_path: Path) -> str:
    """Resolve the weights with hf_hub_download in a fresh offline interpreter (HF_HUB_OFFLINE is
    read when huggingface_hub is imported)."""
    code = ("import sys; from huggingface_hub import hf_hub_download; "
            f"print(hf_hub_download(repo_id={HF_REPO!r}, filename={WEIGHTS_NAME!r}))")
    env = {**os.environ, "HF_HOME": str(hf_home), "HF_HUB_OFFLINE": "1"}
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    resolved = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    if out.returncode or Path(resolved).resolve() != expected_path.resolve():
        raise SystemExit(f"NOT_READY: hf_hub_download does not resolve the pinned weights offline "
                         f"(got {resolved!r}): {out.stderr.strip()[-2000:]}")
    return resolved


def place_cpsam(dest_dir: Path, source: Path | None) -> dict:
    """Copy an explicit source, or download exactly cpsam with the installed Cellpose API."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "cpsam"
    if target.exists():
        raise SystemExit(f"NOT_READY: {target} exists; refusing an unverified cached download")
    if source is not None:
        if not Path(source).is_file():
            raise SystemExit(f"NOT_READY: explicit cpsam source does not exist: {source}")
        shutil.copyfile(source, target)
        origin = str(source)
    else:
        code = """from pathlib import Path
import os
from cellpose import models
current = getattr(models, 'cache_model_path', None)
legacy = getattr(models, 'cache_CPSAM_model_path', None)
if callable(current):
    if 'cpsam' not in getattr(models, 'MODEL_NAMES', ()):
        raise RuntimeError('installed Cellpose does not advertise cpsam support')
    resolved = current('cpsam')
elif callable(legacy):
    resolved = legacy()
else:
    raise RuntimeError('installed Cellpose exposes no supported cpsam cache API')
expected = Path(os.environ['CELLPOSE_LOCAL_MODELS_PATH']) / 'cpsam'
if Path(resolved).resolve() != expected.resolve():
    raise RuntimeError('Cellpose resolved cpsam outside the requested cache')
print(resolved)
"""
        env = {**os.environ, "CELLPOSE_LOCAL_MODELS_PATH": str(dest_dir)}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=3600)
        if out.returncode or not target.is_file():
            raise SystemExit(f"NOT_READY: cellpose could not download cpsam into {dest_dir}: "
                             f"{out.stderr.strip()[-2000:]}")
        origin = "downloaded by cellpose"
    if target.stat().st_size == 0:
        raise SystemExit(f"NOT_READY: empty cpsam weights: {target}")
    return {"source": origin, "path": str(target), "sha256": sha256(target)}


def build(root: Path, *, dinov3_repo: Path, weights: Path, cpsam: Path | None = None,
          commit: str = DINOV3_COMMIT, weights_sha256: str | None = WEIGHTS_SHA256) -> Path:
    """Write $root/assets (refuses to overwrite) and return ASSETS.json."""
    root = Path(root).absolute()
    assets = root / "assets"
    if assets.exists():
        raise SystemExit(f"NOT_READY: {assets} exists; earlier evidence is preserved")
    root.mkdir(parents=True, exist_ok=True)
    # Each attempt keeps its own staging directory, including failures, for diagnosis.
    partial = Path(tempfile.mkdtemp(prefix="assets.partial.", dir=root))
    hub = partial / "torch" / "hub"
    dinov3 = copy_dinov3(Path(dinov3_repo), commit, hub / HUB_DIR_NAME)
    # torch.hub records trusted repos here; present up front, so loading never needs to write it
    (hub / "trusted_list").write_text("facebookresearch_dinov3\n", encoding="utf-8")
    hf = place_hf_weights(Path(weights), partial / "hf", weights_sha256)
    check_hf_offline(partial / "hf", Path(hf["path"]))
    cellpose = place_cpsam(partial / "cellpose", cpsam)
    partial.rename(assets)
    for record in (dinov3, hf, cellpose):          # paths as they are after the rename
        record["path"] = record["path"].replace(str(partial), str(assets), 1)
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "env": tool_env(root), "dinov3_hub": dinov3, "dinov3_weights": hf, "cpsam": cellpose}
    path = assets / "ASSETS.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"ASSETS {path}: DINOv3 {commit[:8]} ({dinov3['n_files']} files), weights {hf['sha256'][:12]}, "
          f"cpsam {cellpose['sha256'][:12]} ({cellpose['source']})")
    return path

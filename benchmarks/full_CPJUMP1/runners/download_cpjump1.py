"""Download the CPJUMP1 1000-view / 5000-channel subset from the manifest's per-file
https_url into DATA/raw/<relative_path>, and copy the manifest to DATA/manifest.json, so
`cli prepare --data DATA` passes. The manifest (runners/manifest.json, SHA256
7cf7a28b...) already lists every file's public cellpainting-gallery URL + the exact local
relative_path prepare() expects -- download is purely mechanical, no reverse engineering.

Public/anonymous S3 over HTTPS (no AWS creds). Resumable: a file whose size already matches
is skipped. Concurrent. Verifies the manifest SHA against the agreed set before, each download
against the server Content-Length (catches a truncated transfer), and the file count after.

  python -m benchmarks.full_CPJUMP1.runners.download_cpjump1 \
      --manifest benchmarks/full_CPJUMP1/runners/manifest.json \
      --out data/CPJUMP1 --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The agreed manifests this downloader will fetch (name -> sha256): the v1 1000-view selection and the two
# non-overlapping field-replicate selections (generate_replica_manifests.py). A manifest whose SHA is not
# here is refused -- the download only ever fetches an agreed, reproducible selection.
KNOWN_SHA = {
    "CPJUMP1-v1-1000views": "7cf7a28b12f703a152da4f2445e3a72aa62ed9078012e227e398ba4f92cadd31",
    "CPJUMP1-replica-seed20261009": "44be6b84bb5ecf51108a110436189a6470d5abc1775d4c5ad22620ab2409f9ce",
    "CPJUMP1-replica-seed20261010": "9e85a049395cceb2e86c0e0d7e7c2fb6eb0215b8d17eb4627a0f183f98a184f1",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _fetch(url: str, dest: Path, expect_bytes: int | None, retries: int = 5) -> tuple[str, int]:
    """Download url->dest unless dest already has the expected size. Returns (status, nbytes)."""
    if dest.exists() and (expect_bytes is None or dest.stat().st_size == expect_bytes):
        return "cached", dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as out:
                clen = r.headers.get("Content-Length")
                shutil.copyfileobj(r, out, length=8 * 1024 * 1024)
            n = tmp.stat().st_size
            # integrity: the whole response must be on disk. Verify against the manifest's byte count when it
            # has one, else against the server's Content-Length -- either way a truncated transfer is rejected.
            want = expect_bytes if expect_bytes is not None else (int(clen) if clen else None)
            if want is not None and n != want:
                raise IOError(f"size {n} != expected {want} (truncated download)")
            tmp.replace(dest)
            return "downloaded", n
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"failed after {retries} tries: {url} ({last})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="benchmarks/full_CPJUMP1/runners/manifest.json")
    ap.add_argument("--out", type=Path, default=Path("data/CPJUMP1"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--url-key", default="https_url", help="manifest file field with the download URL")
    a = ap.parse_args()

    man_path = Path(a.manifest)
    got = _sha256(man_path)
    known = {v: k for k, v in KNOWN_SHA.items()}
    if got not in known:
        sys.exit(f"manifest SHA {got} is not an agreed selection {sorted(KNOWN_SHA)}; refusing")
    manifest = json.loads(man_path.read_text())
    files = manifest["files"]
    print(f"[download] manifest OK: {known[got]} ({len(files)} files) -> {a.out}", flush=True)

    # copy manifest next to raw/ so `cli prepare --data <out>` finds <out>/manifest.json
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(man_path, a.out / "manifest.json")
    raw = a.out / "raw"

    jobs = [(f[a.url_key], raw / f["relative_path"], f.get("bytes") or f.get("size")) for f in files]
    done = {"cached": 0, "downloaded": 0}
    errors = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_fetch, url, dest, nb): dest for url, dest, nb in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                status, _ = fut.result()
                done[status] += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
            if i % 250 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} (downloaded={done['downloaded']} cached={done['cached']} "
                      f"errors={len(errors)})", flush=True)

    present = sum(1 for _, dest, _ in jobs if dest.exists())
    print(f"[download] {present}/{len(jobs)} files on disk; errors={len(errors)}", flush=True)
    if errors:
        for e in errors[:10]:
            print("  ERR:", e)
        sys.exit(f"{len(errors)} files failed; re-run to resume")
    print(f"DOWNLOAD_COMPLETE files={present} views={len(manifest['sites'])} "
          f"genes={len(manifest.get('genes', []))}", flush=True)
    print(f"[download] next: python -m benchmarks.full_CPJUMP1.harness.cli prepare "
          f"--data {a.out} --out {a.out}/prepared", flush=True)


if __name__ == "__main__":
    main()

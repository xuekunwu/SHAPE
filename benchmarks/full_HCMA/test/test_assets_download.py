"""Exercise the real cache subprocess offline with version-specific Cellpose APIs."""
from pathlib import Path

import pytest

from benchmarks.full_HCMA.harness import assets


def install_stub(tmp_path, monkeypatch, api, behavior="valid"):
    package = tmp_path / "stub" / "cellpose"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    code = """import os
from pathlib import Path
MODEL_NAMES = ['cpsam', 'cpsam_v2']
def download(name):
    assert name == 'cpsam', 'must not select a newer default model'
    target = Path(os.environ['CELLPOSE_LOCAL_MODELS_PATH']) / name
"""
    if behavior == "error":
        code += "    raise RuntimeError('download failed deliberately')\n"
    elif behavior == "missing":
        code += "    return str(target)\n"
    else:
        if behavior == "wrong_path":
            code += "    target = target.with_name('cpsam_v2')\n"
        payload = b"" if behavior == "empty" else b"offline cpsam fixture"
        code += f"    target.write_bytes({payload!r})\n    return str(target)\n"
    if api in ("current", "both"):
        code += "def cache_model_path(backbone):\n    return download(backbone)\n"
    if api == "legacy":
        code += "def cache_CPSAM_model_path():\n    return download('cpsam')\n"
    if api == "both":
        code += "def cache_CPSAM_model_path():\n    raise AssertionError('must not retry legacy API')\n"
    if api == "unsupported_model":
        code += "MODEL_NAMES = ['cyto3']\ndef cache_model_path(backbone):\n    return download(backbone)\n"
    (package / "models.py").write_text(code)
    monkeypatch.setenv("PYTHONPATH", str(package.parent))


@pytest.mark.parametrize("api", ["current", "legacy", "both"])
def test_cpsam_download_uses_installed_api_and_isolated_cache(tmp_path, monkeypatch, api):
    install_stub(tmp_path, monkeypatch, api)
    dest = tmp_path / "assets" / "cellpose"
    monkeypatch.setenv("CELLPOSE_LOCAL_MODELS_PATH", str(tmp_path / "wrong_cache"))
    record = assets.place_cpsam(dest, None)
    assert Path(record["path"]) == dest / "cpsam"
    assert (dest / "cpsam").read_bytes() == b"offline cpsam fixture"
    assert record["sha256"] == assets.sha256(dest / "cpsam")
    assert not (tmp_path / "wrong_cache").exists()


@pytest.mark.parametrize("api,behavior", [
    ("none", "valid"), ("unsupported_model", "valid"), ("both", "error"),
    ("current", "missing"), ("current", "wrong_path"), ("current", "empty"),
])
def test_cpsam_download_fails_closed(tmp_path, monkeypatch, api, behavior):
    install_stub(tmp_path, monkeypatch, api, behavior)
    with pytest.raises(SystemExit, match="NOT_READY"):
        assets.place_cpsam(tmp_path / "cellpose", None)


def test_explicit_missing_source_never_triggers_download(tmp_path, monkeypatch):
    install_stub(tmp_path, monkeypatch, "current")
    with pytest.raises(SystemExit, match="explicit cpsam source"):
        assets.place_cpsam(tmp_path / "cellpose", tmp_path / "missing")
    assert not (tmp_path / "cellpose" / "cpsam").exists()


def test_existing_cpsam_is_preserved_not_trusted(tmp_path):
    target = tmp_path / "cpsam"
    target.write_bytes(b"partial download")
    with pytest.raises(SystemExit, match="unverified"):
        assets.place_cpsam(tmp_path, None)
    assert target.read_bytes() == b"partial download"

"""Shared fixtures of the analyzer's SSL tests (CPU, no download).

`stub_backbone` routes `DinoV3Projector` to `StubViT`, a tiny stand-in for the DINOv3 ViT-B/16 with
the dinov3 parameter names (`patch_embed.*`, `cls_token`, `blocks.<i>.*`, `norm.*`), `forward` -> the
CLS token, `forward_features` -> {x_norm_clstoken, x_norm_patchtokens} and blocks that carry
`sample_drop_ratio` (stochastic depth in train mode only, by dinov3's batch-subset rule). `crops` gives 32
synthetic 6-channel crops on two plates (four wells of two sites of two cells each; wells W0 / W1
are "val", W2 / W3 "train" under the metadata key `split`), read through a patched `io.imread`.
"""
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

import shape.tools.cell_state_analyzer.tool as tool_mod

# pytest's importlib mode imports this file's parent packages from their files when none is loaded yet; that
# replaces the `shape.tools` package which `import shape` had just filled (e.g. with `shape.tools.base`), so a
# later dotted lookup such as monkeypatch.setattr("shape.tools.base....") fails in a combined run. Re-attach
# every loaded submodule to its current parent.
for _name, _module in list(sys.modules.items()):
    _parent, _, _child = _name.rpartition(".")
    if _name.startswith("shape.") and _parent in sys.modules and not hasattr(sys.modules[_parent], _child):
        setattr(sys.modules[_parent], _child, _module)

tool_mod.plt.switch_backend("Agg")      # the tool's main-thread loss curve; no display backend in tests

DIM = 768          # the projector's feature width for dinov3_vitb16


class StubBlock(nn.Module):
    def __init__(self, with_ratio=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        self.mlp = nn.Linear(DIM, DIM)
        if with_ratio:
            self.sample_drop_ratio = 0.0

    def forward(self, x):
        # dinov3's SelfAttentionBlock: in training a random subset of max(int(b * (1 - ratio)), 1) of the
        # b rows takes the residual branch, scaled by b / subset; the other rows skip it
        ratio = getattr(self, "sample_drop_ratio", 0.0)
        if not (self.training and ratio > 0):
            return x + self.mlp(self.norm1(x))
        b = len(x)
        keep = max(int(b * (1 - ratio)), 1)
        rows = torch.randperm(b, device=x.device)[:keep]
        return torch.index_add(x, 0, rows, self.mlp(self.norm1(x[rows])), alpha=b / keep)


class StubViT(nn.Module):
    def __init__(self, depth=4, patch=56, with_ratio=True):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, DIM, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.blocks = nn.ModuleList(StubBlock(with_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Identity()

    def forward_features(self, x):
        tokens = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        # the CLS reads the patches (the blocks act per token, so this stands in for attention)
        cls = self.cls_token.expand(len(x), -1, -1) + tokens.mean(dim=1, keepdim=True)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return {"x_norm_clstoken": tokens[:, 0], "x_norm_patchtokens": tokens[:, 1:]}

    def forward(self, x):
        return self.head(self.forward_features(x)["x_norm_clstoken"])


@pytest.fixture
def stub_backbone(monkeypatch, tmp_path):
    """DinoV3Projector builds a StubViT (options: depth, with_ratio) and loads its saved "pretrained"
    weights. Returns {"options", "weights"} (the weights file of the pretrained state)."""
    options = {"depth": 4, "with_ratio": True}
    weights = tmp_path / "stub_vit_pretrained.pth"
    with torch.random.fork_rng():
        torch.manual_seed(0)
        torch.save(StubViT(depth=options["depth"]).state_dict(), weights)
    monkeypatch.setattr(tool_mod, "hf_hub_download", lambda **kwargs: str(weights))
    monkeypatch.setattr(tool_mod.torch.hub, "load", lambda *args, **kwargs: StubViT(**options))
    return {"options": options, "weights": weights}


@pytest.fixture
def stub_vit():
    return StubViT


def make_crops(plates=("P1", "P2"), wells=4, sites=2, cells=2, seed=0, split=True):
    rng = np.random.default_rng(seed)
    images, meta = {}, []
    for p, plate in enumerate(plates):
        for w in range(wells):
            for s in range(sites):
                for c in range(cells):
                    path = f"crop_{plate}_W{w}_s{s}_c{c}.tif"
                    # plates differ in brightness, so per-plate intensity statistics differ
                    images[path] = (40 + rng.random((6, 24, 24)) * (90 + 60 * p)).astype(np.uint8)
                    row = {"plate": plate, "experiment": "E1", "well": f"W{w}", "site": f"W{w}_s{s}",
                           "group": "default"}
                    if split:
                        row["split"] = "val" if w < 2 else "train"
                    meta.append(row)
    return images, meta


@pytest.fixture
def crop_factory():
    return make_crops


@pytest.fixture
def crops(monkeypatch):
    """(paths, metadata, images) of `make_crops()`; `io.imread` reads `images` (extend it for more crops)."""
    images, meta = make_crops()
    monkeypatch.setattr(tool_mod.io, "imread", lambda path: images[path])
    return list(images), meta, images


@pytest.fixture
def cpu_tool(stub_backbone, monkeypatch):
    """The analyzer on the CPU with the stub backbone; clustering (unchanged, scanpy) is skipped."""
    tool = tool_mod.Cell_State_Analyzer_Tool()
    tool.device = torch.device("cpu")
    monkeypatch.setattr(tool, "_compute_umap_and_clustering", lambda *args, **kwargs: None)
    return tool


@pytest.fixture
def run_call(cpu_tool, crops, tmp_path):
    """execute() on the `crops` (or the given paths / metadata) with max_epochs 1 unless set, each call
    in its own output_subdir under tmp_path."""
    def run(subdir="call", paths=None, meta=None, **kwargs):
        kwargs.setdefault("max_epochs", 1)
        return cpu_tool.execute(cell_crops=list(paths or crops[0]),
                                cell_metadata=[dict(m) for m in (meta or crops[1])],
                                query_cache_dir=str(tmp_path), output_subdir=subdir, **kwargs)
    return run


@pytest.fixture
def train_spy(cpu_tool, monkeypatch):
    """Every `_train_model` call: its loader, kwargs, trainable parameter names and model state at entry
    (`calls.spy.before(n, model)`, optional, runs before the real call and may raise)."""
    class Calls(list):
        pass

    calls = Calls()
    real = cpu_tool._train_model

    def spy(model, train_loader, *args, **kwargs):
        calls.append({"loader": train_loader, "kwargs": kwargs,
                      "trainable": [n for n, p in model.named_parameters() if p.requires_grad],
                      "state": {k: v.detach().clone() for k, v in model.state_dict().items()}})
        if spy.before is not None:
            spy.before(len(calls), model)
        return real(model, train_loader, *args, **kwargs)

    spy.before = None
    monkeypatch.setattr(cpu_tool, "_train_model", spy)
    calls.spy = spy
    return calls

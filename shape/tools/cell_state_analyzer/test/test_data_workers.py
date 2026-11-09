"""The analyzer's DataLoader workers (CPU, the stub backbone of conftest.py): the worker-count resolver, the
per-sample random streams that make a sample a pure function of (seed, epoch, index), the per-plate sampler
drawing an epoch once, the nn_other_image refresh rule, and -- where the start method fork exists -- training
runs with 0 and 2 workers giving the same losses, weights and embedding.
"""
import multiprocessing
import sys

import numpy as np
import pytest
import torch

import shape.tools.cell_state_analyzer.tool as tool_mod
from shape.tools.cell_state_analyzer import quant

pytest.importorskip("anndata")
import anndata as ad  # noqa: E402

FORK = "fork" in multiprocessing.get_all_start_methods() and sys.platform != "darwin"
needs_fork = pytest.mark.skipif(not FORK, reason="DataLoader workers are forked; this platform has no safe fork "
                                                 "(Windows, macOS), so the tool loads with 0 workers there")


# --------------------------------------------------------------------- the resolver

@pytest.fixture
def linux(monkeypatch):
    """A fork platform with 24 CPUs and no affinity / Slurm limit unless a test sets one."""
    monkeypatch.setattr(tool_mod.multiprocessing, "get_all_start_methods", lambda: ["fork", "spawn", "forkserver"])
    monkeypatch.setattr(tool_mod.sys, "platform", "linux")
    monkeypatch.setattr(tool_mod.os, "cpu_count", lambda: 24)
    monkeypatch.delattr(tool_mod.os, "sched_getaffinity", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv(tool_mod.LOADER_WORKERS_ENV, raising=False)    # conftest.py sets it to 0
    return monkeypatch


def test_no_workers_without_fork(monkeypatch):
    monkeypatch.setattr(tool_mod.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert tool_mod.resolve_num_workers() == {"num_workers": 0, "usable_cpus": None, "start_method": None}


def test_no_workers_on_macos(linux):
    linux.setattr(tool_mod.sys, "platform", "darwin")
    assert tool_mod.resolve_num_workers()["num_workers"] == 0


def test_workers_are_the_cpus_minus_one(linux):
    assert tool_mod.resolve_num_workers() == {"num_workers": 23, "usable_cpus": 24, "start_method": "fork"}


def test_the_affinity_bounds_the_cpus(linux):
    linux.setattr(tool_mod.os, "sched_getaffinity", lambda pid: set(range(6)), raising=False)
    assert tool_mod.resolve_num_workers()["num_workers"] == 5


@pytest.mark.parametrize("slurm, expected", [("4", 3), ("64", 5), ("1", 0), ("", 5), ("x", 5), ("0", 5)])
def test_slurm_cpus_bound_the_cpus_when_smaller(linux, slurm, expected):
    linux.setattr(tool_mod.os, "sched_getaffinity", lambda pid: set(range(6)), raising=False)
    linux.setenv("SLURM_CPUS_PER_TASK", slurm)
    assert tool_mod.resolve_num_workers()["num_workers"] == expected


def test_one_cpu_gives_no_worker(linux):
    linux.setattr(tool_mod.os, "cpu_count", lambda: None)
    assert tool_mod.resolve_num_workers()["num_workers"] == 0


@pytest.mark.parametrize("cap, expected", [("0", 0), ("4", 4), ("64", 23), ("", 23)])
def test_the_workers_env_caps_the_workers(linux, cap, expected):
    linux.setenv(tool_mod.LOADER_WORKERS_ENV, cap)
    assert tool_mod.resolve_num_workers()["num_workers"] == expected


def test_loader_kwargs():
    assert tool_mod.loader_kwargs(0) == {"num_workers": 0, "pin_memory": False}
    assert tool_mod.loader_kwargs(0, True) == {"num_workers": 0, "pin_memory": True}
    if FORK:
        kwargs = tool_mod.loader_kwargs(3, True)
        assert kwargs["num_workers"] == 3 and kwargs["pin_memory"] is True
        assert kwargs["multiprocessing_context"].get_start_method() == "fork"


def test_workers_get_a_short_multiprocessing_temp_dir_under_a_long_tmpdir(monkeypatch):
    """A long TMPDIR (the benchmark drivers' cache tmp) would overflow the workers' AF_UNIX socket paths."""
    config = multiprocessing.current_process()._config
    monkeypatch.setitem(config, "tempdir", "/scratch/" + "x" * 120)
    made = []
    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix, dir: made.append((prefix, dir)) or "/tmp/pymp-test")
    monkeypatch.setattr("multiprocessing.util.Finalize", lambda *a, **k: None)
    tool_mod.short_multiprocessing_tempdir()
    assert made == [("pymp-", "/tmp")] and config["tempdir"] == "/tmp/pymp-test"
    tool_mod.short_multiprocessing_tempdir()                             # short already: left as it is
    assert len(made) == 1
    if FORK:
        monkeypatch.setitem(config, "tempdir", "/scratch/" + "y" * 120)
        tool_mod.loader_kwargs(2)                                        # a loader with workers shortens it
        assert config["tempdir"] == "/tmp/pymp-test" and len(made) == 2


# --------------------------------------------------------------------- per-sample streams (no worker)

def _dataset(cpu_tool, crops, augmentation, seed=7, pairing="same_well"):
    paths, meta, _ = crops
    units = tool_mod.CropUnits(quant.identities(meta), seed=0)
    return tool_mod.CellCropDataset(paths, transform=cpu_tool._get_augmentation_transform(augmentation=augmentation),
                                    plate_ids=[""] * len(paths), units=units, view_pairing=pairing, seed=seed)


def _sample(dataset, i):
    v1, v2, _, _, identity = dataset[i]
    return v1, v2, identity["partner"]


def _same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]


@pytest.mark.parametrize("augmentation", ["strong", "light"])
def test_a_sample_is_a_function_of_seed_epoch_and_index(cpu_tool, crops, augmentation):
    dataset = _dataset(cpu_tool, crops, augmentation)
    n = len(dataset)
    dataset.set_epoch(3)
    torch.manual_seed(0)
    state = torch.get_rng_state()
    forward = [_sample(dataset, i) for i in range(n)]
    assert torch.equal(torch.get_rng_state(), state)                    # the caller's stream is untouched
    torch.manual_seed(99)                                               # other global and unit streams
    np.random.seed(99)
    dataset.units.rng.integers(10, size=7)
    backward = [_sample(dataset, i) for i in reversed(range(n))][::-1]  # another access order
    assert all(_same(a, b) for a, b in zip(forward, backward))
    threads = torch.get_num_threads()
    try:                                                                # a worker computes at one thread
        torch.set_num_threads(1)
        single = [_sample(dataset, i) for i in range(n)]
    finally:
        torch.set_num_threads(threads)
    assert all(_same(a, b) for a, b in zip(forward, single))
    assert not torch.equal(forward[0][0], forward[0][1])                # the two views differ
    dataset.set_epoch(4)
    later = [_sample(dataset, i) for i in range(n)]
    assert not any(torch.equal(a[0], b[0]) for a, b in zip(forward, later))
    assert [a[2] for a in forward] != [b[2] for b in later]             # the partners are redrawn too
    other = _dataset(cpu_tool, crops, augmentation, seed=8)
    other.set_epoch(3)
    assert not torch.equal(_sample(other, 0)[0], forward[0][0])


def test_partner_streams_follow_the_epoch_for_nn_pairing(cpu_tool, crops):
    dataset = _dataset(cpu_tool, crops, "strong", pairing="nn_other_image")
    n = len(dataset)
    dataset.units.nn = np.tile(np.arange(5, dtype=np.int64), (n, 1))
    first = [_sample(dataset, i)[2] for i in range(n)]
    dataset.units.rng.integers(10, size=7)
    assert [_sample(dataset, i)[2] for i in range(n)] == first
    assert set(first) <= set(range(5))


def test_unit_batch_sampler_draws_an_epoch_at_its_first_batch(crops):
    _, meta, _ = crops
    ids = quant.identities(meta)
    sampler = tool_mod.UnitBatchSampler(tool_mod.CropUnits(ids, seed=0), 5, "per_plate")
    iter(sampler), iter(sampler)                                        # what a multi-worker iterator does
    got = [list(iter(sampler)), list(iter(sampler))]
    fresh = tool_mod.UnitBatchSampler(tool_mod.CropUnits(ids, seed=0), 5, "per_plate")
    assert got == [list(fresh), list(fresh)]


# --------------------------------------------------------------------- the nn_other_image refresh rule

@pytest.fixture
def two_workers(monkeypatch):
    """resolve_num_workers -> 2; every DataLoader's requested worker count is recorded and the loaders load in
    the main process (so this runs on every platform)."""
    requested = []
    real = tool_mod.loader_kwargs
    monkeypatch.setattr(tool_mod, "resolve_num_workers",
                        lambda: {"num_workers": 2, "usable_cpus": 3, "start_method": "fork"})
    monkeypatch.setattr(tool_mod, "loader_kwargs",
                        lambda num_workers, pin_memory=False: requested.append(num_workers) or real(0, pin_memory))
    return requested


@pytest.mark.parametrize("space, epochs, train_workers", [
    ("current", 0.5, 0),            # every 2 of 4 steps: a mid-epoch rebuild, the training loader loads in process
    ("current", 1.5, 0),            # every 6 steps: a rebuild in the middle of the second epoch
    ("current", 1.0, 2),            # at the end of every epoch: the next epoch's workers fork with the new table
    ("current", 2.0, 2),
    ("current", 0.0, 2),            # never rebuilt
    ("frozen", 0.5, 2),             # nn_space=frozen never rebuilds
])
def test_a_mid_epoch_rebuild_keeps_the_training_loader_in_process(run_call, two_workers, space, epochs,
                                                                    train_workers):
    out = run_call(split_key="split", view_pairing="nn_other_image", nn_space=space, nn_k=2,
                   nn_refresh_epochs=epochs, batch_size=4, max_epochs=2)
    config = out["training_config"]
    assert config["num_workers"] == 2 and config["train_num_workers"] == train_workers
    assert config["nn_refresh_steps"] == (tool_mod.nn_refresh_steps(epochs, 4) if space == "current" else 0)
    assert 0 in two_workers if train_workers == 0 else set(two_workers) == {2}


def test_the_worker_counts_are_recorded(run_call, two_workers):
    trained = run_call("trained", split_key="split", batch_size=4)
    assert (trained["training_config"]["num_workers"], trained["training_config"]["train_num_workers"]) == (2, 2)
    frozen = run_call("frozen", adaptation_depth="zero_shot")
    assert frozen["training_config"]["num_workers"] == 2 and "train_num_workers" not in frozen["training_config"]


# --------------------------------------------------------------------- 0 vs 2 forked workers

@pytest.fixture
def losses(monkeypatch):
    """Every InfoNCE value the call computes (each training step's, then the eval loss batches')."""
    seen = []
    real = tool_mod.contrastive_loss

    def record(*args, **kwargs):
        loss = real(*args, **kwargs)
        seen.append(float(loss.detach()))
        return loss

    monkeypatch.setattr(tool_mod, "contrastive_loss", record)
    return seen


CONFIGS = {
    "defaults": {},
    "ssl116_same_well_per_plate": {"augmentation": "light", "view_pairing": "same_well", "batch_sampling": "per_plate",
                                   "contrastive_mask": "same_site", "drop_path": 0.2},
    "nn_refresh_each_epoch": {"view_pairing": "nn_other_image", "nn_space": "current", "nn_k": 2,
                              "nn_refresh_epochs": 1.0},
    "nn_refresh_mid_epoch": {"view_pairing": "nn_other_image", "nn_space": "current", "nn_k": 2,
                             "nn_refresh_epochs": 0.5},
}


@needs_fork
@pytest.mark.parametrize("name", list(CONFIGS))
def test_zero_and_two_workers_train_identically(run_call, losses, monkeypatch, name):
    runs = {}
    for workers in (0, 2):
        monkeypatch.setattr(tool_mod, "resolve_num_workers",
                            lambda workers=workers: {"num_workers": workers, "usable_cpus": workers + 1,
                                                     "start_method": "fork"})
        losses.clear()
        out = run_call(f"{name}_w{workers}", split_key="split", max_epochs=2, batch_size=4, learning_rate=1e-3,
                       **CONFIGS[name])
        runs[workers] = {"out": out, "losses": list(losses),
                         "end": torch.load(out["resume_end"], map_location="cpu", weights_only=True),
                         "X": np.asarray(ad.read_h5ad(out["adata_path"]).X)}
    zero, two = runs[0], runs[2]
    assert two["out"]["training_config"]["num_workers"] == 2
    assert two["out"]["training_config"]["train_num_workers"] == (0 if name == "nn_refresh_mid_epoch" else 2)
    assert len(zero["losses"]) > 8 and zero["losses"] == two["losses"]
    assert zero["out"]["loss_history"] == two["out"]["loss_history"]
    assert zero["end"]["model"].keys() == two["end"]["model"].keys()
    for key, value in zero["end"]["model"].items():
        assert torch.equal(value, two["end"]["model"][key]), key
    rng0, rng2 = zero["end"]["rng"], two["end"]["rng"]
    assert torch.equal(rng0["torch"], rng2["torch"]) and torch.equal(rng0["loader"], rng2["loader"])
    assert rng0["units"] == rng2["units"]
    assert np.array_equal(zero["X"], two["X"])

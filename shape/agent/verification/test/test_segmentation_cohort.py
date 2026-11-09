"""The segmentation cohort (`segmentation_cohort`): its policy -- the calibration size, the groups, the draw, the
consensus, the routing -- and `run` over a set of images with a stand-in per-image run."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from shape.agent.verification import segmentation_cohort as cohort
from shape.utils.schemas import StageResolution, VLRRResult

DEFAULTS = {"diameter": 30.0, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "min_size": 15,
            "color_class_filter": None}
LEVERS = list(DEFAULTS)


@pytest.mark.parametrize("n, size", [(1, 0), (9, 0), (10, 3), (60, 3), (61, 4), (100, 5), (101, 6), (200, 8),
                                     (500, 15), (1000, 28), (1001, 28), (2000, 38), (10000, 118)])
def test_the_calibration_size_is_progressive_at_least_three_and_none_below_ten(n, size):
    assert cohort.calibration_size(n) == size


def test_the_calibration_size_never_drops_as_the_group_grows():
    sizes = [cohort.calibration_size(n) for n in range(1, 3001)]
    assert all(b >= a for a, b in zip(sizes, sizes[1:]))


def profile(specimen="cell_culture", modality="fluorescence", confidence="high"):
    return SimpleNamespace(specimen_context=specimen, modality=modality, confidence=confidence)


def test_the_group_is_the_preflights_specimen_and_modality_and_an_unsure_read_has_none():
    assert cohort.group_key(profile()) == ("cell_culture", "fluorescence")
    assert cohort.group_key(profile(modality="brightfield")) == ("cell_culture", "brightfield")
    assert cohort.group_key(None) is None
    assert cohort.group_key(profile(confidence="low")) is None
    assert cohort.group_key(profile(specimen="unknown")) is None


def test_the_draw_is_random_but_reproducible_and_sorted():
    ids = [f"img{i:02d}" for i in range(40)]
    first = cohort.draw(ids, 5, ("cell_culture", "fluorescence"))
    assert first == cohort.draw(list(reversed(ids)), 5, ("cell_culture", "fluorescence")) == sorted(first)
    assert len(set(first)) == 5 and set(first) <= set(ids)
    assert cohort.draw(ids, 5, ("blood_smear", "brightfield")) != first
    assert cohort.draw(ids[:3], 5, None) == ids[:3]


def test_the_consensus_transfers_only_the_levers_the_calibration_agrees_on():
    records = [{"levers": {"cellprob_threshold": -1.0, "min_size": 20}},
               {"levers": {"cellprob_threshold": -1.1, "min_size": 20}},
               {"levers": {"cellprob_threshold": -0.9, "flow_threshold": 0.6}},
               {"levers": {"cellprob_threshold": -1.0}}]
    params, report = cohort.consensus_params(DEFAULTS, records, ["cellprob_threshold", "flow_threshold", "min_size"])
    assert report["stable"] and report["selected_params"] == {"cellprob_threshold": -1.0}
    assert params["min_size"] == 15 and report["levers"]["min_size"]["support"] == 0.5     # 2 of 4: below 75%
    dispersed = [{"levers": {"cellprob_threshold": v}} for v in (-1.0, -3.0, -0.2, -2.0)]
    _, report = cohort.consensus_params(DEFAULTS, dispersed, ["cellprob_threshold"])
    assert not report["stable"] and report["levers"]["cellprob_threshold"]["reason"] == "numeric calibration is dispersed"
    split = [{"levers": {"color_class_filter": v}} for v in ("keep_high_br", "keep_low_br", "keep_high_br",
                                                            "keep_low_br")]
    _, report = cohort.consensus_params(DEFAULTS, split, ["color_class_filter"])
    assert not report["stable"] and report["levers"]["color_class_filter"]["agreement"] == 0.5
    whole = [{"levers": {"min_size": v}} for v in (20, 21, 20)]
    params, report = cohort.consensus_params(DEFAULTS, whole, ["min_size"])
    assert params["min_size"] == 20 and isinstance(params["min_size"], int)


def test_cpjump1s_routing_sends_the_flagged_images_and_every_image_past_a_fifth():
    ids = [f"i{i}" for i in range(10)]
    routed, report = cohort.route_outliers({i: (["bad"] if i in {"i0", "i1"} else []) for i in ids}, ids)
    assert routed == {"i0", "i1"} and not report["route_all"] and report["initial_outlier_fraction"] == 0.2
    routed, report = cohort.route_outliers({i: (["bad"] if i in {"i0", "i1", "i2"} else []) for i in ids}, ids)
    assert routed == set(ids) and report["route_all"]


# ---- run: a set of images, a stand-in per-image run ---------------------------------------------------------------

def stage(params=None, problems=(), status="satisfied", delivered=True):
    call = "reference"
    return StageResolution(result={"mask_path": "m.npy"} if delivered else None, status=status,
                           params={**DEFAULTS, **(params or {})}, verdict=VLRRResult("PASS", 1.0, {}, action="accept"),
                           reason="cohort_applied", details={"admission": {"decision": "RUN"}, "call": call,
                                                             "measured_problems": {call: list(problems)}})


class Images:
    """`n` images per group (a dict group -> n, the group a (specimen, modality) or None for no read); each run
    answers with `levers(image_id, options)` and `problems(image_id, options)`; the calls are recorded in order."""

    def __init__(self, groups, levers=None, problems=None):
        self.specs, self.profiles, self.calls = [], {}, []
        for g, n in groups.items():
            for k in range(n):
                image_id = f"{'_'.join(g) if g else 'none'}_{k:03d}"
                self.specs.append({"path": f"{image_id}.png", "image_id": image_id})
                self.profiles[image_id] = profile(*g) if g else None
        self.levers = levers or (lambda i, o: {})
        self.problems = problems or (lambda i, o: [])

    def run_one(self, spec, options):
        i = spec["image_id"]
        self.calls.append((i, options))
        res = stage(self.levers(i, options), self.problems(i, options))
        return [res], res.result, res.status, None

    def go(self):
        return cohort.run(self.specs, self.run_one, lambda items, fn: [fn(s) for s in items],
                          lambda spec: self.profiles[spec["image_id"]], DEFAULTS, LEVERS)

    def options(self, image_id):
        return [o for i, o in self.calls if i == image_id]


def test_a_group_below_ten_and_an_image_without_a_read_take_the_per_image_vrr():
    images = Images({("cell_culture", "fluorescence"): 9, None: 2})
    runs, record = images.go()
    assert len(runs) == 11 and record["counts"] == {"per_image_vrr": 11}
    assert all(o is None for _, o in images.calls) and len(images.calls) == 11
    assert record["ungrouped"] == ["none_000", "none_001"]
    assert record["groups"][0]["routing"] == "per_image_small_group" and record["groups"][0]["calibration_size"] == 0


def test_a_group_is_calibrated_its_consensus_applied_and_a_flagged_image_routed_from_the_consensus():
    key = ("blood_smear", "brightfield")
    ids = [f"blood_smear_brightfield_{k:03d}" for k in range(12)]
    sample = cohort.draw(ids, 3, key)
    flagged = next(i for i in ids if i not in sample)                         # a cohort image with a problem
    images = Images({key: 12},
                    levers=lambda i, o: {"cellprob_threshold": -1.0, "diameter": 60.0} if o is None else {},
                    problems=lambda i, o: ["tiny_mask_fraction 0.2 above 0.15"]
                    if i == flagged and o and "cohort_levers" in o else [])
    runs, record = images.go()
    group = record["groups"][0]
    assert group["calibration"] == sample and all(images.options(i) == [None] for i in sample)   # their own VRR
    # the consensus covers the agent-tuned levers only: the diameter is measured per image
    assert group["cohort_levers"] == {"cellprob_threshold": -1.0}
    assert "diameter" not in group["consensus"]["levers"] and group["consensus"]["stable"]
    rest = [i for i in ids if i not in sample]
    assert all(images.options(i)[0] == {"cohort_levers": {"cellprob_threshold": -1.0}} for i in rest)
    assert group["routed"] == [flagged] and group["cohort_problems"] == {flagged: ["tiny_mask_fraction 0.2 above 0.15"]}
    assert images.options(flagged) == [{"cohort_levers": {"cellprob_threshold": -1.0}},
                                       {"start_params": {"cellprob_threshold": -1.0}}]
    assert record["parameter_source"][flagged] == "per_image_vrr"
    assert record["counts"] == {"calibration": 3, "cohort": 8, "per_image_vrr": 1}
    assert len(runs) == 12 and list(record["parameter_source"]) == ids                # every image, in order


def test_an_unstable_consensus_sends_the_rest_of_its_group_per_image():
    values = itertools.cycle([-1.0, -3.0, -0.2])
    images = Images({("tissue_section", "histology"): 10},
                    levers=lambda i, o: {"cellprob_threshold": next(values)} if o is None else {})
    _, record = images.go()
    group = record["groups"][0]
    assert group["routing"] == "per_image_unstable_calibration" and not group["consensus"]["stable"]
    assert record["counts"] == {"calibration": 3, "per_image_vrr": 7}
    assert all(o is None for _, o in images.calls)


def test_only_the_flagged_images_are_routed_however_many_and_no_consensus_starts_them_at_the_defaults():
    flagged = lambda i: int(i[-3:]) % 2 == 0                               # half the group: still only those
    images = Images({("cell_culture", "phase_contrast"): 20},
                    problems=lambda i, o: ["count_miss_fraction 0.3 above 0.15"] if o and flagged(i) else [])
    _, record = images.go()
    group = record["groups"][0]
    rest = [i for i in group["images"] if i not in group["calibration"]]
    assert group["cohort_levers"] == {} and group["routed"] == sorted(i for i in rest if flagged(i))
    assert group["routing"] == {"flagged": len(group["routed"]),
                                "flagged_fraction": round(len(group["routed"]) / len(rest), 6)}
    assert all(images.options(i) == [{"cohort_levers": {}}, None] for i in group["routed"])
    assert all(images.options(i) == [{"cohort_levers": {}}] for i in rest if not flagged(i))
    assert record["counts"] == {"calibration": 3, "cohort": 17 - len(group["routed"]),
                                "per_image_vrr": len(group["routed"])}


def test_the_groups_run_at_the_same_time_within_the_worker_cap():
    """The smears' calibration (sorted first) waits until the fluorescence group has applied its consensus: run one
    group after another, it would wait in vain. Never more than `workers` images at once."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock, state, applied = threading.Lock(), {"now": 0, "peak": 0}, threading.Event()
    images = Images({("cell_culture", "fluorescence"): 10, ("blood_smear", "brightfield"): 10, None: 4})

    def run_one(spec, options):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        if options and spec["image_id"].startswith("cell_culture"):
            applied.set()                                    # a fluorescence image in the cohort mode
        if options is None and spec["image_id"].startswith("blood"):
            assert applied.wait(timeout=10), "the smears' calibration held the other group back"
        with lock:
            state["now"] -= 1
        return images.run_one(spec, options)

    def pool_map(items, fn):
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(fn, items))

    runs, record = cohort.run(images.specs, run_one, pool_map, lambda s: images.profiles[s["image_id"]],
                              DEFAULTS, LEVERS, workers=4)
    assert applied.is_set() and state["peak"] <= 4 and len(runs) == 24
    assert record["counts"] == {"calibration": 6, "cohort": 14, "per_image_vrr": 4}


def test_groups_are_calibrated_apart_and_the_runs_keep_the_order_of_the_specs():
    images = Images({("cell_culture", "fluorescence"): 10, ("cell_culture", "brightfield"): 10})
    runs, record = images.go()
    assert [g["group"] for g in record["groups"]] == [{"specimen_context": "cell_culture", "modality": "brightfield"},
                                                     {"specimen_context": "cell_culture", "modality": "fluorescence"}]
    assert record["counts"] == {"calibration": 6, "cohort": 14}
    assert list(record["parameter_source"]) == [s["image_id"] for s in images.specs]
    assert len(runs) == 20


def test_a_delivery_without_a_mask_or_a_conceded_stage_is_a_problem():
    assert cohort.problems_of([stage()]) == []
    assert cohort.problems_of([stage(problems=["p"])]) == ["p"]
    assert cohort.problems_of([stage(delivered=False)]) == ["no segmentation was delivered"]
    assert cohort.problems_of([stage(status="conceded")]) == ["the stage conceded (cohort_applied)"]


def test_the_image_ids_must_be_unique():
    with pytest.raises(ValueError, match="unique image_id"):
        cohort.run([{"path": "a", "image_id": "x"}, {"path": "b", "image_id": "x"}], None, None, None, DEFAULTS, LEVERS)

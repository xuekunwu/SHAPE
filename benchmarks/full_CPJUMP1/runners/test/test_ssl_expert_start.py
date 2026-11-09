"""The SHAPE arm's SSL expert start (user decisions 2026-10-13): two rounds of progressive unfreezing, round 1
representation_RxRx1's A21 expert configuration at trainable_pct 33, round 2 resuming it at trainable_pct 66."""
import inspect

from benchmarks.full_CPJUMP1.harness import pipeline
from benchmarks.representation_RxRx1.runners.epoch_budget import FixedEpochVerifier
from benchmarks.representation_RxRx1.runners.shape_arm import EXPERT_CONFIG


def test_the_expert_start_is_rxrx1_a21s_at_a_third_then_resumed_at_two_thirds():
    assert [dict(r) for r in pipeline.SSL_EXPERT_CONFIG] == EXPERT_CONFIG
    assert EXPERT_CONFIG == [
        {"view_pairing": "same_well", "contrastive_mask": "same_well", "batch_sampling": "per_plate",
         "intensity_norm": "per_plate", "augmentation": "light", "trainable_pct": 33},
        {"init_from": "parent", "trainable_pct": 66},
    ]


def test_the_verifier_takes_it_and_plans_rounds_one_and_two_as_expert_rounds():
    verifier = FixedEpochVerifier({"max_epochs": 8}, expert_config=[dict(r) for r in pipeline.SSL_EXPERT_CONFIG])
    assert verifier.expert_config == [dict(r) for r in pipeline.SSL_EXPERT_CONFIG]
    assert verifier._expert_round()
    text = verifier.admission_text()
    assert "in round 2, resuming a round-1 call (init_from parent), trainable_pct=66" in text
    assert "trainable_pct=33" in text and "This is round 1." in text
    verifier.rounds.append([1])
    assert verifier._expert_round() and "This is round 2." in verifier.admission_text()
    verifier.rounds.append([2])
    assert not verifier._expert_round()


def test_the_shape_ssl_stage_passes_it_to_its_verifier():
    source = inspect.getsource(pipeline.representation)
    assert "[dict(r) for r in SSL_EXPERT_CONFIG]" in source  # the default (legacy) search policy keeps the two-round start

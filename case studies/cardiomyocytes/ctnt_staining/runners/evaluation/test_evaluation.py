"""Contract tests for failure accounting and scientific aggregation."""
import tempfile
import unittest
from pathlib import Path

from evaluation_common import fraction, write
from process import classify_error, evaluate
from report import donor_macro, manifest


def event(**kwargs):
    return dict(arm="codex", stage="crop", unit="i1", call="c1", status="complete",
                error=None, selected=True, is_tool_attempt=True, **kwargs)


class AccountingTests(unittest.TestCase):
    def test_rejected_request_is_not_native_failure(self):
        decisions = [dict(arm="codex", stage="crop", action="run", outcome="rejected",
                          message="run needs a non-empty JSON list of call specs", has_reason=True)]
        r = evaluate([event()], decisions)["rows"][0]
        self.assertEqual(r["native_errors"], 0)
        self.assertEqual(r["tool_completion_rate"], 1)
        self.assertEqual(r["run_admission_rate"], 0)
        self.assertEqual(r["request_rejection_rate"], 1)
        self.assertEqual(r["rejection_classes"], {"request_schema": 1})

    def test_uncorrected_input_is_not_execution(self):
        e = event()
        e.update(stage="batch", is_tool_attempt=False)
        r = evaluate([e], [])["rows"][0]
        self.assertIsNone(r["logical_attempts"])
        self.assertIsNone(r["tool_completion_rate"])
        self.assertEqual(r["non_tool_input_records"], 1)
        self.assertEqual(r["delivered_units"], 1)

    def test_search_attempts_and_delivery_have_separate_denominators(self):
        first, second = event(), event()
        first.update(call="c0", selected=False)
        r = evaluate([first, second], [])["rows"][0]
        self.assertEqual(r["work_amplification"], 2)
        self.assertEqual(r["selected_attempt_share"], .5)
        self.assertEqual(r["tool_completion_rate"], 1)

    def test_qc_error_class_is_not_schema(self):
        e = event()
        e.update(status="error", error="Segmentation output failed visual QC", selected=False)
        r = evaluate([e], [])["rows"][0]
        self.assertEqual(r["native_error_classes"], {"native_quality_gate": 1})
        self.assertEqual(r["tool_completion_rate"], 0)

    def test_evidence_budget_separate_from_schema(self):
        self.assertEqual(classify_error("inspect rejected: attachments total 28000000 bytes, above 18000000"), "evidence_size_limit")
        self.assertEqual(classify_error("inspect rejected: combined JSON evidence is 700000 characters, above 400000"), "evidence_size_limit")

    def test_unknown_does_not_become_zero(self):
        self.assertIsNone(fraction(0, 0))
        self.assertIsNone(fraction(None, 20))
        with self.assertRaises(ValueError):
            fraction(5, 4)

    def test_uninstrumented_control_remains_unknown(self):
        r = evaluate([], [], expected=[("control", "segment")])["rows"][0]
        self.assertIsNone(r["native_errors"])
        self.assertIsNone(r["logical_attempts"])
        self.assertIsNone(r["request_rejection_rate"])

    def test_donors_have_equal_weight(self):
        values = {"D01": dict(cells=10000, balanced_accuracy=.9),
                  "D02": dict(cells=10, balanced_accuracy=.3),
                  "D03": dict(cells=20, balanced_accuracy=None)}
        self.assertAlmostEqual(donor_macro(values, "balanced_accuracy"), .6)

    def test_manifest_detects_input_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "sample.json", {"value": 1})
            manifest(root)
            manifest(root, verify=True)
            write(root / "sample.json", {"value": 2})
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                manifest(root, verify=True)


if __name__ == "__main__":
    unittest.main()

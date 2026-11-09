"""Scientific denominator, pairing, provenance and native-checkout guards."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import evaluation as ev
import native


def row(arm, image, ap=0.5, status="success"):
    result = {key: "" for key in (*ev.SCORES, *ev.COSTS, *ev.IDENTITY,
              "ground_truth_cell_count", "predicted_cell_count", "mean_matched_iou_at_50",
              "cell_count_error_fraction", "tool_calls", "started_unix", "mask_sha256", "error", "delivered")}
    result.update(image_id=image, arm=arm, evaluation_status=status,
                  source_run="test", source_commit="test", manifest_fingerprint="test",
                  run_kind="sample", run_status="PARTIAL", checkpoint_sha256="weights",
                  image_sha256=image, ground_truth_sha256=image+"gt",
                  image_height_px="32", image_width_px="48")
    if status != "not_completed":
        result.update({key: str(ap) for key in ev.SCORES})
        result.update({key: "2" for key in ev.COSTS})
        result.update(delivered="true" if status == "success" else "false",
                      error="" if status == "success" else "native abort")
    return result


def synthetic():
    data = {a: {i: row(a, i) for i in ("a", "b", "c")} for a in ev.BASELINES}
    data["shape"] = {"a": row("shape", "a", 1.0),
                     "b": row("shape", "b", 0.0, "failed"),
                     "c": row("shape", "c", status="not_completed")}
    data["llm"] = {"c": row("llm", "c")}
    data["react"] = {"c": row("react", "c")}
    return data


def metadata(rows):
    return {"rows": len(rows), "delivered_masks": sum(r["evaluation_status"] == "success" for r in rows),
            "failed_predictions": sum(r["evaluation_status"] == "failed" for r in rows),
            "not_completed": sum(r["evaluation_status"] == "not_completed" for r in rows),
            "recorded_predictions": sum(ev.observed(r) for r in rows),
            "source_run": "test", "source_commit": "test", "manifest_fingerprint": "test",
            "run_kind": "sample", "run_status": "PARTIAL"}


class ComparisonTests(unittest.TestCase):
    def test_failure_in_denominator_missing_excluded_and_exact_pairing(self):
        data = synthetic()
        ids, arms = ev.cohorts(data)["shape_observed"]
        self.assertEqual(ids, ("a", "b"))
        data["shape_seg_co"]["c"]["ap50"] = "1"  # Unobserved SHAPE image must not affect its reference.
        rows, pairs = ev.summarize(data, ids, arms)
        shape = next(r for r in rows if r["arm"] == "shape")
        delta = next(r for r in pairs if r["arm"] == "shape")
        self.assertEqual((shape["n_images"], shape["n_failed"], shape["ap50"]), (2, 1, .5))
        self.assertEqual((delta["ap50_mean_delta"], delta["ap50_wins"], delta["ap50_losses"]), (0, 1, 1))

    def test_missing_never_becomes_zero(self):
        rows = list(synthetic()["shape"].values())
        meta = metadata(rows)
        ev.validate_rows("shape", rows, meta)
        rows[-1]["ap50"] = "0"
        with self.assertRaisesRegex(ValueError, "not_completed"):
            ev.validate_rows("shape", rows, meta)

    def test_duplicate_and_nonfinite_rejected(self):
        rows = list(synthetic()["shape"].values())
        rows[1]["image_id"] = rows[0]["image_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ev.validate_rows("shape", rows, metadata(rows))
        rows = list(synthetic()["shape"].values())
        rows[0]["ap50"] = "nan"
        with self.assertRaisesRegex(ValueError, "invalid"):
            ev.validate_rows("shape", rows, metadata(rows))

    def test_input_gt_and_checkpoint_must_match(self):
        for field in ("image_sha256", "ground_truth_sha256", "checkpoint_sha256"):
            data = synthetic()
            data["shape"]["a"][field] = "changed"
            with self.assertRaises(ValueError):
                ev.cohorts(data)

    def test_full_and_canary_mismatch_rejected(self):
        data = synthetic()
        del data["stardist"]["a"]
        with self.assertRaisesRegex(ValueError, "Full"):
            ev.cohorts(data)
        data = synthetic()
        data["react"] = {"a": row("react", "a")}
        with self.assertRaisesRegex(ValueError, "canary"):
            ev.cohorts(data)

    def test_missing_native_time_remains_unavailable(self):
        rows = [row("stardist", "a"), row("stardist", "b")]
        rows[0]["native_seconds"] = ""
        self.assertIsNone(ev.mean(rows, "native_seconds"))

    def test_native_wrong_or_dirty_checkout_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / native.ENTRY
            script.parent.mkdir(parents=True)
            script.write_text("# native", encoding="utf-8")
            with patch.object(native.subprocess, "check_output", side_effect=["wrong"]):
                with self.assertRaisesRegex(ValueError, "wrong commit"):
                    native.command(folder, native.SOURCE_COMMITS[0], "python", ["--help"])
            with patch.object(native.subprocess, "check_output",
                              side_effect=[native.SOURCE_COMMITS[0], " M shape/solver.py"]):
                with self.assertRaisesRegex(ValueError, "dirty"):
                    native.command(folder, native.SOURCE_COMMITS[0], "python", ["--help"])
            with patch.object(native.subprocess, "check_output",
                              side_effect=[native.SOURCE_COMMITS[0], ""]):
                cmd = native.command(folder, native.SOURCE_COMMITS[0], "python", ["--help"])
                self.assertEqual(cmd, ["python", str(script), "--help"])


@unittest.skipUnless(os.environ.get("CARDIAC_RESULTS_INPUT"), "Set CARDIAC_RESULTS_INPUT to test pinned real data")
class FrozenDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(os.environ["CARDIAC_RESULTS_INPUT"])
        cls.files = ev.build(cls.folder)

    def test_real_counts_and_disjoint_canary(self):
        report = json.loads(self.files["comparison.json"])
        self.assertEqual({k: v["n_images"] for k,v in report["cohorts"].items()},
                         {"full_baselines": 7582, "shape_observed": 151, "llm_react_canary": 20})
        self.assertEqual(report["all_eight_arms_common_images"], 0)
        self.assertEqual(report["shape_canary_common_images"], 0)
        full_star = next(r for r in report["summaries"] if r["arm"] == "stardist" and r["cohort"] == "full_baselines")
        self.assertEqual((full_star["n_failed"], full_star["n_delivered"]), (4, 7578))
        self.assertAlmostEqual(full_star["ap50"], .00027467023334920484, places=16)

    def test_export_hashes_and_published_replay(self):
        export = json.loads(self.files["EXPORT.json"])
        for name, meta in export["files"].items():
            self.assertEqual(ev.sha256(self.files[name]), meta["sha256"])
        self.assertEqual(ev.verify(self.folder, HERE.parent / "results/segmentation_20261020"), 8)

    def test_tampered_source_rejected(self):
        # Manifest is checked before accessing any other file.
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "manifest.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "manifest"):
                ev.load_inputs(folder)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            manifest = (self.folder / "manifest.json").read_bytes()
            (target / "manifest.json").write_bytes(manifest)
            first = next(iter(json.loads(manifest)["files"]))
            (target / first).write_bytes(b"corrupt CSV bytes")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ev.load_inputs(target)


if __name__ == "__main__":
    unittest.main()

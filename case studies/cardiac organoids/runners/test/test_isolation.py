"""Real spawned worker checks; no TensorFlow, images or models."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from stardist_isolation import IsolatedStarDist


def fake_worker(connection, source_root, log_path):
    connection.send(("ready", None))
    while True:
        request = connection.recv()
        if request is None:
            break
        image, _ = request
        if image == "crash":
            os._exit(7)
        if image == "python_error":
            connection.send(("error", "synthetic error"))
        else:
            connection.send(("result", {"mask": [0, 1, 2], "trace": {}}))
    connection.close()


class IsolationTests(unittest.TestCase):
    def test_success_native_exit_and_continuation_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            isolated = IsolatedStarDist(folder, folder, worker_target=fake_worker,
                                        startup_timeout=15, prediction_timeout=15)
            try:
                self.assertEqual(isolated("control", folder)["mask"], [0, 1, 2])
                self.assertEqual(isolated.generation, 1)
                with self.assertRaisesRegex(RuntimeError, "native worker failed"):
                    isolated("crash", folder)
                self.assertEqual(isolated.last["worker_exitcode"], 7)
                self.assertTrue((Path(folder)/"native_failure.json").is_file())
                self.assertEqual(isolated("next", folder)["mask"], [0, 1, 2])
                self.assertEqual(isolated.generation, 2)
            finally:
                isolated.close()

    def test_python_error_keeps_same_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            isolated = IsolatedStarDist(folder, folder, worker_target=fake_worker,
                                        startup_timeout=15, prediction_timeout=15)
            try:
                with self.assertRaisesRegex(RuntimeError, "synthetic error"):
                    isolated("python_error", folder)
                self.assertEqual(isolated("next", folder)["mask"], [0, 1, 2])
                self.assertEqual(isolated.generation, 1)
            finally:
                isolated.close()

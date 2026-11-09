"""Portable crash isolation for the frozen StarDist adapter; no algorithm changes.

This is a maintained rerun adapter, not the byte-exact historical Slurm wrapper.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time


def worker(connection, source_root, log_path):
    with open(log_path, "a", buffering=1, encoding="utf-8") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        root = Path(source_root)
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "case studies/cardiac organoids/runners/segmentation"))
        from common import cellseg
        model = cellseg.StarDist()
        connection.send(("ready", None))
        while True:
            request = connection.recv()
            if request is None:
                break
            image, output = request
            try:
                result = model(Path(image), Path(output))
                connection.send(("result", result))
            except Exception as exc:
                connection.send(("error", f"{type(exc).__name__}: {exc}"))
    connection.close()


class IsolatedStarDist:
    def __init__(self, source_root, recovery_dir, *, startup_timeout=180,
                 prediction_timeout=300, worker_target=worker):
        self.source_root = str(source_root)
        self.recovery = Path(recovery_dir)
        self.recovery.mkdir(parents=True, exist_ok=True)
        self.startup_timeout = startup_timeout
        self.prediction_timeout = prediction_timeout
        self.worker_target = worker_target
        self.process = self.connection = None
        self.generation = 0
        self.last = None

    def dispose(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(10)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def start(self):
        self.dispose()
        self.generation += 1
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.connection = parent
        self.process = ctx.Process(target=self.worker_target,
            args=(child, self.source_root, str(self.recovery / f"worker-{self.generation}.log")))
        self.process.start()
        child.close()
        try:
            if not parent.poll(self.startup_timeout):
                raise RuntimeError("StarDist worker startup timed out")
            kind, _ = parent.recv()
            if kind != "ready":
                raise RuntimeError("StarDist worker initialization failed")
        except BaseException:
            self.dispose()
            raise

    def __call__(self, image, output):
        self.last = None
        if self.process is None or not self.process.is_alive():
            self.start()
        self.last = {"attempted": True, "worker_generation": self.generation,
                     "worker_pid": self.process.pid}
        started = time.monotonic()
        try:
            self.connection.send((str(image), str(output)))
            if not self.connection.poll(self.prediction_timeout):
                raise TimeoutError("StarDist prediction timed out")
            kind, payload = self.connection.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError, TimeoutError, OSError) as exc:
            self.process.join(.1)
            self.last.update(worker_exitcode=self.process.exitcode,
                             seconds=time.monotonic() - started, error=type(exc).__name__)
            self.dispose()
            (Path(output) / "native_failure.json").write_text(
                json.dumps(self.last, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"StarDist native worker failed: {self.last}") from exc
        self.last["seconds"] = time.monotonic() - started
        if kind == "error":
            raise RuntimeError(payload)
        if kind != "result":
            raise RuntimeError("Unknown StarDist worker response")
        payload.setdefault("trace", {})["process_isolation"] = dict(self.last)
        return payload

    def close(self):
        if self.process is not None and self.process.is_alive():
            try:
                self.connection.send(None)
                self.process.join(5)
            except (EOFError, OSError):
                pass
        self.dispose()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-root", required=True, type=Path)
    cli.add_argument("arguments", nargs=argparse.REMAINDER)
    args = cli.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "case studies/cardiac organoids/runners/segmentation"))
    import arms
    import run_organoid_segmentation as runner
    from common import dump, sha256, protect_output, read_manifest
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    native_args = runner.parser().parse_args(arguments)
    if native_args.command != "run" or native_args.arm != "stardist":
        raise ValueError("Isolation is restricted to run --arm stardist")
    protect_output(Path(native_args.run_dir), read_manifest(native_args.manifest, verify=False)["pairs"])
    recovery = Path(native_args.run_dir).resolve() / "stardist" / ("isolation_" + str(time.time_ns()))
    recovery.mkdir(parents=True, exist_ok=False)
    provenance = {"wrapper_sha256": sha256(__file__), "source_root": str(root),
                  "job_id": os.environ.get("SLURM_JOB_ID"), "started_unix": time.time(),
                  "policy": "Unchanged frozen adapter/scorer; failures recorded once, no alternative prediction or retry."}
    dump(recovery / "provenance.json", provenance)
    isolated = IsolatedStarDist(root, recovery)
    original_image, original_make = runner.run_image, arms.make_arm

    def make(name, *args, **kwargs):
        if name != "stardist":
            raise ValueError("Only StarDist is supported")
        return isolated

    def image(arm, pair, output, name):
        isolated.last = None
        record = original_image(arm, pair, output, name)
        record["isolation_wrapper_sha256"] = provenance["wrapper_sha256"]
        if isolated.last:
            record["process_isolation"] = dict(isolated.last)
            record["tool_calls"] = record["native_calls"] = 1
        dump(output / "record.json", record)
        return record

    arms.make_arm, runner.run_image = make, image
    try:
        result = runner.run(native_args)
    finally:
        isolated.close()
        arms.make_arm, runner.run_image = original_make, original_image
    dump(recovery / "completion.json", {"status": result["status"], "n_recorded": result["n_recorded"]})
    return 2 if result["status"] == "COMPLETED_WITH_FAILURES" else 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("workers,failed_smoke", [(2, False), ("all", False), ("all", True)])
def test_parallel_queue_respects_gpu_ownership_and_smoke_dependency(tmp_path, monkeypatch, workers, failed_smoke):
    script = Path(__file__).resolve().parents[1] / "scripts/run_queue.py"
    spec = importlib.util.spec_from_file_location("scheduler_under_test", script)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    for directory in ("configs", "runs", "logs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "configs/resource_policy.json").write_text(json.dumps({
        "max_concurrent_gpu_jobs": workers, "cpu_threads": 4,
        "preprocessing_workers": 2}))
    for number in range(6):
        (tmp_path / "configs" / f"lcf_M{number}_d128.json").write_text(json.dumps({
            "name": f"lcf_M{number}_d128", "method": "lcf", "minimum_free_gpu_mib": 100}))
    devices = [{"index": i, "uuid": f"TEST-{tmp_path.name}-{i}", "free_mib": 20000,
                "utilization_percent": 0} for i in range(4)]
    monkeypatch.setattr(queue, "gpu_snapshot", lambda: devices)
    monkeypatch.setattr(queue, "data_ready", lambda method: True)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: None)
    occupied, launches = set(), []
    peak = 0

    class FakeProcess:
        counter = 0

        def __init__(self, command, **kwargs):
            nonlocal peak
            self.name = Path(command[command.index("--config") + 1]).stem
            assert "--gpu-memory-fraction" not in command
            self.smoke = "--smoke" in command
            self.gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            assert self.gpu not in occupied, "Two experiments assigned to one GPU"
            if not self.smoke:
                assert queue.completed_phase({"name": self.name}, "smoke")
            occupied.add(self.gpu)
            peak = max(peak, len(occupied))
            launches.append((self.name, self.smoke))
            FakeProcess.counter += 1
            self.pid = 900000 + FakeProcess.counter
            self.returncode = None
            self.polls_left = 2

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            self.polls_left -= 1
            if self.polls_left:
                return None
            self.returncode = int(failed_smoke and self.smoke and self.name == "lcf_M0_d128")
            occupied.remove(self.gpu)
            directory = tmp_path / "runs" / ("smoke" if self.smoke else "formal") / self.name
            directory.mkdir(parents=True)
            if not self.returncode:
                (directory / "final").mkdir()
                (directory / "final/adapter.safetensors").touch()
                (directory / "status.json").write_text(json.dumps({
                    "state": "smoke_passed" if self.smoke else "completed"}))
            return self.returncode

    monkeypatch.setattr(queue.subprocess, "Popen", FakeProcess)
    try:
        result = queue.main(argparse.Namespace(gpu="auto", max_workers=0, poll_seconds=0))
    finally:
        for gpu in devices:
            lock = Path(f"/tmp/lcf-gpu-{gpu['uuid']}-{queue.os.getuid()}.lock")
            lock.unlink(missing_ok=True)
    assert result == int(failed_smoke)
    assert peak == (4 if workers == "all" else workers)
    assert not occupied
    assert sum(not smoke for _, smoke in launches) == (5 if failed_smoke else 6)
    if failed_smoke:
        assert ("lcf_M0_d128", False) not in launches

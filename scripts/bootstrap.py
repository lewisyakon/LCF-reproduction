#!/usr/bin/env python3
"""Run checks, then let the GPU scheduler prepare and dispatch each method."""
import fcntl
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def status(state, **details):
    content = {"state": state, "pid": os.getpid(), "time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **details}
    temp = ROOT / "runs/bootstrap_status.json.tmp"
    temp.write_text(json.dumps(content, indent=2))
    os.replace(temp, ROOT / "runs/bootstrap_status.json")
    print(json.dumps(content), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    lock = open(ROOT / "runs/bootstrap.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.wait_for_pid:
        while True:
            try:
                os.kill(args.wait_for_pid, 0)
            except ProcessLookupError:
                break
            status("waiting_for_process", awaited_pid=args.wait_for_pid)
            time.sleep(30)
    try:
        snapshot = subprocess.check_output(["/home/lewis/.local/bin/uv", "pip", "freeze", "--python", sys.executable], text=True)
        (ROOT / "manifests/environment.txt").write_text(snapshot)
    except subprocess.CalledProcessError:
        status("failed", phase="environment_snapshot")
        raise
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    steps = [
        ("check_environment", [sys.executable, "-c", "import torch,transformers,datasets; print(torch.__version__,transformers.__version__,datasets.__version__); assert torch.cuda.is_available()"]),
        ("verify_assets", [sys.executable, "scripts/verify_assets.py"]),
        ("test_implementation", [sys.executable, "-m", "pytest", "-q"]),
        ("parameter_report", [sys.executable, "scripts/parameter_report.py"]),
        ("gpu_queue", [sys.executable, "scripts/run_queue.py", "--gpu", "auto"]),
    ]
    for phase, command in steps:
        if phase == "verify_assets" and (ROOT / "manifests/verified_models.json").exists():
            continue
        status(phase)
        with open(ROOT / "logs" / f"{phase}.log", "a", buffering=1) as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status(phase, child_pid=child.pid)
            code = child.wait()
        if code:
            status("failed", phase=phase, exit_code=code)
            sys.exit(code)
    status("completed")

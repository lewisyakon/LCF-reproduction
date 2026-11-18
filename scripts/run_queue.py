#!/usr/bin/env python3
"""Parallel independent experiments, one process per available physical GPU."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def status(**kwargs):
    data = {"time": now(), "pid": os.getpid(), **kwargs}
    tmp = ROOT / "runs/queue_status.json.tmp"
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, ROOT / "runs/queue_status.json")
    print(json.dumps(data), flush=True)


def gpu_snapshot():
    result = subprocess.check_output(["nvidia-smi",
        "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    devices = []
    for line in result.splitlines():
        index, uuid, free, utilization = [part.strip() for part in line.split(",")]
        devices.append({"index": int(index), "uuid": uuid, "free_mib": int(free),
                        "utilization_percent": int(utilization)})
    return devices


def eligible_gpus(devices, minimum, max_utilization, allowed_gpu="auto"):
    candidates = [gpu for gpu in devices if gpu["free_mib"] >= minimum
                  and gpu["utilization_percent"] <= max_utilization
                  and (allowed_gpu == "auto" or gpu["index"] == int(allowed_gpu))]
    return sorted(candidates, key=lambda gpu: (gpu["utilization_percent"], -gpu["free_mib"], gpu["index"]))


def reserve_gpu(args, minimum, max_utilization, occupied):
    devices = [gpu for gpu in gpu_snapshot() if gpu["uuid"] not in occupied]
    for gpu in eligible_gpus(devices, minimum, max_utilization, args.gpu):
        gpu_lock = open(f"/tmp/lcf-gpu-{gpu['uuid']}-{os.getuid()}.lock", "w")
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            gpu_lock.close()
            continue
        refreshed = next((item for item in gpu_snapshot() if item["uuid"] == gpu["uuid"]), None)
        if refreshed and eligible_gpus([refreshed], minimum, max_utilization, args.gpu):
            return refreshed, gpu_lock
        gpu_lock.close()
    return None


def data_ready(method):
    return ((ROOT / "data" / method / "dataset_dict.json").is_file()
            and (ROOT / "manifests" / f"{method}_data.json").is_file())


def completed_phase(job, phase):
    directory = ROOT / "runs" / ("smoke" if phase == "smoke" else "formal") / job["name"]
    state_file = directory / "status.json"
    expected = "smoke_passed" if phase == "smoke" else "completed"
    return ((directory / "final/adapter.safetensors").is_file() and state_file.is_file()
            and json.loads(state_file.read_text()).get("state") == expected)


def main(args):
    lock = open(ROOT / "runs/queue.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy = json.loads((ROOT / "configs/resource_policy.json").read_text())
    cpu_threads = str(policy["cpu_threads"])
    max_utilization = policy.get("max_idle_utilization_percent", 10)
    configured_limit = policy.get("max_concurrent_gpu_jobs", "all")
    limit = args.max_workers or (len(gpu_snapshot()) if configured_limit == "all" else int(configured_limit))
    if limit < 1:
        raise ValueError("At least one GPU worker is required")
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1")
    env.update(OMP_NUM_THREADS=cpu_threads, MKL_NUM_THREADS=cpu_threads)
    jobs = []
    for path in sorted((ROOT / "configs").glob("*_d128.json")):
        config = json.loads(path.read_text())
        job = {"name": config["name"], "method": config["method"], "config": str(path),
               "minimum_free_gpu_mib": config["minimum_free_gpu_mib"],
               "gpu_memory_fraction": None,
               "state": "pending", "phase": "smoke"}
        if completed_phase(job, "training"):
            job.update(state="completed", phase="training")
        elif completed_phase(job, "smoke"):
            job["phase"] = "training"
        jobs.append(job)
    active = {}
    preparer = None
    failed_data = set()
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    original_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not stopping:
            for name, run in list(active.items()):
                code = run["child"].poll()
                if code is None:
                    continue
                job = run["job"]
                run["log"].close()
                run["lock"].close()
                del active[name]
                if code != 0 or not completed_phase(job, job["phase"]):
                    job.update(state="failed", exit_code=code, error="Inspect the per-run log for details")
                elif job["phase"] == "smoke":
                    job.update(state="pending", phase="training")
                else:
                    job["state"] = "completed"
            if preparer and preparer["child"].poll() is not None:
                method = preparer["method"]
                if preparer["child"].returncode != 0 or not data_ready(method):
                    failed_data.add(method)
                preparer["log"].close()
                preparer = None
            for job in jobs:
                if job["method"] in failed_data and job["state"] == "pending":
                    job.update(state="failed", error=f"Data preparation failed; see logs/prepare_data_{job['method']}.log")
            if not preparer:
                missing = sorted({job["method"] for job in jobs if job["state"] == "pending"
                                  and not data_ready(job["method"]) and job["method"] not in failed_data})
                if missing:
                    method = missing[0]
                    log = open(ROOT / "logs" / f"prepare_data_{method}.log", "a", buffering=1)
                    command = [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--method", method,
                               "--workers", str(policy["preprocessing_workers"])]
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    preparer = {"method": method, "child": child, "log": log}
            # Finish passed smoke tests before taking another new experiment.
            for job in sorted(jobs, key=lambda item: item["phase"] != "training"):
                if len(active) >= limit:
                    break
                if job["state"] != "pending" or not data_ready(job["method"]):
                    continue
                reservation = reserve_gpu(args, job["minimum_free_gpu_mib"], max_utilization,
                                          {run["gpu"]["uuid"] for run in active.values()})
                if reservation is None:
                    continue
                gpu, gpu_lock = reservation
                command = [sys.executable, str(ROOT / "scripts/train.py"), "--config", job["config"],
                           "--cpu-threads", cpu_threads]
                if job["phase"] == "smoke":
                    command.append("--smoke")
                log = open(ROOT / "logs" / f"{job['name']}.{'smoke' if job['phase'] == 'smoke' else 'train'}.log", "a", buffering=1)
                try:
                    child = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=gpu["uuid"]),
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    gpu_lock.close()
                    raise
                job.update(state="running", gpu=gpu["index"], gpu_uuid=gpu["uuid"], child_pid=child.pid)
                active[job["name"]] = {"job": job, "child": child, "gpu": gpu, "lock": gpu_lock, "log": log}
            terminal = all(job["state"] in ("completed", "failed") for job in jobs)
            state = ("finished_with_errors" if any(job["state"] == "failed" for job in jobs) else "completed") if terminal else (
                "running" if active else "preparing_data" if preparer else "waiting_for_gpu")
            status(state=state, mode="parallel", gpu_selection=args.gpu, max_workers=limit,
                   active_gpu_jobs=len(active), jobs=jobs, devices=gpu_snapshot(),
                   data_preparation={"method": preparer["method"], "pid": preparer["child"].pid} if preparer else None)
            if terminal:
                return int(state == "finished_with_errors")
            time.sleep(args.poll_seconds)
        status(state="stopping", mode="parallel", jobs=jobs)
        return 130
    finally:
        for run in list(active.values()) + ([preparer] if preparer else []):
            if run["child"].poll() is None:
                os.killpg(run["child"].pid, signal.SIGTERM)
                run["child"].wait()
            run["log"].close()
            if "lock" in run:
                run["lock"].close()
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="auto", help="auto selects any idle GPU; a physical index optionally restricts selection")
    parser.add_argument("--max-workers", type=int, default=0, help="0 uses the resource policy; all means one worker per physical GPU")
    parser.add_argument("--poll-seconds", type=float, default=10)
    sys.exit(main(parser.parse_args()))

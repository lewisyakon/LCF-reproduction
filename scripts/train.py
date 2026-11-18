#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from datasets import load_from_disk
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lcf.adapter import CommunicationAdapter
from lcf.runtime import load_frozen, sample_loss, answer


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, adapter, optimizer, scheduler, step, cursor, config, metrics):
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in adapter.state_dict().items()}, str(path / "adapter.safetensors.tmp"))
    os.replace(path / "adapter.safetensors.tmp", path / "adapter.safetensors")
    torch.save({"adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, "cursor": cursor, "rng": rng_state(), "config": config}, path / "training_state.pt.tmp")
    os.replace(path / "training_state.pt.tmp", path / "training_state.pt")
    write_json(path / "config.json", config)
    write_json(path / "metrics.json", metrics)


def qa_scores(prediction, reference):
    import re, string
    from collections import Counter
    def norm(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())
    p, r = norm(prediction), norm(reference)
    if p in ("yes", "no", "noanswer") and p != r:
        return float(p == r), 0.0
    if r in ("yes", "no", "noanswer") and p != r:
        return float(p == r), 0.0
    pt, rt = p.split(), r.split()
    common = sum((Counter(pt) & Counter(rt)).values())
    return float(p == r), 2 * common / (len(pt) + len(rt)) if pt and rt else float(pt == rt)


def main(args):
    config = json.loads(Path(args.config).read_text())
    method, seed = config["method"], config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = ROOT / ("runs/smoke" if args.smoke else "runs/formal") / config["name"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "final/adapter.safetensors").exists() and not args.smoke:
        print("Final checkpoint already exists; leaving completed run intact", flush=True)
        return
    write_json(output / "status.json", {"state": "loading", "name": config["name"], "pid": os.getpid(),
               "started_at": now(), "smoke": args.smoke,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    torch_dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    sender_path, receiver_path = [ROOT / "models" / config[x] for x in ("sender", "receiver")]
    sender = load_frozen(sender_path, args.sender_device or args.device, torch_dtype)
    receiver = load_frozen(receiver_path, args.device, torch_dtype)
    adapter = CommunicationAdapter(sender.config, receiver.config, method, config["d"], config["dropout"]).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(receiver_path, local_files_only=True)
    ds = load_from_disk(ROOT / "data" / method)
    train, validation = ds["train"], ds["validation"]
    batch = 1 if args.smoke else config["effective_batch"]
    total_steps = args.smoke_steps if args.smoke else (300 if method == "lcf" else math.ceil(len(train) / batch))
    warmup = min(total_steps, 200 if method == "lcfx" else math.ceil(total_steps * 0.1))
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    config.update(total_steps=total_steps, actual_microbatch=1, actual_grad_accumulation=batch,
                  logical_per_device_batch=10 if method == "lcf" else 16,
                  loss_weighting="Token mean within paper microbatch, mean across accumulation groups",
                  warmup_steps=warmup, device=args.device, sender_device=args.sender_device or args.device,
                  base_dtype=str(torch_dtype), adapter_parameter_dtype="float32", compute_dtype=str(torch_dtype),
                  cpu_threads=args.cpu_threads, gpu_memory_fraction=None,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  checkpoint_selection="final", training_samples=len(train), validation_samples=len(validation),
                  validation_limit=args.validation_limit or "full",
                  smoke=args.smoke, software={"python": platform.python_version(), "torch": torch.__version__})
    verified = ROOT / "manifests/verified_models.json"
    if verified.exists():
        assets = json.loads(verified.read_text())
        config["model_revisions"] = {name: assets[name]["revision"] for name in (config["sender"], config["receiver"])}
    data_manifest = ROOT / "manifests" / f"{method}_data.json"
    config["data_manifest_sha256"] = hashlib.sha256(data_manifest.read_bytes()).hexdigest()
    write_json(output / "config.json", config)
    status = {"state": "loading", "name": config["name"], "started_at": now(), "pid": os.getpid(),
              "step": 0, "total_steps": total_steps, "smoke": args.smoke,
              "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
              "frozen_parameters": sum(p.numel() for m in (sender, receiver) for p in m.parameters())}
    step, cursor = 0, 0
    resume = output / "latest/training_state.pt"
    if resume.exists() and not args.smoke:
        state = torch.load(resume, map_location=args.device, weights_only=False)
        if any(state["config"][key] != config[key] for key in ("name", "method", "sender", "receiver", "d", "total_steps")):
            raise ValueError("Resume checkpoint/config mismatch")
        adapter.load_state_dict(state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, cursor = state["step"], state["cursor"]
        # torch.load(map_location=cuda) also moves CPU RNG state; restore to CPU.
        state["rng"]["torch"] = state["rng"]["torch"].cpu()
        state["rng"]["cuda"] = [s.cpu() for s in state["rng"]["cuda"]]
        restore_rng(state["rng"])
        status["resumed_from_step"] = step
        status["resumed_from_cursor"] = cursor
        print(json.dumps({"event": "resumed", "step": step, "cursor": cursor,
                          "checkpoint": str(resume), "gpu_memory_fraction": None}), flush=True)
        del state
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed)).tolist()
    def precision():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else contextlib.nullcontext()
    metric_log = open(output / "metrics.jsonl", "a", buffering=1)
    status.update(state="training", step=step)
    write_json(output / "status.json", status)
    print(json.dumps(status), flush=True)
    initial_smoke_parameters = [p.detach().clone() for p in adapter.parameters()] if args.smoke else None
    try:
        while step < total_steps:
            started = time.monotonic()
            adapter.train()
            adapter.set_temperature(step, total_steps)
            optimizer.zero_grad(set_to_none=True)
            count = min(batch, len(train) - cursor) if method == "lcfx" else batch
            loss_sum = 0.0
            logical_batch = 1 if args.smoke else (10 if method == "lcf" else 16)
            groups = math.ceil(count / logical_batch)
            examples = [train[order[(cursor + offset) % len(train)]] for offset in range(count)]
            for micro in range(count):
                example = examples[micro]
                group_start = (micro // logical_batch) * logical_batch
                group_tokens = sum(len(e["response_ids"]) for e in examples[group_start:group_start + logical_batch])
                weight = len(example["response_ids"]) / group_tokens / groups
                with precision():
                    loss = sample_loss(sender, receiver, adapter, example)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss on {example['id']}")
                (loss * weight).backward()
                loss_sum += loss.item() * weight
                cursor += 1
                if micro % 16 == 0:
                    status.update(step=step, microbatch=micro + 1, accumulation=count, cursor=cursor,
                                  loss=loss.item(), updated_at=now())
                    write_json(output / "status.json", status)
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
            if args.smoke and float(norm) <= 0:
                raise AssertionError("Smoke produced zero communication-module gradients")
            if any(p.grad is not None for m in (sender, receiver) for p in m.parameters()):
                raise AssertionError("Frozen LLM received gradients")
            optimizer.step()
            scheduler.step()
            step += 1
            metrics = {"step": step, "train_loss": loss_sum, "grad_norm": float(norm),
                       "lr": scheduler.get_last_lr()[0], "seconds": time.monotonic() - started,
                       "examples": cursor, "time": now()}
            if args.device.startswith("cuda"):
                metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            if not args.smoke and (step % config["eval_steps"] == 0 or step == total_steps):
                adapter.eval()
                losses, ems, f1s = [], [], []
                limit = args.validation_limit or len(validation)
                with torch.no_grad(), precision():
                    for index, example in enumerate(validation):
                        if index >= limit:
                            break
                        losses.append(sample_loss(sender, receiver, adapter, example).item())
                        if method == "lcfx":
                            pred = answer(sender, receiver, adapter, example, tokenizer)
                            ref = tokenizer.decode(example["response_ids"], skip_special_tokens=True)
                            em, f1 = qa_scores(pred, ref)
                            ems.append(em)
                            f1s.append(f1)
                metrics.update(validation_loss=sum(losses) / len(losses), validation_items=len(losses))
                if ems:
                    metrics.update(validation_em=sum(ems) / len(ems), validation_f1=sum(f1s) / len(f1s))
            metric_log.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            # Save every optimizer step for safe continuation of long accumulations.
            save_checkpoint(output / "latest", adapter, optimizer, scheduler, step, cursor, config, metrics)
            if step % config["save_steps"] == 0:
                save_checkpoint(output / f"checkpoint-{step:06d}", adapter, optimizer, scheduler, step, cursor, config, metrics)
            status.update(state="training", step=step, metrics=metrics, updated_at=now())
            write_json(output / "status.json", status)
        if args.smoke:
            changed = any(not torch.equal(before, after) for before, after in zip(initial_smoke_parameters, adapter.parameters()))
            if not changed:
                raise AssertionError("Smoke optimizer did not change adapter weights")
            metrics["smoke_checks"] = {"finite_loss": True, "nonzero_adapter_gradients": True,
                                       "frozen_model_gradients_absent": True, "adapter_weights_changed": changed}
        save_checkpoint(output / "final", adapter, optimizer, scheduler, step, cursor, config, metrics)
        status.update(state="smoke_passed" if args.smoke else "completed", finished_at=now(), step=step)
        write_json(output / "status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}", updated_at=now())
        write_json(output / "status.json", status)
        raise
    finally:
        metric_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sender-device")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--validation-limit", type=int, default=0, help="0 evaluates complete independent validation split")
    main(parser.parse_args())

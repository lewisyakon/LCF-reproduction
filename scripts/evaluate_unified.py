#!/usr/bin/env python3
"""Unified stage-1 evaluation for the reproduced LCF and LCF-X adapters.

The task inventory, protocol views, scorers, and frozen prompt archive are the
same ones used by the T2T evaluation.  Results are append-only and resumable.
LCF is evaluated only on native P1 because its adapter requires token-aligned
source and receiver caches.  LCF-X is evaluated on every applicable protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache


ROOT = Path(__file__).resolve().parents[1]
COMMON_ROOT = Path(
    os.environ.get("LCF_COMMON_SCRIPTS", ROOT / "reproduction" / "current" / "scripts")
)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from lcf.adapter import CommunicationAdapter  # noqa: E402
from lcf.runtime import load_frozen  # noqa: E402
import evaluate_t2t_frozen_prompts as common  # noqa: E402


PAIR_CONFIGS = {
    "M1": ("Qwen3-0.6B", "Qwen3-1.7B"),
    "M2": ("Qwen3-0.6B", "Qwen3-4B"),
    "M3": ("Qwen3-4B", "Qwen3-0.6B"),
}
ALL_CONDITIONS = (
    ("gsm8k", "P1"),
    ("gsm8k", "P4"),
    ("hotpotqa", "P1"),
    ("hotpotqa", "P2"),
    ("hotpotqa", "P3"),
    ("hotpotqa", "P4"),
    ("humaneval", "P1"),
    ("humaneval", "P4"),
)
RESULT_SCHEMA_VERSION = 2
DEFAULT_PROMPT_ZIP = ROOT / "reproduction" / "prompts" / "提示词模板.zip"
DEFAULT_OUTPUT_ROOT = ROOT / "evaluations" / "unified_stage1_d128"
RECEIVER_MAX_NEW_TOKENS = {"gsm8k": 1024, "hotpotqa": 64, "humaneval": 1024}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_conditions(method: str):
    if method == "lcf":
        return tuple(item for item in ALL_CONDITIONS if item[1] == "P1")
    return ALL_CONDITIONS


def unsupported_conditions(method: str):
    if method != "lcf":
        return []
    return [
        {
            "dataset": dataset,
            "protocol": protocol,
            "status": "unsupported_native_adapter_path",
            "reason": (
                "The reproduced base LCF projector requires source sequence length to equal "
                "the receiver-cache sequence length. This protocol presents different views; "
                "padding or fabricated token alignment is not a valid implementation."
            ),
            "text_fallback": "separate_baseline_not_part_of_native_LCF_adapter_evaluation",
        }
        for dataset, protocol in ALL_CONDITIONS
        if protocol != "P1"
    ]


def cache_layers(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(key, value) for key, value in cache]


def tensor_bytes(layers: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
    return sum(t.numel() * t.element_size() for pair in layers for t in pair)


def payload_shape(layers: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
    keys = [tuple(pair[0].shape) for pair in layers]
    values = [tuple(pair[1].shape) for pair in layers]
    return {
        "layer_count": len(layers),
        "key_shape": list(keys[0]) if keys else None,
        "value_shape": list(values[0]) if values else None,
        "uniform_key_shapes": len(set(keys)) <= 1,
        "uniform_value_shapes": len(set(values)) <= 1,
        "dtype": str(layers[0][0].dtype) if layers else None,
    }


def timed_cuda(function):
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    value = function()
    finished.record()
    finished.synchronize()
    return value, float(started.elapsed_time(finished))


def prompt_ids_and_offsets(tokenizer, visible_view: str, task_prompt: dict[str, str]):
    # Cache methods do not emit a sender message.  Both caches therefore use the
    # frozen downstream task instruction; this also preserves LCF's required P1
    # token alignment and matches its same-input training protocol.
    messages = [
        {"role": "system", "content": task_prompt["receiver_system"]},
        {"role": "user", "content": visible_view + "\n\n" + task_prompt["receiver_task"]},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    template_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if isinstance(template_ids, torch.Tensor):
        template_ids = template_ids.tolist()
    if template_ids and isinstance(template_ids[0], list):
        template_ids = template_ids[0]
    if ids != list(template_ids):
        raise AssertionError("Rendered prompt and tokenized chat template differ")
    return ids, list(encoded["offset_mapping"]), rendered, messages


def source_evidence(record: dict[str, Any], protocol: str):
    return record["source_evidence"] if protocol == "P2" else record["all_evidence"]


def locate_spans(
    dataset: str,
    record: dict[str, Any],
    protocol: str,
    rendered: str,
    offsets: list[tuple[int, int]],
) -> list[list[int]]:
    if dataset == "hotpotqa":
        targets = [f"[{item['title']}]\n{item['text']}" for item in source_evidence(record, protocol)]
    else:
        targets = [f"<TASK>\n{record['q']}\n</TASK>"]
    spans: list[list[int]] = []
    cursor = 0
    for target in targets:
        start = rendered.find(target, cursor)
        if start < 0:
            raise ValueError(f"Could not locate LCF-X source span in rendered prompt: {target[:80]!r}")
        end = start + len(target)
        indices = [index for index, (a, b) in enumerate(offsets) if b > start and a < end]
        if not indices:
            raise ValueError("A source span mapped to no tokens")
        spans.append([indices[0], indices[-1] + 1])
        cursor = end
    if not spans:
        raise ValueError("LCF-X requires at least one nonempty source span")
    return spans


def pool_source(adapter: CommunicationAdapter, source_cache, spans):
    summaries = [pool(key, value, spans) for pool, (key, value) in zip(adapter.pools, source_cache)]
    return summaries


def map_source_layers(adapter: CommunicationAdapter, summaries):
    if adapter.layer_map is None:
        return summaries
    keys, values = [torch.stack(items) for items in zip(*summaries)]
    keys = torch.einsum("rs,sbhnd->rbhnd", adapter.layer_map.weight, keys)
    values = torch.einsum("rs,sbhnd->rbhnd", adapter.layer_map.weight, values)
    return list(zip(keys.unbind(), values.unbind()))


def fuse_with_breakdown(adapter: CommunicationAdapter, source, receiver_cache):
    pairs = list(zip(adapter.projectors, source, receiver_cache))

    def down_projection():
        result = []
        for projector, (sk, sv), (rk, rv) in pairs:
            n = rk.shape[2]
            if sk.shape[2] not in (1, n):
                raise ValueError("LCF requires aligned tokens; LCF-X requires singleton summaries")
            flatten = lambda x: x.transpose(1, 2).flatten(2)
            inputs = torch.cat(
                [
                    flatten(sk).expand(-1, n, -1),
                    flatten(sv).expand(-1, n, -1),
                    flatten(rk),
                    flatten(rv),
                ],
                dim=-1,
            )
            result.append(projector.down(inputs))
        return result

    reduced, down_ms = timed_cuda(down_projection)
    converted, mlp_ms = timed_cuda(lambda: [projector.mlp(z) for (projector, *_), z in zip(pairs, reduced)])

    def expand_projection():
        result = []
        for (projector, _, (rk, _)), z in zip(pairs, converted):
            zk, zv = z.chunk(2, dim=-1)
            batch, n = rk.shape[0], rk.shape[2]
            reshape = lambda x: x.view(batch, n, projector.heads, projector.head_dim).transpose(1, 2)
            alpha_k = projector.alpha_k(z).sigmoid().transpose(1, 2).unsqueeze(-1)
            alpha_v = projector.alpha_v(z).sigmoid().transpose(1, 2).unsqueeze(-1)
            result.append((reshape(projector.up_k(zk)), reshape(projector.up_v(zv)), alpha_k, alpha_v))
        return result

    expanded, up_ms = timed_cuda(expand_projection)

    def gated_fusion():
        result = DynamicCache()
        for index, ((projector, _, (rk, rv)), (uk, uv, ak, av)) in enumerate(zip(pairs, expanded)):
            shape = (rk.shape[0], projector.heads, rk.shape[2], 1)
            key = rk + projector._gate(projector.gate_k, shape).to(rk.dtype) * ak * uk
            value = rv + projector._gate(projector.gate_v, shape).to(rv.dtype) * av * uv
            result.update(key, value, index)
        return result

    fused, fusion_ms = timed_cuda(gated_fusion)
    return fused, {
        "down_projection_cuda_ms": down_ms,
        "nonlinear_conversion_cuda_ms": mlp_ms,
        "up_projection_and_head_weight_cuda_ms": up_ms,
        "gate_and_cache_fusion_cuda_ms": fusion_ms,
        "joint_transform_and_fusion_cuda_ms": down_ms + mlp_ms + up_ms + fusion_ms,
    }


def max_difference(left, right) -> float:
    differences = []
    for (lk, lv), (rk, rv) in zip(cache_layers(left), cache_layers(right)):
        differences.extend([(lk.float() - rk.float()).abs().max().item(), (lv.float() - rv.float()).abs().max().item()])
    return max(differences, default=0.0)


@torch.inference_mode()
def decode(
    receiver,
    cache,
    current_token: int,
    tokenizer,
    context_limit: int,
    prefix_tokens: int,
    max_new_tokens: int,
):
    current = torch.tensor([[current_token]], dtype=torch.long, device=receiver.device)
    generated: list[int] = []
    forward_ms: list[float] = []
    selection_ms = 0.0
    finish_reason = "context_limit"
    eos_ids = set()
    for value in (tokenizer.eos_token_id, getattr(receiver.generation_config, "eos_token_id", None)):
        if value is None:
            continue
        eos_ids.update(value if isinstance(value, (list, tuple, set)) else [value])
    context_available = max(0, context_limit - prefix_tokens)
    available = min(context_available, max_new_tokens)
    finish_reason = "max_new_tokens" if max_new_tokens < context_available else "context_limit"
    for _ in range(available):
        try:
            def forward():
                output = receiver.model(input_ids=current, past_key_values=cache, use_cache=True, return_dict=True)
                logits = receiver.lm_head(output.last_hidden_state[:, -1:])
                return output.past_key_values, logits

            (cache, logits), elapsed = timed_cuda(forward)
            forward_ms.append(elapsed)
            started = time.perf_counter()
            next_token = logits.float().argmax(dim=-1)
            token = int(next_token.item())
            selection_ms += (time.perf_counter() - started) * 1000
            generated.append(token)
            if token in eos_ids:
                finish_reason = "eos"
                break
            current = next_token
        except torch.OutOfMemoryError:
            finish_reason = "cuda_oom"
            torch.cuda.empty_cache()
            break
    return generated, {
        "finish_reason": finish_reason,
        "forward_cuda_ms": sum(forward_ms),
        "forward_calls": len(forward_ms),
        "token_selection_and_d2h_wall_ms": selection_ms,
    }


def distribution(values: list[float]) -> dict[str, Any]:
    return common.distribution(values)


def prior_state(path: Path):
    completed = set()
    metrics: dict[str, list[float]] = defaultdict(list)
    timings: dict[str, list[float]] = defaultdict(list)
    communications: dict[str, list[float]] = defaultdict(list)
    tokens: dict[str, list[float]] = defaultdict(list)
    finishes = Counter()
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("result_schema_version") != RESULT_SCHEMA_VERSION:
                    raise ValueError(f"Result schema mismatch in {path}")
                if row["record_id"] in completed:
                    raise ValueError(f"Duplicate result {row['record_id']} in {path}")
                completed.add(row["record_id"])
                for destination, source in (
                    (metrics, row["metrics"]),
                    (timings, row["timing_ms"]),
                    (communications, row["communication_bytes"]),
                    (tokens, row["token_counts"]),
                ):
                    common.add_values(destination, source)
                finishes[row["termination"]["receiver_finish_reason"]] += 1
    return completed, metrics, timings, communications, tokens, finishes


def normalized_uuid(value: str) -> str:
    return value.strip().lower().removeprefix("gpu-").replace("-", "")


def assert_expected_gpu() -> None:
    expected = os.environ.get("LCF_EXPECTED_GPU_UUID")
    if not expected:
        return
    actual = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    if normalized_uuid(actual) != normalized_uuid(expected):
        raise RuntimeError(
            f"CUDA enumeration selected GPU UUID {actual}, expected {expected}; refusing to run on the wrong card"
        )


def runtime_environment() -> dict[str, Any]:
    import pyarrow
    import safetensors
    import transformers

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    actual_uuid = str(properties.uuid)
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    matches = [row.split(", ") for row in rows if normalized_uuid(row.split(", ")[1]) == normalized_uuid(actual_uuid)]
    if len(matches) != 1:
        raise RuntimeError(f"Could not map CUDA UUID {actual_uuid} to one physical nvidia-smi device")
    gpu = matches[0]
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "pyarrow": pyarrow.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "attention_backend": "torch_sdpa",
        "generation_backend": "custom_greedy_dynamic_cache",
        "batch_size": 1,
        "base_and_adapter_compute_dtype": "torch.bfloat16 under CUDA autocast",
        "adapter_parameter_storage_dtype": "torch.float32",
        "logical_cuda_device": str(torch.cuda.current_device()),
        "physical_cuda_device": int(gpu[0]),
        "gpu_uuid": gpu[1],
        "gpu_name": gpu[2],
        "nvidia_driver": gpu[3],
        "gpu_total_memory_mib": int(gpu[4]),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timing_backends": {"cuda_operations": "torch.cuda.Event", "cpu_and_wall": "time.perf_counter"},
    }


def parameter_report(adapter: CommunicationAdapter, sender, receiver) -> dict[str, Any]:
    categories = defaultdict(int)
    for name, parameter in adapter.named_parameters():
        if name.startswith("pools."):
            category = "pooling"
        elif name.startswith("layer_map."):
            category = "layer_mapping"
        else:
            category = "joint_projection_conversion_fusion"
        categories[category] += parameter.numel()
    return {
        **categories,
        "deduplicated_trainable_total": sum(p.numel() for p in adapter.parameters()),
        "base_models_excluded": {
            "sender_frozen_parameters": sum(p.numel() for p in sender.parameters()),
            "receiver_frozen_parameters": sum(p.numel() for p in receiver.parameters()),
        },
        "deployment": {
            "pooling": "upstream physical GPU 1 for LCF-X; not applicable for LCF",
            "layer_mapping_projection_conversion_fusion": "downstream physical GPU 1",
        },
    }


def model_report(model, name: str, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "model_id": name,
        "path": str(ROOT / "models" / name),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "loaded_weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "frozen": True,
        "deployment": "logical cuda:0 / physical Device 1",
    }


def build_summary(
    *, method, pair, config, dataset, protocol, predictions_path, expected, excluded,
    processed_this_run, accumulators, environment, parameters, models, prompt_zip,
    prompt_bundle, checkpoint, checkpoint_sha256,
):
    _, metrics, timings, communications, tokens, finishes = accumulators
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "method": method.upper().replace("LCFX", "LCF-X"),
        "experiment_stage": "stage1_unified_test_and_base_overhead",
        "condition_id": f"{method}__{pair}__{dataset}__{protocol}",
        "pair": pair,
        "direction": f"{config['sender']} -> {config['receiver']}",
        "dataset": dataset,
        "protocol": protocol,
        "native_protocol_support": True,
        "completed_rows": len(accumulators[0]),
        "expected_rows": expected,
        "source_dataset_rows": expected + excluded,
        "protocol_ineligible_rows_excluded": excluded,
        "processed_rows_this_run": processed_this_run,
        "quality_statistics": common.quality_statistics(metrics),
        "relative_to_same_condition_t2t_baseline": {
            "status": "pending_paired_post_analysis_after_both_result_sets_complete",
            "raw_per_sample_metrics_available": True,
        },
        "timing_statistics_ms": {name: distribution(values) for name, values in timings.items()},
        "communication_statistics_bytes": {name: distribution(values) for name, values in communications.items()},
        "token_statistics": {name: distribution(values) for name, values in tokens.items()},
        "termination_counts": dict(finishes),
        "experiment_configuration": {
            "weights": str(checkpoint),
            "weights_sha256": checkpoint_sha256,
            "training_source": "OpenHermes-2.5 500K" if method == "lcf" else "HotpotQA bridge complementary evidence",
            "training_method": "paper-derived LCF reproduction" if method == "lcf" else "paper-derived LCF-X with XKV heterogeneous extension",
            "d": config["d"],
            "K": None,
            "k": None,
            "transmission_position": "full_source_KV_after_prefill" if method == "lcf" else "global_source_summary_after_hierarchical_pooling",
            "test_split": "GSM8K test / HotpotQA bridge validation / HumanEval test",
        },
        "module_parameters": parameters,
        "communication_representation": {
            "actual_payload": "all source KV layers" if method == "lcf" else "one pooled K and V slot per source layer and KV head",
            "tensor_dtype": "torch.bfloat16" if method == "lcf" else "torch.float32 (actual hierarchical-pool output)",
            "metadata_encoding": "canonical UTF-8 JSON",
            "network_transport_time_included": False,
            "centralized_execution_note": "Logical payload bytes are exact tensor nbytes; both frozen models execute on the same measured GPU.",
        },
        "operation_deployment": {
            "upstream": ["prompt tokenization on CPU", "source prefill on physical GPU 1"] + (["hierarchical span/global pooling on physical GPU 1"] if method == "lcfx" else []),
            "downstream": ["receiver local prefill", "layer mapping when required", "down projection", "nonlinear conversion", "up projection and head weighting", "gated cache fusion", "greedy final decode"],
            "downstream_device": "physical GPU 1",
        },
        "prompt": {
            "archive": str(prompt_zip),
            "zip_sha256": sha256_file(prompt_zip),
            "schema": prompt_bundle["schema"],
            "version": prompt_bundle["version"],
            "archive_status_at_evaluation": prompt_bundle["status"],
            "chat_template": prompt_bundle["chat_template"],
            "cache_method_adaptation": "receiver_system + visible protocol view + receiver_task; applied symmetrically to both caches",
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "enable_thinking": False,
            "max_new_tokens": RECEIVER_MAX_NEW_TOKENS[dataset],
            "limit_policy": "HotpotQA follows the XKV/LCF-X paper's receiver limit of 64; GSM8K and HumanEval use fixed 1024-token task limits",
            "hard_ceiling": "min(fixed task limit, remaining receiver context)",
        },
        "measurement_environment": environment,
        "model_deployments": models,
        "predictions": str(predictions_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("lcf", "lcfx"), required=True)
    parser.add_argument("--pair", choices=sorted(PAIR_CONFIGS), required=True)
    parser.add_argument("--prompt-zip", type=Path, default=DEFAULT_PROMPT_ZIP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--only-condition")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / "configs" / f"{args.method}_{args.pair}_d128.json"
    config = json.loads(config_path.read_text())
    prompt_bundle = common.load_prompt_bundle(args.prompt_zip)
    conditions = []
    inventory = {}
    for dataset, protocol in selected_conditions(args.method):
        condition_id = f"{args.method}__{args.pair}__{dataset}__{protocol}"
        if args.only_condition is not None and args.only_condition != condition_id:
            continue
        expected, excluded = common.expected_and_excluded(dataset, protocol)
        inventory[condition_id] = {"expected_rows": expected, "excluded_rows": excluded}
        conditions.append((dataset, protocol, condition_id))
    if not conditions:
        raise ValueError(f"No matching supported condition: {args.only_condition}")

    pair_root = args.output_root / args.method / args.pair
    pair_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(pair_root / "protocol_support.json", {
        "method": args.method,
        "pair": args.pair,
        "native_supported": [{"dataset": d, "protocol": p} for d, p in selected_conditions(args.method)],
        "unsupported": unsupported_conditions(args.method),
        "policy_source": "/home/lewis/文档/子问题四_统一复现与变量实验计划.md",
    })
    if args.validate_only:
        print(json.dumps({"method": args.method, "pair": args.pair, "conditions": inventory, "unsupported": unsupported_conditions(args.method)}, ensure_ascii=False, indent=2))
        return

    torch.manual_seed(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda:0"
    assert_expected_gpu()
    sender_name, receiver_name = PAIR_CONFIGS[args.pair]
    sender_tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / sender_name, local_files_only=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / receiver_name, local_files_only=True)
    sender = load_frozen(ROOT / "models" / sender_name, device, torch.bfloat16)
    receiver = load_frozen(ROOT / "models" / receiver_name, device, torch.bfloat16)
    adapter = CommunicationAdapter(sender.config, receiver.config, args.method, config["d"], config["dropout"]).to(device)
    checkpoint = ROOT / "runs" / "formal" / config["name"] / "final" / "adapter.safetensors"
    adapter.load_state_dict(load_file(checkpoint, device=device), strict=True)
    adapter.eval()
    adapter.set_temperature(config.get("total_steps", 300), config.get("total_steps", 300))
    environment = runtime_environment()
    parameters = parameter_report(adapter, sender, receiver)
    models = {"sender": model_report(sender, sender_name, "upstream"), "receiver": model_report(receiver, receiver_name, "downstream")}
    checkpoint_sha256 = sha256_file(checkpoint)
    run_status = pair_root / "run_status.json"
    atomic_write_json(run_status, {
        "state": "running", "method": args.method, "pair": args.pair, "pid": os.getpid(),
        "started_at": now(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "conditions": [item[2] for item in conditions],
    })

    try:
        for dataset, protocol, condition_id in conditions:
            task_prompt = prompt_bundle["per_task"][dataset]
            condition_root = pair_root / dataset / protocol
            condition_root.mkdir(parents=True, exist_ok=True)
            predictions_path = condition_root / "predictions.jsonl"
            summary_path = condition_root / "summary.json"
            accumulators = prior_state(predictions_path)
            completed, metrics, timings, communications, tokens, finishes = accumulators
            expected_rows, excluded_rows = common.expected_and_excluded(dataset, protocol)
            effective_expected = min(expected_rows, args.limit) if args.limit is not None else expected_rows
            processed_this_run = 0

            with predictions_path.open("a", encoding="utf-8") as output:
                eligible_ordinal = 0
                for record in common.records_for(dataset):
                    if protocol == "P2" and not record["p2_eligible"]:
                        continue
                    if args.limit is not None and eligible_ordinal >= args.limit:
                        break
                    eligible_ordinal += 1
                    if record["record_id"] in completed:
                        continue

                    torch.cuda.reset_peak_memory_stats()
                    inference_started = time.perf_counter()
                    assembly_started = time.perf_counter()
                    sender_view, receiver_view = common.protocol_views(record, protocol, task_prompt["p0"])
                    sender_ids, sender_offsets, sender_rendered, sender_messages = prompt_ids_and_offsets(sender_tokenizer, sender_view, task_prompt)
                    receiver_ids, _, receiver_rendered, receiver_messages = prompt_ids_and_offsets(receiver_tokenizer, receiver_view, task_prompt)
                    if len(receiver_ids) < 2:
                        raise ValueError("Receiver prompt must contain at least two tokens")
                    spans = locate_spans(dataset, record, protocol, sender_rendered, sender_offsets) if args.method == "lcfx" else None
                    if args.method == "lcf" and sender_ids != receiver_ids:
                        raise ValueError("Native LCF P1 requires exactly aligned token IDs")
                    sender_model_ids = sender_ids[:-1] if args.method == "lcf" else sender_ids
                    prompt_assembly_cpu_ms = (time.perf_counter() - assembly_started) * 1000
                    if len(sender_model_ids) > sender.config.max_position_embeddings or len(receiver_ids) > receiver.config.max_position_embeddings:
                        raise ValueError(f"Prompt exceeds model context for {record['record_id']}")

                    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                        source_output, sender_prefill_ms = timed_cuda(lambda: sender.model(
                            input_ids=torch.tensor([sender_model_ids], dtype=torch.long, device=device),
                            use_cache=True, return_dict=True,
                        ))
                        receiver_output, receiver_prefill_ms = timed_cuda(lambda: receiver.model(
                            input_ids=torch.tensor([receiver_ids[:-1]], dtype=torch.long, device=device),
                            use_cache=True, return_dict=True,
                        ))
                        source_cache = cache_layers(source_output.past_key_values)
                        receiver_cache = cache_layers(receiver_output.past_key_values)
                        original_kv_bytes = tensor_bytes(source_cache)
                        pooling_ms = 0.0
                        layer_mapping_ms = 0.0
                        if args.method == "lcfx":
                            payload, pooling_ms = timed_cuda(lambda: pool_source(adapter, source_cache, spans))
                            if adapter.layer_map is None:
                                transformed_source, layer_mapping_ms = payload, 0.0
                            else:
                                transformed_source, layer_mapping_ms = timed_cuda(lambda: map_source_layers(adapter, payload))
                        else:
                            payload = source_cache
                            transformed_source = [source_cache[index] for index in adapter.mapping]

                        packaging_started = time.perf_counter()
                        payload_description = payload_shape(payload)
                        wire_metadata = {
                            "condition_id": condition_id,
                            "sample_id": record["record_id"],
                            "schema_version": 1,
                            "representation": "full_source_kv" if args.method == "lcf" else "pooled_global_source_summary",
                            "shape": payload_description,
                            "span_count": len(spans) if spans is not None else None,
                        }
                        metadata_json = json.dumps(wire_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        packaging_cpu_ms = (time.perf_counter() - packaging_started) * 1000
                        unpack_started = time.perf_counter()
                        if json.loads(metadata_json)["sample_id"] != record["record_id"]:
                            raise RuntimeError("Metadata round trip failed")
                        unpack_cpu_ms = (time.perf_counter() - unpack_started) * 1000

                        fused_cache, fusion_timings = fuse_with_breakdown(adapter, transformed_source, receiver_cache)
                        reference_max_abs_difference = None
                        if args.verify_reference:
                            reference = adapter(source_cache, receiver_cache, spans)
                            reference_max_abs_difference = max_difference(fused_cache, reference)
                            if reference_max_abs_difference > 1e-4:
                                raise AssertionError(f"Decomposed adapter differs from centralized reference: {reference_max_abs_difference}")

                        generation_started = time.perf_counter()
                        generated_ids, generation = decode(
                            receiver, fused_cache, receiver_ids[-1], receiver_tokenizer,
                            receiver.config.max_position_embeddings, len(receiver_ids),
                            RECEIVER_MAX_NEW_TOKENS[dataset],
                        )
                        generation_wall_ms = (time.perf_counter() - generation_started) * 1000

                    detokenize_started = time.perf_counter()
                    prediction = receiver_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    detokenize_cpu_ms = (time.perf_counter() - detokenize_started) * 1000
                    inference_wall_ms = (time.perf_counter() - inference_started) * 1000
                    scoring_started = time.perf_counter()
                    sample_metrics, execution = common.score_prediction(dataset, prediction, record)
                    scoring_wall_ms = (time.perf_counter() - scoring_started) * 1000
                    payload_bytes = tensor_bytes(payload)
                    metadata_bytes = len(metadata_json.encode("utf-8"))
                    cuda_sum = sender_prefill_ms + receiver_prefill_ms + pooling_ms + layer_mapping_ms + fusion_timings["joint_transform_and_fusion_cuda_ms"] + generation["forward_cuda_ms"]
                    timing = {
                        "prompt_assembly_and_tokenization_cpu_ms": prompt_assembly_cpu_ms,
                        "sender_prefill_cuda_ms": sender_prefill_ms,
                        "source_pooling_cuda_ms": pooling_ms,
                        "sender_packaging_cpu_ms": packaging_cpu_ms,
                        "receiver_unpack_cpu_ms": unpack_cpu_ms,
                        "receiver_local_prefill_cuda_ms": receiver_prefill_ms,
                        "source_layer_mapping_cuda_ms": layer_mapping_ms,
                        **fusion_timings,
                        "receiver_final_decode_cuda_ms": generation["forward_cuda_ms"],
                        "receiver_generation_wall_ms": generation_wall_ms,
                        "receiver_token_selection_and_d2h_wall_ms": generation["token_selection_and_d2h_wall_ms"],
                        "receiver_detokenize_cpu_ms": detokenize_cpu_ms,
                        "end_to_end_inference_wall_ms": inference_wall_ms,
                        "end_to_end_minus_measured_cuda_ms": max(0.0, inference_wall_ms - cuda_sum),
                        "quality_scoring_wall_ms": scoring_wall_ms,
                        "network_transport_ms": 0.0,
                    }
                    communication = {
                        "source_original_kv_cache_bytes": original_kv_bytes,
                        "transmitted_tensor_bytes": payload_bytes,
                        "wire_metadata_json_utf8_bytes": metadata_bytes,
                        "logical_transmission_bytes": payload_bytes + metadata_bytes,
                        "supplemental_original_text_utf8_bytes": 0,
                        "generated_message_utf8_bytes": 0,
                    }
                    token_counts = {
                        "sender_visible_input_tokens": len(sender_tokenizer.encode(sender_view, add_special_tokens=False)),
                        "sender_prompt_tokens": len(sender_ids),
                        "sender_prefill_tokens": len(sender_model_ids),
                        "receiver_local_input_tokens": len(receiver_tokenizer.encode(receiver_view, add_special_tokens=False)),
                        "receiver_prompt_tokens": len(receiver_ids),
                        "receiver_prefill_tokens": len(receiver_ids) - 1,
                        "receiver_output_tokens": len(generated_ids),
                    }
                    result = {
                        "result_schema_version": RESULT_SCHEMA_VERSION,
                        "method": args.method.upper().replace("LCFX", "LCF-X"),
                        "record_id": record["record_id"],
                        "task_id": record.get("task_id"),
                        "condition_id": condition_id,
                        "pair": args.pair,
                        "direction": f"{sender_name} -> {receiver_name}",
                        "dataset": dataset,
                        "protocol": protocol,
                        "prediction": prediction,
                        "reference": record.get("answer"),
                        "metrics": sample_metrics,
                        "humaneval_execution": execution,
                        "token_counts": token_counts,
                        "input_bytes": {
                            "question_utf8_bytes": len(record["q"].encode("utf-8")),
                            "sender_visible_view_utf8_bytes": len(sender_view.encode("utf-8")),
                            "receiver_local_view_utf8_bytes": len(receiver_view.encode("utf-8")),
                            "sender_rendered_prompt_utf8_bytes": len(sender_rendered.encode("utf-8")),
                            "receiver_rendered_prompt_utf8_bytes": len(receiver_rendered.encode("utf-8")),
                        },
                        "communication_bytes": communication,
                        "communication_payload": wire_metadata,
                        "timing_ms": timing,
                        "termination": {
                            "sender_finish_reason": "prefill_completed",
                            "receiver_finish_reason": generation["finish_reason"],
                            "receiver_context_capacity_tokens": receiver.config.max_position_embeddings,
                            "configured_max_new_tokens": RECEIVER_MAX_NEW_TOKENS[dataset],
                            "receiver_forward_calls": generation["forward_calls"],
                        },
                        "protocol_compliance": {
                            "native_adapter_path": True,
                            "exact_token_alignment_verified": True if args.method == "lcf" else None,
                            "p2_source_support_paragraphs": 1 if protocol == "P2" else None,
                            "p2_source_distractor_paragraphs": 4 if protocol == "P2" else None,
                            "p2_receiver_support_paragraphs": 1 if protocol == "P2" else None,
                            "p2_receiver_distractor_paragraphs": 4 if protocol == "P2" else None,
                            "source_pool_span_count": len(spans) if spans is not None else None,
                        },
                        "experiment_configuration": {
                            "d": config["d"], "K": None, "k": None, "batch_size": 1,
                            "dtype": "bfloat16", "enable_thinking": False, "do_sample": False,
                            "max_new_tokens": RECEIVER_MAX_NEW_TOKENS[dataset],
                            "transmission_position": "full_source_KV_after_prefill" if args.method == "lcf" else "global_source_summary_after_hierarchical_pooling",
                        },
                        "memory": {
                            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                        },
                        "centralized_reference_max_abs_cache_difference": reference_max_abs_difference,
                    }
                    output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    output.flush()
                    completed.add(record["record_id"])
                    processed_this_run += 1
                    common.add_values(metrics, sample_metrics)
                    common.add_values(timings, timing)
                    common.add_values(communications, communication)
                    common.add_values(tokens, token_counts)
                    finishes[generation["finish_reason"]] += 1

                    if processed_this_run == 1 or len(completed) % 10 == 0 or len(completed) == effective_expected:
                        atomic_write_json(summary_path, build_summary(
                            method=args.method, pair=args.pair, config=config, dataset=dataset, protocol=protocol,
                            predictions_path=predictions_path, expected=effective_expected, excluded=excluded_rows,
                            processed_this_run=processed_this_run, accumulators=accumulators, environment=environment,
                            parameters=parameters, models=models, prompt_zip=args.prompt_zip, prompt_bundle=prompt_bundle,
                            checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256,
                        ))
                        atomic_write_json(run_status, {
                            "state": "running", "method": args.method, "pair": args.pair, "pid": os.getpid(),
                            "condition_id": condition_id, "completed": len(completed), "expected": effective_expected,
                            "updated_at": now(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        })
                    print(json.dumps({
                        "condition_id": condition_id, "record_id": record["record_id"],
                        "completed": len(completed), "expected": effective_expected,
                        "metrics": sample_metrics, "finish": generation["finish_reason"],
                    }, ensure_ascii=False), flush=True)

            atomic_write_json(summary_path, build_summary(
                method=args.method, pair=args.pair, config=config, dataset=dataset, protocol=protocol,
                predictions_path=predictions_path, expected=effective_expected, excluded=excluded_rows,
                processed_this_run=processed_this_run, accumulators=accumulators, environment=environment,
                parameters=parameters, models=models, prompt_zip=args.prompt_zip, prompt_bundle=prompt_bundle,
                checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256,
            ))
            if len(completed) != effective_expected:
                raise RuntimeError(f"Incomplete condition {condition_id}: {len(completed)} != {effective_expected}")

        atomic_write_json(pair_root / "pair_complete.json", {
            "method": args.method, "pair": args.pair, "state": "completed", "completed_at": now(),
            "conditions": [item[2] for item in conditions], "checkpoint_sha256": checkpoint_sha256,
        })
        atomic_write_json(run_status, {"state": "completed", "method": args.method, "pair": args.pair, "finished_at": now()})
    except BaseException as error:
        atomic_write_json(run_status, {
            "state": "failed", "method": args.method, "pair": args.pair,
            "error": f"{type(error).__name__}: {error}", "updated_at": now(),
        })
        raise


if __name__ == "__main__":
    main()

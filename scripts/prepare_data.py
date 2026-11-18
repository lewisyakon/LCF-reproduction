#!/usr/bin/env python3
"""Materialize paper training sets from the local, read-only asset mirror."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from datasets.table import MemoryMappedTable
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("LCF_SOURCE_ASSETS_ROOT", ROOT))
OH = ASSETS / "cache/huggingface/datasets/teknium___open_hermes-2.5/default/0.0.0/b82037821055c377bed0d495e72e46de3bc72e84/open_hermes-2.5-train.arrow"
TOKENIZER = None
TEST_HASHES = set()


def load_cached_openhermes(path):
    # The shared cache was written by a newer datasets release, whose JSON
    # feature metadata uses "List". Infer features from the unchanged physical
    # Arrow schema instead; keep the original file memory-mapped and read-only.
    table = MemoryMappedTable.from_file(str(path)).replace_schema_metadata(None)
    return Dataset(table)


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def digest(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-0.6B", local_files_only=True)
    return TOKENIZER


def messages_of(row):
    # Match extract_openhermes_messages in the pinned public C2C source:
    # exclude system messages, strip user text, preserve assistant text.
    roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    result = []
    for message in row["conversations"]:
        role = roles.get(message.get("from") or message.get("role"))
        if role is not None:
            content = str(message.get("value", ""))
            result.append({"role": role, "content": content.strip() if role == "user" else content})
    return result


def openhermes(row, index):
    tok = tokenizer()
    messages = messages_of(row)
    result = {"id": f"openhermes:{index}", "sender_ids": [], "receiver_ids": [],
              "response_ids": [], "spans": [], "status": "invalid", "question_hash": ""}
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return result
    user_hashes = {digest(m["content"]) for m in messages if m["role"] == "user"}
    result["question_hash"] = digest(json.dumps(messages[:-1], ensure_ascii=False, sort_keys=True))
    if user_hashes & TEST_HASHES:
        result["status"] = "test_exact_overlap"
        return result
    # Match the public C2C filter: render the complete conversation, <=2048 tokens.
    full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
    if len(full) > 2048:
        result["status"] = "overlength"
        return result
    prompt = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if full[:len(prompt)] != prompt:
        result["status"] = "template_prefix_mismatch"
        return result
    answer = full[len(prompt):]
    if not answer or len(prompt) < 2:
        return result
    result.update(sender_ids=prompt[:-1], receiver_ids=prompt, response_ids=answer,
                  spans=[[0, len(prompt) - 1]], status="ok")
    return result


def evidence_prompt(question, paragraphs):
    tok = tokenizer()
    heading = "Use the evidence to answer the question. Give only the short answer.\nQuestion: " + question + "\n\nEvidence:\n"
    blocks = [f"[{title}]\n{''.join(sentences)}\n\n" for title, sentences in paragraphs]
    content = heading + "".join(blocks)
    rendered = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    encoded = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    content_start = rendered.index(content)
    pos = content_start + len(heading)
    spans = []
    offsets = encoded["offset_mapping"]
    for block in blocks:
        end = pos + len(block)
        indices = [i for i, (a, b) in enumerate(offsets) if b > pos and a < end]
        if not indices:
            raise ValueError("Empty paragraph span")
        spans.append([indices[0], indices[-1] + 1])
        pos = end
    return encoded["input_ids"], spans


def hotpot(row, index):
    import random
    result = {"id": row["id"], "sender_ids": [], "receiver_ids": [], "response_ids": [],
              "spans": [], "status": "invalid", "question_hash": digest(row["question"])}
    if result["question_hash"] in TEST_HASHES:
        result["status"] = "test_exact_overlap"
        return result
    context = list(zip(row["context"]["title"], row["context"]["sentences"]))
    support = set(row["supporting_facts"]["title"])
    positives = [p for p in context if p[0] in support]
    negatives = [p for p in context if p[0] not in support]
    if len(positives) != 2 or len(negatives) != 8:
        return result
    # Stable per-ID assignment; no Python randomized hash, no answer used here.
    rng = random.Random(int(hashlib.sha256(("0:" + row["id"]).encode()).hexdigest()[:16], 16))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    sender, receiver = [positives[0]] + negatives[:4], [positives[1]] + negatives[4:]
    rng.shuffle(sender)
    rng.shuffle(receiver)
    sid, spans = evidence_prompt(row["question"], sender)
    rid, _ = evidence_prompt(row["question"], receiver)
    answer = tokenizer()(row["answer"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    # Preserve every paragraph: overlength records are explicitly excluded, never
    # silently truncate away supporting facts or all answer tokens.
    if len(sid) > 2048 or len(rid) + len(answer) > 2048:
        result["status"] = "overlength"
        return result
    result.update(sender_ids=sid, receiver_ids=rid, response_ids=answer,
                  spans=spans, status="ok")
    return result


def test_hashes():
    rows = []
    for name, field in [("GSM8K", "question"), ("HotpotQA", "question"), ("HumanEval", "prompt")]:
        files = sorted((ASSETS / "datasets" / name).rglob("*test*.parquet"))
        if name == "HotpotQA":
            files = sorted((ASSETS / "datasets" / name).rglob("*validation*.parquet"))
        if not files:
            raise FileNotFoundError(f"Cannot audit reserved test data for {name}")
        ds = load_dataset("parquet", data_files=[str(p) for p in files], split="train",
                          cache_dir=str(ROOT / "data/cache"))
        rows.extend(digest(q) for q in ds[field])
    return set(rows)


def prepare(method, workers):
    output = ROOT / "data" / method
    if output.exists():
        ds = load_from_disk(output)
        print(method, {k: len(v) for k, v in ds.items()}, "already prepared", flush=True)
        return
    if method == "lcf":
        raw = load_cached_openhermes(OH).select(range(500000))
        fn, seed, holdout = openhermes, 42, 0.01
    else:
        files = sorted((ASSETS / "datasets/HotpotQA/distractor").glob("train*.parquet"))
        raw = load_dataset("parquet", data_files=[str(p) for p in files], split="train",
                           cache_dir=str(ROOT / "data/cache"))
        raw = raw.filter(lambda x: x["type"] == "bridge")
        fn, seed, holdout = hotpot, 0, 0.028
    initial = len(raw)
    processed = raw.map(fn, with_indices=True, num_proc=workers, remove_columns=raw.column_names,
                        cache_file_name=str(ROOT / "data/cache" / f"{method}-tokenized.arrow"),
                        desc=f"Preparing {method}")
    from collections import Counter
    counts = Counter(processed["status"])
    processed = processed.filter(lambda x: x["status"] == "ok")
    # Keep duplicate conversations/questions in a single split to avoid leakage.
    permutation = torch.randperm(len(processed), generator=torch.Generator().manual_seed(seed)).tolist()
    ndev = math.ceil(len(processed) * holdout)
    dev_indices = permutation[-ndev:]
    hashes = processed["question_hash"]
    dev_hashes = {hashes[i] for i in dev_indices}
    train_indices = [i for i in permutation[:-ndev] if hashes[i] not in dev_hashes]
    datasets = DatasetDict(train=processed.select(train_indices), validation=processed.select(dev_indices))
    # Publish a fully written dataset atomically so interrupted preparation can
    # safely reuse map caches rather than mistake partial output for completion.
    import shutil
    staging = ROOT / "data" / f".{method}-preparing"
    if staging.exists():
        shutil.rmtree(staging)
    datasets.save_to_disk(str(staging))
    staging.rename(output)
    manifest = {"method": method, "seed": seed, "raw_candidates": initial, "filter_counts": dict(counts),
                "train": len(train_indices), "validation": ndev, "max_sequence_length": 2048,
                "effective_batch": 260 if method == "lcf" else 256,
                "total_optimizer_steps": 300 if method == "lcf" else math.ceil(len(train_indices) / 256),
                "dedup_across_train_validation": True,
                "test_audit": "Exact normalized user/question/prompt SHA256; no semantic or substring audit",
                "reserved_test_hashes": len(TEST_HASHES),
                "train_ids_sha256": hashlib.sha256("\n".join(datasets["train"]["id"]).encode()).hexdigest(),
                "validation_ids_sha256": hashlib.sha256("\n".join(datasets["validation"]["id"]).encode()).hexdigest(),
                "source": str(OH) if method == "lcf" else str(ASSETS / "datasets/HotpotQA/distractor"),
                "source_revision": "b82037821055c377bed0d495e72e46de3bc72e84" if method == "lcf" else "1908d6afbbead072334abe2965f91bd2709910ab"}
    (ROOT / f"manifests/{method}_data.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lcf", "lcfx", "all"], default="all")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    (ROOT / "data/cache").mkdir(exist_ok=True)
    TEST_HASHES = test_hashes()
    for method in (["lcfx", "lcf"] if args.method == "all" else [args.method]):
        prepare(method, args.workers)

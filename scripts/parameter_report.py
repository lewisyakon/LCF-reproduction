#!/usr/bin/env python3
import gc
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lcf.adapter import CommunicationAdapter


if __name__ == "__main__":
    report = {}
    for path in sorted((ROOT / "configs").glob("*_d128.json")):
        cfg = json.loads(path.read_text())
        source = AutoConfig.from_pretrained(ROOT / "models" / cfg["sender"], local_files_only=True)
        receiver = AutoConfig.from_pretrained(ROOT / "models" / cfg["receiver"], local_files_only=True)
        adapter = CommunicationAdapter(source, receiver, cfg["method"], cfg["d"])
        parts = {name: sum(p.numel() for p in module.parameters()) for name, module in adapter.named_children()}
        total = sum(p.numel() for p in adapter.parameters())
        assert sum(parts.values()) == total
        report[cfg["name"]] = {"total": total, "components": parts, "bf16_deployment_bytes": total * 2,
            "source_layers": source.num_hidden_layers, "receiver_layers": receiver.num_hidden_layers,
            "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(cfg["name"], total, parts, flush=True)
        del adapter
        gc.collect()
    (ROOT / "manifests/adapter_parameters.json").write_text(json.dumps(report, indent=2))

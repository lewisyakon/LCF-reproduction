#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for path in [root / "runs/bootstrap_status.json", root / "runs/queue_status.json"]:
    if path.exists():
        print(path.name, json.dumps(json.loads(path.read_text()), ensure_ascii=False))
for path in sorted((root / "runs/formal").glob("*/status.json")):
    print(path.parent.name, json.dumps(json.loads(path.read_text()), ensure_ascii=False))
for path in sorted((root / "manifests").glob("*_data.json")):
    data = json.loads(path.read_text())
    print(path.name, "train", data["train"], "validation", data["validation"], "steps", data["total_optimizer_steps"])

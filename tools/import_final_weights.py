#!/usr/bin/env python3
"""Import the six final LCF/LCF-X adapters and build their checksum manifest."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/mnt/hd2/lewis/LCFReproduction/runs/formal")
RUNS = [
    ("lcf", "M1", "lcf_M1_d128"),
    ("lcf", "M2", "lcf_M2_d128"),
    ("lcf", "M3", "lcf_M3_d128"),
    ("lcfx", "M1", "lcfx_M1_d128"),
    ("lcfx", "M2", "lcfx_M2_d128"),
    ("lcfx", "M3", "lcfx_M3_d128"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"existing destination has wrong size: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parameter_report = json.loads((REPO / "manifests/adapter_parameters.json").read_text())
    records = []
    for method, pair, run in RUNS:
        source = SOURCE_ROOT / run / "final"
        destination = REPO / "weights" / method / pair
        weight_source = source / "adapter.safetensors"
        weight_destination = destination / "adapter.safetensors"
        hardlink_or_copy(weight_source, weight_destination)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "config.json", destination / "config.json")
        shutil.copy2(source / "metrics.json", destination / "metrics.json")
        report = parameter_report[run]
        records.append(
            {
                "id": f"{method}_{pair}",
                "method": method,
                "pair": pair,
                "path": str(weight_destination.relative_to(REPO)),
                "parameters": report["total"],
                "file_bytes": weight_destination.stat().st_size,
                "sha256": sha256(weight_destination),
                "source_path": str(weight_source),
                "config": str((destination / "config.json").relative_to(REPO)),
                "metrics": str((destination / "metrics.json").relative_to(REPO)),
                "load_api": "safetensors.torch.load_file",
            }
        )
        print(f"imported {weight_destination}")
    manifest = {
        "schema_version": 1,
        "created_at": "2026-11-18",
        "entries": records,
        "total_file_bytes": sum(record["file_bytes"] for record in records),
        "excluded": "training_state.pt and intermediate checkpoints",
    }
    (REPO / "weights/manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

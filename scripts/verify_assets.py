#!/usr/bin/env python3
"""Rehash reused model files; do not modify shared assets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    original = json.loads((ROOT / "manifests/reused_assets.json").read_text())
    result = {}
    for name, model in original["models"].items():
        directory = ROOT / "models" / name
        result[name] = {"path": str(directory.resolve()), "revision": model["revision"], "files": {}}
        for filename, expected in model["weights"].items():
            path = directory / filename
            actual = sha(path)
            ok = actual == expected["sha256"] and path.stat().st_size == expected["size"]
            result[name]["files"][filename] = {"sha256": actual, "bytes": path.stat().st_size, "ok": ok}
            print(name, filename, "OK" if ok else "FAIL", flush=True)
            if not ok:
                raise ValueError(f"Model verification failed: {path}")
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            result[name]["files"][filename] = {"sha256": sha(directory / filename)}
    hashes = [m["files"]["tokenizer.json"]["sha256"] for m in result.values()]
    if len(set(hashes)) != 1:
        raise ValueError("Tokenizers differ: shared-context LCF needs explicit token alignment")
    (ROOT / "manifests/verified_models.json").write_text(json.dumps(result, indent=2))

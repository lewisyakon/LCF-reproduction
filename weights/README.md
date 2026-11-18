# Included LCF/LCF-X adapters

This directory contains the six completed `d=128` inference adapters. Binary
weights are tracked with Git LFS; `git lfs pull` materializes them after a
clone. `manifest.json` records the source run, model direction, parameter
count, byte size, and SHA-256 checksum.

Each directory contains `adapter.safetensors`, the exact final training
configuration, and final metrics. Optimizer/training state is intentionally
excluded because it is not needed for inference.

The unified evaluator automatically falls back to these files when the
original `runs/formal/...` tree is absent.

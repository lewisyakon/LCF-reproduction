#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
# PyPI's Linux torch 2.6.0 wheel includes CUDA 12.4. Use locally verified wheels
# when present; resolve remaining CUDA dependencies from PyPI.
task_torch_spec='torch==2.6.0'
if [[ -f wheels/torch-2.6.0-cp312-cp312-manylinux1_x86_64.whl ]]; then
  task_torch_spec="$PWD/wheels/torch-2.6.0-cp312-cp312-manylinux1_x86_64.whl"
fi
task_extra_wheels=()
if [[ -f wheels/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl ]]; then
  task_extra_wheels+=("$PWD/wheels/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl")
fi
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple \
  "$task_torch_spec" "${task_extra_wheels[@]}" \
  'transformers==4.51.3' 'datasets==3.5.0' \
  'accelerate==1.6.0' 'safetensors==0.5.3' 'pytest>=8,<9'
uv pip freeze --python .venv/bin/python > manifests/environment.txt

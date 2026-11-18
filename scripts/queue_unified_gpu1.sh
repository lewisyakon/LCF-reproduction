#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LCF_PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT="$ROOT/evaluations/unified_stage1_d128"
LOG_DIR="$OUTPUT/logs"
STATUS="$OUTPUT/queue_status.json"
LOCK="/run/user/$(id -u)/lcf-unified-stage1-gpu1.lock"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "Another LCF unified GPU1 queue holds $LOCK" >&2
    exit 2
fi

write_status() {
    local state="$1"
    local method="${2:-}"
    local pair="${3:-}"
    local detail="${4:-}"
    "$PYTHON" - "$STATUS" "$state" "$method" "$pair" "$detail" <<'PY'
import json, os, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

path = Path(sys.argv[1])
value = {
    "state": sys.argv[2],
    "method": sys.argv[3] or None,
    "pair": sys.argv[4] or None,
    "detail": sys.argv[5] or None,
    "queue_pid": os.getppid(),
    "physical_gpu": 1,
    "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, path)
PY
}

wait_for_gpu1() {
    while true; do
        local free_mib
        free_mib=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 20000 )); then
            return
        fi
        write_status waiting_for_gpu "$1" "$2" "GPU1 free memory ${free_mib:-unknown} MiB; require 20000 MiB"
        sleep 30
    done
}

run_one() {
    local method="$1"
    local pair="$2"
    local complete="$OUTPUT/$method/$pair/pair_complete.json"
    local log="$LOG_DIR/${method}_${pair}.log"
    if [[ -f "$complete" ]]; then
        write_status skipping_completed "$method" "$pair" "$complete"
        return
    fi
    wait_for_gpu1 "$method" "$pair"
    write_status running "$method" "$pair" "$log"
    echo "[$(date --iso-8601=seconds)] starting $method $pair on physical GPU1" >>"$log"
    if env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        ${LCF_EXPECTED_GPU_UUID:+LCF_EXPECTED_GPU_UUID="$LCF_EXPECTED_GPU_UUID"} \
        TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
        "$PYTHON" "$ROOT/scripts/evaluate_unified.py" \
        --method "$method" --pair "$pair" --output-root "$OUTPUT" >>"$log" 2>&1; then
        :
    else
        local code=$?
        write_status failed "$method" "$pair" "exit=$code log=$log"
        return "$code"
    fi
    echo "[$(date --iso-8601=seconds)] completed $method $pair" >>"$log"
}

write_status starting "" "" "serial order: lcf M1-M3, then lcfx M1-M3"
for method in lcf lcfx; do
    for pair in M1 M2 M3; do
        if ! run_one "$method" "$pair"; then
            exit 1
        fi
    done
done
write_status completed "" "" "$OUTPUT"

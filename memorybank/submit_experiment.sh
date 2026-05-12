#!/bin/bash

set -euo pipefail

########################################
# Usage check
########################################
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml>"
    exit 1
fi

CONFIG_PATH=$1

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

########################################
# Project paths (EDIT if needed)
########################################
PROJ=/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena/memorybank
LLM_CLIENTS="$PROJ/configs/llm_clients.yaml"

########################################
# Extract fields from YAML
########################################
read -r MODEL_NAME RESULT_DIR TEST_START_IDX TEST_END_IDX WALL_TIME < <(python3 - <<EOF
import yaml
from pathlib import Path

config_path = "$CONFIG_PATH"
clients_path = "$LLM_CLIENTS"

with open(config_path) as f:
    cfg = yaml.safe_load(f)

llm_name = cfg.get("llm_config_name", "qwen3")

with open(clients_path) as f:
    clients = yaml.safe_load(f)

model = clients.get("clients", {}).get(llm_name, {}).get("model_name", "Qwen/Qwen3-8B")

# Derive result_dir: use explicit field or fall back to config filename stem
result_dir = cfg.get("result_dir") or f"memorybank/results_{Path(config_path).stem}"

test_start = cfg.get("test_start_idx", 0)
test_end   = cfg.get("test_end_idx",   812)
wall_time  = cfg.get("wall_time", "") or ""

print(model, result_dir, test_start, test_end, wall_time)
EOF
)

echo "Resolved model:      $MODEL_NAME"
echo "Result dir:          $RESULT_DIR"
echo "Task range:          $TEST_START_IDX .. $TEST_END_IDX"

########################################
# Map model → GPU constraint + time
########################################
if [[ "$MODEL_NAME" =~ 27B ]]; then
    GPU_CONSTRAINT="vram80"
    TIME="4:00:00"
    PARTITION="superpod-a100"
elif [[ "$MODEL_NAME" =~ 9B ]]; then
    GPU_CONSTRAINT="vram40|vram48"
    TIME="2:00:00"
    PARTITION="superpod-a100"
elif [[ "$MODEL_NAME" =~ 0\.8B ]]; then
    GPU_CONSTRAINT="vram16|vram23|vram40"
    TIME="1:00:00"
    PARTITION="gpu"
elif [[ "$MODEL_NAME" =~ 8B ]]; then
    GPU_CONSTRAINT="vram80"
    TIME="3:00:00"
    PARTITION="superpod-a100"
elif [[ "$MODEL_NAME" =~ 4B ]]; then
    GPU_CONSTRAINT="vram23|vram40"
    TIME="2:00:00"
    PARTITION="gpu"
else
    echo "WARNING: Unknown model size, defaulting to safe config"
    GPU_CONSTRAINT="vram40|vram48|vram80"
    TIME="4:00:00"
    PARTITION="superpod-a100"
fi

# Override default time with YAML wall_time if specified
[[ -n "$WALL_TIME" ]] && TIME="$WALL_TIME"

echo "Selected GPU constraint: $GPU_CONSTRAINT"
echo "Selected time: $TIME"
echo "Selected partition: $PARTITION"

########################################
# Submit single CPU job: launches services, polls health, then submits GPU job
########################################
SVC_JOB_ID=$(sbatch --parsable \
    --job-name="wa_svc_$(basename $CONFIG_PATH .yaml)" \
    --partition="cpu" \
    --time="10:00:00" \
    --export=ALL,WEBARENA_CONFIG="$CONFIG_PATH",MODEL_NAME="$MODEL_NAME",RESULT_DIR="$RESULT_DIR",TEST_START_IDX="$TEST_START_IDX",TEST_END_IDX="$TEST_END_IDX",GPU_CONSTRAINT="$GPU_CONSTRAINT",GPU_PARTITION="$PARTITION",GPU_TIME="$TIME",NODEDIR="${NODEDIR:-}" \
    "$PROJ/run_webarena_services.sh")

echo "Submitted services job: $SVC_JOB_ID"
echo "Services job will poll for health and submit the GPU job automatically."
echo "Safe to disconnect."

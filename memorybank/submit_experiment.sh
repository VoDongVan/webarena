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
WA_PROJ="$(dirname $PROJ)"
NODEDIR="$WA_PROJ/../webarena_build/homepage"

########################################
# Extract fields from YAML
########################################
read -r MODEL_NAME RESULT_DIR TEST_START_IDX TEST_END_IDX < <(python3 - <<EOF
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

print(model, result_dir, test_start, test_end)
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
    TIME="8:00:00"
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

echo "Selected GPU constraint: $GPU_CONSTRAINT"
echo "Selected time: $TIME"
echo "Selected partition: $PARTITION"

########################################
# Submit CPU services job
########################################
SVC_JOB_ID=$(sbatch --parsable \
    --job-name="wa_svc_$(basename $CONFIG_PATH .yaml)" \
    --partition="cpu" \
    --time="$TIME" \
    --export=ALL \
    "$PROJ/run_webarena_services.sh")

echo "Submitted services job: $SVC_JOB_ID"

########################################
# Poll until all services are healthy
########################################
declare -A SVC_PORTS=(
    [shopping]=7770
    [shopping_admin]=7780
    [reddit]=9999
    [gitlab]=8023
    [wikipedia]=8888
    [homepage]=4399
)

echo "Waiting for WebArena services to be ready..."
for attempt in $(seq 1 180); do
    ALL_OK=true

    for svc in "${!SVC_PORTS[@]}"; do
        node_file="$NODEDIR/.${svc}_node"
        port="${SVC_PORTS[$svc]}"

        if [[ ! -f "$node_file" ]]; then
            ALL_OK=false
            continue
        fi

        host=$(cat "$node_file")
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
            "http://${host}:${port}" 2>/dev/null || true)
        code=${code:-000}
        [[ "$code" =~ ^[23] ]] || ALL_OK=false
    done

    if [[ "$ALL_OK" == "true" ]]; then
        echo "All services ready (attempt $attempt, $((attempt * 15))s elapsed)"
        break
    fi

    if [[ $attempt -eq 180 ]]; then
        echo "ERROR: Services not ready after 45 minutes. Cancelling services job."
        scancel "$SVC_JOB_ID"
        exit 1
    fi

    sleep 15
done

########################################
# Submit GPU experiment job
########################################
GPU_JOB_ID=$(sbatch --parsable \
    --job-name="wa_$(basename $CONFIG_PATH .yaml)" \
    --partition="$PARTITION" \
    --constraint="$GPU_CONSTRAINT" \
    --time="$TIME" \
    --export=ALL,WEBARENA_CONFIG="$CONFIG_PATH",MODEL_NAME="$MODEL_NAME",RESULT_DIR="$RESULT_DIR",TEST_START_IDX="$TEST_START_IDX",TEST_END_IDX="$TEST_END_IDX",SVC_JOB_ID="$SVC_JOB_ID" \
    "$PROJ/run_experiment.sh")

echo "Submitted GPU job: $GPU_JOB_ID"
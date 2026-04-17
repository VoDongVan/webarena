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
# Extract model from YAML
########################################
MODEL_NAME=$(python3 - <<EOF
import yaml

config_path = "$CONFIG_PATH"
clients_path = "$LLM_CLIENTS"

with open(config_path) as f:
    cfg = yaml.safe_load(f)

llm_name = cfg.get("llm_config_name", "qwen3")

with open(clients_path) as f:
    clients = yaml.safe_load(f)

model = clients.get("clients", {}).get(llm_name, {}).get("model_name", "Qwen/Qwen3-8B")

print(model)
EOF
)

echo "Resolved model: $MODEL_NAME"

########################################
# Map model → GPU constraint + time
########################################
if [[ "$MODEL_NAME" =~ 27B ]]; then
    GPU_CONSTRAINT="vram80"
    TIME="6:00:00"
elif [[ "$MODEL_NAME" =~ 0\.8B ]]; then
    GPU_CONSTRAINT="vram16|vram23|vram40"
    TIME="1:00:00"
elif [[ "$MODEL_NAME" =~ 8B ]]; then
    GPU_CONSTRAINT="vram40|vram48"
    TIME="3:00:00"
elif [[ "$MODEL_NAME" =~ 4B ]]; then
    GPU_CONSTRAINT="vram23|vram40"
    TIME="2:00:00"
else
    echo "WARNING: Unknown model size, defaulting to safe config"
    GPU_CONSTRAINT="vram40|vram48|vram80"
    TIME="4:00:00"
fi

echo "Selected GPU constraint: $GPU_CONSTRAINT"
echo "Selected time: $TIME"

########################################
# Submit job
########################################
sbatch \
    --job-name="wa_$(basename $CONFIG_PATH .yaml)" \
    --constraint="$GPU_CONSTRAINT" \
    --time="$TIME" \
    --export=ALL,WEBARENA_CONFIG="$CONFIG_PATH",MODEL_NAME="$MODEL_NAME" \
    run_experiment.sh
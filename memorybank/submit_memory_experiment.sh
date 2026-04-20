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
# Project paths
########################################
PROJ=/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena/memorybank
LLM_CLIENTS="$PROJ/configs/llm_clients.yaml"

########################################
# Extract all fields from YAML
########################################
read -r MODEL_NAME RETRIEVER_TYPE RETRIEVER_PORT TOP_K EXTRACTION_MODEL \
     MEMORY_SAVE_PATH TEST_START_IDX TEST_END_IDX MEMORIES_INIT_PATH < <(python3 - <<EOF
import yaml

config_path = "$CONFIG_PATH"
clients_path = "$LLM_CLIENTS"

with open(config_path) as f:
    cfg = yaml.safe_load(f)

with open(clients_path) as f:
    clients = yaml.safe_load(f)

# Main model
llm_name = cfg.get("llm_config_name", "qwen3")
model = clients.get("clients", {}).get(llm_name, {}).get("model_name", "Qwen/Qwen3-8B")

# Memory fields
retriever_type  = cfg.get("retriever_type", "bm25")
retriever_port  = str(cfg.get("retriever_port", 8020))
top_k           = str(cfg.get("top_k", 3))
test_start      = str(cfg.get("test_start_idx", 0))
test_end        = str(cfg.get("test_end_idx", 1))
memories_init   = cfg.get("memories_init_path", "") or ""
memory_save     = cfg.get("memory_save_path", "") or ""

# Resolve optional extraction model
extraction_llm_name = cfg.get("extraction_llm_config_name") or ""
extraction_model = ""
if extraction_llm_name:
    extraction_model = (
        clients.get("clients", {})
               .get(extraction_llm_name, {})
               .get("model_name", "")
    )

print(model, retriever_type, retriever_port, top_k, extraction_model,
      memory_save, test_start, test_end, memories_init, sep="\n")
EOF
)

echo "Resolved model:          $MODEL_NAME"
echo "Retriever type:          $RETRIEVER_TYPE  port: $RETRIEVER_PORT"
echo "Top-k:                   $TOP_K"
echo "Extraction model:        ${EXTRACTION_MODEL:-<same as main>}"
echo "Memory save path:        ${MEMORY_SAVE_PATH:-<none>}"
echo "Memories init path:      ${MEMORIES_INIT_PATH:-<none>}"
echo "Task range:              $TEST_START_IDX .. $TEST_END_IDX"

########################################
# Map model → GPU constraint + time
########################################
if [[ "$MODEL_NAME" =~ 27B ]]; then
    GPU_CONSTRAINT="vram80"
    TIME="6:00:00"
    PARTITION="superpod-a100"
elif [[ "$MODEL_NAME" =~ 0\.8B ]]; then
    GPU_CONSTRAINT="vram16|vram23|vram40"
    TIME="1:00:00"
    PARTITION="gpu"
elif [[ "$MODEL_NAME" =~ 8B ]]; then
    GPU_CONSTRAINT="vram40|vram48"
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
echo "Selected time:           $TIME"
echo "Selected partition:      $PARTITION"

########################################
# Submit job
########################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch \
    --job-name="wamem_$(basename $CONFIG_PATH .yaml)" \
    --partition="$PARTITION" \
    --constraint="$GPU_CONSTRAINT" \
    --time="$TIME" \
    --export=ALL,\
WEBARENA_CONFIG="$CONFIG_PATH",\
MODEL_NAME="$MODEL_NAME",\
RETRIEVER_TYPE="$RETRIEVER_TYPE",\
RETRIEVER_PORT="$RETRIEVER_PORT",\
TOP_K="$TOP_K",\
EXTRACTION_MODEL="$EXTRACTION_MODEL",\
MEMORY_SAVE_PATH="$MEMORY_SAVE_PATH",\
TEST_START_IDX="$TEST_START_IDX",\
TEST_END_IDX="$TEST_END_IDX",\
MEMORIES_INIT_PATH="$MEMORIES_INIT_PATH" \
    "$SCRIPT_DIR/run_memory_experiment.sh"

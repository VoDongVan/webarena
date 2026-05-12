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
mapfile -t _values < <(python3 - <<EOF
import yaml

config_path = "$CONFIG_PATH"
clients_path = "$LLM_CLIENTS"

with open(config_path) as f:
    cfg = yaml.safe_load(f)

with open(clients_path) as f:
    clients = yaml.safe_load(f)

llm_name = cfg.get("llm_config_name", "qwen3")
model = clients.get("clients", {}).get(llm_name, {}).get("model_name", "Qwen/Qwen3-8B")

retriever_type  = cfg.get("retriever_type", "bm25")
retriever_port  = str(cfg.get("retriever_port", 8020))
top_k           = str(cfg.get("top_k", 3))
test_start      = str(cfg.get("test_start_idx", 0))
test_end        = str(cfg.get("test_end_idx", 1))
memories_init   = cfg.get("memories_init_path", "") or ""
memory_save     = cfg.get("memory_save_path", "") or ""
result_dir      = cfg.get("result_dir", "") or ""
wall_time       = cfg.get("wall_time", "") or ""

extraction_llm_name = cfg.get("extraction_llm_config_name") or ""
extraction_model = ""
if extraction_llm_name:
    extraction_model = (
        clients.get("clients", {})
               .get(extraction_llm_name, {})
               .get("model_name", "")
    )

embedding_model = cfg.get("embedding_model", "") or ""
gpu_count = "1"

print(model)
print(retriever_type)
print(retriever_port)
print(top_k)
print(extraction_model)
print(memory_save)
print(test_start)
print(test_end)
print(memories_init)
print(result_dir)
print(wall_time)
print(embedding_model)
print(gpu_count)
EOF
)

MODEL_NAME="${_values[0]}"
RETRIEVER_TYPE="${_values[1]}"
RETRIEVER_PORT="${_values[2]}"
TOP_K="${_values[3]}"
EXTRACTION_MODEL="${_values[4]}"
MEMORY_SAVE_PATH="${_values[5]}"
TEST_START_IDX="${_values[6]}"
TEST_END_IDX="${_values[7]}"
MEMORIES_INIT_PATH="${_values[8]}"
RESULT_DIR="${_values[9]}"
WALL_TIME="${_values[10]:-}"
EMBEDDING_MODEL="${_values[11]:-}"
GPU_COUNT="${_values[12]:-1}"

echo "Resolved model:          $MODEL_NAME"
echo "Retriever type:          $RETRIEVER_TYPE  port: $RETRIEVER_PORT"
echo "Top-k:                   $TOP_K"
echo "Extraction model:        ${EXTRACTION_MODEL:-<same as main>}"
echo "Memory save path:        ${MEMORY_SAVE_PATH:-<none>}"
echo "Memories init path:      ${MEMORIES_INIT_PATH:-<none>}"
echo "Task range:              $TEST_START_IDX .. $TEST_END_IDX"
echo "Result dir:              ${RESULT_DIR:-<default>}"
[[ "$RETRIEVER_TYPE" == "dense" ]] && echo "Embedding model:         ${EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"
echo "GPU count:               $GPU_COUNT"

########################################
# Map model → GPU constraint + time
########################################
if [[ "$MODEL_NAME" =~ 27B ]]; then
    GPU_CONSTRAINT="vram80"
    TIME="6:00:00"
    PARTITION="superpod-a100"
elif [[ "$MODEL_NAME" =~ 9B ]]; then
    GPU_CONSTRAINT="vram40|vram48"
    TIME="2:00:00"
    PARTITION="superpod-a100"
elif [[ "$MODEL_NAME" =~ 8B ]]; then
    GPU_CONSTRAINT="vram40|vram48"
    TIME="3:00:00"
    PARTITION="superpod-a100"
elif [[ "$MODEL_NAME" =~ 4B ]]; then
    GPU_CONSTRAINT="vram23|vram40"
    TIME="2:00:00"
    PARTITION="gpu"
elif [[ "$MODEL_NAME" =~ 0\.8B ]]; then
    GPU_CONSTRAINT="vram16|vram23|vram40"
    TIME="1:00:00"
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
echo "Selected time:           $TIME"
echo "Selected partition:      $PARTITION"

########################################
# Submit single CPU job: launches services, polls health, then submits GPU job
########################################
SVC_JOB_ID=$(sbatch --parsable \
    --job-name="wamem_svc_$(basename $CONFIG_PATH .yaml)" \
    --partition="cpu" \
    --time="10:00:00" \
    --export=ALL,\
WEBARENA_CONFIG="$CONFIG_PATH",\
MODEL_NAME="$MODEL_NAME",\
RESULT_DIR="$RESULT_DIR",\
TEST_START_IDX="$TEST_START_IDX",\
TEST_END_IDX="$TEST_END_IDX",\
GPU_CONSTRAINT="$GPU_CONSTRAINT",\
GPU_PARTITION="$PARTITION",\
GPU_TIME="$TIME",\
GPU_SCRIPT="$PROJ/run_memory_experiment.sh",\
RETRIEVER_TYPE="$RETRIEVER_TYPE",\
RETRIEVER_PORT="$RETRIEVER_PORT",\
TOP_K="$TOP_K",\
EXTRACTION_MODEL="$EXTRACTION_MODEL",\
MEMORY_SAVE_PATH="$MEMORY_SAVE_PATH",\
MEMORIES_INIT_PATH="$MEMORIES_INIT_PATH",\
EMBEDDING_MODEL="$EMBEDDING_MODEL",\
GPU_COUNT="$GPU_COUNT",\
NODEDIR="${NODEDIR:-}" \
    "$PROJ/run_webarena_services.sh")

echo "Submitted services job: $SVC_JOB_ID"
echo "Services job will poll for health and submit the GPU job automatically."
echo "Safe to disconnect."

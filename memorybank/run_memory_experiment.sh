#!/bin/bash
#SBATCH -J wa_memory
#SBATCH -p superpod-a100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -o /scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena/memorybank/logs/wa_%j.out

set -euo pipefail

PROJ=/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena
BUILDDIR=$PROJ/../webarena_build
NODEDIR=$BUILDDIR/homepage

CONFIG="${WEBARENA_CONFIG}"
VLLM_MODEL="${MODEL_NAME}"
RESULT_DIR="${RESULT_DIR:-memorybank/results_memory}"
TEST_START_IDX="${TEST_START_IDX:-0}"
TEST_END_IDX="${TEST_END_IDX:-1}"
SVC_JOB_ID="${SVC_JOB_ID:-}"

# Memory-specific env vars
RETRIEVER_TYPE="${RETRIEVER_TYPE:-bm25}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8020}"
TOP_K="${TOP_K:-3}"
EXTRACTION_MODEL="${EXTRACTION_MODEL:-}"
MEMORY_SAVE_PATH="${MEMORY_SAVE_PATH:-}"
MEMORIES_INIT_PATH="${MEMORIES_INIT_PATH:-}"

echo "=== Starting memory job on $(hostname) at $(date) ==="
echo "Config:           $CONFIG"
echo "Model:            $VLLM_MODEL"
echo "Retriever type:   $RETRIEVER_TYPE  port: $RETRIEVER_PORT"
echo "Top-k:            $TOP_K"
echo "Extraction model: ${EXTRACTION_MODEL:-<same as main>}"
echo "Memory save path: ${MEMORY_SAVE_PATH:-<none>}"
echo "Task range:       $TEST_START_IDX .. $TEST_END_IDX"

########################################
# Cleanup
########################################
cleanup() {
    echo "Cleaning up..."
    [[ -n "${VLLM_PID:-}" ]] && kill $VLLM_PID 2>/dev/null || true
    [[ -n "${RETRIEVER_PID:-}" ]] && kill $RETRIEVER_PID 2>/dev/null || true
    [[ -n "${SVC_JOB_ID:-}" ]] && scancel "$SVC_JOB_ID" 2>/dev/null || true
}
trap cleanup EXIT

########################################
# Env
########################################
module load conda/latest
module load cuda/12.6
conda activate webarena

mkdir -p "$PROJ/memorybank/logs"
cd "$PROJ"

export HF_HOME=/work/pi_rrahimi_umass_edu/vanvo/huggingface
export VLLM_API_KEY="abc"
export MKL_SERVICE_FORCE_INTEL=1

########################################
# Wait for WebArena services (started by CPU services job)
########################################
wait_for_services() {
    declare -A SVC_PORTS=(
        [shopping]=7770
        [shopping_admin]=7780
        [reddit]=9999
        [gitlab]=8023
        [wikipedia]=8888
        [homepage]=4399
    )

    for attempt in $(seq 1 180); do
        ALL_OK=true

        for svc in "${!SVC_PORTS[@]}"; do
            port="${SVC_PORTS[$svc]}"
            node_file="$NODEDIR/.${svc}_node"

            [[ ! -f "$node_file" ]] && ALL_OK=false && continue

            host=$(cat "$node_file")
            code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
                "http://${host}:${port}" 2>/dev/null || true)
            code=${code:-000}

            [[ "$code" =~ ^[23] ]] || ALL_OK=false
        done

        [[ "$ALL_OK" == "true" ]] && return 0
        sleep 15
    done

    return 1
}

########################################
# vLLM startup
########################################
start_vllm() {
    python -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --port 8010 \
        --host 0.0.0.0 \
        --api-key abc \
        --gpu-memory-utilization 0.85 \
        --max-model-len 128000 \
        --dtype auto \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3xml \
        --trust-remote-code \
        > "$PROJ/memorybank/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &

    VLLM_PID=$!

    for i in $(seq 1 60); do
        code=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer abc" \
            "http://localhost:8010/v1/models" || true)

        [[ "$code" == "200" ]] && return 0
        sleep 30
    done

    return 1
}

########################################
# Retrieval server startup
########################################
start_retrieval_server() {
    local retriever_args="--port $RETRIEVER_PORT --retriever $RETRIEVER_TYPE"
    [[ -n "$MEMORIES_INIT_PATH" ]] && retriever_args+=" --memories $MEMORIES_INIT_PATH"

    python /scratch3/workspace/vdvo_umass_edu-CS696_S26/memorybank/retrieval/server.py \
        $retriever_args \
        > "$PROJ/memorybank/logs/retriever_${SLURM_JOB_ID}.log" 2>&1 &
    RETRIEVER_PID=$!

    for i in $(seq 1 30); do
        code=$(curl -s -o /dev/null -w "%{http_code}" \
            "http://localhost:${RETRIEVER_PORT}/health" || true)
        [[ "$code" == "200" ]] && return 0
        sleep 5
    done

    return 1
}

########################################
# Parallel startup: wait for services + start vLLM + start retriever
########################################
wait_for_services &
WA_PID=$!

start_vllm &
VLLM_INIT_PID=$!

start_retrieval_server &
RETRIEVAL_INIT_PID=$!

wait $WA_PID             || { echo "WebArena services not ready"; exit 1; }
wait $VLLM_INIT_PID      || { echo "vLLM startup failed"; exit 1; }
wait $RETRIEVAL_INIT_PID || { echo "Retrieval server startup failed"; exit 1; }

echo "=== All services ready at $(date) ==="

########################################
# Export WebArena endpoints
########################################
export WA_SHOPPING="http://$(cat $NODEDIR/.shopping_node):7770"
export WA_SHOPPING_ADMIN="http://$(cat $NODEDIR/.shopping_admin_node):7780"
export WA_REDDIT="http://$(cat $NODEDIR/.reddit_node):9999"
export WA_GITLAB="http://$(cat $NODEDIR/.gitlab_node):8023"
export WA_WIKIPEDIA="http://$(cat $NODEDIR/.wikipedia_node):8888"
export WA_HOMEPAGE="http://$(cat $NODEDIR/.homepage_node):4399"
export WA_MAP="http://localhost:1"

export SHOPPING="$WA_SHOPPING"
export SHOPPING_ADMIN="$WA_SHOPPING_ADMIN"
export REDDIT="$WA_REDDIT"
export GITLAB="$WA_GITLAB"
export WIKIPEDIA="$WA_WIKIPEDIA"
export HOMEPAGE="$WA_HOMEPAGE"
export MAP="$WA_MAP"

export OPENAI_API_KEY="${VLLM_API_KEY}"
export EVAL_LLM_MODEL="${VLLM_MODEL}"

########################################
# Run experiment
########################################
sleep 60

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

echo "=== Running memory retrieval experiment ==="
python "$PROJ/scripts/generate_test_data.py"

mkdir -p ./.auth

MEMORY_ARGS=()
MEMORY_ARGS+=(--retriever_server_url "http://localhost:${RETRIEVER_PORT}")
MEMORY_ARGS+=(--top_k "$TOP_K")
[[ -n "$EXTRACTION_MODEL" ]] && MEMORY_ARGS+=(--extraction_model "$EXTRACTION_MODEL")
[[ -n "$MEMORY_SAVE_PATH" ]] && MEMORY_ARGS+=(--memory_save_path "$PROJ/$MEMORY_SAVE_PATH")

python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s_memory.json \
  --provider vllm \
  --model "$VLLM_MODEL" \
  --max_tokens 2048 \
  --test_start_idx "$TEST_START_IDX" \
  --test_end_idx "$TEST_END_IDX" \
  --exclude_sites map \
  --result_dir "$PROJ/$RESULT_DIR" \
  "${MEMORY_ARGS[@]}"

echo "=== Done at $(date) ==="

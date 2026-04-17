#!/bin/bash
#SBATCH -J wa_baseline
#SBATCH -p superpod-a100
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH -t 3:00:00
#SBATCH -o /scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena/memorybank/logs/wa_%j.out

set -euo pipefail

PROJ=/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena
BUILDDIR=$PROJ/../webarena_build
NODEDIR=$BUILDDIR/homepage

CONFIG="${WEBARENA_CONFIG}"
VLLM_MODEL="${MODEL_NAME}"

echo "=== Starting job on $(hostname) at $(date) ==="
echo "Config: $CONFIG"
echo "Model: $VLLM_MODEL"

########################################
# Cleanup
########################################
cleanup() {
    echo "Cleaning up..."
    [[ -n "${VLLM_PID:-}" ]] && kill $VLLM_PID 2>/dev/null || true
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
# WebArena startup
########################################
start_webarena() {
    rm -f "$NODEDIR"/.{shopping,shopping_admin,reddit,gitlab,wikipedia,homepage}_node
    bash "$BUILDDIR/launch_all.sh"

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
            code=$(curl -s -o /dev/null -w "%{http_code}" \
                "http://${host}:${port}" || true)
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
# Parallel startup
########################################
start_webarena &
WA_PID=$!

start_vllm &
VLLM_INIT_PID=$!

wait $WA_PID || { echo "WebArena failed"; exit 1; }
wait $VLLM_INIT_PID || { echo "vLLM failed"; exit 1; }

########################################
# Export endpoints
# Bridge WA_* names to what browser_env/env_config.py expects
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

# vLLM accepts any key; OPENAI_API_KEY satisfies the env-var check in openai_utils.py
export OPENAI_API_KEY="${VLLM_API_KEY}"

########################################
# Run experiment
########################################
# Give services extra time after HTTP-200 before login forms are usable
sleep 60

# Ensure project root is in Python path regardless of how scripts are invoked
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

echo "=== Running experiment ==="
python "$PROJ/scripts/generate_test_data.py"

mkdir -p ./.auth

python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --provider vllm \
  --model "$VLLM_MODEL" \
  --max_tokens 2048 \
  --test_start_idx 0 \
  --test_end_idx 100 \
  --exclude_sites map \
  --result_dir "$PROJ/memorybank/results"

echo "=== Done ==="

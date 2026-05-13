#!/bin/bash
#SBATCH -J wa_embed
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40|vram48|vram80
#SBATCH -t 30:00:00
#SBATCH -o /scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena/memorybank/logs/wa_embed_%j.out
#SBATCH --mail-type=NONE

set -euo pipefail

PROJ=/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena
NODEDIR="${NODEDIR:-$PROJ/../webarena_build/homepage}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-AQ-MedAI/Diver-Retriever-4B}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8101}"

echo "=== Starting embedding server on $(hostname) at $(date) ==="
echo "Model:   $EMBEDDING_MODEL"
echo "Port:    $EMBEDDING_PORT"
echo "NODEDIR: $NODEDIR"

########################################
# Env
########################################
module load conda/latest
module load cuda/12.6
conda activate webarena

mkdir -p "$PROJ/memorybank/logs"
mkdir -p "$NODEDIR"

export HF_HOME=/work/pi_rrahimi_umass_edu/vanvo/huggingface
export MKL_SERVICE_FORCE_INTEL=1

########################################
# Cleanup: remove node file on exit
########################################
cleanup() {
    echo "Embedding server shutting down at $(date)"
    rm -f "$NODEDIR/.embedding_node"
}
trap cleanup EXIT

# Clear any stale node file from a previous run
rm -f "$NODEDIR/.embedding_node"

########################################
# Start vLLM pooling server
########################################
python -m vllm.entrypoints.openai.api_server \
    --model "$EMBEDDING_MODEL" \
    --port "$EMBEDDING_PORT" \
    --host 0.0.0.0 \
    --api-key abc \
    --runner pooling \
    --gpu-memory-utilization 0.50 \
    --dtype auto \
    --trust-remote-code \
    > "$PROJ/memorybank/logs/embedding_${SLURM_JOB_ID}.log" 2>&1 &

EMBEDDING_PID=$!

########################################
# Wait for server to be ready, then publish hostname
########################################
echo "Waiting for embedding server to be ready..."
for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer abc" \
        "http://localhost:${EMBEDDING_PORT}/v1/models" || true)
    if [[ "$code" == "200" ]]; then
        echo "Embedding server ready after $((i * 30))s"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "ERROR: Embedding server failed to start after 30 minutes"
        exit 1
    fi
    sleep 30
done

echo "$(hostname)" > "$NODEDIR/.embedding_node"
echo "Published hostname $(hostname) to $NODEDIR/.embedding_node"

########################################
# Stay alive until the GPU job cancels this job
########################################
wait $EMBEDDING_PID

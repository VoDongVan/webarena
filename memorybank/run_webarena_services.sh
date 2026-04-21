#!/bin/bash
#SBATCH -J wa_services
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 10:00:00
#SBATCH -o /scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena/memorybank/logs/wa_svc_%j.out
#SBATCH --mail-type=NONE
set -euo pipefail

PROJ=/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena
BUILDDIR=$PROJ/../webarena_build
NODEDIR=$BUILDDIR/homepage

# Passed via --export from submit_experiment.sh
CONFIG="${WEBARENA_CONFIG}"
MODEL_NAME="${MODEL_NAME}"
RESULT_DIR="${RESULT_DIR}"
TEST_START_IDX="${TEST_START_IDX}"
TEST_END_IDX="${TEST_END_IDX}"
GPU_CONSTRAINT="${GPU_CONSTRAINT}"
GPU_PARTITION="${GPU_PARTITION}"
GPU_TIME="${GPU_TIME}"

echo "=== Starting WebArena services on $(hostname) at $(date) ==="

module load conda/latest
conda activate webarena

mkdir -p "$PROJ/memorybank/logs"

rm -f "$NODEDIR"/.{shopping,shopping_admin,reddit,gitlab,wikipedia,homepage}_node
bash "$BUILDDIR/launch_all.sh"

echo "=== Services launched. Polling for health... ==="

declare -A SVC_PORTS=(
    [shopping]=7770
    [shopping_admin]=7780
    [reddit]=9999
    [gitlab]=8023
    [wikipedia]=8888
    [homepage]=4399
)

for attempt in $(seq 1 360); do
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

    if [[ $attempt -eq 360 ]]; then
        echo "ERROR: Services not ready after 90 minutes. Aborting."
        exit 1
    fi

    sleep 15
done

########################################
# Submit GPU experiment job now that services are healthy
########################################
GPU_JOB_ID=$(sbatch --parsable \
    --job-name="wa_$(basename "$CONFIG" .yaml)" \
    --partition="$GPU_PARTITION" \
    --constraint="$GPU_CONSTRAINT" \
    --time="$GPU_TIME" \
    --export=ALL,WEBARENA_CONFIG="$CONFIG",MODEL_NAME="$MODEL_NAME",RESULT_DIR="$RESULT_DIR",TEST_START_IDX="$TEST_START_IDX",TEST_END_IDX="$TEST_END_IDX",SVC_JOB_ID="$SLURM_JOB_ID" \
    "$PROJ/memorybank/run_experiment.sh")

echo "Submitted GPU job: $GPU_JOB_ID"
echo "=== Keeping services alive until GPU job completes ==="
sleep infinity

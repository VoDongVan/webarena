#!/bin/bash
# Submit one or more experiments simultaneously, each with its own isolated
# WebArena service set (separate NODEDIR per experiment).
#
# Usage:
#   bash memorybank/submit_parallel_experiments.sh <config1.yaml> [config2.yaml ...]
#
# Examples:
#   # All three v3 experiments
#   bash memorybank/submit_parallel_experiments.sh \
#       memorybank/configs/webarena_baseline_27b_300tasks_v3.yaml \
#       memorybank/configs/webarena_memory_bm25_27b_300tasks_v3.yaml \
#       memorybank/configs/webarena_memory_dense_27b_300tasks_v3.yaml
#
#   # Baseline + dense only
#   bash memorybank/submit_parallel_experiments.sh \
#       memorybank/configs/webarena_baseline_27b_300tasks_v3.yaml \
#       memorybank/configs/webarena_memory_dense_27b_300tasks_v3.yaml
#
# Memory vs. baseline is detected automatically: configs with "memory" in their
# filename use submit_memory_experiment.sh; others use submit_experiment.sh.
# NODEDIR is derived from the config filename so each experiment is isolated.

set -euo pipefail

PROJ=/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena
BUILDDIR=$PROJ/../webarena_build

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config1.yaml> [config2.yaml ...]"
    exit 1
fi

for cfg in "$@"; do
    [[ -f "$cfg" ]] || { echo "ERROR: config not found: $cfg"; exit 1; }
done

echo "=== Submitting ${#@} experiment(s) in parallel ==="
echo ""

for cfg in "$@"; do
    name=$(basename "$cfg" .yaml)
    nodedir="$BUILDDIR/nodedir_${name}"

    if [[ "$name" == *memory* ]]; then
        submit_script="$PROJ/memorybank/submit_memory_experiment.sh"
    else
        submit_script="$PROJ/memorybank/submit_experiment.sh"
    fi

    echo "--- $name"
    NODEDIR="$nodedir" bash "$submit_script" "$cfg"
    echo "    NODEDIR: $nodedir"
    echo ""
done

echo "=== All experiments submitted. Each has its own isolated service set. ==="

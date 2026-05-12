#!/bin/bash
# Submit baseline, BM25, and dense experiments simultaneously, each with its own
# isolated WebArena service set (separate NODEDIR per experiment).
#
# Usage:
#   bash memorybank/submit_parallel_experiments.sh
#
# Each experiment reads from and writes to its own nodedir_* directory under
# webarena_build/, so the three service sets never share node-discovery files.

set -euo pipefail

PROJ=/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena
BUILDDIR=$PROJ/../webarena_build

BASELINE_CONFIG="$PROJ/memorybank/configs/webarena_baseline_27b_300tasks_v3.yaml"
BM25_CONFIG="$PROJ/memorybank/configs/webarena_memory_bm25_27b_300tasks_v3.yaml"
DENSE_CONFIG="$PROJ/memorybank/configs/webarena_memory_dense_27b_300tasks_v3.yaml"

for cfg in "$BASELINE_CONFIG" "$BM25_CONFIG" "$DENSE_CONFIG"; do
    [[ -f "$cfg" ]] || { echo "ERROR: config not found: $cfg"; exit 1; }
done

echo "=== Submitting three parallel experiments ==="

NODEDIR="$BUILDDIR/nodedir_baseline_v3" \
    bash "$PROJ/memorybank/submit_experiment.sh" "$BASELINE_CONFIG"

NODEDIR="$BUILDDIR/nodedir_bm25_v3" \
    bash "$PROJ/memorybank/submit_memory_experiment.sh" "$BM25_CONFIG"

NODEDIR="$BUILDDIR/nodedir_dense_v3" \
    bash "$PROJ/memorybank/submit_memory_experiment.sh" "$DENSE_CONFIG"

echo ""
echo "=== All three experiments submitted ==="
echo "Each uses its own isolated service set:"
echo "  baseline → $BUILDDIR/nodedir_baseline_v3"
echo "  bm25     → $BUILDDIR/nodedir_bm25_v3"
echo "  dense    → $BUILDDIR/nodedir_dense_v3"

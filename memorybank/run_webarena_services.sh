#!/bin/bash
#SBATCH -J wa_services
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 8:00:00
#SBATCH -o /scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena/memorybank/logs/wa_svc_%j.out
#SBATCH --mail-type=NONE
set -euo pipefail

PROJ=/scratch3/workspace/vdvo_umass_edu-CS696_S26/webarena
BUILDDIR=$PROJ/../webarena_build
NODEDIR=$BUILDDIR/homepage

echo "=== Starting WebArena services on $(hostname) at $(date) ==="

module load conda/latest
conda activate webarena

mkdir -p "$PROJ/memorybank/logs"

rm -f "$NODEDIR"/.{shopping,shopping_admin,reddit,gitlab,wikipedia,homepage}_node
bash "$BUILDDIR/launch_all.sh"

echo "=== Services launched. Keeping alive until cancelled. ==="
sleep infinity

#!/usr/bin/env bash
#SBATCH --job-name=p15-pilot
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
set -euo pipefail
python="$PWD/.venv/bin/python"
out="results/${SLURM_JOB_ID}"
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1
for flow in os ws is; do
    for workload in tiny alexnet_conv alexnet_fc; do
        run="${flow}_${workload}"
        side=32
        format=gemm
        [[ "$workload" != tiny ]] || side=4
        [[ "$workload" != alexnet_conv ]] || format=conv
        cat > "$out/$run.cfg" <<CFG
[general]
run_name = $run
[architecture_presets]
ArrayHeight: $side
ArrayWidth: $side
IfmapSramSzkB: 256
FilterSramSzkB: 256
OfmapSramSzkB: 256
IfmapOffset: 0
FilterOffset: 100000000
OfmapOffset: 200000000
Dataflow: $flow
Bandwidth: 32
MemoryBanks: 1
[run_presets]
InterfaceBandwidth: CALC
CFG
        "$python" -m scalesim.scale -c "$out/$run.cfg" \
            -t "experiments/$workload.csv" -i "$format" -p "$out"
        test -s "$out/$run/COMPUTE_REPORT.csv"
    done
done

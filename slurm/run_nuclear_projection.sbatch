#!/bin/bash
#SBATCH --job-name=nuclear_projection
#SBATCH --output=logs/nuclear_projection_%j.out
#SBATCH --error=logs/nuclear_projection_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --account=bsc21
#SBATCH --qos=gp_bsccase
#SBATCH --partition=gpp

set -euo pipefail

# Keep NumPy/SciPy and optional XLA work inside the allocated CPU set.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Adjust WORKDIR when the repository has a different location on the server.
WORKDIR="${WORKDIR:-$HOME/QuantumMachineLearningFermions}"
RUN_ID="${SLURM_JOB_ID:-local}"
OUTDIR="${OUTDIR:-$WORKDIR/results/nuclear_projection/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTDIR" "$WORKDIR/logs"
cd "$WORKDIR"

# Activate the repository virtual environment when one is present. Otherwise,
# submit from an environment containing NumPy, SciPy and PFAPACK.
if [[ -f "$WORKDIR/.venv/bin/activate" ]]; then
    source "$WORKDIR/.venv/bin/activate"
fi
export PYTHONPATH="$WORKDIR/src/NSMFermions:$WORKDIR/benchmarks:${PYTHONPATH:-}"

# Select "grid", "neon-hf", or "all" at submission time. Examples:
# sbatch --export=ALL,RUN_MODE=grid,INTERACTION=cki,MASS=8,M_MAX=9,J_MAX=4 \
#   slurm/run_nuclear_projection.sbatch
# sbatch --export=ALL,RUN_MODE=grid,INTERACTION=usdb,MASS=20,M_MAX=17,J_MAX=6 \
#   slurm/run_nuclear_projection.sbatch
# sbatch --export=ALL,RUN_MODE=neon-hf,ISOTOPES="20 22 24" \
#   slurm/run_nuclear_projection.sbatch
RUN_MODE="${RUN_MODE:-grid}"
INTERACTION="${INTERACTION:-usdb}"
MASS="${MASS:-20}"
M_MAX="${M_MAX:-3}"
J_MAX="${J_MAX:-3}"
INTRINSIC_METHOD="${INTRINSIC_METHOD:-hf}"
STARTS="${STARTS:-8}"
MAXITER="${MAXITER:-500}"
SEED="${SEED:-8}"
FAF_ORDER="${FAF_ORDER:-2}"
ISOTOPES="${ISOTOPES:-20 22 24}"

run_grid() {
    command=(
        "$PYTHON_BIN" -u benchmarks/projection_grid_convergence.py
        --interaction "$INTERACTION"
        --mass "$MASS"
        --m-max "$M_MAX"
        --j-max "$J_MAX"
        --intrinsic-method "$INTRINSIC_METHOD"
        --starts "$STARTS"
        --maxiter "$MAXITER"
        --seed "$SEED"
        --faf-order "$FAF_ORDER"
        --output-dir "$OUTDIR"
    )
    if [[ -n "${NUMBER_GRID:-}" ]]; then
        read -r number_grid_n number_grid_z <<< "$NUMBER_GRID"
        command+=(--number-grid "$number_grid_n" "$number_grid_z")
    fi
    "${command[@]}"
}

run_neon_hf() {
    read -r -a isotope_values <<< "$ISOTOPES"
    "$PYTHON_BIN" -u benchmarks/usdb_ne_hf.py \
        --isotopes "${isotope_values[@]}" \
        --starts "$STARTS" \
        --maxiter "$MAXITER" \
        --seed "$SEED" \
        --output-dir "$OUTDIR"
}

case "$RUN_MODE" in
    grid)
        run_grid
        ;;
    neon-hf)
        run_neon_hf
        ;;
    all)
        run_neon_hf
        run_grid
        ;;
    *)
        echo "RUN_MODE must be grid, neon-hf, or all" >&2
        exit 2
        ;;
esac

echo "Results written to $OUTDIR"

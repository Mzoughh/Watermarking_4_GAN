#!/bin/bash
#
# Evaluation script (metrics + attacks) on Jean Zay (V100)
#
#

#SBATCH -A lfb@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --job-name=sg2_eval_v100
#SBATCH --output=EVAL-%j.out

set -eo pipefail
module purge
module load pytorch-gpu/py3/2.0.0
set -u
export PYTHONUNBUFFERED=1

# --- Specific to V100 architecture (sm_70)
export TORCH_CUDA_ARCH_LIST="7.0"

# Robust CUDA JIT toolchain selection
export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDAHOSTCXX="$(which g++)"

# Job-local CUDA extensions cache to avoid stale builds
export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR:-$JOBSCRATCH}/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"
rm -rf "$TORCH_EXTENSIONS_DIR"/*

# -----------------------------
# Paths (ADAPT)
# -----------------------------
CODEDIR="$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF"
NETWORK_NAMES=('T4G' 'T4G' 'T4G')
NETWORK_NUMBERS=('001100' '000275' '000425')
CONFIGS=('R2D' 'ZE' 'FG')
SIZES=('256' '128' '128')
DATASETS=('ffhq' 'ffhq' 'CelebA')

cd "$CODEDIR"

for i in "${!NETWORK_NAMES[@]}"; do
    NAME="${NETWORK_NAMES[$i]}"
    NUMBER="${NETWORK_NUMBERS[$i]}"
    METRICS="${NAME}_extraction_with_multimedia_attacks"
    NETWORK_W="./training_run_experiences_${SIZES[$i]}_configs_${CONFIGS[$i]}/${NAME}_${SIZES[$i]}_${SIZES[$i]}_${DATASETS[$i]}/PROD_0/network-snapshot-${NUMBER}.pkl"

    echo "==== Running no attack ===="
    python calc_metrics_after_network_attacks.py \
        --metrics="$METRICS" \
        --network="$NETWORK_W" \
        --attack_name="none"

#     echo "==== Running pruning attacks ===="
#     for PERCENT in 5 15 20 25 30 35 40 45 
#     do
#         echo "Running pruning with $PERCENT% sparsity..."
#         python calc_metrics_after_network_attacks.py \
#             --metrics="$METRICS" \
#             --network="$NETWORK_W" \
#             --attack_name="pruning" \
#             --attacks_parameters="$PERCENT"
#     done

#     echo "==== Running quantization attacks ===="
#     for BITS in  6 7 8 9 10 
#     do
#         echo "Running quantization with $BITS bits..."
#         python calc_metrics_after_network_attacks.py \
#             --metrics="$METRICS" \
#             --network="$NETWORK_W" \
#             --attack_name="quantization" \
#             --attacks_parameters="$BITS"
#     done

#     echo "==== Noise attacks ===="
#     for POWER in  1 3 6 9 12
#     do
#       echo "Running quantization with $POWER power..."
#       python calc_metrics_after_network_attacks.py \
#         --metrics="$METRICS" \
#         --network="$NETWORK_W" \
#         --attack_name="noise" \
#         --attacks_parameters="$POWER"
#     done

done

echo " ===> Evaluation completed successfully on V100 <=== "
#!/bin/bash

# =================================================================
# Training script for fine-tuning StyleGAN2-ADA with UCHI  method on Jean Zay (V100)
# =================================================================

#SBATCH -A lfb@v100                 # Project/account
#SBATCH --partition=gpu_p2          # V100 GPU partition
#SBATCH --qos=qos_gpu-t3            # GPU QoS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --cpus-per-task=10          # 10 CPU cores
#SBATCH --hint=nomultithread        # Physical cores only
#SBATCH --time=18:00:00             # Max walltime
#SBATCH --job-name=sg2_UCHI_V100     # Job name
#SBATCH --output=UCHI_256_afhq-%j.out         # Log file: sg2_UCHI_v100-<jobid>.out

# -----------------------------
# 1. Load environment (PyTorch)
# -----------------------------
set -eo pipefail
module purge
module load pytorch-gpu/py3/2.0.0
set -u 
export PYTHONUNBUFFERED=1           

# -----------------------------
# 2. Paths and config 
# -----------------------------

CODEDIR="$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF"
SNAP=1                              
OUTDIR="$CODEDIR/training_run_experiences/UCHI_256_256_afhq"
DATA_ZIP="$CODEDIR/datasets/afhq256x256.zip"
RESUME="$CODEDIR/vanilla_weights/vanilla_afhq_256x256_e1600.pkl"
CFG="watermarking"
PATH_CONF="$CODEDIR/configs/watermarking_dict_conf_UCHI.json"
GPUS=1
METRICS="UCHI_extraction,fid50k_full"

# --- SEPECIFIC ACCORDING THE ARCHITECTURE: CUDA arch for V100 (sm_70)
export TORCH_CUDA_ARCH_LIST="7.0"

# -----------------------------
# 3. SCRATCH DISQUE
# -----------------------------
cp -f "$DATA_ZIP" "$JOBSCRATCH/"
DATA_LOCAL="$JOBSCRATCH/$(basename "$DATA_ZIP")"
ls -lh "$DATA_LOCAL"

# -----------------------------
# 4. Run training
# -----------------------------
cd "$CODEDIR"

# Robust CUDA JIT toolchain selection
export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDAHOSTCXX="$(which g++)"

# Job-local CUDA extensions cache to avoid stale builds
export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR:-$JOBSCRATCH}/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"
rm -rf "$TORCH_EXTENSIONS_DIR"/*

python train.py \
    --snap="$SNAP" \
    --outdir="$OUTDIR" \
    --data="$DATA_LOCAL" \
    --resume="$RESUME" \
    --cfg="$CFG" \
    --gpus="$GPUS" \
    --metrics="$METRICS" \
    --aug=noaug \
    --water_config_path="$PATH_CONF"

echo " ===> Training completed successfully on V100 <=== "


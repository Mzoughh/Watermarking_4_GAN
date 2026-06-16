#!/bin/bash

# =================================================================
# Training script for fine-tuning StyleGAN2-ADA with T4G  method on Jean Zay (V100)
# =================================================================

#SBATCH -A lfb@v100                 # Project/account
#SBATCH --partition=gpu_p2          # V100 GPU partition
#SBATCH --qos=qos_gpu-t4                        # <-- LONG JOBS: up to 100h on V100 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --cpus-per-task=10          # 10 CPU cores
#SBATCH --hint=nomultithread        # Physical cores only
#SBATCH --time=30:30:00             # Max walltime
#SBATCH --job-name=spec_sg2_T4G_V100_ffhq_256      # Job name
#SBATCH --output=T4G_256_ffhq-%j.out         # Log file: sg2_T4G_v100-<jobid>.out

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
OUTDIR="$CODEDIR/training_run_experiences_256_configs_R2E/T4G_256_256_ffhq"
DATA_ZIP="$CODEDIR/datasets/ffhq256x256.zip"
RESUME="$CODEDIR/vanilla_weights/ffhq_256x256.pkl"
CFG="watermarking"
PATH_CONF="$CODEDIR/configs_R2E/configs_ffhq/configs_256/watermarking_dict_conf_T4G.json"
GPUS=1
METRICS="T4G_extraction,fid50k_full"

# --- SEPECIFIC ACCORDING THE ARCHITECTURE: CUDA arch for V100 (sm_70)
export TORCH_CUDA_ARCH_LIST="7.0"

# -----------------------------
# 3. SCRATCH DISQUE
# -----------------------------
# cp -f "$DATA_ZIP" "$JOBSCRATCH/"
# DATA_LOCAL="$JOBSCRATCH/$(basename "$DATA_ZIP")"
# ls -lh "$DATA_LOCAL"

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
    --data="$DATA_ZIP" \
    --resume="$RESUME" \
    --cfg="$CFG" \
    --gpus="$GPUS" \
    --metrics="$METRICS" \
    --aug=noaug \
    --water_config_path="$PATH_CONF"

echo " ===> Training completed successfully on V100 <=== "


#!/bin/bash

# =================================================================
# Training script for fine-tuning StyleGAN2-ADA with TONDI  method on Jean Zay (V100)
# =================================================================

#SBATCH -A lfb@v100                 # Project/account
#SBATCH --partition=gpu_p2          # V100 GPU partition
#SBATCH --qos=qos_gpu-t3            # GPU QoS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --cpus-per-task=10          # 10 CPU cores
#SBATCH --hint=nomultithread        # Physical cores only
#SBATCH --time=19:30:00             # Max walltime
#SBATCH --job-name=sg2_TONDI_V100_CelebA_256      # Job name
#SBATCH --output=TONDI_256_CelebA-%j.out         # Log file: sg2_TONDI_v100-<jobid>.out


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
OUTDIR="$CODEDIR/training_run_experiences_256_configs_Bb/TONDI_256_256_CelebA"
DATA_ZIP="$CODEDIR/datasets/CelebA256x256.zip"
RESUME="$CODEDIR/vanilla_weights/CelebA_256x256.pkl"
RESUME_CUSTOM="$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_128_configs_F/T4G_128_128_ffhq/00000-ffhq128x128-watermarking-noaug-resumecustom/network-snapshot-002200.pkl"
CFG="watermarking"
PATH_CONF="$CODEDIR/configs_Bb/configs_CelebA/configs_256/watermarking_dict_conf_TONDI.json"
GPUS=1
METRICS="TONDI_extraction,fid50k_full"

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
    --kimg=450 \
    --snap="$SNAP" \
    --outdir="$OUTDIR" \
    --data="$DATA_ZIP" \
    --resume="$RESUME_CUSTOM" \
    --cfg="$CFG" \
    --gpus="$GPUS" \
    --metrics="$METRICS" \
    --aug=noaug \
    --water_config_path="$PATH_CONF"

echo " ===> Training completed successfully on V100 <=== "


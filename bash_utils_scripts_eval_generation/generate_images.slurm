#!/bin/bash
#
# Image generation script with StyleGAN2 on Jean Zay (V40)
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

# -----------------------------
# Paths (ADAPT)
# -----------------------------
CODEDIR="$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF"

# -----------------------------
# Réseaux à évaluer (ADAPT)
# Format (champs optionnels après le chemin):
#   "Nom|/chemin/vers/network.pkl|OUT_TAG|SEEDS|TRUNC|NOISE_MODE|CLASS_IDX|TRIGGER"
# Exemples:
#   "Vanilla|.../CelebA_256x256.pkl"                        -> tout par défaut
#   "Vanilla|.../CelebA_256x256.pkl|UW|0-99|1.0|none|"      -> overrides
#   "Cond|.../cond.pkl|W|88|0.7|const|12"                   -> class conditionnelle
#   "T4G|.../network.pkl|UW|0-9|1.0|none||1"                -> trigger activé
# -----------------------------

SEEDS_T=7

NETWORK_CONFIGS=(
#     "Vanilla|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/vanilla_weights/CelebA_256x256.pkl|UW|4|1.0|none||0"
#     "T4G|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_R2E/T4G_256_256_CelebA/PROD_0/network-snapshot-002500.pkl|UW|4|1.0|none||0"
#     "T4G|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_R2E/T4G_256_256_CelebA/PROD_0/network-snapshot-002500.pkl|W|4|1.0|none||1"
#     "TONDI|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_C/TONDI_256_256_CelebA/PROD_0/network-snapshot-001775.pkl|W|4|1.0|none||0"
#     "IPR|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_Hj/IPR_256_256_CelebA/PROD_0/network-snapshot-001325.pkl|UW|4|1.0|none||0"
#     "T4G|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_R2E/T4G_256_256_CelebA/PROD_0/network-snapshot-002500.pkl|W|4|1.0|none||1"
# )
    "Vanilla|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/vanilla_weights/CelebA_256x256.pkl|UW|${SEEDS_T}|1.0|none||0"
    "OURS|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_R2E/T4G_256_256_CelebA/PROD_0/network-snapshot-002500.pkl|UW|${SEEDS_T}|1.0|none||0"
    "OURS|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_R2E/T4G_256_256_CelebA/PROD_0/network-snapshot-002500.pkl|W|${SEEDS_T}|1.0|none||1"
    "OURSC|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_R2EC/T4G_256_256_CelebA/PROD_0/network-snapshot-000900.pkl|UW|${SEEDS_T}|1.0|none||0"
    "OURSC|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_R2EC/T4G_256_256_CelebA/PROD_0/network-snapshot-000900.pkl|W|${SEEDS_T}|1.0|none||1"
#     "TONDI|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_C/TONDI_256_256_ffhq/PROD_0/network-snapshot-001975.pkl|W|${SEEDS_T}|1.0|none||0"
#     "IPR|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_Hk/IPR_256_256_ffhq/PROD_0/network-snapshot-001300.pkl|UW|${SEEDS_T}|1.0|none||0"
#     "IPR|$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences_256_configs_Hk/IPR_256_256_ffhq/PROD_0/network-snapshot-001300.pkl|W|${SEEDS_T}|1.0|none||1"
# 
)

# Racine de sortie (un sous-dossier par réseau sera créé)
OUTROOT="generated_image_W/Mode"

DEFAULT_OUT_TAG="UW"
DEFAULT_SEEDS="88"
DEFAULT_NOISE_MODE="none"
DEFAULT_TRUNC="1.0"
DEFAULT_CLASS_IDX=""
DEFAULT_TRIGGER="0"

# V40 => sm_70
export TORCH_CUDA_ARCH_LIST="7.0"
export CC=gcc
export CXX=g++

cd "$CODEDIR"

echo "=== Génération d'images StyleGAN2 sur Jean Zay ==="
echo "Code:           $CODEDIR"
echo "Sortie racine:  $OUTROOT"
echo "Graines (défaut):     $DEFAULT_SEEDS"
echo "Mode bruit (défaut):  $DEFAULT_NOISE_MODE"
echo "Troncature (défaut):  $DEFAULT_TRUNC"
echo "Mode sortie (défaut): $DEFAULT_OUT_TAG"
if [[ -n "$DEFAULT_CLASS_IDX" ]]; then
    echo "Classe (défaut):      $DEFAULT_CLASS_IDX"
fi
echo "================================="

if [[ ${#NETWORK_CONFIGS[@]} -eq 0 ]]; then
    echo "ERREUR: NETWORK_CONFIGS est vide. Ajoutez au moins un réseau."
    exit 1
fi

for cfg in "${NETWORK_CONFIGS[@]}"; do
    IFS='|' read -r NAME NETWORK OUT_TAG SEEDS TRUNC NOISE_MODE CLASS_IDX TRIGGER <<< "$cfg"

    OUT_TAG="${OUT_TAG:-$DEFAULT_OUT_TAG}"
    SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
    TRUNC="${TRUNC:-$DEFAULT_TRUNC}"
    NOISE_MODE="${NOISE_MODE:-$DEFAULT_NOISE_MODE}"
    CLASS_IDX="${CLASS_IDX:-$DEFAULT_CLASS_IDX}"
    TRIGGER="${TRIGGER:-$DEFAULT_TRIGGER}"

    OUTDIR="$OUTROOT/${NAME}_${OUT_TAG}"

    echo ""
    echo "--- Réseau: $NAME ---"
    echo "Modèle:         $NETWORK"
    echo "Dossier sortie: $OUTDIR"
    echo "Graines:        $SEEDS"
    echo "Mode bruit:     $NOISE_MODE"
    echo "Troncature:     $TRUNC"
    echo "Trigger:        $TRIGGER"
    if [[ -n "$CLASS_IDX" ]]; then
        echo "Classe:         $CLASS_IDX"
    fi

    if [[ ! -f "$NETWORK" ]]; then
        echo "ERREUR: fichier réseau introuvable: $NETWORK"
        exit 1
    fi

    mkdir -p "$OUTDIR"

    GEN_ARGS=(
        --network="$NETWORK"
        --outdir="$OUTDIR"
        --seeds="$SEEDS"
        --noise-mode="$NOISE_MODE"
        --trunc="$TRUNC"
    )

    if [[ -n "$CLASS_IDX" ]]; then
        GEN_ARGS+=(--class="$CLASS_IDX")
    fi

    case "${TRIGGER,,}" in
        1|true|yes|y|on)
            GEN_ARGS+=(--trigger)
            ;;
    esac

    python generate.py "${GEN_ARGS[@]}"
done

echo " ===> Génération terminée pour tous les réseaux <=== "
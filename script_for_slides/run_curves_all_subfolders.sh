#!/bin/bash

# Script pour exécuter curves_training_plot.py sur tous les sous-dossiers
# Usage: ./run_curves_all_subfolders.sh <input_folder>

# Vérifier qu'un argument est fourni
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_folder>"
    echo "Example: $0 training_run_experiences"
    exit 1
fi

INPUT_FOLDER=$1

# Vérifier que le dossier existe
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Erreur: $INPUT_FOLDER n'est pas un répertoire valide"
    exit 1
fi

echo "Traitement récursif de tous les sous-dossiers dans: $INPUT_FOLDER"
echo "================================================"

# Utiliser find pour parcourir tous les sous-dossiers récursivement
# -mindepth 1 : exclut le dossier parent lui-même
# -type d : seulement les répertoires
while IFS= read -r -d '' subfolder; do
    echo ""
    echo "Processing: $subfolder"
    # Activer --no-normalize uniquement si 'T4G' apparaît dans le nom du dossier (chemin)
    if [[ "$subfolder" == *"T4G"* ]]; then
        python script_for_slides/curves_training_plot.py --no-normalize "$subfolder"
    else
        python script_for_slides/curves_training_plot.py "$subfolder"
    fi
    
    # Vérifier le code de retour
    if [ $? -eq 0 ]; then
        echo "✓ Succès pour $subfolder"
    else
        echo "✗ Échec pour $subfolder"
    fi
done < <(find "$INPUT_FOLDER" -mindepth 1 -type d -print0)

echo ""
echo "================================================"
echo "Traitement terminé!"

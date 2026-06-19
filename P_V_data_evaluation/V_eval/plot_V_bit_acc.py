import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_json_list(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Cas 1 : le JSON est directement une liste
    if isinstance(data, list):
        return data

    # Cas 2 : le JSON est un dictionnaire avec une seule clé
    if isinstance(data, dict):
        if len(data) == 1:
            return list(data.values())[0]
        else:
            raise ValueError(
                f"Le fichier {path} contient plusieurs clés. "
                "Le JSON doit être une liste ou un dictionnaire avec une seule liste."
            )

    raise ValueError(f"Format JSON non supporté dans {path}")


def main(mask_json, bitacc_json, output_prefix):
    # Chargement des données
    mask_values = load_json_list(mask_json)
    bit_acc_values = load_json_list(bitacc_json)

    # Vérification de la taille des listes
    if len(mask_values) != len(bit_acc_values):
        raise ValueError(
            f"Les deux listes n'ont pas la même taille : "
            f"{len(mask_values)} valeurs de masquage contre "
            f"{len(bit_acc_values)} valeurs de bit accuracy."
        )

    # Création du DataFrame
    df = pd.DataFrame({
        "Mask value": mask_values,
        "Bit accuracy": bit_acc_values
    })

    # Moyenne par valeur de masquage
    result = (
        df.groupby("Mask value", as_index=False)
        .agg(
            **{
                "Mean bit accuracy": ("Bit accuracy", "mean"),
                "Std bit accuracy": ("Bit accuracy", "std"),
                "Count": ("Bit accuracy", "count")
            }
        )
        .sort_values("Mask value")
    )

    # Arrondi pour affichage propre
    result_display = result.copy()
    result_display["Mean bit accuracy"] = result_display["Mean bit accuracy"].round(4)
    result_display["Std bit accuracy"] = result_display["Std bit accuracy"].round(4)

    # Sauvegarde CSV
    csv_path = f"{output_prefix}_mean_bit_accuracy_table.csv"
    result_display.to_csv(csv_path, index=False)

    # Création du tableau en image
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(result_display) + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=result_display.values,
        colLabels=result_display.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Style simple du tableau
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")

        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4C72B0")
        else:
            cell.set_facecolor("#F2F2F2" if row % 2 == 0 else "white")

    png_path = f"{output_prefix}_mean_bit_accuracy_table.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("CSV saved:", csv_path)
    print("Table image saved:", png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask_json",
        type=str,
        required=True,
        help="JSON file containing masking values"
    )

    parser.add_argument(
        "--bitacc_json",
        type=str,
        required=True,
        help="JSON file containing bit accuracy values"
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        default="masking_analysis",
        help="Prefix for saved files"
    )

    args = parser.parse_args()

    main(args.mask_json, args.bitacc_json, args.output_prefix)


# python plot_V_bit_acc.py \
# --mask_json mask_values.json \
# --bitacc_json bit_accuracy.json \
# --output_prefix masking_analysis
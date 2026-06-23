import json
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json_list(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if len(data) == 1:
            return list(data.values())[0]
        else:
            raise ValueError(
                f"{path} contient plusieurs clés. "
                "Le fichier doit être une liste ou un dictionnaire avec une seule liste."
            )

    raise ValueError(f"Format JSON non supporté pour {path}")


def min_position_distance(mask, target_mask):
    """
    Distance numérique minimale entre deux masques.

    Exemple :
    mask = [77, 366, 430]
    target = [78, 367, 426]

    On teste toutes les permutations possibles pour éviter
    que l'ordre des indices influence la distance.
    """
    mask = list(mask)
    target_mask = list(target_mask)

    best_dist = None

    for perm in itertools.permutations(mask):
        dist = sum(abs(a - b) for a, b in zip(perm, target_mask))

        if best_dist is None or dist < best_dist:
            best_dist = dist

    return best_dist


def exact_mask_distance(mask, target_mask):
    """
    Distance basée sur le nombre exact de positions communes.

    Si les 3 positions sont identiques : distance = 0
    Si 2 positions communes : distance = 1
    Si 1 position commune : distance = 2
    Si 0 position commune : distance = 3
    """
    mask_set = set(mask)
    target_set = set(target_mask)

    common = len(mask_set.intersection(target_set))
    distance = len(target_mask) - common

    return distance, common


def parse_threshold(value):
    if value is None:
        return None

    value = str(value).strip().lower()

    if value in ["no", "none", "false", "off"]:
        return None

    return float(value)


def main(idx_json, bitacc_json, target_mask, threshold, output_prefix, bins):
    # =========================
    # Load data
    # =========================
    idx_list = load_json_list(idx_json)
    bit_acc_list = load_json_list(bitacc_json)

    if len(idx_list) != len(bit_acc_list):
        raise ValueError(
            f"Les deux listes n'ont pas la même taille : "
            f"{len(idx_list)} masks vs {len(bit_acc_list)} bit accuracies."
        )

    rows = []

    for idx, bit_acc in zip(idx_list, bit_acc_list):
        bit_acc = float(bit_acc)

        exact_dist, common_positions = exact_mask_distance(idx, target_mask)
        pos_dist = min_position_distance(idx, target_mask)

        row = {
            "mask": idx,
            "bit_accuracy": bit_acc,
            "common_positions": common_positions,
            "exact_distance": exact_dist,
            "position_distance": pos_dist
        }

        if threshold is not None:
            row["above_threshold"] = bit_acc >= threshold

        rows.append(row)

    df = pd.DataFrame(rows)

    # =========================
    # Save detailed table
    # =========================
    detailed_csv = f"{output_prefix}_detailed_results.csv"
    df.to_csv(detailed_csv, index=False)

    # =========================
    # Summary table
    # =========================
    summary_agg = {
        "count": ("bit_accuracy", "count"),
        "mean_bit_accuracy": ("bit_accuracy", "mean"),
        "std_bit_accuracy": ("bit_accuracy", "std"),
        "min_bit_accuracy": ("bit_accuracy", "min"),
        "max_bit_accuracy": ("bit_accuracy", "max"),
        "mean_position_distance": ("position_distance", "mean")
    }

    if threshold is not None:
        summary_agg["high_ba_count"] = ("above_threshold", "sum")

    summary = (
        df.groupby("common_positions", as_index=False)
        .agg(**summary_agg)
        .sort_values("common_positions")
    )

    if threshold is not None:
        summary["high_ba_ratio"] = summary["high_ba_count"] / summary["count"]

    summary_display = summary.copy()

    for col in [
        "mean_bit_accuracy",
        "std_bit_accuracy",
        "min_bit_accuracy",
        "max_bit_accuracy",
        "mean_position_distance",
        "high_ba_ratio"
    ]:
        if col in summary_display.columns:
            summary_display[col] = summary_display[col].round(4)

    summary_csv = f"{output_prefix}_summary_table.csv"
    summary_display.to_csv(summary_csv, index=False)

    # =========================
    # Save summary table as PNG
    # =========================
    fig, ax = plt.subplots(figsize=(13, 0.55 * len(summary_display) + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=summary_display.values,
        colLabels=summary_display.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")

        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4C72B0")
        else:
            cell.set_facecolor("#F2F2F2" if row % 2 == 0 else "white")

    table_png = f"{output_prefix}_summary_table.png"
    plt.tight_layout()
    plt.savefig(table_png, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # Clean histogram
    # =========================
    bit_acc = df["bit_accuracy"].values
    position_dist = df["position_distance"].values
    common_pos = df["common_positions"].values

    counts, bin_edges = np.histogram(bit_acc, bins=bins, range=(0.4, 1.0))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    fig, ax = plt.subplots(figsize=(11, 5))

    bin_annotation_rows = []

    for i in range(len(counts)):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        center = bin_centers[i]
        count = counts[i]

        in_bin = (bit_acc >= left) & (bit_acc < right)

        if i == len(counts) - 1:
            in_bin = (bit_acc >= left) & (bit_acc <= right)

        ax.bar(
            center,
            count,
            width=bin_widths[i] * 0.9,
            color="#4C72B0",
            edgecolor="black",
            alpha=0.85
        )

        if count > 0:
            bin_annotation_rows.append([
                f"{left:.2f}-{right:.2f}",
                int(count),
                f"{np.mean(position_dist[in_bin]):.1f}",
                f"{np.mean(common_pos[in_bin]):.2f}"
            ])

    ax.set_xlabel("Bit accuracy")
    ax.set_ylabel("Number of masks")
    ax.set_title("Bit accuracy distribution")
    ax.set_xlim(0.4, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)

    hist_png = f"{output_prefix}_bit_accuracy_histogram.png"
    plt.tight_layout()
    plt.savefig(hist_png, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # Save histogram bin table separately
    # =========================
    bin_table_columns = ["BA bin", "Count", "Mean dist", "Mean common"]

    def save_bin_table(rows, csv_path, png_path, header_color):
        table_df = pd.DataFrame(rows, columns=bin_table_columns)
        table_df.to_csv(csv_path, index=False)

        if len(table_df) == 0:
            table_df = pd.DataFrame(
                [["No bin", 0, "-", "-"]],
                columns=bin_table_columns
            )

        fig, ax = plt.subplots(figsize=(8, 0.45 * len(table_df) + 1.2))
        ax.axis("off")

        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")

            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor("#F2F2F2" if row % 2 == 0 else "white")

        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()

    bins_csv = f"{output_prefix}_bins_table.csv"
    bins_png = f"{output_prefix}_bins_table.png"

    save_bin_table(
        bin_annotation_rows,
        bins_csv,
        bins_png,
        "#4C72B0"
    )

    # =========================
    # Scatter plot
    # Color by number of common positions
    # =========================
    fig, ax = plt.subplots(figsize=(8, 5))

    colors_map = {
        0: "#0072B2",  # blue
        1: "#009E73",  # green
        2: "#D55E00",  # red
        3: "#F0E442",  # yellow
    }

    markers_map = {
        0: "o",
        1: "s",
        2: "^",
        3: "D",
    }

    labels_map = {
        0: "0 common position",
        1: "1 common position",
        2: "2 common positions",
        3: "3 common positions / target mask",
    }

    for n_common in sorted(df["common_positions"].unique()):
        sub = df[df["common_positions"] == n_common]

        ax.scatter(
            sub["position_distance"],
            sub["bit_accuracy"],
            color=colors_map[int(n_common)],
            marker=markers_map[int(n_common)],
            label=labels_map[int(n_common)],
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
            s=45
        )

    if threshold is not None:
        ax.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold = {threshold}"
        )

    ax.set_xlabel("Numerical distance to target mask")
    ax.set_ylabel("Bit accuracy")
    ax.set_title("Bit accuracy according to distance and common positions")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    scatter_png = f"{output_prefix}_distance_vs_bit_accuracy.png"
    plt.tight_layout()
    plt.savefig(scatter_png, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # Scatter plot with exact common positions only
    # =========================
    fig, ax = plt.subplots(figsize=(7, 5))

    for n_common in sorted(df["common_positions"].unique()):
        sub = df[df["common_positions"] == n_common]

        jitter = np.random.normal(0, 0.04, size=len(sub))

        ax.scatter(
            sub["common_positions"] + jitter,
            sub["bit_accuracy"],
            color=colors_map[int(n_common)],
            marker=markers_map[int(n_common)],
            label=labels_map[int(n_common)],
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
            s=45
        )

    if threshold is not None:
        ax.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold = {threshold}"
        )

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Number of common positions with target mask")
    ax.set_ylabel("Bit accuracy")
    ax.set_title("Bit accuracy vs exact mask overlap")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    overlap_png = f"{output_prefix}_common_positions_vs_bit_accuracy.png"
    plt.tight_layout()
    plt.savefig(overlap_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved files:")
    print(detailed_csv)
    print(summary_csv)
    print(table_png)
    print(hist_png)
    print(bins_csv)
    print(bins_png)
    print(scatter_png)
    print(overlap_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--idx_json",
        type=str,
        required=True,
        help="JSON file containing the list of masks / indices"
    )

    parser.add_argument(
        "--bitacc_json",
        type=str,
        required=True,
        help="JSON file containing the list of bit accuracies"
    )

    parser.add_argument(
        "--target_mask",
        type=str,
        default="78,367,426",
        help="Target mask positions, example: 78,367,426"
    )

    parser.add_argument(
        "--threshold",
        dest="threshold",
        type=str,
        default="0.65",
        help="Bit accuracy threshold. Use 'no' to remove threshold lines and threshold-based summary columns."
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for the histogram"
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        default="mask_position_analysis",
        help="Prefix for saved files"
    )

    args = parser.parse_args()

    target_mask = [int(x) for x in args.target_mask.split(",")]

    main(
        idx_json=args.idx_json,
        bitacc_json=args.bitacc_json,
        target_mask=target_mask,
        threshold=parse_threshold(args.threshold),
        output_prefix=args.output_prefix,
        bins=args.bins
    )


# python plot_mask_position_analysis.py \
#     --idx_json idx_list_total_FP.json \
#     --bitacc_json bit_acc_list_total_FP.json \
#     --target_mask 78,367,426 \
#     --threshold no \
#     --output_prefix position_analysis
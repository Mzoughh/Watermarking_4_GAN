import os
import json
import argparse
import csv
import re
import matplotlib.pyplot as plt


# ---------- Config ----------
IGNORED_METRICS = {"uchida_hamming_dist"}

# Types d'attaques connus (pour parser les noms de fichiers)
KNOWN_ATTACKS = {
    "none",
    "pruning",
    "quantization",
    "noise",
}

# Normalisation des noms de métriques (colonnes cohérentes)
METRIC_ALIASES = {
    "bit_acc": "uchida_bit_acc",
    "bit_accuracy": "uchida_bit_acc",
    "uchida_bit_acc": "uchida_bit_acc",
    "fid": "fid50k_full",
}


def normalize_metric_key(metric_key: str) -> str:
    return METRIC_ALIASES.get(metric_key, metric_key)


# ---------- Helpers labels ----------

def latex_escape(s: str) -> str:
    """
    Échappe les caractères spéciaux LaTeX.
    IMPORTANT: remplacer "\" en premier pour éviter re-escape.
    """
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for orig, repl in replacements:
        s = s.replace(orig, repl)
    return s


def metric_latex_label(metric: str) -> str:
    mapping = {
        "fid50k_full": r"FID$_{50k}$ ($\downarrow$)",
        "uchida_bit_acc": r"Bit acc. ($\uparrow$)",
    }
    if metric in mapping:
        return mapping[metric]
    return latex_escape(metric)


def metric_png_label(metric: str) -> str:
    """
    Label lisible pour PNG (matplotlib) : PAS de LaTeX escaping.
    """
    mapping = {
        "fid50k_full": "FID50k (↓)",
        "uchida_bit_acc": "Bit acc. (↑)",
    }
    if metric in mapping:
        return mapping[metric]
    return metric.replace("_", " ")


# ---------- Parsing du nom de fichier ----------

def parse_attack_from_filename(fname: str):
    """
    Exemples:
      metric-TONDI_extraction-none-1.jsonl     -> attack_type=none,         attack_name=none-1
      metric-TONDI_extraction-pruning-5.jsonl  -> attack_type=pruning,      attack_name=pruning-5
      metric-XYZ-quantization-8.jsonl          -> attack_type=quantization, attack_name=quantization-8

    Robuste si le "metric name" contient des '-': on prend le DERNIER token
    qui matche un KNOWN_ATTACKS.
    """
    base = os.path.basename(fname)
    stem, _ = os.path.splitext(base)

    if not stem.startswith("metric-"):
        return "unknown", "unknown"

    rest = stem[len("metric-"):]
    parts = rest.split("-")

    idxs = [i for i, p in enumerate(parts) if p in KNOWN_ATTACKS]
    if not idxs:
        return "unknown", "unknown"

    i = idxs[-1]
    attack_type = parts[i]
    attack_name = "-".join(parts[i:])  # ex: none-1, pruning-5, quantization-8...
    return attack_type, attack_name


# ---------- Lecture JSON / JSONL ----------

def read_metric_file(path: str):
    """
    Supporte .json (1 objet) et .jsonl (lignes JSON, on prend la 1ère valide).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)

    if ext == ".jsonl":
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                return json.loads(line)
        raise json.JSONDecodeError("Empty jsonl", doc="", pos=0)

    # fallback
    with open(path, "r") as f:
        return json.load(f)


# ---------- Chargement des métriques ----------

def load_evaluation_metrics_per_type(directory: str):
    files = [
        f for f in os.listdir(directory)
        if f.startswith("metric-")
        and os.path.isfile(os.path.join(directory, f))
        and (f.endswith(".json") or f.endswith(".jsonl"))
    ]

    if not files:
        print(f"Aucun fichier 'metric-*.(json|jsonl)' trouvé dans {directory}")
        return {}, {}

    tables_by_type = {}   # attack_type -> { attack_name -> { metric_key: value } }
    metrics_by_type = {}  # attack_type -> set(metric_key)

    for fname in sorted(files):
        path = os.path.join(directory, fname)

        try:
            data = read_metric_file(path)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON/JSONL invalide dans {fname} : {e}")
            continue

        results = data.get("results", {})
        if not isinstance(results, dict) or not results:
            print(f"[WARN] Champ 'results' vide ou invalide dans : {fname}")
            continue

        attack_type, attack_name = parse_attack_from_filename(fname)

        type_table = tables_by_type.setdefault(attack_type, {})
        type_metrics_set = metrics_by_type.setdefault(attack_type, set())
        attack_metrics = type_table.setdefault(attack_name, {})

        for metric_key, value in results.items():
            canon = normalize_metric_key(metric_key)
            type_metrics_set.add(canon)
            attack_metrics[canon] = value

    return tables_by_type, metrics_by_type


# ---------- Sauvegarde CSV / LaTeX / PNG ----------

def save_table_csv(attacks, metrics, table_for_type, output_path: str):
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([""] + metrics)
        for attack in attacks:
            row = [attack] + [table_for_type.get(attack, {}).get(m, "") for m in metrics]
            writer.writerow(row)
    print(f"CSV : {output_path}")


def save_table_latex(attacks, metrics, table_for_type, output_path: str, attack_type: str):
    with open(output_path, "w") as f:
        f.write("%% Auto-generated evaluation table (attack type: %s)\n" % attack_type)
        f.write("\\begin{table}[ht]\n\\centering\n")

        col_spec = "l" + "c" * len(metrics)
        f.write("\\begin{tabular}{%s}\n" % col_spec)
        f.write("\\hline\n")

        header_cells = [""] + [metric_latex_label(m) for m in metrics]
        f.write(" & ".join(header_cells) + " \\\\\n")
        f.write("\\hline\n")

        for attack in attacks:
            row_cells = [latex_escape(attack)]
            for metric in metrics:
                val = table_for_type.get(attack, {}).get(metric, "")
                if isinstance(val, float):
                    row_cells.append(f"{val:.4f}")
                elif isinstance(val, int):
                    row_cells.append(str(val))
                else:
                    row_cells.append(latex_escape(str(val)))
            f.write(" & ".join(row_cells) + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\caption{Évaluation des métriques pour l'attaque \\texttt{%s}.}\n" % latex_escape(attack_type))
        f.write("\\label{tab:%s_metrics}\n" % attack_type.replace(" ", "_"))
        f.write("\\end{tabular}\n\\end{table}\n")

    print(f"LaTeX : {output_path}")


def save_table_png(attacks, metrics, table_for_type, output_path: str, attack_type: str):
    col_labels = [metric_png_label(m) for m in metrics]

    cell_text = []
    for attack in attacks:
        row = []
        for metric in metrics:
            val = table_for_type.get(attack, {}).get(metric, "")
            row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        cell_text.append(row)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 9

    fig_width = max(6, 1.2 * len(metrics))
    fig_height = max(2.5, 0.5 * (len(attacks) + 1))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table_obj = ax.table(
        cellText=cell_text,
        rowLabels=attacks,
        colLabels=col_labels,
        loc="center"
    )

    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(9)
    table_obj.scale(1.2, 1.2)

    # ax.set_title(f"Attack type: {attack_type}", pad=10)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"PNG : {output_path}")


# ---------- Tri attaques (avec quantization décroissant) ----------

def numeric_tuple_from_name(name: str):
    nums = re.findall(r"\d+\.?\d*", name)
    return tuple(float(x) for x in nums) if nums else ()


def attack_sort_key_asc(name: str):
    t = numeric_tuple_from_name(name)
    if t:
        return (0, t, name)
    return (1, (float("inf"),), name)


def attack_sort_key_quant_desc(name: str):
    t = numeric_tuple_from_name(name)
    if t:
        t_inv = tuple(-x for x in t)  # décroissant
        return (0, t_inv, name)
    return (1, (float("inf"),), name)


def sort_attacks(attacks, attack_type: str, baseline_label: str):
    base = []
    if baseline_label and baseline_label in attacks:
        base = [baseline_label]
        attacks = [a for a in attacks if a != baseline_label]

    if attack_type == "quantization":
        attacks = sorted(attacks, key=attack_sort_key_quant_desc)
    else:
        attacks = sorted(attacks, key=attack_sort_key_asc)

    return base + attacks


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Dossier contenant les fichiers metric-*.json/.jsonl")
    args = parser.parse_args()

    tables_by_type, metrics_by_type = load_evaluation_metrics_per_type(args.directory)
    if not tables_by_type:
        print("Rien à générer.")
        return

    # ---- Baseline = la première ligne (tri asc) du type d'attaque 'none' (ex: none-1) ----
    baseline_label = None
    baseline_metrics = None

    if "none" in tables_by_type and tables_by_type["none"]:
        baseline_candidates = sorted(list(tables_by_type["none"].keys()), key=attack_sort_key_asc)
        baseline_label = baseline_candidates[0]  # ex: "none-1"
        baseline_metrics = tables_by_type["none"][baseline_label]

    # ---- Injecte baseline dans toutes les tables ----
    if baseline_label and baseline_metrics:
        for attack_type, table_for_type in tables_by_type.items():
            if baseline_label not in table_for_type:
                table_for_type[baseline_label] = baseline_metrics.copy()
                metrics_by_type.setdefault(attack_type, set()).update(baseline_metrics.keys())

    # ---- Génération ----
    for attack_type, table_for_type in tables_by_type.items():
        attacks = sort_attacks(list(table_for_type.keys()), attack_type, baseline_label)

        all_metrics = sorted(metrics_by_type.get(attack_type, []))
        metrics = [m for m in all_metrics if m not in IGNORED_METRICS]
        if not attacks or not metrics:
            continue

        csv_path = os.path.join(args.directory, f"evaluation_metrics_table_{attack_type}.csv")
        tex_path = os.path.join(args.directory, f"evaluation_metrics_table_{attack_type}.tex")
        png_path = os.path.join(args.directory, f"evaluation_metrics_table_{attack_type}.png")

        save_table_csv(attacks, metrics, table_for_type, csv_path)
        save_table_latex(attacks, metrics, table_for_type, tex_path, attack_type)
        save_table_png(attacks, metrics, table_for_type, png_path, attack_type)


if __name__ == "__main__":
    main()

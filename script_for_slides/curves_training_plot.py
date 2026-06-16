import os
import json
import argparse
import matplotlib.pyplot as plt
import re
from itertools import cycle
import math

IGNORED_METRICS = {"uchida_hamming_dist"}

# Palette de couleurs agréable (matplotlib "tab" colors)
BASE_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:brown",
    "tab:purple",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

# Mapping des noms internes de métriques T4G vers les labels affichés
T4G_LABEL_MAP = {
    "fid50k_full": "FID",
    "perceptual_metric": "SSIM (on/off)",
    "bit_accs_avc": "Bit-Acc (on)",
    "bit_accs_avg": "Bit-Acc (on)",
    "bit_accs_avg_vanilla": "Bit-Acc (off)",
}

# Métriques T4G qui vont sur l'axe gauche (FID)
T4G_LEFT_METRICS = {"fid50k_full"}

# Couleurs fixes pour chaque métrique T4G
T4G_COLOR_MAP = {
    "fid50k_full": "tab:blue",
    "perceptual_metric": "tab:green",
    "bit_accs_avc": "tab:orange",
    "bit_accs_avg": "tab:orange",
    "bit_accs_avg_vanilla": "tab:brown",
}


def is_t4g_experiment(metrics_items):
    """Renvoie True si les métriques détectées correspondent à une expérience T4G."""
    names = {n for n, _ in metrics_items}
    return bool(names & {"bit_accs_avc", "bit_accs_avg", "bit_accs_avg_vanilla", "perceptual_metric"})
COLOR_CYCLE = cycle(BASE_COLORS)


def parse_step_from_snapshot(snapshot_pkl: str, fallback: int) -> int:
    """
    Extrait le numéro de step à partir de 'network-snapshot-000010.pkl'.
    Si impossible, renvoie fallback.
    """
    if not snapshot_pkl:
        return fallback

    m = re.search(r"(\d+)(?=\.pkl$)", snapshot_pkl)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return fallback


def load_all_metrics(directory: str):
    """
    Parcourt tous les fichiers commençant par 'metric-' dans le dossier,
    lit chaque ligne comme un JSON, et agrège toutes les métriques trouvées
    dans un dict : { metric_name: [(step, value), ...], ... }.
    Ignore les métriques listées dans IGNORED_METRICS.
    """
    metric_files = sorted(
        f for f in os.listdir(directory)
        if f.startswith("metric-") and os.path.isfile(os.path.join(directory, f))
    )

    if not metric_files:
        print(f"Aucun fichier 'metric-*' trouvé dans {directory}")
        return {}

    all_metrics = {}  # metric_name -> list of (continuous_step, value)

    for fname in metric_files:
        path = os.path.join(directory, fname)
        print(f"Lecture du fichier de métriques : {path}")

        # Permet de recoller automatiquement des finetunings successifs
        # dans un même fichier metric-*.jsonl quand snapshot_pkl repart à 000000.
        segment_offset = 0
        prev_raw_step = None
        prev_continuous_step = None
        last_positive_delta = 1

        with open(path, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Ligne ignorée dans {fname} (JSON invalide) : {e}")
                    continue

                snapshot_pkl = data.get("snapshot_pkl", "")
                raw_step = parse_step_from_snapshot(snapshot_pkl, idx)

                if prev_raw_step is not None:
                    raw_delta = raw_step - prev_raw_step

                    if raw_delta > 0:
                        last_positive_delta = raw_delta
                    elif raw_delta <= 0 and prev_continuous_step is not None:
                        segment_offset = prev_continuous_step + max(1, last_positive_delta)

                step = raw_step + segment_offset
                prev_raw_step = raw_step
                prev_continuous_step = step

                results = data.get("results", {})
                for metric_name, value in results.items():
                    if metric_name in IGNORED_METRICS:
                        continue
                    all_metrics.setdefault(metric_name, []).append((step, value))

    return all_metrics


def _build_title(metric_names):
    metric_names = list(metric_names)
    n = len(metric_names)
    if n == 1:
        return f"Évolution de {metric_names[0]} pendant l'entraînement"
    elif n == 2:
        return f"Évolution de {metric_names[0]} et {metric_names[1]} pendant l'entraînement"
    else:
        return f"Évolution de {n} métriques pendant l'entraînement"


def plot_all_metrics(all_metrics: dict, directory: str, normalize: bool = True):
    """
    Trace les métriques trouvées.

    - Si 0 métrique : ne fait rien.
    - Si 1 métrique  : un seul axe Y, valeurs brutes.
    - Si 2 métriques : double axe Y (twinx), valeurs brutes, couleurs différentes.
    - Si >2 métriques:
        * si normalize=True  : normalisation par métrique dans [0, 1] sur un seul axe.
        * si normalize=False : seulement 2 axes Y (gauche + droite),
                               métriques réparties automatiquement par ordre de grandeur.
                               Le label de chaque axe liste les métriques qui y sont associées.
    """
    metrics_items = [(name, points) for name, points in all_metrics.items()
                     if name not in IGNORED_METRICS]

    if not metrics_items:
        print("Aucune métrique à tracer.")
        return

    # Tri par nom pour stabilité
    metrics_items.sort(key=lambda x: x[0])
    n_metrics = len(metrics_items)

    # Reset du cycle de couleurs à chaque figure
    global COLOR_CYCLE
    COLOR_CYCLE = cycle(BASE_COLORS)

    # --- Cas 1 : une seule métrique ---
    if n_metrics == 1:
        metric_name, points = metrics_items[0]
        points = sorted(points, key=lambda x: x[0])
        points = points[::2]
        steps = [p[0] for p in points]
        values = [p[1] for p in points]

        color = next(COLOR_CYCLE)

        plt.figure()
        plt.plot(steps, values, marker="o", label=metric_name, color=color)
        plt.xlabel("Training step")
        plt.ylabel(metric_name, color=color)
        plt.tick_params(axis='y', labelcolor=color)
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.tight_layout()

    # --- Cas 2 : double axe Y ---
    elif n_metrics == 2:
        (m1, pts1), (m2, pts2) = metrics_items

        pts1 = sorted(pts1, key=lambda x: x[0])
        pts2 = sorted(pts2, key=lambda x: x[0])
        pts1 = pts1[::2]
        pts2 = pts2[::2]
        steps1 = [p[0] for p in pts1]
        values1 = [p[1] for p in pts1]
        steps2 = [p[0] for p in pts2]
        values2 = [p[1] for p in pts2]

        c1 = next(COLOR_CYCLE)
        c2 = next(COLOR_CYCLE)

        fig, ax1 = plt.subplots()

        l1, = ax1.plot(steps1, values1, marker="o", label=m1, color=c1)
        ax1.set_xlabel("k-imgs")
        ax1.set_ylabel(m1, color=c1)
        ax1.tick_params(axis='y', labelcolor=c1)
        ax1.spines["left"].set_color(c1)
        ax1.grid(True)

        ax2 = ax1.twinx()
        l2, = ax2.plot(steps2, values2, marker="s", linestyle="--", label=m2, color=c2)
        ax2.set_ylabel(m2, color=c2)
        ax2.tick_params(axis='y', labelcolor=c2)
        ax2.spines["right"].set_color(c2)

        ax1.legend([l1, l2], [m1, m2], loc="lower right")
        fig.tight_layout()

    # --- Cas > 2 métriques ---
    else:
        # (A) Normalisé => 1 seul axe
        if normalize:
            plt.figure()

            for metric_name, points in metrics_items:
                points = sorted(points, key=lambda x: x[0])
                points = points[::2]
                steps = [p[0] for p in points]
                values = [p[1] for p in points]

                vmin = min(values)
                vmax = max(values)
                if vmax > vmin:
                    plot_values = [(v - vmin) / (vmax - vmin) for v in values]
                else:
                    plot_values = [0.5] * len(values)

                color = next(COLOR_CYCLE)
                plt.plot(steps, plot_values, marker="o", label=metric_name, color=color)

            plt.xlabel("Training step")
            plt.ylabel("Valeur normalisée de la métrique")
            plt.grid(True)
            plt.legend(loc="lower right")
            plt.tight_layout()

        # (B) Brut => seulement 2 axes Y (gauche + droite)
        else:
            def metric_scale(points):
                vals = [v for _, v in points]
                vmin, vmax = min(vals), max(vals)
                r = vmax - vmin
                return r if r > 0 else max(1e-12, abs(vmax), abs(vmin), 1e-12)

            def axis_label(side_name: str, group):
                names = [n for n, _ in group]
                if not names:
                    return f"({side_name})"
                max_chars = 60
                s = ", ".join(names)
                if len(s) > max_chars:
                    if len(names) > 4:
                        s = ", ".join(names[:2]) + ", …, " + ", ".join(names[-2:])
                    else:
                        s = s[:max_chars - 1] + "…"
                return f"{s}"

            # Liste enrichie (name, sorted_points, scale)
            items = []
            for name, pts in metrics_items:
                pts_sorted = sorted(pts, key=lambda x: x[0])
                scale = metric_scale(pts_sorted)
                items.append((name, pts_sorted, scale))

            # Trier par ordre de grandeur (log10)
            items.sort(key=lambda t: math.log10(t[2]), reverse=True)

            left_group = []
            right_group = []
            left_center = None
            right_center = None

            # Répartition automatique sur 2 axes selon proximité d'échelle
            for name, pts, scale in items:
                s = math.log10(scale)
                if left_center is None:
                    left_group.append((name, pts))
                    left_center = s
                elif right_center is None:
                    right_group.append((name, pts))
                    right_center = s
                else:
                    if abs(s - left_center) <= abs(s - right_center):
                        left_group.append((name, pts))
                        left_center = (left_center * (len(left_group) - 1) + s) / len(left_group)
                    else:
                        right_group.append((name, pts))
                        right_center = (right_center * (len(right_group) - 1) + s) / len(right_group)

            fig, axL = plt.subplots()
            axR = axL.twinx()

            t4g = is_t4g_experiment(metrics_items)

            # Pour T4G : répartition forcée FID à gauche, reste à droite
            if t4g:
                left_group = [(n, p) for n, p in metrics_items if n in T4G_LEFT_METRICS]
                right_group = [(n, p) for n, p in metrics_items if n not in T4G_LEFT_METRICS]

            axL.set_xlabel("k-imgs" if t4g else "Training step", fontsize=12)
            axL.grid(True)
            axL.tick_params(axis='x', labelsize=12)
            axL.tick_params(axis='y', labelsize=12)
            axR.tick_params(axis='y', labelsize=12)

            lines = []
            labels = []


            # Axe gauche
            for metric_name, points in left_group:
                points = points[::2]
                steps = [p[0] for p in points]
                values = [p[1] for p in points]
                color = T4G_COLOR_MAP.get(metric_name, next(COLOR_CYCLE)) if t4g else next(COLOR_CYCLE)
                # FID en carré et label (off)
                if metric_name == "fid50k_full":
                    marker = "s"  # carré
                    display_label = "FID (off)"
                else:
                    marker = "o"
                    display_label = T4G_LABEL_MAP.get(metric_name, metric_name) if t4g else metric_name
                alpha = 0.45 if (t4g and metric_name in T4G_LEFT_METRICS) else 1.0

                line, = axL.plot(
                    steps, values,
                    marker=marker, linestyle="-",
                    label=display_label, color=color, alpha=alpha
                )
                lines.append(line)
                labels.append(display_label)

            # Axe droit
            bit_acc_on_done = False
            for metric_name, points in right_group:
                points = points[::2]
                steps = [p[0] for p in points]
                values = [p[1] for p in points]
                color = T4G_COLOR_MAP.get(metric_name, next(COLOR_CYCLE)) if t4g else next(COLOR_CYCLE)
                # Bit-Acc (on) en rond et en dash, une seule fois
                if not bit_acc_on_done and metric_name in ["bit_accs_avc", "bit_accs_avg"]:
                    marker = "o"
                    linestyle = "--"  # dash
                    display_label = "Bit-Acc (on)"
                    alpha = 1.0
                    bit_acc_on_done = True
                # Style spécial pour SSIM et Bit-Acc vanilla
                elif metric_name == "perceptual_metric":
                    marker = "o"
                    linestyle = "-"
                    display_label = T4G_LABEL_MAP.get(metric_name, metric_name)
                    alpha = 0.45 if (t4g and metric_name == "perceptual_metric") else 1.0
                elif metric_name == "bit_accs_avg_vanilla":
                    marker = "s"  # carré
                    linestyle = "--"  # pointillé
                    display_label = T4G_LABEL_MAP.get(metric_name, metric_name)
                    alpha = 1.0
                else:
                    marker = "s"
                    linestyle = "--"
                    display_label = T4G_LABEL_MAP.get(metric_name, metric_name) if t4g else metric_name
                    alpha = 1.0
                line, = axR.plot(
                    steps, values,
                    marker=marker, linestyle=linestyle,
                    label=display_label, color=color, alpha=alpha
                )
                lines.append(line)
                labels.append(display_label)

            # Labels des axes
            if t4g:
                axL.set_ylabel("FID", fontsize=12)
                axR.set_ylabel("Bit-Acc and SSIM", fontsize=12)
            else:
                axL.set_ylabel(axis_label("axe gauche", left_group), fontsize=12)
                axR.set_ylabel(axis_label("axe droit", right_group), fontsize=12)

            legend_loc = "lower left" if t4g else "lower right"
            legend_anchor = (0.08, 0.02) if t4g else None
            axL.legend(lines, labels, loc=legend_loc,
                       **(dict(bbox_to_anchor=legend_anchor) if legend_anchor else {}), fontsize=12)

            fig.tight_layout()

    output_path = os.path.join(directory, "metrics_training_curves.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Figure sauvegardée dans : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Trace les métriques trouvées dans les fichiers 'metric-*'."
    )
    parser.add_argument(
        "directory",
        help="Dossier contenant les fichiers metric-* (ex: metric-fid50k_full.jsonl, metric-uchida_extraction.jsonl, ...)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Désactiver la normalisation quand il y a plus de 2 métriques."
    )
    args = parser.parse_args()

    all_metrics = load_all_metrics(args.directory)
    plot_all_metrics(all_metrics, args.directory, normalize=not args.no_normalize)


if __name__ == "__main__":
    main()

""" 
Trace 6 graphes (2 lignes × 3 colonnes) :
    Ligne 1 – Bit  : T4G (Ours) + TONDI  vs  noise / pruning / quantization
    Ligne 2 – SSIM : T4G (Ours) + IPR    vs  noise / pruning / quantization
Axe droit (log) : FID (on) gris + FID (off) bleu sur chaque subplot.

Usage:
    python plot_bitacc_vs_attacks.py <dossier_CelebA> [<dossier_vanilla_T4G_eval>]

    <dossier_CelebA>          : contient T4G/, TONDI/, IPR/ (chacun avec evaluation/)
    <dossier_vanilla_T4G_eval>: dossier evaluation/ du modèle vanilla (FID on)
                                Défaut: ../SG2_evaluation_final_bis/best_weights/CelebA/T4G/evaluation
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path

# ─── Méthodes ────────────────────────────────────────────────────────────────

# Ligne 1 : Bit-Accuracy
BITACC_METHODS = {
    "T4G": {
        "paths": [
            "T4G/evaluation/metrics_summary.json",  # ancien format
            "T4G/metrics_summary.json",             # nouveau format
        ],
        "extractor": lambda e: e["T4G_extraction"]["bit_accs_avc"],
        "color": "tab:orange",
        "label": r"Ours – Bit$_{\mathrm{on}}$",
        "marker": "o",
    },
    "TONDI": {
        "paths": [
            "TONDI/evaluation/metrics_summary.json",
            "TONDI/metrics_summary.json",
        ],
        "extractor": lambda e: e["TONDI_extraction"],
        "color": "sandybrown",
        "label": r"[13] – Bit$_{\mathrm{on}}$",
        "marker": "^",
    },
}

# Ligne 2 : SSIM
SSIM_METHODS = {
    "T4G": {
        "paths": [
            "T4G/evaluation/metrics_summary.json",
            "T4G/metrics_summary.json",
        ],
        "extractor": lambda e: e["T4G_extraction"]["perceptual_metric"],
        "color": "darkgreen",
        "label": r"Ours – SSIM$_{\mathrm{off}}$",
        "marker": "o",
    },
    "IPR": {
        "paths": [
            "IPR/evaluation/metrics_summary.json",
            "IPR/metrics_summary.json",
        ],
        "extractor": lambda e: e["IPR_extraction"]["ipr_SSIM"],
        "color": "mediumseagreen",
        "label": r"[39] – SSIM$_{\mathrm{on}}$",
        "marker": "^",
    },
}

# ─── FID ─────────────────────────────────────────────────────────────────────

FID_ON_COLOR  = "tab:gray"
FID_OFF_COLOR = "tab:blue"
FID_ALPHA     = 0.45

ATTACK_TYPES = ["noise", "pruning", "quantization"]

ATTACK_X_LABELS = {
    "noise":        r"Noise (% of $\sigma_w$)",
    "pruning":      "Pruning rate (%)",
    "quantization": "Quantization (bits)",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_summary(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def resolve_first_existing(base_dir: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = base_dir / rel
        if p.exists():
            return p
    return base_dir / candidates[0]


def extract_attack_data(summary: dict, attack_type: str, extractor, reverse: bool = False):
    points = []
    for key, entry in summary.items():
        parts = key.rsplit("-", 1)
        if len(parts) != 2:
            continue
        atype, level_str = parts
        if atype != attack_type:
            continue
        try:
            level = int(level_str)
        except ValueError:
            continue
        try:
            val = extractor(entry)
        except (KeyError, TypeError):
            continue
        points.append((level, val))
    points.sort(key=lambda p: p[0], reverse=reverse)
    return [p[0] for p in points], [p[1] for p in points]


def get_baseline(summary: dict, extractor):
    for key, entry in summary.items():
        if key.startswith("none"):
            try:
                return extractor(entry)
            except (KeyError, TypeError):
                pass
    return None


def prepend_baseline(xs, ys, summary, extractor):
    b = get_baseline(summary, extractor)
    if b is not None:
        return [0] + xs, [b] + ys
    return xs, ys


# ─── Tracé d'une cellule (1 attaque × 1 métrique) ────────────────────────────

def plot_cell(ax, attack_type, methods_summaries, methods_cfg,
              fid_on_summary, fid_off_summary,
              fid_on_extractor, fid_off_extractor):
    is_quant = (attack_type == "quantization")
    ax_fid = ax.twinx()
    ax_fid.set_yscale("log")

    # ticks : affichage homogène pour SSIM/Bit (valeurs ~[0,1])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    # FID (log) : chiffres significatifs
    ax_fid.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:.3g}"))
    # x : niveaux entiers attendus, on garde un affichage propre
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    all_lines, all_labels = [], []

    # ── Courbes principales (axe gauche) ──
    for name, cfg in methods_cfg.items():
        summary = methods_summaries.get(name, {})
        if not summary:
            continue
        xs, ys = extract_attack_data(summary, attack_type, cfg["extractor"], reverse=is_quant)
        if not xs:
            continue
        if not is_quant:
            xs, ys = prepend_baseline(xs, ys, summary, cfg["extractor"])
        l, = ax.plot(xs, ys, marker=cfg["marker"], color=cfg["color"],
                     label=cfg["label"], linewidth=1.8, markersize=4)
        all_lines.append(l)
        all_labels.append(cfg["label"])

    # ── FID (on) : gris ──
    xs_on, ys_on = extract_attack_data(fid_on_summary, attack_type, fid_on_extractor, reverse=is_quant)
    if xs_on:
        if not is_quant:
            xs_on, ys_on = prepend_baseline(xs_on, ys_on, fid_on_summary, fid_on_extractor)
        l_on, = ax_fid.plot(xs_on, ys_on, marker="o", linestyle="--",
                            color=FID_ON_COLOR, linewidth=1.4, markersize=3,
                            alpha=FID_ALPHA, label=r"Ours – FID$_{\mathrm{on}}$")
        all_lines.append(l_on)
        all_labels.append(r"Ours – FID$_{\mathrm{on}}$")

    # ── FID (off) : bleu ──
    xs_off, ys_off = extract_attack_data(fid_off_summary, attack_type, fid_off_extractor, reverse=is_quant)
    if xs_off:
        if not is_quant:
            xs_off, ys_off = prepend_baseline(xs_off, ys_off, fid_off_summary, fid_off_extractor)
        l_off, = ax_fid.plot(xs_off, ys_off, marker="o", linestyle="--",
                             color=FID_OFF_COLOR, linewidth=1.4, markersize=3,
                             alpha=FID_ALPHA, label=r"Ours – FID$_{\mathrm{off}}$")
        all_lines.append(l_off)
        all_labels.append(r"Ours – FID$_{\mathrm{off}}$")

    ax.set_xlabel(ATTACK_X_LABELS[attack_type], fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.tick_params(labelsize=10)
    ax_fid.tick_params(labelsize=10)

    if is_quant:
        ax.set_xlim(10.3, 5.7)

    ax_fid.set_ylabel("")

    return all_lines, all_labels


# ─── Fonction principale ──────────────────────────────────────────────────────

def plot(celeba_dir: Path, vanilla_eval_dir: Path):
    # Charger les summaries
    def load(path):
        return load_summary(path) if path.exists() else {}

    bitacc_summaries = {
        n: load(resolve_first_existing(celeba_dir, cfg["paths"])) for n, cfg in BITACC_METHODS.items()
    }
    ssim_summaries = {
        n: load(resolve_first_existing(celeba_dir, cfg["paths"])) for n, cfg in SSIM_METHODS.items()
    }

    fid_off_summary = bitacc_summaries.get("T4G", {})
    fid_off_extractor = lambda e: e["fid50k_full"]

    fid_on_summary = load(vanilla_eval_dir / "metrics_summary.json")
    fid_on_extractor = lambda e: e["fid50k_full"]
    # Fallback: si pas de summary vanilla, on utilise la métrique "trigger" de T4G
    # (souvent loggée comme fid50k_full_trigger).
    if not fid_on_summary and fid_off_summary:
        fid_on_summary = fid_off_summary
        fid_on_extractor = lambda e: e["fid50k_full_trigger"]

    # BMVC : 2 lignes × 3 colonnes (colonnes = attaques, lignes = métriques)
    # On garde la même taille de sous-figure qu'avant (ancien: 6.875×9.28 pour 3×2)
    # => nouvelle figure ~ (6.875/2*3) × (9.28/3*2)
    fig, axes = plt.subplots(2, 3, figsize=(10.3125, 6.1866666666666665))

    legend_handles: dict = {}  # label -> handle (dédupliqué)

    for col, attack_type in enumerate(ATTACK_TYPES):
        # Ligne 0 : Bit
        lines, labels = plot_cell(axes[0, col], attack_type,
                                  bitacc_summaries, BITACC_METHODS,
                                  fid_on_summary, fid_off_summary,
                                  fid_on_extractor, fid_off_extractor)
        for l, lbl in zip(lines, labels):
            if lbl not in legend_handles:
                legend_handles[lbl] = l

        # Ligne 1 : SSIM
        lines, labels = plot_cell(axes[1, col], attack_type,
                                  ssim_summaries, SSIM_METHODS,
                      fid_on_summary, fid_off_summary,
                      fid_on_extractor, fid_off_extractor)
        for l, lbl in zip(lines, labels):
            if lbl not in legend_handles:
                legend_handles[lbl] = l

        # Titres de colonnes (attaque)
        axes[0, col].set_title(attack_type.capitalize(), fontsize=11, pad=3)

    # Uniformiser l'échelle de l'axe gauche par ligne.
    # Ligne 0 (Bit-Accuracy) : focus sur la zone haute.
    for ax in axes[0, :].flat:
        ax.set_ylim(0.4, 1.0)
    # Ligne 1 (SSIM) : focus sur la zone haute.
    for ax in axes[1, :].flat:
        ax.set_ylim(0.7, 1.0)

    # Titres de lignes
    axes[0, 0].set_ylabel(r"Bit$_{\mathrm{on}}$", fontsize=11)
    axes[1, 0].set_ylabel(r"SSIM", fontsize=11)

    # Légende commune sous tous les graphes
    legend = fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        fontsize=10,
        frameon=True,
    )
    # S'assure que la légende reste au-dessus de tout (dont la ligne séparatrice)
    legend.set_zorder(30)
    # Marge basse plus grande pour que la légende ne recouvre pas les subplots.
    plt.tight_layout(rect=[0, 0.085, 1, 1])

    # Ligne de séparation entre les deux rangées (Bit / SSIM)
    # Calculée à partir des *tightbbox* (incluant ticklabels + xlabel), pour éviter
    # de recouvrir les textes de la 1ère rangée ou les graphes de la 2e.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    row0_bottom_disp = min(ax.get_tightbbox(renderer).y0 for ax in axes[0, :].flat)
    row1_top_disp = max(ax.get_tightbbox(renderer).y1 for ax in axes[1, :].flat)
    gap_disp = row0_bottom_disp - row1_top_disp
    if gap_disp > 0:
        pad = 0.10 * gap_disp
        y_sep_disp = row1_top_disp + 0.5 * gap_disp
        y_sep_disp = min(row0_bottom_disp - pad, max(row1_top_disp + pad, y_sep_disp))
        _x, y_sep = fig.transFigure.inverted().transform((0, y_sep_disp))
    else:
        # Fallback : si jamais les bboxes se chevauchent (cas extrême), on revient
        # à une séparation basée sur les positions d'axes.
        top_row_bottom = min(ax.get_position().y0 for ax in axes[0, :].flat)
        bottom_row_top = max(ax.get_position().y1 for ax in axes[1, :].flat)
        y_sep = 0.5 * (top_row_bottom + bottom_row_top)

    # Ligne sur toute la largeur de la figure.
    x_left, x_right = 0.0, 1.0
    fig.add_artist(
        Line2D(
            [x_left, x_right],
            [y_sep, y_sep],
            transform=fig.transFigure,
            color="black",
            linewidth=1.0,
            alpha=0.35,
            zorder=0,
            clip_on=False,
        )
    )

    out_path = celeba_dir / "bitacc_ssim_vs_attacks.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure sauvegardée : {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        celeba_dir = Path(__file__).parent.parent / "best_weights" / "CelebA"
    else:
        celeba_dir = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        vanilla_eval_dir = Path(sys.argv[2])
    else:
        vanilla_eval_dir = (Path.cwd() / ".." / "SG2_evaluation_final_bis" /
                            "best_weights" / "CelebA" / "T4G" / "evaluation").resolve()

    if vanilla_eval_dir.is_file() and vanilla_eval_dir.name == "metrics_summary.json":
        vanilla_eval_dir = vanilla_eval_dir.parent

    if not celeba_dir.exists():
        print(f"Dossier introuvable : {celeba_dir}")
        sys.exit(1)
    if not vanilla_eval_dir.exists():
        print(f"[WARN] Dossier vanilla introuvable : {vanilla_eval_dir}")

    plot(celeba_dir, vanilla_eval_dir)

# (pytorch-gpu-2.0.0+py3.10.9) [uak91sd@jean-zay1: SG2_evaluation_final]$ python bash_utils_scripts/plot_bitacc_vs_attacks.py ./best_weights/CelebA
# Figure sauvegardée : best_weights/CelebA/bitacc_ssim_vs_attacks.png
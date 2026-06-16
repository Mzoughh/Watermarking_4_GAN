#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from PIL import Image
from typing import List

EXTS = {".png"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_png(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    return sorted(files, key=lambda p: natural_key(p.name))


def open_rgba(p: Path) -> Image.Image:
    return Image.open(p).convert("RGBA")


def concat_h(images: List[Image.Image]) -> Image.Image:
    """Concatène horizontalement sans resize. Aligne en haut."""
    if not images:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    total_w = sum(im.width for im in images)
    max_h = max(im.height for im in images)
    out = Image.new("RGBA", (total_w, max_h), (0, 0, 0, 0))

    x = 0
    for im in images:
        out.paste(im, (x, 0), im)  # conserve alpha
        x += im.width
    return out


def concat_v(top: Image.Image, bottom: Image.Image) -> Image.Image:
    """Empile verticalement sans resize. Aligne à gauche."""
    w = max(top.width, bottom.width)
    h = top.height + bottom.height
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(top, (0, 0), top)
    out.paste(bottom, (0, top.height), bottom)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Sauvegarde un PNG par attaque en collant les PNG tels quels (pas de resize). "
                    "Ligne 1: generated_images, Ligne 2: trigger_images. "
                    "Le dossier 'none' sert de baseline (préfixe) pour toutes les attaques. "
                    "Cas spécial: 'quantization' -> ordre inversé."
    )
    ap.add_argument("parent", type=str, help="Chemin vers le dossier parent (ex: images_debug_eval)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Dossier de sortie (défaut: <parent>/_png_composites_raw)")
    args = ap.parse_args()

    parent = Path(args.parent).expanduser().resolve()
    if not parent.exists() or not parent.is_dir():
        raise SystemExit(f"[ERREUR] Dossier parent introuvable: {parent}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (parent / "_png_composites_raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline optionnelle
    baseline_dir = parent / "none"
    base_gen_paths = list_png(baseline_dir / "generated_images")
    base_trig_paths = list_png(baseline_dir / "trigger_images")

    base_gen_imgs = [open_rgba(p) for p in base_gen_paths]
    base_trig_imgs = [open_rgba(p) for p in base_trig_paths]

    # Attaques
    attack_dirs = sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))

    for attack_path in attack_dirs:
        name = attack_path.name
        if name == "none":
            continue

        atk_gen_paths = list_png(attack_path / "generated_images")
        atk_trig_paths = list_png(attack_path / "trigger_images")

        # ✅ Cas spécial: quantization est inversé
        if name == "quantization":
            atk_gen_paths = list(reversed(atk_gen_paths))
            atk_trig_paths = list(reversed(atk_trig_paths))

        atk_gen_imgs = [open_rgba(p) for p in atk_gen_paths]
        atk_trig_imgs = [open_rgba(p) for p in atk_trig_paths]

        row_top = concat_h(base_gen_imgs + atk_gen_imgs)
        row_bottom = concat_h(base_trig_imgs + atk_trig_imgs)

        composite = concat_v(row_top, row_bottom)

        out_path = out_dir / f"{name}.png"
        composite.save(out_path, "PNG")
        print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()

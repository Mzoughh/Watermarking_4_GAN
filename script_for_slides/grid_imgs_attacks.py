import os
import re
import argparse
from PIL import Image


def attack_sort_key(severity: str, attack_type: str):
    """
    Clé de tri pour les SEVERITÉS d'une même attaque :

    - 'none' en premier
    - pour 'quant'/'quantization' : du moins sévère au plus sévère
        -> typiquement nbits décroissant (8 -> 4 -> 2)
        -> si multi-paramètres (ex: 2_8), on trie d'abord sur le dernier nombre
           (souvent nbits), puis sur les autres, tous en décroissant.
    - pour les autres (ex: pruning, noise) : tri croissant (5, 10, 50)
    - si pas de nombre, tri lexicographique en dernier.
    """
    if severity in ("none", "", None):
        return (0, (), severity or "")

    nums = re.findall(r"\d+\.?\d*", str(severity))
    if nums:
        numeric_tuple = tuple(float(x) for x in nums)

        if attack_type.startswith("quant"):
            # Décroissant : on trie d'abord sur le dernier param (souvent nbits),
            # puis sur les autres (tous en décroissant).
            last = -numeric_tuple[-1]
            rest = tuple(-x for x in numeric_tuple[:-1])
            return (1, (last,) + rest, str(severity))
        else:
            # Croissant normal : pruning/noise/etc.
            return (1, numeric_tuple, str(severity))

    return (2, (float("inf"),), str(severity))


def parse_attack_from_filename(fname: str):
    """
    À partir d'un nom de fichier du type :
        fakes_after_pruning_5.png
        fakes_after_pruning_50.png
        fakes_after_quantization_2_8.png
        fakes_after_none.png

    renvoie:
        attack_type   : 'pruning', 'quantization', 'none', ...
        severity_name : '5', '50', '2_8', 'none', ...
    """
    base = os.path.basename(fname)
    stem, _ = os.path.splitext(base)

    prefix = "fakes_after_"
    if not stem.startswith(prefix):
        return stem, ""

    rest = stem[len(prefix):]  # ex: 'pruning_5', 'quantization_2_8', 'none'

    if rest == "" or rest == "none":
        return "none", "none"

    parts = rest.split("_", 1)
    if len(parts) == 1:
        return parts[0], ""
    attack_type, severity = parts[0], parts[1]
    return attack_type, severity


def extract_first_face(img: Image.Image) -> Image.Image:
    """
    Extrait le premier visage (tuile en haut à gauche) d'une grille 4x4.
    """
    w, h = img.size
    tile_w = w // 4
    tile_h = h // 4
    return img.crop((0, 0, tile_w, tile_h))


def find_before_attack_image(directory: str):
    """
    Cherche un fichier 'fake_before_attack*.png' ou 'fakes_before_attack*.png'
    pour servir de référence 'avant attaque'.
    """
    candidates = [
        f for f in os.listdir(directory)
        if f.lower().endswith(".png")
        and (f.startswith("fake_before_attack") or f.startswith("fakes_before_attack"))
    ]
    if not candidates:
        return None
    return os.path.join(directory, sorted(candidates)[0])


def build_composites_by_attack_type(directory: str, pattern_prefix: str = "fakes_after_"):
    """
    Pour chaque type d'attaque (pruning, quantization, ...), construit une image
    où l'on aligne horizontalement le premier visage de chaque grille,
    trié par sévérité (au sens de l'impact).
    Au début de chaque ligne, on ajoute le visage issu de fake_before_attack.
    Ensuite, on ajoute (si présent) fakes_after_none.png.
    """
    files = [
        f for f in os.listdir(directory)
        if f.startswith(pattern_prefix) and f.lower().endswith(".png")
    ]

    if not files:
        print(f"Aucun fichier '{pattern_prefix}*.png' trouvé dans {directory}")
        return

    before_attack_path = find_before_attack_image(directory)
    if before_attack_path:
        print(f"Image 'before attack' utilisée : {os.path.basename(before_attack_path)}")
    else:
        print("Aucune image 'fake_before_attack*.png' trouvée, pas de référence avant attaque.")

    attacks_by_type = {}  # attack_type -> list of (severity_name, filepath)
    baseline_after_none = None

    for f in files:
        attack_type, severity = parse_attack_from_filename(f)
        full_path = os.path.join(directory, f)

        if attack_type == "none":
            baseline_after_none = full_path
            continue

        attacks_by_type.setdefault(attack_type, []).append((severity, full_path))

    if not attacks_by_type and not before_attack_path and not baseline_after_none:
        print("Uniquement une baseline 'none' trouvée, rien à comparer.")
        return

    # Taille d'une tuile : priorité à before-attack, sinon none, sinon une attaque quelconque.
    sample_img_path = before_attack_path or baseline_after_none
    if sample_img_path is None:
        any_type = next(iter(attacks_by_type.keys()))
        sample_img_path = attacks_by_type[any_type][0][1]

    with Image.open(sample_img_path) as im0:
        example_face = extract_first_face(im0)
        tile_w, tile_h = example_face.size

    for attack_type, entries in attacks_by_type.items():
        # tri adapté au type
        entries.sort(key=lambda x: attack_sort_key(x[0], attack_type))

        faces = []
        labels = []

        # 1) before
        if before_attack_path is not None:
            with Image.open(before_attack_path) as im_before:
                faces.append(extract_first_face(im_before))
                labels.append("before")

        # 2) none (après-attaque)
        if baseline_after_none is not None:
            with Image.open(baseline_after_none) as im_none:
                faces.append(extract_first_face(im_none))
                labels.append("none")

        print(f"\nType d'attaque : {attack_type}")
        print("Ordre des sévérités :")

        # 3) toutes les sévérités
        for severity, path in entries:
            with Image.open(path) as im:
                faces.append(extract_first_face(im))
            labels.append(severity)
            print(f"  - {severity}  <- {os.path.basename(path)}")

        if not faces:
            continue

        n = len(faces)
        composite = Image.new("RGB", (tile_w * n, tile_h))

        for i, face in enumerate(faces):
            composite.paste(face, (i * tile_w, 0))

        out_name = f"first_faces_{attack_type}.png"
        out_path = os.path.join(directory, out_name)
        composite.save(out_path)
        print(f"Image composée sauvegardée pour '{attack_type}' dans : {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extrait le premier visage de chaque grille fakes_after_*.png et, "
            "pour chaque type d'attaque (pruning, quantization, ...), génère "
            "une image montrant la dégradation en fonction des paramètres. "
            "Ajoute au début le visage du fichier fake_before_attack*.png."
        )
    )
    parser.add_argument(
        "directory",
        help="Dossier contenant les fichiers fakes_after_*.png et fake_before_attack*.png"
    )
    args = parser.parse_args()

    build_composites_by_attack_type(args.directory)


if __name__ == "__main__":
    main()

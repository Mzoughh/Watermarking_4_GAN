import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

# Structure: { "noise-1": { "fid50k_full": [...], "IPR_extraction": [...] }, ... }
aggregated = defaultdict(dict)

pattern = re.compile(r"metric-(.+?)-(\w+)-(\d+)\.jsonl")

for jsonl_file in sorted(folder.glob("*.jsonl")):
    match = pattern.match(jsonl_file.name)
    if not match:
        continue

    metric_name = match.group(1)   # ex: fid50k_full, IPR_extraction
    attack_type = match.group(2)   # ex: noise, pruning, quantization, none
    attack_level = match.group(3)  # ex: 1, 10, 25

    attack_key = f"{attack_type}-{attack_level}"

    with open(jsonl_file, "r") as f:
        first_line = f.readline().strip()

    if first_line:
        data = json.loads(first_line)
        results = data.get("results", {})
        # Si plusieurs métriques dans results (ex: T4G), on les garde toutes
        # Si une seule (ex: fid50k_full, IPR_extraction), on prend la valeur directement
        if len(results) == 1:
            val = next(iter(results.values()))
        else:
            val = results
        aggregated[attack_key][metric_name] = val

# Trier par type d'attaque puis niveau
def sort_key(k):
    parts = k.rsplit("-", 1)
    return (parts[0], int(parts[1]))

output = {k: aggregated[k] for k in sorted(aggregated.keys(), key=sort_key)}

out_path = folder / "metrics_summary.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Fichier généré : {out_path}")
print(f"{len(output)} clés d'attaque enregistrées.")

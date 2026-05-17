from pathlib import Path
import csv

def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_dicts_to_csv(rows, csv_path):
    ensure_parent(csv_path)
    if not rows:
        return
    keys = rows[0].keys()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

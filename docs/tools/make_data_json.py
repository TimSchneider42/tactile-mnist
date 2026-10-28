#!/usr/bin/env python3
"""Write the data of the project page's baseline bar charts to one JSON file.

Reads <data dir>/<task>/<method>/{eval,train}_mean_loss_detailed.csv from every given data directory (the
baseline-data/ of this repository and of ap_gym) and writes, per task and method, the split the cards show (eval if
recorded, else train) and its per-seed curves: {"tasks": {task: {method: {"split": ..., "seeds": [[...], ...]}}}}.
Missing values are null; values keep 6 significant digits. score_bars.js computes the scores from these curves.
build_site.sh calls this; to run it by hand:  python3 docs/tools/make_data_json.py <out file> <data dir>...
"""
import csv
import json
import math
import sys
from pathlib import Path

SPLITS = ("eval", "train")


def value(cell):
    try:
        x = float(cell)
    except ValueError:
        return None
    return float(f"{x:.6g}") if math.isfinite(x) else None


def seeds(path):
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    cols = [i for i, h in enumerate(rows[0]) if h.strip().startswith("seed_")]
    return [[value(r[i]) if i < len(r) else None for r in rows[1:]] for i in cols]


out, data_dirs = Path(sys.argv[1]), [Path(d) for d in sys.argv[2:]]
tasks = {}
for data in data_dirs:
    for task in sorted(p for p in data.iterdir() if p.is_dir()):
        for method in sorted(p for p in task.iterdir() if p.is_dir()):
            for split in SPLITS:
                csv_path = method / f"{split}_mean_loss_detailed.csv"
                if csv_path.is_file():
                    tasks.setdefault(task.name, {})[method.name] = {
                        "split": split,
                        "seeds": seeds(csv_path),
                    }
                    break

out.write_text(json.dumps({"tasks": tasks}, separators=(",", ":")) + "\n")
print(
    f"wrote {out}: {len(tasks)} tasks, {sum(map(len, tasks.values()))} methods, "
    f"{out.stat().st_size / 1e3:.0f} kB"
)

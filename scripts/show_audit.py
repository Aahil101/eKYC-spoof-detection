"""Print a dataset_audit.json summary (from the Kaggle dataset_audit notebook)."""
from __future__ import annotations

import json
import sys

r = json.load(open(sys.argv[1]))
print("input_roots", r["input_roots"])
for k, v in r["datasets"].items():
    print(f"== {k}  mount={v.get('mount')}  files={v.get('n_files')}  partial={v.get('partial')} "
          f"sec={v.get('seconds')} {v.get('error', '')}")
    print("   ext", v.get("extensions"), " layout", v.get("expected_layout"))
    d = v.get("files_per_dir_depth4", {})
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    for kk, vv in list(d.items())[:lim]:
        print("     ", kk, vv)

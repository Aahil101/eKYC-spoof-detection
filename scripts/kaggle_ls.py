"""List a Kaggle dataset's directories (metadata only). Usage: python scripts/kaggle_ls.py <ref> [grep]"""
from __future__ import annotations

import collections
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi


def ls(api, ref: str) -> list[str]:
    out, tok = [], None
    while True:
        for attempt in range(6):
            try:
                r = api.dataset_list_files(ref, page_token=tok, page_size=1000)
                break
            except Exception as e:
                if "429" not in str(e) or attempt == 5:
                    raise
                time.sleep(5 * 2 ** attempt)
        out += [f.name for f in r.files]
        tok = getattr(r, "next_page_token", None)
        if not tok or len(out) >= 200_000:
            return out
        time.sleep(0.5)


if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()
    names = ls(api, sys.argv[1])
    pat = sys.argv[2] if len(sys.argv) > 2 else None
    dirs = collections.Counter("/".join(x.split("/")[:-1]) for x in names)
    for d, c in sorted(dirs.items()):
        print(f"{c:7d}  {d}")
    if pat:
        print("--- matching", pat)
        for x in names:
            if pat in x:
                print(x)

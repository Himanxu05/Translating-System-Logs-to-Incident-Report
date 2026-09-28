"""Download Loghub and write data/train.jsonl and data/test.jsonl.

    python scripts/build_dataset.py
"""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logreport.dataset import build, write_jsonl
from logreport.loghub import download


def main() -> None:
    root = download(ROOT / "data" / "loghub")
    data = build(root)
    for name, rows in data.items():
        write_jsonl(ROOT / "data" / f"{name}.jsonl", rows)
        by = Counter((r["system"], r["report"]["status"]) for r in rows)
        systems = sorted({s for s, _ in by})
        print(f"{name}: {len(rows)} windows")
        for s in systems:
            print(f"  {s:<12} incident {by[(s, 'incident')]:>4}  normal {by[(s, 'normal')]:>4}")
        cats = Counter(r["report"]["category"] for r in rows)
        print("  categories:", dict(cats.most_common()))


if __name__ == "__main__":
    main()

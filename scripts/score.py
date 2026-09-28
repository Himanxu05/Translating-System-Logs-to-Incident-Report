"""Score every predictions/*.jsonl file against data/test.jsonl and print a table.

    python scripts/score.py
    python scripts/score.py --by-system
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logreport.dataset import read_jsonl
from logreport.metrics import score_one, summarize

COLS = ["valid json", "status", "category", "severity", "component", "evidence F1",
        "invented component"]


def fmt(v: float) -> str:
    return "-" if v != v else f"{v:.0%}"  # nan -> "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-system", action="store_true")
    args = ap.parse_args()

    test = {r["id"]: r for r in read_jsonl(ROOT / "data" / "test.jsonl")}
    files = sorted((ROOT / "predictions").glob("*.jsonl"))
    if not files:
        sys.exit("no predictions/*.jsonl yet")

    print("| model | test set | n | " + " | ".join(COLS) + " |")
    print("|---|---|---|" + "---|" * len(COLS))
    for f in files:
        preds = {p["id"]: p["output"] for p in read_jsonl(f)}
        missing = [i for i in test if i not in preds]
        groups = {"seen systems": [], "unseen systems": [], "all": []}
        by_system: dict[str, list] = {}
        for tid, ex in test.items():
            if tid not in preds:
                continue
            s = score_one(preds[tid], ex["report"], ex["log"])
            groups["seen systems" if ex["split"] == "seen" else "unseen systems"].append(s)
            groups["all"].append(s)
            by_system.setdefault(ex["system"], []).append(s)
        if args.by_system:
            groups.update({f"  {k}": v for k, v in sorted(by_system.items())})
        for name, scores in groups.items():
            x = summarize(scores)
            if not x:
                continue
            cells = " | ".join(fmt(x[c]) for c in COLS)
            print(f"| {f.stem} | {name} | {x['n']} | {cells} |")
        if missing:
            print(f"| {f.stem} | ({len(missing)} test windows not predicted yet) |")
    print("\nstatus/category/severity/evidence: vs the reference report. component: same name or "
          "one contains the other, incidents only.\ninvented component: named something that "
          "isn't in the log.")


if __name__ == "__main__":
    main()

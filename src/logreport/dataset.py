"""Cut each system's log into windows of consecutive lines and build train/test sets.

Split:
- Train systems: training windows come from the first 80% of the file, overlapping.
- "seen" test: non-overlapping windows from the last 20% of the train systems.
- "unseen" test: whole files of systems the model never sees in training. This is
  the real test: a model that memorised templates (like v1 did) falls apart here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .loghub import load_system
from .report import reference_report

TRAIN_SYSTEMS = ["Linux", "HDFS", "Hadoop", "Spark", "BGL", "Windows", "Thunderbird", "Proxifier"]
UNSEEN_SYSTEMS = ["OpenSSH", "Apache", "Zookeeper", "OpenStack"]

WINDOW = 15
TRAIN_STRIDE = 8
MAX_LINE_CHARS = 300


def windows(df, start: int, end: int, stride: int, size: int = WINDOW):
    rows = df.iloc[start:end].to_dict("records")
    for i in range(0, len(rows) - size + 1, stride):
        yield rows[i:i + size]


def make_example(system: str, rows: list[dict], split: str) -> dict:
    lines = [r["raw"][:MAX_LINE_CHARS] for r in rows]
    return {
        "id": f"{system}-{rows[0]['LineId']}",
        "system": system,
        "split": split,
        "log": "\n".join(f"{i}: {line}" for i, line in enumerate(lines, start=1)),
        "report": reference_report(system, rows).model_dump(),
    }


def sample_balanced(examples: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Up to n examples, half incidents / half normal when the data allows it."""
    inc = [e for e in examples if e["report"]["status"] == "incident"]
    nor = [e for e in examples if e["report"]["status"] == "normal"]
    rng.shuffle(inc)
    rng.shuffle(nor)
    k = min(len(inc), n // 2)
    picked = inc[:k] + nor[:n - k]
    if len(picked) < n:  # not enough normal ones, top up with incidents
        picked += inc[k:k + n - len(picked)]
    return sorted(picked, key=lambda e: e["id"])


def build(root: Path, seed: int = 13, seen_per_system: int = 6,
          unseen_per_system: int = 30, normal_ratio: float = 1.0) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    train, test = [], []
    for system in TRAIN_SYSTEMS:
        df = load_system(root, system)
        cut = int(len(df) * 0.8)
        train += [make_example(system, w, "train") for w in windows(df, 0, cut, TRAIN_STRIDE)]
        seen = [make_example(system, w, "seen") for w in windows(df, cut, len(df), WINDOW)]
        test += sample_balanced(seen, seen_per_system, rng)
    for system in UNSEEN_SYSTEMS:
        df = load_system(root, system)
        pool = [make_example(system, w, "unseen") for w in windows(df, 0, len(df), WINDOW)]
        test += sample_balanced(pool, unseen_per_system, rng)
    return {"train": downsample_normal(train, normal_ratio, rng), "test": test}


def downsample_normal(rows: list[dict], ratio: float, rng: random.Random) -> list[dict]:
    """Most windows are normal. Keep all incidents and about `ratio` x as many normal
    windows, spread over the systems, so the model doesn't learn to always say normal."""
    inc = [r for r in rows if r["report"]["status"] == "incident"]
    nor = [r for r in rows if r["report"]["status"] == "normal"]
    rng.shuffle(nor)
    by_system: dict[str, list[dict]] = {}
    for r in nor:
        by_system.setdefault(r["system"], []).append(r)
    keep, want = [], int(len(inc) * ratio)
    while len(keep) < want and any(by_system.values()):
        for rs in by_system.values():  # round-robin so every system keeps some
            if rs and len(keep) < want:
                keep.append(rs.pop())
    out = inc + keep
    rng.shuffle(out)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]

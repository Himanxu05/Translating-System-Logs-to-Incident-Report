"""Keyword baseline predictions -> predictions/rules.jsonl"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logreport import rules
from logreport.dataset import read_jsonl, write_jsonl

test = read_jsonl(ROOT / "data" / "test.jsonl")
write_jsonl(ROOT / "predictions" / "rules.jsonl",
            [{"id": ex["id"], "output": rules.predict(ex["log"]).to_json()} for ex in test])
print(f"wrote {len(test)} predictions")

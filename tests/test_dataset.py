import random

import pandas as pd

from logreport.dataset import downsample_normal, make_example, sample_balanced, windows
from logreport.loghub import load_system


def ex(i, status):
    return {"id": f"s-{i}", "system": "s" if i % 2 else "t", "report": {"status": status}}


def test_windows_and_example_format():
    df = pd.DataFrame({"LineId": [str(i) for i in range(1, 11)], "raw": [f"line {i}" for i in range(10)],
                       "EventId": ["E0"] * 10, "Content": ["c"] * 10, "Component": [""] * 10})
    ws = list(windows(df, 0, 10, stride=3, size=4))
    assert [w[0]["LineId"] for w in ws] == ["1", "4", "7"]
    e = make_example("Spark", ws[0], "train")
    assert e["log"].splitlines()[0] == "1: line 0" and e["report"]["status"] == "normal"


def test_sample_balanced_halves():
    pool = [ex(i, "incident") for i in range(10)] + [ex(i, "normal") for i in range(10, 30)]
    picked = sample_balanced(pool, 8, random.Random(0))
    assert sum(p["report"]["status"] == "incident" for p in picked) == 4 and len(picked) == 8


def test_downsample_keeps_all_incidents():
    rows = [ex(i, "incident") for i in range(5)] + [ex(i, "normal") for i in range(5, 50)]
    out = downsample_normal(rows, 1.0, random.Random(0))
    assert sum(r["report"]["status"] == "incident" for r in out) == 5 and len(out) == 10
    assert {r["system"] for r in out if r["report"]["status"] == "normal"} == {"s", "t"}


def test_label_prefix_is_stripped(tmp_path):
    d = tmp_path / "BGL"
    d.mkdir()
    (d / "BGL_2k.log").write_text("KERNDTLB 1117838570 RAS KERNEL FATAL data TLB error\n- 111 RAS ok\n")
    pd.DataFrame({"LineId": ["1", "2"], "Label": ["KERNDTLB", "-"], "EventId": ["E55", "E1"],
                  "Content": ["data TLB error", "ok"]}).to_csv(d / "BGL_2k.log_structured.csv", index=False)
    df = load_system(tmp_path, "BGL")
    assert df["raw"].tolist() == ["1117838570 RAS KERNEL FATAL data TLB error", "111 RAS ok"]

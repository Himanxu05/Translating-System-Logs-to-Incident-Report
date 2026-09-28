"""Download the Loghub 2k samples.

Loghub (https://github.com/logpai/loghub) is a collection of real system logs
released for research. Each system has a 2,000-line sample plus a parsed
version (`*_structured.csv`) where every line is matched to a template
(EventId). The data is free for research/academic use; see LICENSE_LOGHUB.txt
next to the downloaded files.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/logpai/loghub/master"
SYSTEMS = ["Linux", "OpenSSH", "Apache", "HDFS", "Hadoop", "Spark", "BGL", "Zookeeper",
           "OpenStack", "Windows", "Thunderbird", "Proxifier"]

# In these two the raw line starts with the expert anomaly label ("-" or e.g.
# "KERNDTLB"). It has to be removed or the model could read the answer off the input.
LABEL_PREFIXED = {"BGL", "Thunderbird"}


def download(dest: Path, systems: list[str] = SYSTEMS) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    lic = dest / "LICENSE_LOGHUB.txt"
    if not lic.exists():
        urllib.request.urlretrieve(f"{BASE}/LICENSE", lic)
    for s in systems:
        for name in (f"{s}_2k.log", f"{s}_2k.log_structured.csv"):
            path = dest / s / name
            if not path.exists():
                path.parent.mkdir(exist_ok=True)
                urllib.request.urlretrieve(f"{BASE}/{s}/{name}", path)
    return dest


def load_system(root: Path, system: str) -> pd.DataFrame:
    """One row per log line: the raw text the model sees, plus the parsed fields."""
    df = pd.read_csv(root / system / f"{system}_2k.log_structured.csv", dtype=str,
                     keep_default_na=False)
    raw = (root / system / f"{system}_2k.log").read_text(encoding="utf-8",
                                                         errors="replace").splitlines()
    if len(raw) != len(df):
        raise ValueError(f"{system}: {len(raw)} raw lines but {len(df)} parsed rows")
    if system in LABEL_PREFIXED:
        raw = [line.split(" ", 1)[1] if " " in line else line for line in raw]
    df["raw"] = raw
    df["system"] = system
    return df

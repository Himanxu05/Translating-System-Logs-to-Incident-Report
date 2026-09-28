"""Keyword baseline, a port of v1's `rule_based_baseline` to the new report format.

The keywords are v1's originals plus ones taken from the *training* systems only.
I deliberately didn't add anything seen in the unseen test systems (OpenSSH, Apache,
Zookeeper, OpenStack), otherwise the baseline would be quietly tuned on the test set.
"""

from __future__ import annotations

import re

from .report import NORMAL, Report

KEYWORDS = [
    # (category, severity, patterns) - checked in this order
    # v1: cpu / oom / disk / timeout+gateway / connection refused+db / exit code+failed
    # train systems: Linux (auth), Hadoop + Proxifier (network), BGL + HDFS (hardware, storage)
    ("hardware", "critical", ["tlb error", "data storage interrupt", "machine check"]),
    ("authentication", "medium", ["authentication failure", "authentication failed",
                                  "check pass"]),
    ("network", "medium", ["timeout", "gateway", "connection refused", "connection reset",
                           "no route to host", "retrying connect", "could not connect",
                           "error in contacting"]),
    ("storage", "medium", ["disk", "i/o", "mount failed", "exception while serving blk",
                           "failed to renew lease"]),
    ("service", "high", ["out-of-memory", "oom", "exited abnormally", "exit code", "failed"]),
]
LINE_RE = re.compile(r"^(\d+): (.*)$")


def predict(log: str) -> Report:
    hits: dict[str, list[int]] = {}
    for line in log.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        no, text = int(m.group(1)), m.group(2).lower()
        for category, _, patterns in KEYWORDS:
            if any(p in text for p in patterns):
                hits.setdefault(category, []).append(no)
                break
    if not hits:
        return NORMAL
    category = max(hits, key=lambda c: len(hits[c]))
    severity = next(sev for cat, sev, _ in KEYWORDS if cat == category)
    return Report(status="incident", category=category, severity=severity, component=None,
                  evidence=hits[category],
                  summary=f"{len(hits[category])} {category} line(s) matched by keywords.")

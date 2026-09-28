"""Score predicted reports against the reference reports.

Every model writes a predictions file with one {"id", "output"} per test window
(`output` is the raw text it produced). Scoring parses that text, so a model that
writes invalid JSON is penalised the same way everywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .report import Report, parse_report


def _f1(pred: set[int], gold: set[int]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def same_component(pred: str | None, ref: str | None) -> bool:
    """Lenient match: "sshd" counts for "sshd(pam_unix)", "QuorumCnxManager" for
    "QuorumCnxManager$RecvWorker". Too-short names don't count."""
    if not pred or not ref:
        return pred == ref
    p, r = pred.lower().strip(), ref.lower().strip()
    return p == r or (len(p) >= 3 and p in r) or (len(r) >= 3 and r in p)


def _in_log(value: str, log: str) -> bool:
    body = re.sub(r"^\d+: ", "", log, flags=re.M)  # ignore our own "1: " line prefixes
    return value in body


@dataclass
class Score:
    valid: bool
    status: bool
    category: bool
    severity: bool
    all_three: bool
    component: bool | None  # None when the reference is normal
    evidence_f1: float
    invented_component: bool  # named a component that isn't in the log at all


def score_one(output: str, reference: dict, log: str) -> Score:
    ref = Report.model_validate(reference)
    pred = parse_report(output)
    if pred is None:
        return Score(False, False, False, False, False,
                     None if ref.status == "normal" else False, 0.0, False)
    status = pred.status == ref.status
    category = pred.category == ref.category
    severity = pred.severity == ref.severity
    component = None if ref.status == "normal" else same_component(pred.component, ref.component)
    invented = bool(pred.component) and not _in_log(pred.component, log)
    return Score(True, status, category, severity, status and category and severity, component,
                 _f1(set(pred.evidence), set(ref.evidence)), invented)


def summarize(scores: list[Score]) -> dict[str, float]:
    n = len(scores)
    if n == 0:
        return {}
    comp = [s.component for s in scores if s.component is not None]

    def rate(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "n": n,
        "valid json": rate([s.valid for s in scores]),
        "status": rate([s.status for s in scores]),
        "category": rate([s.category for s in scores]),
        "severity": rate([s.severity for s in scores]),
        "all three": rate([s.all_three for s in scores]),
        "component": rate(comp),
        "evidence F1": rate([s.evidence_f1 for s in scores]),
        "invented component": rate([s.invented_component for s in scores]),
    }

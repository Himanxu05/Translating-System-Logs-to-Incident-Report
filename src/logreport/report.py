"""The incident report format, how a window of lines gets its reference report,
and how model output is parsed back into a report."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .labels import CATEGORIES, SEVERITIES, component, line_label

Category = Literal["authentication", "network", "storage", "hardware", "service", "permission",
                   "configuration", "none"]
Severity = Literal["medium", "high", "critical", "none"]

CATEGORY_WORDS = {
    "authentication": "authentication failure", "network": "network error",
    "storage": "storage error", "hardware": "hardware fault", "service": "service failure",
    "permission": "permission error", "configuration": "configuration problem",
}


class Report(BaseModel):
    status: Literal["incident", "normal"]
    category: Category
    severity: Severity
    component: str | None = None
    evidence: list[int] = Field(default_factory=list)  # 1-based line numbers in the window
    summary: str = ""

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)


NORMAL = Report(status="normal", category="none", severity="none", component=None, evidence=[],
                summary="No problems found in these log lines.")


def reference_report(system: str, rows: list[dict]) -> Report:
    """The expected report for a window, built from the line labels."""
    problems = []
    for i, row in enumerate(rows, start=1):
        lab = line_label(system, row)
        if lab:
            problems.append((i, row, *lab))
    if not problems:
        return NORMAL

    # main category: the one with the most problem lines, ties -> more severe
    counts = Counter(cat for _, _, cat, _ in problems)
    worst = {cat: max(SEVERITIES.index(sev) for _, _, c, sev in problems if c == cat)
             for cat in counts}
    category = max(counts, key=lambda c: (counts[c], worst[c], -CATEGORIES.index(c)))
    main = [p for p in problems if p[2] == category]
    severity = SEVERITIES[worst[category]]

    first = next(p for p in main if p[3] == severity)
    comp = component(system, first[1])
    evidence = [i for i, *_ in main]
    what = CATEGORY_WORDS[category] + ("s" if len(evidence) > 1 else "")
    source = f" from {comp}" if comp else ""
    example = first[1]["Content"].strip()
    if len(example) > 120:
        example = example[:117] + "..."
    summary = f"{len(evidence)} {what}{source}, e.g. \"{example}\""
    return Report(status="incident", category=category, severity=severity, component=comp,
                  evidence=evidence, summary=summary)


JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_report(text: str) -> Report | None:
    """Pull the first JSON object out of model output. None if it isn't a valid report."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(data.get("evidence"), list):
        data["evidence"] = [int(x) for x in data["evidence"]
                            if isinstance(x, int | str) and str(x).isdigit()]
    try:
        return Report.model_validate(data)
    except ValidationError:
        return None

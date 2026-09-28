import json

from logreport.labels import component, line_label
from logreport.report import NORMAL, parse_report, reference_report


def row(eid, raw, content="x", **kw):
    return {"EventId": eid, "raw": raw, "Content": content, "Label": "-", "Component": "", **kw}


SSH_FAIL = "Dec 10 06:55:46 LabSZ sshd[24200]: Failed password for root from 1.2.3.4 port 22 ssh2"
SSH_BYE = "Dec 10 06:55:47 LabSZ sshd[24200]: Received disconnect from 1.2.3.4: 11: Bye Bye [preauth]"


def test_openssh_rules_and_component():
    assert line_label("OpenSSH", row("E9", SSH_FAIL)) == ("authentication", "medium")
    assert line_label("OpenSSH", row("E24", SSH_BYE)) is None
    assert component("OpenSSH", row("E9", SSH_FAIL)) == "sshd"


def test_bgl_uses_expert_labels():
    assert line_label("BGL", row("E55", "x", Label="KERNDTLB")) == ("hardware", "critical")
    assert line_label("BGL", row("E55", "x", Label="-")) is None


def test_component_must_appear_in_the_line():
    r = row("E3", "[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6")
    assert component("Apache", r) == "mod_jk"
    assert component("HDFS", row("E3", "no component here", Component="dfs.DataNode")) is None


def test_reference_report_picks_most_frequent_category():
    rows = [row("E9", SSH_FAIL, "Failed password for root"), row("E24", SSH_BYE),
            row("E11", "sshd[1]: fatal: Write failed: Connection reset by peer"),
            row("E9", SSH_FAIL, "Failed password for root")]
    r = reference_report("OpenSSH", rows)
    assert (r.status, r.category, r.severity, r.component) == ("incident", "authentication",
                                                               "medium", "sshd")
    assert r.evidence == [1, 4]
    assert r.summary.startswith("2 authentication failures from sshd")


def test_normal_window():
    assert reference_report("OpenSSH", [row("E24", SSH_BYE)]) == NORMAL


def test_parse_report_handles_fences_and_string_evidence():
    text = '```json\n{"status": "incident", "category": "network", "severity": "high", ' \
           '"component": "x", "evidence": ["3", 5, "line 7"], "summary": "s"}\n```'
    r = parse_report(text)
    assert r.category == "network" and r.evidence == [3, 5]


def test_parse_report_rejects_bad_output():
    assert parse_report("no json here") is None
    assert parse_report('{"status": "maybe"}') is None
    assert parse_report("<think>{}</think>" + json.dumps(NORMAL.model_dump())) == NORMAL

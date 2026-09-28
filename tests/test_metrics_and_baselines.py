import json

from logreport.lstm import target_tokens, tokens_to_report
from logreport.metrics import same_component, score_one, summarize
from logreport.report import NORMAL
from logreport.rules import predict

LOG = "1: Dec 10 LabSZ sshd[1]: Failed password for root\n2: Dec 10 LabSZ sshd[1]: Bye Bye"
REF = {"status": "incident", "category": "authentication", "severity": "medium",
       "component": "sshd", "evidence": [1], "summary": "s"}


def out(**kw):
    return json.dumps({**REF, **kw})


def test_perfect_prediction():
    s = score_one(out(), REF, LOG)
    assert s.all_three and s.component and s.evidence_f1 == 1.0 and not s.invented_component


def test_invented_component_is_caught():
    s = score_one(out(component="sshd(pam_unix)"), REF, LOG)
    assert s.component  # lenient match still counts it as the right program
    assert s.invented_component  # but that string is not in the log


def test_line_prefixes_dont_count_as_log_text():
    assert score_one(out(component="1:"), REF, LOG).invented_component


def test_invalid_output_scores_zero():
    s = score_one("sorry, I can't", REF, LOG)
    assert not s.valid and not s.status and s.evidence_f1 == 0.0


def test_same_component():
    assert same_component("sshd", "sshd(pam_unix)")
    assert same_component("QuorumCnxManager", "QuorumCnxManager$RecvWorker")
    assert not same_component("ab", "abc")  # too short to count
    assert not same_component(None, "sshd")


def test_summarize_rates():
    scores = [score_one(out(), REF, LOG), score_one(out(category="network"), REF, LOG)]
    x = summarize(scores)
    assert x["n"] == 2 and x["category"] == 0.5 and x["status"] == 1.0


def test_rules_baseline():
    r = predict("1: sshd: authentication failure; user=root\n2: all good\n3: no route to host")
    assert r.category == "authentication" and r.evidence == [1]
    assert predict("1: all good") == NORMAL


def test_lstm_output_round_trip():
    text = tokens_to_report(target_tokens(REF))
    assert json.loads(text) == {**REF, "summary": ""}
    assert tokens_to_report(["status", "incident"]) == "status incident"  # incomplete -> invalid

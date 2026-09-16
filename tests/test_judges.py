import asyncio
import json

import pytest

from genval.core import prompts as P
from genval.engine.dispatch import fire
from genval.tasks.verify import judges as J


def _q(n=4, key="b", qid="q1"):
    return {"id": qid, "answerKey": key, "unit": "KC-1",
            "question": {"stem": "Which principle applies?",
                         "choices": [{"label": chr(97 + i), "text": f"choice {i}"} for i in range(n)]}}


def test_render_deterministic_nota_last():
    c1 = J.render_call("backend", _q(), 0)
    c2 = J.render_call("backend", _q(), 0)
    assert c1 == c2
    assert c1["id"] == J.verify_call_id("backend", "q1", 0) == "backend|q1|p0"
    assert (c1["item"], c1["perm_index"], c1["n_choices"], c1["prompt_version"]) == ("q1", 0, 4, P.VERIFY_PROMPT_VERSION)
    assert c1["prompt"].splitlines()[-1] == "E. None of the above"
    assert sorted(c1["permutation"]) == [0, 1, 2, 3]
    lines = c1["prompt"].splitlines()
    assert lines[-5:-1] == [f"{L}. choice {i}" for L, i in zip("ABCD", c1["permutation"])]   # displayed order = perm
    assert lines[0] == "Answer this multiple-choice question." and "Which principle applies?" in lines
    assert J.render_call("backend", _q(), 1)["id"] != c1["id"]
    perms = {tuple(J.render_call("backend", _q(), k)["permutation"]) for k in range(J.N_PERMS)}
    assert len(perms) > 1
    assert J.render_call("backend", _q(), 0, "e-learning")["prompt"].startswith(
        "Answer this multiple-choice question about e-learning.")


def test_build_calls_one_per_perm():
    calls = J.build_calls("m", [_q(qid="a"), _q(qid="b")])
    assert len(calls) == 2 * J.N_PERMS
    assert len({c["id"] for c in calls}) == len(calls)
    assert [c["item"] for c in calls] == ["a"] * J.N_PERMS + ["b"] * J.N_PERMS
    assert [c["perm_index"] for c in calls] == list(range(J.N_PERMS)) * 2
    assert J.build_calls("m", []) == []


def test_parse_vote_variants():
    perm = [2, 0, 3, 1]
    assert J.parse_vote(" C", perm, 4) == chr(97 + perm[2])
    assert J.parse_vote("(b).", perm, 4) == chr(97 + perm[1])
    assert J.parse_vote("The answer is A", perm, 4) == chr(97 + perm[0])
    assert J.parse_vote("E", perm, 4) == "nota"       # NOTA is position n
    assert J.parse_vote("F", perm, 4) is None         # out of range
    assert J.parse_vote("no idea", perm, 4) is None
    assert J.parse_vote("", perm, 4) is None
    assert J.parse_vote(None, perm, 4) is None
    assert J.parse_vote("D", [0, 1, 2], 3) == "nota"


def test_decide_verdicts():
    assert J.decide({0: "b", 1: "b", 2: "a"}, "b") == "verified"
    assert J.decide({0: "a", 1: "a", 2: "b"}, "b") == "disputed"
    assert J.decide({0: "nota", 1: "nota", 2: "b"}, "b") == "nota"
    assert J.decide({0: "a", 1: "b", 2: "c"}, "b") == "unverified"
    assert J.decide({0: None, 1: None, 2: "b"}, "b") == "unverified"
    assert J.decide({0: "b", 1: "b"}, "b") == "verified"     # early majority
    assert J.decide({0: "a", 1: "b"}, "b") == "incomplete"   # third could decide
    assert J.decide({}, "b") == "incomplete"


def test_fire_resume_and_errors(tmp_path):
    calls = J.build_calls("m", [_q(qid=f"q{i}") for i in range(3)])

    async def worker(call):
        if call["item"] == "q2":
            raise RuntimeError("boom")
        return {"response": " A"}

    raw = tmp_path / "raw.jsonl"
    n_ok, n_err = asyncio.run(fire(calls, worker, raw, 4, "t"))
    assert (n_ok, n_err) == (2 * J.N_PERMS, J.N_PERMS)
    recs = [json.loads(ln) for ln in raw.read_text().splitlines()]
    assert len(recs) == 3 * J.N_PERMS
    for rec in recs:                                          # merged records: the call travels with its vote
        assert {"id", "item", "perm_index", "permutation", "n_choices", "prompt"} <= set(rec)
        assert rec.get("error") == "RuntimeError: boom" if rec["item"] == "q2" else rec["response"] == " A"

    fired = []

    async def worker2(call):
        fired.append(call["id"])
        return {"response": " A"}

    n_ok, n_err = asyncio.run(fire(calls, worker2, raw, 4, "t"))
    assert (n_ok, n_err) == (J.N_PERMS, 0)                 # only q2's calls refire
    assert fired and all("|q2|" in i for i in fired)
    votes = J.collect_votes(raw)
    assert set(votes) == {"q0", "q1", "q2"}
    assert all(len(v["votes"]) == J.N_PERMS and v["n_choices"] == 4 for v in votes.values())
    for item, v in votes.items():                          # " A" = the displayed first choice = permutation[0]
        for rec in recs:
            if rec["item"] == item and "error" not in rec:
                assert v["votes"][rec["perm_index"]] == chr(97 + rec["permutation"][0])
    assert asyncio.run(fire(calls, worker2, raw, 4, "t")) == (0, 0)


def test_collect_votes_missing_file(tmp_path):
    assert dict(J.collect_votes(tmp_path / "nope.jsonl")) == {}
    votes = J.collect_votes(tmp_path / "nope.jsonl")
    assert votes["unseen"] == {"votes": {}, "n_choices": None}       # a defaultdict for judge_questions


JUDGES = ("j1", "j2", "j3")


def test_unanimous_verdict_cases():
    ok = {j: {0: "b", 1: "b", 2: "b"} for j in JUDGES}
    r = J.unanimous_verdict(ok, "b", JUDGES)
    assert r["verdict"] == "verified" and r["passed"] and r["complete"] and r["distractor_votes"] == 0
    assert r["verdicts"] == {j: "verified" for j in JUDGES}
    assert r["votes"]["j1"] == {"p0": "b", "p1": "b", "p2": "b"}
    nota = {**ok, "j3": {0: "b", 1: "b", 2: "nota"}}                   # majority holds, NOTA is not a distractor
    assert J.unanimous_verdict(nota, "b", JUDGES)["verdict"] == "verified"
    unparsed = {**ok, "j3": {0: "b", 1: None, 2: "b"}}
    assert J.unanimous_verdict(unparsed, "b", JUDGES)["passed"]
    one_distractor = {**ok, "j2": {0: "b", 1: "b", 2: "c"}}           # majority holds, but a distractor vote
    r = J.unanimous_verdict(one_distractor, "b", JUDGES)
    assert r["verdict"] == "disputed" and r["distractor_votes"] == 1 and not r["passed"]
    assert r["verdicts"] == {j: "verified" for j in JUDGES}
    other = {**ok, "j3": {0: "c", 1: "c", 2: "c"}}                    # a judge majority for another choice
    r = J.unanimous_verdict(other, "b", JUDGES)
    assert r["verdict"] == "disputed" and r["verdicts"]["j3"] == "disputed"
    missing = {**ok, "j2": {0: "b", 1: "b"}}                          # unanswered call
    r = J.unanimous_verdict(missing, "b", JUDGES)
    assert r["verdict"] == "incomplete" and not r["complete"] and not r["passed"]
    assert J.unanimous_verdict({}, "b", JUDGES)["verdict"] == "incomplete"
    all_nota = {**ok, "j1": {0: "nota", 1: "nota", 2: "nota"}}
    assert J.unanimous_verdict(all_nota, "b", JUDGES)["verdict"] == "nota"
    split = {**ok, "j1": {0: "nota", 1: None, 2: "b"}}
    assert J.unanimous_verdict(split, "b", JUDGES)["verdict"] == "unverified"


def test_majority_verdict():
    r = J.majority_verdict({"j1": {0: "b", 1: "a", 2: "b"}}, "b", ["j1"])
    assert r["verdict"] == "verified" and r["passed"] and r["complete"] and r["distractor_votes"] is None
    assert r["votes"] == {"j1": {"p0": "b", "p1": "a", "p2": "b"}} and r["verdicts"] == {"j1": "verified"}
    assert J.majority_verdict({"j1": {0: "a", 1: "a", 2: "b"}}, "b", ["j1"])["verdict"] == "disputed"
    assert J.majority_verdict({"j1": {0: "b", 1: "b"}}, "b", ["j1"]) == J.majority_verdict(
        {"j1": {0: "b", 1: "b"}}, "b", ["j1"]) and not J.majority_verdict({"j1": {0: "b", 1: "b"}}, "b", ["j1"])["complete"]
    assert J.majority_verdict({}, "b", ["j1"])["verdict"] == "incomplete"
    with pytest.raises(ValueError):
        J.majority_verdict({}, "b", ["j1", "j2"])


def _votes(per_item):
    """{judge: {item: {votes...}}} -> the shape judge_questions expects."""
    return {j: {item: {"votes": v, "n_choices": 4} for item, v in items.items()} for j, items in per_item.items()}


def test_judge_questions_rows_and_kept_unanimous():
    qs = [_q(qid="good"), _q(qid="bad"), _q(qid="unseen")]
    votes = _votes({"j1": {"good": {0: "b", 1: "b", 2: "b"}, "bad": {0: "b", 1: "b", 2: "b"}},
                    "j2": {"good": {0: "b", 1: "b", 2: "nota"}, "bad": {0: "c", 1: "b", 2: "b"}},
                    "j3": {"good": {0: "b", 1: "b", 2: "b"}, "bad": {0: "b", 1: "b", 2: "b"}}})
    rows, kept = J.judge_questions(qs, votes, JUDGES, "unanimous")
    assert [r["verdict"] for r in rows] == ["verified", "disputed", "incomplete"]
    assert [q["id"] for q in kept] == ["good"] and kept[0]["verify_verdict"] == "verified"
    assert "verify_verdict" not in qs[1]
    row = rows[0]
    assert (row["id"], row["unit"], row["n_choices"], row["answer_key"], row["rule"]) == ("good", "KC-1", 4, "b", "unanimous")
    assert row["judges"] == list(JUDGES) and row["judge_verdicts"] == {j: "verified" for j in JUDGES}
    assert row["votes"]["j2"] == {"p0": "b", "p1": "b", "p2": "nota"} and row["distractor_votes"] == 0
    assert rows[1]["distractor_votes"] == 1 and rows[1]["complete"]
    assert rows[2]["n_choices"] is None and rows[2]["votes"] == {j: {} for j in JUDGES} and not rows[2]["complete"]


def test_judge_questions_majority_requires_exactly_one_judge():
    qs = [_q(qid="good"), _q(qid="bad")]
    votes = _votes({"j1": {"good": {0: "b", 1: "a", 2: "b"}, "bad": {0: "a", 1: "a", 2: "b"}}})
    rows, kept = J.judge_questions(qs, votes, ["j1"], "majority")
    assert [r["verdict"] for r in rows] == ["verified", "disputed"] and [q["id"] for q in kept] == ["good"]
    assert rows[0]["rule"] == "majority" and rows[0]["distractor_votes"] is None and rows[0]["judges"] == ["j1"]
    with pytest.raises(ValueError, match="exactly one judge"):
        J.judge_questions(qs, votes, ["j1", "j2"], "majority")
    with pytest.raises(ValueError):
        J.judge_questions(qs, votes, [], "majority")


def test_parse_judge_spec_and_defaults():
    assert J.parse_judge_spec("claude-sonnet-5@vertex") == ("claude-sonnet-5", "vertex")
    assert J.parse_judge_spec("gemini-3.7-flash@api") == ("gemini-3.7-flash", "api")
    assert J.parse_judge_spec("gemini-3.7-flash") == ("gemini-3.7-flash", None)
    assert J.RULES == ("unanimous", "majority") and J.N_PERMS == 3
    assert all(J.parse_judge_spec(s)[1] == "vertex" for s in J.DEFAULT_JUDGES)
    with pytest.raises(ValueError, match="no worker"):
        J.judge_worker("gpt-5@api")


def test_raw_path_naming_includes_prompt_version(tmp_path):
    assert J.raw_path(tmp_path, "claude-sonnet-5") == tmp_path / f"verify-raw-claude-sonnet-5-{P.VERIFY_PROMPT_VERSION}.jsonl"
    assert J.raw_path(str(tmp_path), "m").name == "verify-raw-m-v2s-1.1.jsonl"

import pytest

from genval.tasks.filter import LAK2026, PROFILES, V2, FilterRules, filter_questions, passes, rule_counts, rule_failures


def _q(stem="What is the primary benefit of spaced practice?",
       choices=("Better retention", "Faster reading", "Less effort", "More stress"),
       feedback="Spacing strengthens memory consolidation over time.", qid="q1", **extra):
    return {"id": qid, "type": "Multiple Choice",
            "question": {"stem": stem,
                         "choices": [{"label": lab, "text": t} for lab, t in zip("abcd", choices)]},
            "answerKey": "a", "feedback": feedback, "explanation": feedback, **extra}


def test_profiles_are_registered():
    assert PROFILES == {"lak2026": LAK2026, "v2": V2}
    assert LAK2026 == FilterRules(name="lak2026")
    assert (LAK2026.min_choice_chars, LAK2026.explanation_field, LAK2026.num_choices) == (5, "explanation", 4)
    assert (V2.min_choice_chars, V2.explanation_field, V2.num_choices) == (2, "feedback", None)
    assert V2.with_(name="v3").name == "v3" and V2.with_(name="v3").dedup_stems


# --- V2 (close-the-loop V1 rules) ---

def test_clean_question_passes_v2():
    assert rule_failures(_q(), V2) == [] and passes(_q(), V2)


@pytest.mark.parametrize("mutation,expected", [
    (dict(stem="Too short"), "short_stem"),
    (dict(stem="  Too short   "), "short_stem"),
    (dict(choices=("Better retention", "", "Less effort", "More stress")), "empty_choice"),
    (dict(choices=("Better retention", "   ", "Less effort", "More stress")), "empty_choice"),
    (dict(choices=("Same", "same", "Less effort", "More stress")), "duplicate_choice"),
    (dict(choices=("Same  thing.", "same thing", "Less effort", "More stress")), "duplicate_choice"),
    (dict(choices=("Better retention", "A", "Less effort", "More stress")), "short_choice"),
    (dict(choices=("Better retention", "All of the above.", "Less effort", "x y")), "all_or_none_of_the_above"),
    (dict(choices=("None of the above", "Better retention", "Less effort", "x y")), "all_or_none_of_the_above"),
    (dict(choices=("True", "Better retention", "Less effort", "More stress")), "dichotomous_choice"),
    (dict(choices=("Better retention", "Less effort", "More stress", "no")), "dichotomous_choice"),
    (dict(choices=("Both a and b", "Better", "Less", "More")), "both_neither_prefix"),
    (dict(choices=("Neither a nor b", "Better", "Less", "More")), "both_neither_prefix"),
    (dict(feedback="ok"), "short_explanation"),
    (dict(feedback=None), "short_explanation"),
])
def test_each_rule_fires_v2(mutation, expected):
    assert expected in rule_failures(_q(**mutation), V2)


def test_short_domain_answers_survive_v2():
    # "CTA"/"KLI"-style answers are valid in this domain (rule floor is 2 chars)
    assert rule_failures(_q(choices=("CTA", "KLI", "DFA", "AFM")), V2) == []


def test_multiple_failures_in_table_order_v2():
    q = _q(stem="Short", choices=("True", "True", "Both a and b", "All of the above"), feedback="")
    assert rule_failures(q, V2) == ["short_stem", "duplicate_choice", "all_or_none_of_the_above",
                                    "dichotomous_choice", "both_neither_prefix", "short_explanation"]


def test_dedup_within_unit_keeps_first_v2():
    a, b = _q(qid="a"), _q(qid="b", stem="  What is the primary   benefit of spaced practice?  ")
    c = _q(qid="c", stem="Which analysis reveals hidden knowledge components?")
    d = _q(qid="d", stem="Too short")                                # rejected for its own reason, not dedup
    e = _q(qid="e")                                                  # a later exact duplicate
    kept, rejects = filter_questions([a, b, c, d, e], V2)
    assert [q["id"] for q in kept] == ["a", "c"]
    assert rejects == [{"id": "b", "rules": ["duplicate_stem_in_unit"]},
                       {"id": "d", "rules": ["short_stem"]},
                       {"id": "e", "rules": ["duplicate_stem_in_unit"]}]
    assert rule_counts(rejects) == {"duplicate_stem_in_unit": 2, "short_stem": 1}
    # a rejected question does not claim its stem
    kept, rejects = filter_questions([_q(qid="x", feedback="ok"), _q(qid="y")], V2)
    assert [q["id"] for q in kept] == ["y"] and rejects[0]["rules"] == ["short_explanation"]


# --- LAK 2026 (the Phi-2 study as published) vs V2 ---

def test_clean_question_passes_lak2026():
    assert rule_failures(_q(), LAK2026) == [] and rule_failures(_q()) == []      # LAK2026 is the default
    assert not rule_failures(_q(stem="Ten chars!"), LAK2026)                    # exactly 10 raw chars


def test_all_of_the_above_with_trailing_period():
    q = _q(choices=("Better retention", "All of the above.", "Less effort", "More stress"))
    assert "all_or_none_of_the_above" not in rule_failures(q, LAK2026)          # legacy lower().strip()
    assert "all_or_none_of_the_above" in rule_failures(q, V2)                   # period-stripped
    bare = _q(choices=("Better retention", " All Of The Above ", "Less effort", "More stress"))
    assert "all_or_none_of_the_above" in rule_failures(bare, LAK2026)


def test_choice_floor_is_5_under_lak2026_and_2_under_v2():
    q = _q(choices=("Four", "Better retention", "Less effort", "More stress"))
    assert rule_failures(q, LAK2026) == ["short_choice"]
    assert rule_failures(q, V2) == []
    assert rule_failures(_q(choices=("Fives", "Better retention", "Less effort", "More stress")), LAK2026) == []
    one = _q(choices=("A", "Better retention", "Less effort", "More stress"))
    assert "short_choice" in rule_failures(one, V2) and "short_choice" in rule_failures(one, LAK2026)


def test_explanation_is_raw_under_lak2026_and_normalized_under_v2():
    assert rule_failures(_q(explanation="  short   ", feedback="fine and long enough"), LAK2026) == []   # 10 raw chars
    assert rule_failures(_q(feedback="  short   ", explanation="fine and long enough"), V2) == ["short_explanation"]
    # each profile reads its own field: explanation under LAK2026, feedback under V2
    assert rule_failures(_q(feedback="fine and long enough", explanation="ok"), LAK2026) == ["short_explanation"]
    assert rule_failures(_q(feedback="fine and long enough", explanation="ok"), V2) == []
    assert rule_failures(_q(feedback="ok", explanation="fine and long enough"), LAK2026) == []
    q = _q(feedback="Spacing strengthens memory consolidation over time.")
    del q["explanation"]
    assert rule_failures(q, LAK2026) == ["short_explanation"] and rule_failures(q, V2) == []
    assert rule_failures(_q(explanation=None, feedback=None), LAK2026) == ["short_explanation"]


def test_missing_choice_only_under_lak2026():
    three = _q(choices=("Better retention", "Faster reading", "Less effort"))
    assert rule_failures(three, LAK2026) == ["missing_choice"]
    assert rule_failures(three, V2) == []
    # a duplicate already explains the shortfall, so missing_choice is not added on top
    dup = _q(choices=("Same thing", "same thing", "Less effort", "More stress"))
    assert rule_failures(dup, LAK2026) == ["duplicate_choice"]
    assert rule_failures(_q(choices=("Better retention",) * 2 + ("Less effort", "More stress")), V2) == ["duplicate_choice"]


def test_both_neither_prefix_needs_a_space_under_v2():
    q = _q(choices=("bothering", "Better retention", "Less effort", "More stress"))
    assert rule_failures(q, LAK2026) == ["both_neither_prefix"]
    assert rule_failures(q, V2) == []
    both = _q(choices=("both of them", "Better retention", "Less effort", "More stress"))
    assert "both_neither_prefix" in rule_failures(both, LAK2026)
    assert "both_neither_prefix" in rule_failures(both, V2)


def test_no_stem_dedup_under_lak2026():
    kept, rejects = filter_questions([_q(qid="a"), _q(qid="b")], LAK2026)
    assert [q["id"] for q in kept] == ["a", "b"] and rejects == []


def test_filter_questions_reject_records_and_counts():
    qs = [_q(qid="a", stem="Short"), _q(qid="b"), _q(qid="c", stem="Short", feedback="ok")]
    kept, rejects = filter_questions(qs, V2)
    assert [q["id"] for q in kept] == ["b"]
    assert rejects == [{"id": "a", "rules": ["short_stem"]}, {"id": "c", "rules": ["short_stem", "short_explanation"]}]
    assert rule_counts(rejects) == {"short_stem": 2, "short_explanation": 1}
    assert rule_counts([]) == {}

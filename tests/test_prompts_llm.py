import random

from genval.core import prompts as P

COURSE = "a graduate course on e-learning design"
LO = "design a rubric aligned with a learning objective"


def _q(qid, stem, key="a"):
    return {"id": qid, "type": "Multiple Choice",
            "question": {"stem": stem,
                         "choices": [{"label": lab, "text": t}
                                     for lab, t in zip("abcd", ("Retention", "Speed", "Fatigue", "Boredom"))]},
            "answerKey": key}


EXEMPLARS = [_q("elearning-mcq-1-norm", "What does spaced practice improve?", "a"),
             _q("elearning-mcq-2-norm", "Which schedule spreads study sessions?", "b"),
             _q("elearning-mcq-3-norm", "Why does cramming fade quickly?", "c"),
             _q("elearning-mcq-4-norm", "  Which cue supports retrieval?  ", "d")]


def test_lo_prompt_with_course_phrase():
    prompt = P.render_lo_prompt(LO, 5, COURSE)
    assert prompt.splitlines()[0] == ("Write 5 NEW multiple-choice practice questions for a graduate course "
                                      "on e-learning design that assess this skill:")
    assert f'"{LO}"' in prompt
    assert "exactly 5 objects" in prompt
    assert prompt == P.LLM_LO_TEMPLATE.format(n=5, lo=LO, course=f" for {COURSE}")


def test_lo_prompt_without_course_phrase():
    for course in (None, "", "   "):
        prompt = P.render_lo_prompt(LO, 5, course)
        assert prompt.splitlines()[0] == "Write 5 NEW multiple-choice practice questions that assess this skill:"
    assert P.render_lo_prompt(LO, 5) == P.render_lo_prompt(LO, 5, None)
    assert P.course_phrase(f"  {COURSE} ") == f" for {COURSE}"
    assert P.course_phrase(None) == P.course_phrase("") == ""


def test_lo_prompt_count_scales_and_nothing_else_changes():
    p3 = P.render_lo_prompt("lo", 3)
    assert "Write 3 NEW" in p3 and "exactly 3 objects" in p3
    assert p3 == P.LLM_LO_TEMPLATE.format(n=3, lo="lo", course="")
    assert P.render_lo_prompt("lo", 3) == p3                          # nothing random


def test_render_exemplar_format():
    text = P.render_exemplar(2, EXEMPLARS[3])
    assert text == ("Example 2:\nWhich cue supports retrieval?\n"
                    "a) Retention\nb) Speed\nc) Fatigue\nd) Boredom\nAnswer: d")


def test_exemplar_prompt_contains_every_exemplar_and_the_target_count():
    prompt, perm = P.render_exemplar_prompt(EXEMPLARS, 5, random.Random(0))
    assert sorted(perm) == sorted(q["id"] for q in EXEMPLARS)
    for q in EXEMPLARS:
        assert q["question"]["stem"].strip() in prompt
        assert f"Answer: {q['answerKey']}" in prompt
    assert prompt.startswith(P.LLM_EXEMPLAR_PREAMBLE)
    assert prompt.endswith(P.LLM_EXEMPLAR_INSTRUCTIONS.format(n=5))
    assert "Write 5 NEW multiple-choice practice questions" in prompt
    assert "exactly 5 objects" in prompt
    # examples are numbered in the shown order
    assert [f"Example {i}:" in prompt for i in range(1, 5)] == [True] * 4 and "Example 5:" not in prompt
    first = prompt.split("Example 1:\n", 1)[1].splitlines()[0]
    assert first == next(q for q in EXEMPLARS if q["id"] == perm[0])["question"]["stem"].strip()


def test_exemplar_prompt_is_label_free():
    prompt, _ = P.render_exemplar_prompt(EXEMPLARS, 5, random.Random(0))
    for q in EXEMPLARS:
        assert q["id"] not in prompt                                  # no bookkeeping reaches the model
    assert "KC-" not in prompt and "skill:" not in prompt             # and no objective text either


def test_exemplar_permutation_is_rng_driven_and_deterministic():
    p1, perm1 = P.render_exemplar_prompt(EXEMPLARS, 5, random.Random(7))
    p2, perm2 = P.render_exemplar_prompt(EXEMPLARS, 5, random.Random(7))
    assert p1 == p2 and perm1 == perm2
    perms = {tuple(P.render_exemplar_prompt(EXEMPLARS, 5, random.Random(s))[1]) for s in range(20)}
    assert len(perms) > 1
    assert [q["id"] for q in EXEMPLARS] == [q["id"] for q in EXEMPLARS]   # the input list is not shuffled in place


def test_verify_prompt_with_and_without_domain():
    body = "Which principle applies?\n\nA. x\nB. y\nC. None of the above"
    with_domain = P.render_verify_prompt(body, "e-learning and instructional design")
    assert with_domain == ("Answer this multiple-choice question about e-learning and instructional design.\n"
                           "Reply with only the letter of the single best answer.\n\n" + body)
    bare = ("Answer this multiple-choice question.\n"
            "Reply with only the letter of the single best answer.\n\n" + body)
    assert P.render_verify_prompt(body) == P.render_verify_prompt(body, None) == P.render_verify_prompt(body, " ") == bare
    assert P.domain_phrase(" e-learning ") == " about e-learning" and P.domain_phrase(None) == ""


def test_version_constants_are_pinned():
    assert P.PROMPT_VERSION == 1
    assert P.LLM_LO_PROMPT_VERSION == 1
    assert P.LLM_EXEMPLAR_PROMPT_VERSION == 1
    assert P.VERIFY_PROMPT_VERSION == "v2s-1.1"
    assert P.VERIFY_NOTA_TEXT == "None of the above"

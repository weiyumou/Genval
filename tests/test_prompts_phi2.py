"""Golden tests for genval's prompt registry (the Phi-2 pipeline half).

Every assertion here is an exact published string (LAK 2026). If one of
these fails, a prompt changed — that is a scientific decision requiring a
PROMPT_VERSION bump, not a wording cleanup. The question-rendering and
congruity strings stay in ``kcluster.core.prompts`` and are tested there.
"""

from genval.core import prompts


def test_lo_alignment_scaffolds():
    assert prompts.LO_ALIGNMENT_MARGINAL_CONTEXT == "{q_type}:\n"
    assert prompts.LO_ACTIONS_HEADER == "The exercise below is designed to test whether a student can {lo}."
    assert prompts.LO_FACTS_HEADER == "The exercise below is designed to test whether a student knows:\n{lo}."


def test_qgen_scaffolds():
    assert prompts.QGEN_SEED_ACTIONS == (
        "The exercises below are designed to test whether a student can {std}.\n\n{header}")
    assert prompts.QGEN_SEED_FACTS == (
        'The exercises below are designed to test whether a student understands the following facts:\n'
        '"{std}."\n\n{header}')
    assert prompts.QGEN_MCQ_HEADER == "Multiple Choice (best out of {num_choices} options):\n1."
    assert prompts.QGEN_SOLUTION_PREFIX == "\n\nSolution:\nThe correct answer is"
    assert prompts.QGEN_EXPLANATION_PREFIX == "\n\nExplanation:\n"


def test_judge_wordings():
    assert prompts.GPT_JUDGE_SYSTEM_PROMPT == (
        "You are an expert at answering multiple choice questions. "
        "If none of the options a-d are correct, choose e for 'None of the above'. "
        "Provide your answer (letter a-e) and explanation in the JSON format specified."
    )
    assert prompts.JUDGE_Q1_PAIRED == "Answer the following two questions:\n\nQ1. {question}"
    assert prompts.JUDGE_Q1_SINGLE == "Answer the following question:\n\n{question}"
    assert prompts.JUDGE_Q2_LOGPROB == (
        "Does the above question help teachers test whether a student can {lo}?\na) Yes\nb) No")
    assert prompts.JUDGE_Q2_TEXT == "Does the following question test whether a student can **{lo}**?\n\n{question}"
    assert prompts.JUDGE_PREFILL_Q1 == "The answer to Q1 is **"
    assert prompts.JUDGE_PREFILL_Q2 == "The answer to Q2 is **"


def test_prompt_version_is_declared():
    assert prompts.PROMPT_VERSION == 1

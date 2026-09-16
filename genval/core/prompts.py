"""The prompt registry: every string genval sends to a model, in one place.

These strings are published artifacts — the LAK 2026 paper quotes the Phi-2
scaffolds and judge wordings verbatim, and the close-the-loop study
registered the LLM templates before generation — so they are versioned:
any change to a template's text is a scientific decision and must bump its
version, not a wording cleanup. Golden tests (tests/test_prompts.py) pin the
exact strings, and the archive-backed reproduction tests assert that the
rendered prompts equal the ones logged during the studies.

Question rendering (``Exercise n:`` headers, choice lines, ``Answer:``) and
the congruity contexts stay in ``kcluster.core.prompts``; genval imports
them rather than duplicating them.
"""

from kcluster.core.prompts import SPACE  # noqa: F401  (re-exported for the Phi-2 tasks)

#: Version of the Phi-2 scaffolds, alignment headers and judge wordings
#: below (carried over unchanged from kcluster PROMPT_VERSION = 1).
PROMPT_VERSION = 1

# --- LO alignment (LAK 2026; tasks/align/phi2.py). The marginal is the bare
# type line — deliberately NOT the congruity grid's "Exercise 2:" header ---
LO_ALIGNMENT_MARGINAL_CONTEXT = "{q_type}:\n"
LO_ACTIONS_HEADER = "The exercise below is designed to test whether a student can {lo}."
LO_FACTS_HEADER = "The exercise below is designed to test whether a student knows:\n{lo}."

# --- Phi-2 incremental MCQ generation (LAK 2026; tasks/generate/phi2.py) ---
QGEN_SEED_ACTIONS = "The exercises below are designed to test whether a student can {std}.\n\n{header}"
QGEN_SEED_FACTS = ('The exercises below are designed to test whether a student understands the following facts:\n'
                   '"{std}."\n\n{header}')
QGEN_MCQ_HEADER = "Multiple Choice (best out of {num_choices} options):\n1."
QGEN_SOLUTION_PREFIX = "\n\nSolution:\nThe correct answer is"
QGEN_EXPLANATION_PREFIX = "\n\nExplanation:\n"

# --- LLM judges of the LAK 2026 study arms (papers/lak2026/scripts/) ---
GPT_JUDGE_SYSTEM_PROMPT = (
    "You are an expert at answering multiple choice questions. "
    "If none of the options a-d are correct, choose e for 'None of the above'. "
    "Provide your answer (letter a-e) and explanation in the JSON format specified."
)
JUDGE_Q1_PAIRED = "Answer the following two questions:\n\nQ1. {question}"
JUDGE_Q1_SINGLE = "Answer the following question:\n\n{question}"
JUDGE_Q2_LOGPROB = "Does the above question help teachers test whether a student can {lo}?\na) Yes\nb) No"
JUDGE_Q2_TEXT = "Does the following question test whether a student can **{lo}**?\n\n{question}"
JUDGE_PREFILL_Q1 = "The answer to Q1 is **"
JUDGE_PREFILL_Q2 = "The answer to Q2 is **"


# --- LLM generation from a learning objective (close-the-loop design v2.4
# ZS_TEMPLATE, registered 2026-08-29; tasks/generate/llm.py) ---
LLM_LO_PROMPT_VERSION = 1

LLM_LO_TEMPLATE = """\
Write {n} NEW multiple-choice practice questions{course} that assess this skill:

"{lo}"

For each question provide:
- a stem
- four answer choices labeled a)-d), exactly one of which is correct
- the correct answer key
- a short feedback explanation (2-4 sentences) that a learner sees after
  answering, explaining why the key is correct and the distractors are not

Each question must require the skill, not merely restate it: a student
should not be able to answer from the wording of the skill statement alone.

Return a JSON array of exactly {n} objects, each with keys:
"stem" (string), "choices" (array of exactly 4 strings, without labels),
"answer" (one of "a", "b", "c", "d"), "feedback" (string).
"""


def course_phrase(course: str | None) -> str:
    """The optional course clause: ``" for a graduate course on e-learning design"``."""
    return f" for {course.strip()}" if course and course.strip() else ""


def render_lo_prompt(lo: str, n: int, course: str | None = None) -> str:
    """The zero-shot prompt for one call: the template with the objective text
    substituted verbatim (no permutation, nothing random)."""
    return LLM_LO_TEMPLATE.format(n=n, lo=lo, course=course_phrase(course))


# --- LLM generation from exemplar questions (close-the-loop design v2.1–v2.3
# seed-demonstration template; the generator never sees an objective) ---
LLM_EXEMPLAR_PROMPT_VERSION = 1

LLM_EXEMPLAR_PREAMBLE = "Below are example multiple-choice questions that all assess one skill.\n"

LLM_EXEMPLAR_INSTRUCTIONS = """\
Write {n} NEW multiple-choice practice questions that assess the same
skill as the examples. For each question provide:
- a stem
- four answer choices labeled a)-d), exactly one of which is correct
- the correct answer key
- a short feedback explanation (2-4 sentences) that a learner sees after
  answering, explaining why the key is correct and the distractors are not

Do not copy or lightly paraphrase any example; write questions a student
could not answer merely by having memorized the examples.

Return a JSON array of exactly {n} objects, each with keys:
"stem" (string), "choices" (array of exactly 4 strings, without labels),
"answer" (one of "a", "b", "c", "d"), "feedback" (string).
"""


def render_exemplar(idx: int, q) -> str:
    """One exemplar as shown to the generator: ``Example i:``, stem, choices, key."""
    lines = [f"Example {idx}:", q["question"]["stem"].strip()]
    for choice in q["question"]["choices"]:
        lines.append(f"{choice['label']}) {choice['text']}")
    lines.append(f"Answer: {q['answerKey']}")
    return "\n".join(lines)


def render_exemplar_prompt(exemplars: list, n: int, rng) -> tuple[str, list[str]]:
    """Build one exemplar-mode prompt; returns (prompt_text, exemplar_id_permutation).

    The exemplars are shown in an order drawn from ``rng`` (a per-call
    seeded ``random.Random``), so reruns rebuild identical prompts.
    """
    perm = list(exemplars)
    rng.shuffle(perm)
    body = "\n\n".join(render_exemplar(i + 1, q) for i, q in enumerate(perm))
    prompt = f"{LLM_EXEMPLAR_PREAMBLE}\n{body}\n\n{LLM_EXEMPLAR_INSTRUCTIONS.format(n=n)}"
    return prompt, [q["id"] for q in perm]


# --- Answer agreement by LLM judges (close-the-loop instrument v2s-1.1;
# tasks/verify/judges.py). The prompt is one question with its choices under
# a seeded permutation and "None of the above" appended last. ---
VERIFY_PROMPT_VERSION = "v2s-1.1"
VERIFY_NOTA_TEXT = "None of the above"
VERIFY_TEMPLATE = ("Answer this multiple-choice question{domain}.\nReply with only the letter of the "
                   "single best answer.\n\n{body}")


def domain_phrase(domain: str | None) -> str:
    """The optional domain clause: ``" about e-learning and instructional design"``."""
    return f" about {domain.strip()}" if domain and domain.strip() else ""


def render_verify_prompt(body: str, domain: str | None = None) -> str:
    return VERIFY_TEMPLATE.format(body=body, domain=domain_phrase(domain))

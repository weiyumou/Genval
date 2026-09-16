"""Syntactic filtering: the rule-based exclusion criteria of LAK 2026 Table 1.

One rule set, two registered profiles (docs/DESIGN.md D5):

* :data:`LAK2026` — the Phi-2 study as published: stems under 10 characters,
  empty/duplicate choices, choices under 5 characters, "all/none of the
  above", dichotomous choices, choices starting with "both"/"neither",
  explanations under 10 characters. Normalization is ``lower().strip()``.
* :data:`V2` — the close-the-loop revision for a domain with valid short
  answers ("CTA", "KLI"): choice floor 2, the explanation lives in
  ``feedback``, normalization also collapses whitespace and strips a
  trailing period, the both/neither rule requires a following space, and
  exact-stem duplicates within a unit are dropped (first occurrence kept).

:func:`rule_failures` names every rule a question violates (empty list =
pass), so rejects are auditable; :func:`filter_questions` applies a profile
to a list and returns survivors plus reject records. No model is involved.
"""

import re
from collections import Counter
from dataclasses import dataclass, replace

FLAWED_CHOICES = frozenset({"all of the above", "none of the above"})
DICHOTOMOUS = frozenset({"yes", "no", "true", "false"})


@dataclass(frozen=True)
class FilterRules:
    name: str
    min_stem_chars: int = 10
    min_choice_chars: int = 5
    min_explanation_chars: int = 10
    explanation_field: str = "explanation"
    #: ``strip``: ``lower().strip()`` (LAK 2026); ``collapse``: also collapse
    #: whitespace and strip a trailing period (close-the-loop V1).
    normalization: str = "strip"
    #: LAK 2026 matched ``startswith("both")``; V1 requires ``"both "``.
    both_neither_space: bool = False
    #: Drop a question whose normalized stem repeats an earlier one in the same batch.
    dedup_stems: bool = False
    #: When set, a question must carry exactly this many distinct choices
    #: (LAK 2026 rejected "fewer than num_choices distinct choices").
    num_choices: int | None = 4

    def with_(self, **changes) -> "FilterRules":
        return replace(self, **changes)


LAK2026 = FilterRules(name="lak2026")
V2 = FilterRules(name="v2", min_choice_chars=2, explanation_field="feedback", normalization="collapse",
                 both_neither_space=True, dedup_stems=True, num_choices=None)
PROFILES = {p.name: p for p in (LAK2026, V2)}


def normalize(text, rules: FilterRules) -> str:
    text = str(text if text is not None else "")
    if rules.normalization == "collapse":
        return re.sub(r"\s+", " ", text).strip().lower().rstrip(".")
    return text.lower().strip()


def rule_failures(q, rules: FilterRules = LAK2026) -> list[str]:
    """Names of every rule the question violates (empty = pass). Order is the
    order of Table 1: stem, choices, explanation."""
    failures = []
    stem = q["question"]["stem"]
    choices = [c["text"] for c in q["question"]["choices"]]
    normed = [normalize(c, rules) for c in choices]
    distinct = set(normed)
    prefixes = ("both ", "neither ") if rules.both_neither_space else ("both", "neither")

    if len(stem.strip() if rules.normalization == "collapse" else stem) < rules.min_stem_chars:
        failures.append("short_stem")
    if any(not c for c in normed):
        failures.append("empty_choice")
    if len(distinct) != len(normed):
        failures.append("duplicate_choice")
    if rules.num_choices is not None and len(distinct) < rules.num_choices:
        if "duplicate_choice" not in failures:
            failures.append("missing_choice")
    if any(0 < len(c) < rules.min_choice_chars for c in normed):
        failures.append("short_choice")
    if any(c in FLAWED_CHOICES for c in normed):
        failures.append("all_or_none_of_the_above")
    if any(c in DICHOTOMOUS for c in normed):
        failures.append("dichotomous_choice")
    if any(c.startswith(prefixes) for c in normed):
        failures.append("both_neither_prefix")
    explanation = q.get(rules.explanation_field, "")
    # LAK 2026 measured the raw explanation; V1 measures the normalized feedback.
    explanation_len = (len(normalize(explanation, rules)) if rules.normalization == "collapse"
                       else len(str(explanation if explanation is not None else "")))
    if explanation_len < rules.min_explanation_chars:
        failures.append("short_explanation")
    return failures


def passes(q, rules: FilterRules = LAK2026) -> bool:
    return not rule_failures(q, rules)


def filter_questions(questions: list, rules: FilterRules = LAK2026) -> tuple[list, list[dict]]:
    """Apply a profile; returns (survivors, reject records ``{"id", "rules"}``)."""
    kept, rejects, seen_stems = [], [], set()
    for q in questions:
        failures = rule_failures(q, rules)
        if rules.dedup_stems:
            key = normalize(q["question"]["stem"], rules)
            if key in seen_stems:
                failures.append("duplicate_stem_in_unit")
        if failures:
            rejects.append({"id": q["id"], "rules": failures})
        else:
            if rules.dedup_stems:
                seen_stems.add(normalize(q["question"]["stem"], rules))
            kept.append(q)
    return kept, rejects


def rule_counts(rejects: list[dict]) -> Counter:
    c = Counter()
    for r in rejects:
        c.update(r["rules"])
    return c

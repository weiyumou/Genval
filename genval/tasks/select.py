"""MCQ selection by KC coverage (LAK 2026).

After clustering an LO's validated questions into KCs, ``select_mcq`` keeps
every cluster exemplar and, if more questions are needed, samples the rest
inversely proportional to their cluster's size — favoring coverage of small
clusters over redundancy in large ones. ``build_report`` renders selected
questions as a human-readable table keyed to the course standards.
"""

import copy

import numpy as np
import pandas as pd
from kcluster.core.question import Question


def select_mcq(kc: pd.DataFrame, questions: list[Question],
               mcq_per_lo: int = 6, rng: np.random.Generator = None) -> list[Question]:
    inv_val_counts = kc.value_counts("KC-raw", sort=True).rdiv(1.0)
    exemplars = [int(x.split("-")[1]) for x in inv_val_counts.index.tolist()]

    # all exemplars are selected; if not enough, randomly sample others
    if (n := mcq_per_lo - len(exemplars)) > 0:
        weights = kc["KC-raw"].apply(lambda x: inv_val_counts.loc[x])
        weights.loc[exemplars] = 0.0
        smp = kc["KC-raw"].sample(n=n, replace=False, weights=weights, random_state=rng)
        exemplars.extend(smp.index.tolist())

    return [questions[idx] for idx in exemplars[:mcq_per_lo]]


def build_report(questions: list[Question], std_to_code: dict[str, str]) -> pd.DataFrame:
    records = []
    for q in questions:
        choices = copy.deepcopy(q["question"]["choices"])
        ans_txt = choices.pop(ord(q["answerKey"]) - ord("a"))["text"]
        q_dict = {"Standard Code": None, "Standard Text": q["lo"][0].upper() + q["lo"][1:].rstrip(".*") + ".",
                  "Stem": q["question"]["stem"], "Answer": ans_txt, "Key": q["answerKey"], }
        q_dict["Standard Code"] = std_to_code[q_dict["Standard Text"].rstrip(".")]
        for idx, item in enumerate(choices, 1):
            q_dict[f"Distractor {idx}"] = item["text"]
        q_dict |= {"Explanation": q["explanation"], "KC": q["kc"]}
        records.append(q_dict)

    return pd.DataFrame.from_records(records)


# --- The study's per-LO similarity (LAK 2026 KC selection) ---

def study_similarity(marginals: np.ndarray, conditionals: np.ndarray) -> np.ndarray:
    """The question-congruity matrix the LAK 2026 study clustered on, from
    the raw scores of one LO's ``PairQuestion`` grid (``conditionals[i, j]``
    = log P(Q_j | Q_i), ``marginals[j]`` = log P(Q_j)).

    This is the study's KCluster snapshot as of its KC-selection runs
    (January 2025): the conditionals log-softmaxed over the *conditioning*
    axis (column-wise), the marginals log-softmaxed over the questions and
    subtracted per row, then symmetrized. The snapshot's later revision
    (2025-01-31) normalized row-wise instead, and KCluster's canonical
    ``PointwiseMutualInfo`` (EDM 2025, its decision D1) normalizes through
    the joint distribution; neither reproduces the study's archived
    ``pmi-kc.csv``, this one does exactly (tests/repro). Computed in float32
    like the original.
    """
    from scipy.special import log_softmax

    marg = log_softmax(np.asarray(marginals, dtype=np.float32))
    mat = log_softmax(np.asarray(conditionals, dtype=np.float32), axis=0) - marg[:, None]
    return (mat + mat.T) / 2


def load_study_similarity(pmi_dir: str, n_questions: int) -> np.ndarray:
    """``study_similarity`` over the score shards of one LO (``run_pmi.py`` layout)."""
    from kcluster.core.pmi import PointwiseMutualInfo

    pmi = PointwiseMutualInfo.from_shards(pmi_dir, n_questions, n_questions, symmetric=False, normalize=False)
    # The raw [marginals; conditionals] the shards hold, before any estimator is applied.
    return study_similarity(pmi._vec, pmi._mat)

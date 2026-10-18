"""LO-alignment: matching questions to learning objectives (LAK 2026).

``QuestionLO`` flattens one rectangular LO×question scoring job into a
linear index space of (context, text) pairs: the first ``n_questions``
items are the marginals — the scored question under its bare type header —
and item ``n + i*n + j`` scores question ``j`` conditioned on learning
objective ``i``. The reassembled ``pmi_mat`` is therefore rectangular with
rows = LOs (conditioning) and columns = questions, and uses the raw PMI
estimator only (``normalize=False``; the joint-normalization path is
square-only by design).

LOs come in two grammatical shapes with different scaffolds: ``actions``
("... test whether a student can <verb phrase>") and ``facts`` ("... test
whether a student knows:\\n<statement>").
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.core.question import Question
from kcluster.io.jsonl import dump_questions, load_questions

from genval.core import prompts


def distinct_los(questions: list[Question]) -> list[str]:
    """The questions' learning objectives, once each, in order of first appearance.

    Every rank of a multi-GPU run is its own process and rebuilds the scoring
    grid from this list, while only rank 0's copy is saved for
    ``classify_from_pmi`` to read the scores back with. So the order has to be
    the same in every process, which a set's is not: it follows the string
    hashes, and those are salted per process.
    """
    return list(dict.fromkeys(q["lo"] for q in questions))


class QuestionLO:
    def __init__(self, questions: list[Question], los: list[str], lo_type: str):
        match lo_type:
            case "actions":
                self.los = [lo[0].lower() + lo[1:].rstrip().rstrip(".") for lo in los]
            case "facts":
                self.los = [lo.rstrip().rstrip(".") for lo in los]
            case _:
                raise ValueError(f"Invalid lo_type: '{lo_type}'")

        self.questions, self.lo_type = questions, lo_type

    def __getitem__(self, index):
        n = len(self.questions)
        if index < n:
            q = self.questions[index]
            return prompts.LO_ALIGNMENT_MARGINAL_CONTEXT.format(q_type=q.q_type), str(q)

        lo_idx, q_idx = (index - n) // n, (index - n) % n
        lo, q = self.los[lo_idx], self.questions[q_idx]
        header = None
        match self.lo_type:
            case "actions":
                header = prompts.LO_ACTIONS_HEADER.format(lo=lo)
            case "facts":
                header = prompts.LO_FACTS_HEADER.format(lo=lo)

        return f"{header}\n\n" + prompts.LO_ALIGNMENT_MARGINAL_CONTEXT.format(q_type=q.q_type), str(q)

    def __len__(self):
        return len(self.questions) * len(self.los) + len(self.questions)


def classify_from_pmi(root_dir: str, topk: int = 3) -> pd.DataFrame:
    """Aggregate a scored classify run into top-k LO predictions per question.

    Reads the score shards plus the ``args-pmi-*.json`` / ``los-*.json``
    breadcrumbs the classify command writes, saves the questions whose true
    ``lo`` appears in the top-k (``matched-top{k}.jsonl``) and the full
    prediction table (``classified-top{k}.csv``).
    """
    # Read the args json file to extract data_path
    [args_f] = glob.glob(os.path.join(root_dir, "args-pmi-*.json"))
    with open(args_f, "r") as f:
        args = json.load(f)

    # Load questions from data_path
    all_questions = load_questions(args["data_path"])

    # Load LOs
    [los_f] = glob.glob(os.path.join(root_dir, "los-*.json"))
    with open(los_f, "r") as f:
        all_los = json.load(f)

    pmi = PointwiseMutualInfo.from_shards(root_dir, len(all_los), len(all_questions),
                                          symmetric=False, normalize=False)

    # Top K predictions (descending, per question)
    preds = np.argsort(-pmi.pmi_mat, axis=0)[:topk].T  # (n_questions, topk)
    records, matched = [], []
    for idx, q in enumerate(all_questions):
        d, is_matched = q.flat_dict, False
        for k, p in enumerate(preds[idx], 1):
            d |= {f"pred_lo_{k}": all_los[p]}
            is_matched |= q["lo"] == all_los[p]
        records.append(d)
        if is_matched:
            matched.append(q)

    # Save matched questions
    dump_questions(matched, os.path.join(root_dir, f"matched-top{topk}.jsonl"))
    print(f"Matched {len(matched)} questions out of {len(all_questions)}")

    # Save classification results
    res_df = pd.DataFrame.from_records(records)
    res_df.to_csv(os.path.join(root_dir, f"classified-top{topk}.csv"), index=False)

    return res_df


# --- Grid scoring beyond the single-process classify command ---

def question_meta(q) -> dict:
    return {"id": q["id"], "lo": q.get("lo"), "unit": q.get("unit")}


def save_relevance(root_dir: str, los: list[str], questions: list[Question], tag: str) -> str:
    """Assemble the scored shards into the LO x question relevance grid and
    persist it (``relevance-<tag>.npy``, the raw ``[marginals; conditionals]``
    layout of ``PointwiseMutualInfo.save``) with row/column metadata."""
    pmi = PointwiseMutualInfo.from_shards(root_dir, len(los), len(questions), symmetric=False, normalize=False)
    path = os.path.join(root_dir, f"relevance-{tag}.npy")
    pmi.save(path)
    with open(os.path.join(root_dir, f"cols-{tag}.json"), "w") as f:
        json.dump([question_meta(q) for q in questions], f)
    return path


def load_relevance(root_dir: str, tag: str):
    """(relevance matrix rows = LOs x cols = questions, LOs, column metadata)."""
    pmi = PointwiseMutualInfo.from_npy(os.path.join(root_dir, f"relevance-{tag}.npy"),
                                       symmetric=False, normalize=False)
    with open(os.path.join(root_dir, f"los-{tag}.json")) as f:
        los = json.load(f)
    with open(os.path.join(root_dir, f"cols-{tag}.json")) as f:
        cols = json.load(f)
    return pmi.pmi_mat, los, cols


def score_grid(questions: list[Question], los: list[str], lo_type: str, llm_path: str, output_dir: str,
               batch_size: int = 16, num_workers: int = 2, pad_to_multiple_of: int = 8,
               devices=-1, strategy: str = "auto", **model_args) -> bool:
    """Score the LO x question grid with Lightning (multi-GPU DDP when
    launched with one task per GPU). Returns True on the rank that owns the
    output (global zero); other ranks return False after the barrier."""
    from functools import partial

    import lightning as L
    from kcluster.engine.local import CustomWriter, LargeLangModel, LogProbScorer, collate_pair
    from torch.utils.data import DataLoader

    tokenizer = LargeLangModel.load_tokenizer(llm_path)
    model = LogProbScorer(llm_path, trust_remote_code=True, **model_args)
    ds = QuestionLO(questions, los, lo_type)
    dl = DataLoader(ds, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers,
                    collate_fn=partial(collate_pair, tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of))
    writer = CustomWriter(output_dir=output_dir, write_interval="epoch")
    trainer = L.Trainer(accelerator="gpu", strategy=strategy, devices=devices, callbacks=[writer], logger=False)
    trainer.predict(model, dataloaders=dl, return_predictions=False)
    trainer.strategy.barrier()
    return bool(trainer.is_global_zero)

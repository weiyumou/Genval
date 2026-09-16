"""``genval classify``: LO alignment by relevance (LAK 2026 §Alignment).

Scores ``Relevance(L, Q) = log P(Q | L) - log P(Q)`` for every (objective,
question) pair on Phi-2, then keeps the questions whose top-k objectives
include their own ``lo``. Input is either a question JSONL (``--data_path``)
or an LLM pool (``--pool_dir --model``, the survivors of ``--stage``, written
back per unit as ``v3-<model>.jsonl``). The objectives are the distinct
``lo`` values of the input unless ``--los_path`` (a JSON list or one per
line) supplies the candidate set. Multi-GPU: launch one task per GPU
(``srun --ntasks=<gpus>``); rank 0 aggregates.
"""

import argparse
import json
import os

from kcluster.io.jsonl import dump_questions, load_questions
from transformers.utils import logging

from genval.core.units import load_units
from genval.paths import default_output_dir, prepare_output_dir, step_dir
from genval.tasks.align.phi2 import classify_from_pmi, save_relevance, score_grid


def read_los(path: str) -> list[str]:
    with open(path) as f:
        text = f.read()
    try:
        los = json.loads(text)
        if isinstance(los, list):
            return [str(x) for x in los]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(args):
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    pool_dir = getattr(args, "pool_dir", None) or (step_dir("llm", getattr(args, "run_dir", None))
                                                   if not getattr(args, "data_path", None) else None)
    if getattr(args, "data_path", None):
        data_path = args.data_path
        units = None
    elif pool_dir:
        from genval.tasks.generate.rounds import screens_dir, stage_questions
        units = load_units(os.path.join(pool_dir, "units.jsonl"))
        sdir = screens_dir(pool_dir, getattr(args, "screens_id", None))
        pooled = [q for u in units for q in stage_questions(args.model, u, sdir, args.stage)]
        os.makedirs(os.path.join(sdir, "classify"), exist_ok=True)
        data_path = os.path.join(sdir, "classify", f"{args.stage}-{args.model}.jsonl")
        dump_questions(pooled, data_path)
        args.output_dir = getattr(args, "output_dir", None) or os.path.join(sdir, "classify")
    else:
        raise SystemExit("give --data_path, or --pool_dir/--run_dir with --model")

    output_dir = getattr(args, "output_dir", None) or default_output_dir("classify", getattr(args, "run_dir", None))
    args.output_dir = prepare_output_dir(output_dir)
    print(f"*** Writing results to {args.output_dir} ***")

    # Read all questions and collect their LOs
    all_questions = load_questions(data_path)
    all_los = read_los(args.los_path) if getattr(args, "los_path", None) else list(set(q["lo"] for q in all_questions))

    fname = os.path.splitext(os.path.basename(data_path))[0]
    is_zero = score_grid(all_questions, all_los, args.lo_type, args.llm_path, args.output_dir,
                         batch_size=args.batch_size, num_workers=args.num_workers,
                         pad_to_multiple_of=args.pad_to_multiple_of, torch_dtype=__import__("torch").float16)
    if not is_zero:
        return  # rank 0 owns aggregation

    # Save LOs
    with open(os.path.join(args.output_dir, f"los-{fname}.json"), "w") as f:
        json.dump(all_los, f)

    # Save arguments
    with open(os.path.join(args.output_dir, f"args-pmi-{fname}.json"), "w") as f:
        json.dump(vars(args) | {"data_path": data_path}, f, indent=2)

    # Aggregate into top-k predictions per question, and keep the assembled grid
    save_relevance(args.output_dir, all_los, all_questions, fname)
    classify_from_pmi(args.output_dir, topk=args.topk)

    if units is not None:
        matched = {q["id"] for q in load_questions(os.path.join(args.output_dir, f"matched-top{args.topk}.jsonl"))}
        for u in units:
            qs = [q for q in stage_questions(args.model, u, sdir, args.stage) if q["id"] in matched]
            udir = os.path.join(sdir, str(u.subdir))
            if os.path.isdir(udir):
                dump_questions(qs, os.path.join(udir, f"v3-{args.model}.jsonl"))


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", default=argparse.SUPPRESS, type=str,
                        help="Path to a jsonl file of questions with an 'lo' field")
    parser.add_argument("--pool_dir", default=argparse.SUPPRESS, type=str,
                        help="An LLM pool (the llm-generate output dir) instead of --data_path")
    parser.add_argument("--model", default=argparse.SUPPRESS, type=str, help="Generator model of the pool")
    parser.add_argument("--stage", default="v2", type=str, help="Pool stage to align (default: v2 survivors)")
    parser.add_argument("--screens_id", default=argparse.SUPPRESS, type=str, help="Pool screens dir (default: latest)")
    parser.add_argument("--lo_type", required=True, type=str, choices=("actions", "facts"),
                        help="Type of LOs in the data")
    parser.add_argument("--los_path", default=argparse.SUPPRESS, type=str,
                        help="Candidate objectives (JSON list or one per line); default: the data's distinct 'lo'")
    parser.add_argument("--topk", type=int, default=3, help="Number of LO predictions to keep per question")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--output_dir", type=str, default=argparse.SUPPRESS, help="Path to the output directory")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Shared run folder; each step writes to <run_dir>/<step> (env: GENVAL_RUN_DIR)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader")
    parser.add_argument("--pad_to_multiple_of", type=int, default=8, help="Pad to multiple of")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())

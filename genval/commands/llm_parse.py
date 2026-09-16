"""``genval llm-parse``: raw LLM responses of one round -> question JSONL per unit."""

import argparse
import os

from genval.core.units import load_units
from genval.paths import step_dir
from genval.tasks.generate.parse import parse_unit
from genval.tasks.generate.rounds import latest_round, rounds_root


def resolve_pool(args) -> str:
    pool = getattr(args, "pool_dir", None) or step_dir("llm", getattr(args, "run_dir", None))
    if not pool or not os.path.isdir(pool):
        raise SystemExit("give --pool_dir (the llm-generate output dir) or --run_dir / GENVAL_RUN_DIR")
    return pool


def main(args):
    pool = resolve_pool(args)
    units = load_units(os.path.join(pool, "units.jsonl"))
    rdir = (rounds_root(pool) / args.round) if getattr(args, "round", None) else latest_round(pool)
    if rdir is None or not rdir.exists():
        raise SystemExit(f"no such round under {rounds_root(pool)}")
    total_q = total_e = n_units = 0
    for unit in units:
        nq, ne = parse_unit(unit, args.model, rdir)
        n_units += bool(nq or ne)
        total_q += nq
        total_e += ne
    print(f"{args.model} [{rdir.name}]: parsed {total_q} questions from {n_units} units "
          f"({total_e} malformed entries -> parse-errors-{args.model}.jsonl)")


def add_arguments(parser):
    parser.add_argument("--model", required=True, type=str, help="Generator model whose raw files to parse")
    parser.add_argument("--pool_dir", default=argparse.SUPPRESS, type=str, help="The llm-generate output dir")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str, help="Shared run folder (env: GENVAL_RUN_DIR)")
    parser.add_argument("--round", default=argparse.SUPPRESS, type=str, help="Round to parse (default: latest)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())

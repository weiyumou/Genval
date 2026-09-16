"""``genval filter``: rule-based syntactic filtering (LAK 2026 Table 1).

Two inputs: an LLM pool (``--pool_dir --model``: every unit's parsed
questions unioned across rounds -> ``screens/<sid>/<unit>/v1-<model>.jsonl``
plus ``v1-rejects-<model>.jsonl`` with the failed rule names), or a single
question file (``--input --output``). ``--profile`` selects the registered
rule set (``v2`` for LLM output with a ``feedback`` field, ``lak2026`` for
Phi-2 output with an ``explanation`` field).
"""

import argparse
import json
import os

from kcluster.io.jsonl import dump_questions, load_questions

from genval.commands.llm_parse import resolve_pool
from genval.core.units import load_units, unit_dir
from genval.tasks.filter import PROFILES, filter_questions, rule_counts
from genval.tasks.generate.rounds import list_rounds, pool_questions, screens_dir


def write_rejects(path, rejects: list[dict]) -> None:
    if rejects:
        with open(path, "w") as f:
            for r in rejects:
                f.write(json.dumps(r) + "\n")
    elif os.path.exists(path):
        os.unlink(path)


def main(args):
    rules = PROFILES[args.profile]
    if getattr(args, "input", None):
        questions = load_questions(args.input)
        kept, rejects = filter_questions(questions, rules)
        dump_questions(kept, args.output)
        write_rejects(os.path.splitext(args.output)[0] + "-rejects.jsonl", rejects)
        print(f"{args.profile}: kept {len(kept)}/{len(questions)} -> {args.output}")
        for rule, n in rule_counts(rejects).most_common():
            print(f"  {rule}: {n}")
        return

    pool = resolve_pool(args)
    units = load_units(os.path.join(pool, "units.jsonl"))
    sdir = screens_dir(pool, getattr(args, "screens_id", None))
    rounds = [r.name for r in list_rounds(pool)]
    total_in = total_kept = 0
    all_rejects, per_unit_kept = [], []
    for unit in units:
        questions = pool_questions(args.model, unit, pool)
        if not questions:
            continue
        udir = unit_dir(sdir, unit)
        udir.mkdir(parents=True, exist_ok=True)
        kept, rejects = filter_questions(questions, rules)
        dump_questions(kept, str(udir / f"v1-{args.model}.jsonl"))
        write_rejects(udir / f"v1-rejects-{args.model}.jsonl", rejects)
        total_in += len(questions)
        total_kept += len(kept)
        per_unit_kept.append(len(kept))
        all_rejects.extend(rejects)
    sdir.mkdir(parents=True, exist_ok=True)
    meta_path = sdir / "screens.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"screens_id": sdir.name, "rounds": rounds, "filter_profile": args.profile})
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    per_unit_kept.sort()
    n = len(per_unit_kept)
    print(f"{args.model} [screens/{sdir.name}, rounds {'+'.join(rounds)}]: kept {total_kept}/{total_in} "
          f"(per-unit min={per_unit_kept[0] if n else 0} median={per_unit_kept[n // 2] if n else 0})")
    for rule, count in rule_counts(all_rejects).most_common():
        print(f"  {rule}: {count}")


def add_arguments(parser):
    parser.add_argument("--profile", default="v2", choices=sorted(PROFILES), help="Rule profile (default v2)")
    parser.add_argument("--model", default=argparse.SUPPRESS, type=str, help="Generator model (pool input)")
    parser.add_argument("--pool_dir", default=argparse.SUPPRESS, type=str, help="The llm-generate output dir")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str, help="Shared run folder (env: GENVAL_RUN_DIR)")
    parser.add_argument("--screens_id", default=argparse.SUPPRESS, type=str,
                        help="Screens dir name (default: the latest round, e.g. r02)")
    parser.add_argument("--input", default=argparse.SUPPRESS, type=str, help="A single question JSONL to filter")
    parser.add_argument("--output", default=argparse.SUPPRESS, type=str, help="Where to write the survivors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())

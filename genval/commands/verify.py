"""``genval verify``: answer agreement by LLM judges over a pool's survivors.

Reads ``screens/<sid>/<unit>/<stage>-<model>.jsonl`` (default stage v1),
fires the pending judge calls (resume-safe; rerun to retry errors), and
writes per unit ``verify-report-<model>.jsonl`` (every question's votes and
verdict) and ``v2-<model>.jsonl`` (survivors). Judges are ``model@transport``
specs; the default trio is the close-the-loop instrument. Secrets come from
the environment (``uv run --env-file .env ...``).
"""

import argparse
import json
import os
from collections import Counter

from kcluster.io.jsonl import dump_questions

from genval.commands.llm_parse import resolve_pool
from genval.core.units import load_units, unit_dir
from genval.engine.dispatch import fire, run, successful_ids
from genval.tasks.generate.rounds import screens_dir, stage_questions
from genval.tasks.verify import judges as J


async def fire_all(specs, questions, sdir, domain, concurrency, model):
    for spec in specs:
        name, _ = J.parse_judge_spec(spec)
        await fire(J.build_calls(name, questions, domain), J.judge_worker(spec), J.raw_path(sdir, name),
                   concurrency, f"verify [{model} / {name}]")


def main(args):
    pool = resolve_pool(args)
    units = load_units(os.path.join(pool, "units.jsonl"))
    sdir = screens_dir(pool, getattr(args, "screens_id", None))
    specs = [s.strip() for s in args.judges.split(",") if s.strip()]
    names = [J.parse_judge_spec(s)[0] for s in specs]
    per_unit = [(u, stage_questions(args.model, u, sdir, args.stage)) for u in units]
    per_unit = [(u, qs) for u, qs in per_unit if qs]
    questions = [q for _, qs in per_unit for q in qs]
    domain = getattr(args, "domain", None)
    if args.dry_run:
        print(f"{args.model} [screens/{sdir.name}] rule={args.rule}: {len(questions)} {args.stage} survivors")
        for name in names:
            calls = J.build_calls(name, questions, domain)
            done = successful_ids(J.raw_path(sdir, name))
            print(f"  {name}: {len(calls)} calls, {len(done)} done, {sum(c['id'] not in done for c in calls)} to fire")
        if questions:
            print(f"--- sample prompt ---\n{J.render_call(names[0], questions[0], 0, domain)['prompt']}")
        return
    run(fire_all(specs, questions, sdir, domain, args.concurrency, args.model))

    votes = {name: J.collect_votes(J.raw_path(sdir, name)) for name in names}
    totals = Counter()
    for unit, qs in per_unit:
        rows, kept = J.judge_questions(qs, votes, names, args.rule)
        udir = unit_dir(sdir, unit)
        with open(udir / f"verify-report-{args.model}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
                totals[row["verdict"]] += 1
        dump_questions(kept, str(udir / f"v2-{args.model}.jsonl"))
    print(f"{args.model} [screens/{sdir.name}]: rule={args.rule} " +
          " ".join(f"{k}={totals[k]}" for k in ("verified", "disputed", "nota", "unverified", "incomplete")))
    if totals["incomplete"]:
        print(f"  {totals['incomplete']} items still have unanswered calls — rerun the same command to retry")


def add_arguments(parser):
    parser.add_argument("--model", required=True, type=str, help="Generator model whose survivors to verify")
    parser.add_argument("--judges", default=",".join(J.DEFAULT_JUDGES), type=str,
                        help="Comma-separated judge specs model@transport (default: the close-the-loop trio)")
    parser.add_argument("--rule", default="unanimous", choices=J.RULES,
                        help="unanimous = every judge verifies, no distractor votes; majority = one verifier")
    parser.add_argument("--domain", default=argparse.SUPPRESS, type=str,
                        help='Domain phrase for the judge prompt, e.g. "e-learning and instructional design"')
    parser.add_argument("--stage", default="v1", type=str, help="Which survivors to verify (default v1)")
    parser.add_argument("--pool_dir", default=argparse.SUPPRESS, type=str, help="The llm-generate output dir")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str, help="Shared run folder (env: GENVAL_RUN_DIR)")
    parser.add_argument("--screens_id", default=argparse.SUPPRESS, type=str, help="Screens dir name (default: latest)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true", help="Count pending calls, no API use")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())

# Examples

A synthetic sample you can run the pipelines on immediately. The 15
questions in `data/sample-mcq.jsonl` are original to this repository
(written for it, derived from no course or dataset) and form three obvious
topic clusters — simple machines, states of matter, food chains. Each
question carries an `lo` field matching `data/standards/actions/sample.txt`,
and `data/units.jsonl` turns those three objectives into generation units
for the LLM pipeline (two sample questions each as exemplars).

## Generate-then-validate (LAK 2026): standards → new MCQs

Needs a local Phi-2 checkout and a GPU (`pip install "genval[local]"`):

    hf download microsoft/phi-2 --local-dir phi-2

    genval generate --llm_path phi-2 --std_dir examples/data/standards \
        --config_path examples/data/qgen-config.toml --qs_per_std 8
    genval validate --llm_path phi-2 --root_dir <generate-run-dir>

`generate` grows one MCQ per seed incrementally (stem → choices →
answer → explanation) for each standard; `validate` keeps only
complete questions whose answer clears the permutation-averaged confidence
threshold (`--prob_thd`, default 0.9), sorted by perplexity.

`--config_path` supplies per-step generation settings (TOML or JSON, one
table per step);
[`qgen-config.toml`](data/qgen-config.toml) holds the LAK 2026 study's.
It is effectively required: asking for more than one question per standard
needs sampling on the stem step, which the transformers defaults (greedy)
cannot do.

## LO alignment (LAK 2026): match questions to objectives

Also on local Phi-2 (`pip install "genval[local]"`):

    export DATA_PATH=examples/data/sample-mcq.jsonl
    genval classify --llm_path phi-2 --data_path $DATA_PATH --lo_type actions

Writes `classified-top3.csv` (top-3 predicted LOs per question) and
`matched-top3.jsonl` (questions whose true `lo` is in the top 3).

## LLM pipeline (Gemini / Claude)

No GPU needed (`pip install "genval[llm]"`, or `[gemini]` / `[claude]` for
one engine). The four stages share one pool directory; every call is
id-stamped and resumable.

    genval llm-generate --backend gemini --model gemini-2.5-flash \
        --std_dir examples/data/standards --target 10 --per_call 5 \
        --output_dir results/llm-demo
    genval llm-parse --model gemini-2.5-flash --pool_dir results/llm-demo
    genval filter --model gemini-2.5-flash --pool_dir results/llm-demo
    genval verify --model gemini-2.5-flash --pool_dir results/llm-demo \
        --judges "gemini-2.5-flash@api" --rule majority
    genval classify --llm_path phi-2 --pool_dir results/llm-demo \
        --model gemini-2.5-flash --lo_type actions

`llm-generate` writes one unit per standards line (or per line of a units
file: `--units examples/data/units.jsonl --exemplar_bank
examples/data/sample-mcq.jsonl` generates from the exemplars instead);
`llm-parse` turns the raw responses into question JSONL; `filter` applies
the rule-based syntactic criteria (LAK 2026 Table 1); `verify` asks the
listed judges to answer every permutation of each question's choices and
keeps the items on which they agree with the key; the final `classify`
runs the Phi-2 LO-alignment grid on the survivors and so needs the local
checkout above.

Secrets come from a `.env` file loaded by `uv run --env-file .env genval ...`:
`GEMINI_API_KEY` for `gemini@api`, `VERTEX_PROJECT_ID` / `VERTEX_REGION`
for `@vertex`, and `ANTHROPIC_API_KEY` for `claude@api`.

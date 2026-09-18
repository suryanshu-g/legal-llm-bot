# legal-llm-bot

https://github.com/suryanshu-g/legal-llm-bot

A retrieval-grounded assistant for Indian criminal law, built around the
1 July 2024 transition from the colonial-era codes to the new ones.

| Old code | Replaced by | Sections |
|---|---|---|
| Indian Penal Code, 1860 | **BNS** — Bharatiya Nyaya Sanhita, 2023 | 358 |
| Code of Criminal Procedure, 1973 | **BNSS** — Bharatiya Nagarik Suraksha Sanhita, 2023 | 531 |
| Indian Evidence Act, 1872 | **BSA** — Bharatiya Sakshya Adhiniyam, 2023 | 170 |

Generic chatbots, trained on data spanning the changeover, routinely mix up old
and new section numbers. The premise here is that a model answering from
retrieved authoritative text should do better on "which section applies" and
"what does section X say" than one relying on parametric memory.

Capstone project for a Data Science / Agentic AI programme. Submitted under
**Government and Public Administration** — the official industry list has no
"Legal Services" category, and codified criminal law administered by police,
prosecutors and courts sits squarely inside public administration.

> **This is not legal advice.** The bot describes what the law and reported
> decisions say and cites its source. It does not advise, does not suggest ways
> to evade liability, and does not represent anyone.

## Status

**Phases 1 through 3.5 are complete.** Retrieval takes the old-to-new
correspondence from 0% to 99% on held-out questions. Phase 3.6, which teaches the
model to answer "no", is built and awaiting a Colab run.

| Deliverable | |
|---|---|
| Fine-tuning QA pairs | 14,524 |
| Train / validation / test | 11,751 / 1,369 / 676 (leakage-safe, group-level) |
| Confusion test set | 44 held-out old-vs-new questions |
| Retrieval chunks | 2,819 |
| Concordance rows | 1,324 |
| BNSS First Schedule | 465 rows over 288 BNS sections |
| Case summaries | 201 records over 174 judgments, 30 topics |
| Section coverage | BNS 358/358 · BNSS 531/531 · BSA 170/170 |
| Retrieval index | 2,819 chunks, bge-small-en-v1.5 + FAISS, 96% recall@3 |
| Context-augmented training | 11,751 / 1,369 rows, 70% positive · 20% distractor · 10% none |
| Context shape | up to 4 passages per prompt, assembled by `bot.py` itself |
| Negation training shapes | 1,006 rows: collision · merged · no_single |

**Retrieval works.** On 676 held-out questions the model names the right
provisions **92.7% of the time with retrieval and 42.2% without**. On the
questions that matter most — what a provision became after 1 July 2024 — it was
right in **0 of 214** cases in Phase 2 and is now right in **213 of 214**.
Retrieval is within 0.5 F1 of its own oracle, so little is left to gain there.

**What still does not work is the confusion set**, stuck at 23.3% through three
runs. The cause is now identified and is a gap in the data rather than the
architecture: 97.7% of its answers open with a negation, against 0.3% of the
training answers, and 33 of its 44 questions are shapes the training set did not
contain at all. Phase 3.6 adds them. Full numbers, the costs and the caveats in
[`RESULTS.md`](RESULTS.md).

Read [`data/DATA_REPORT.md`](data/DATA_REPORT.md) for how the data was verified and
what is wrong with it. [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) holds the full
project context.

- **Phase 2** — fine-tune Flan-T5-base on Colab (T4, ≤25 epochs) using
  `train.jsonl` and `val.jsonl`. Notebook ready at
  [`notebooks/finetune_flan_t5.ipynb`](notebooks/finetune_flan_t5.ipynb) —
  step-by-step Colab instructions in
  [`notebooks/COLAB_SETUP.md`](notebooks/COLAB_SETUP.md).
- **Phase 2.5** — retrieval index built; a second notebook,
  [`notebooks/finetune_flan_t5_contextaware.ipynb`](notebooks/finetune_flan_t5_contextaware.ipynb),
  fine-tunes the model to read a retrieved passage.
- **Phase 3** — the retriever and the context-aware model are wired into one
  assistant, [`scripts/bot.py`](scripts/bot.py), driven by
  [`notebooks/run_bot.ipynb`](notebooks/run_bot.ipynb).
- **Phase 3.5** — retrain on context shaped the way the bot serves it, using
  [`notebooks/finetune_flan_t5_small_contextaware.ipynb`](notebooks/finetune_flan_t5_small_contextaware.ipynb).
  Data rebuilt; the run itself is pending.
- **Phase 3.6** — add the negation question shapes the training data lacked
  ([`scripts/build_negation_dataset.py`](scripts/build_negation_dataset.py)) and
  retrain. Data built; the run itself is pending.
- **Phase 4** — run `confusion_test_set.jsonl` against both this bot and a
  general-purpose LLM to test the core claim; paper and video.

### Asking it something

```bash
python scripts/bot.py "Which BNS section replaced IPC Section 302?"
python scripts/bot.py --sources-only "Is an offence under BNS Section 303 bailable?"
```

Point `--model-dir` (or `$LEGAL_BOT_MODEL`) at the fine-tuned model to get a
composed answer; without one the bot runs in sources-only mode and returns the
retrieved law itself rather than inventing prose. Either way every answer carries
its citations and the informational-only disclaimer.

**The counterpart lookup is the Phase 3 result.** Phase 2.5 found retrieval
barely helped the confusion set because 33 of its 44 answers name two or more
provisions while the run supplied a single passage. Looking counterparts up in
the concordance rather than hoping the embedder surfaces them takes context
completeness on that set from **34% to 98%**.

**Phase 3.5 fixed a format mismatch behind it.** Every context-augmented
training example held one passage while the bot serves four, so the model read the
first and ignored the rest. Training data is now generated by calling the bot's own
assembly code, which makes the two impossible to drift apart. That took test-set
citation accuracy from 86.8% to 92.7% — and left the confusion set untouched,
which is what Phase 3.6 addresses.

## Layout

```
data/
  raw/                      source documents, organised by act
    source_manifest.json    URL, retrieval date and SHA-256 for every fetch
    mapping/                official BPRD and UP Police concordance PDFs
    case_law/               Indian Kanoon cache and case_summaries.json
  processed/
    mapping_table.csv           old -> new section concordance
    mapping_disagreements.csv   where the sources disagree, and why
    bnss_schedule.csv           First Schedule: cognizable / bailable / court
    bnss_schedule_disagreements.csv
    finetune_dataset.jsonl      instruction / input / output
    finetune_dataset_index.jsonl  aligned qa_type + provenance
    train.jsonl / val.jsonl / test.jsonl   group-level split, no fact straddles
    confusion_test_set.jsonl    held-out old-vs-new questions for Phase 4
    retrieval_corpus.jsonl      one chunk per section, schedule entry or judgment
    train_index.jsonl / val_index.jsonl / test_index.jsonl
                                aligned qa_type per split row, for per-type metrics
    train_context.jsonl / val_context.jsonl
                                context-augmented training data
    negation_train.jsonl / negation_val.jsonl
                                collision / merged / no_single shapes, + indexes
    negation_report.json        pools, quotas and the leakage checks
    retrieval_index.faiss       FAISS index over the corpus
    retrieval_index_meta.jsonl  FAISS row -> chunk metadata
    retrieval_index_config.json embedding model, dim, build checks
    split_report.json           split sizes and qa_type balance
    validation_report.json      output of the validation run
  DATA_REPORT.md
scripts/
  lawlib.py                 shared: cached polite fetching, PDF line geometry
  scrape_acts.py            fetch and parse the six acts
  scrape_case_law.py        harvest case summaries from Indian Kanoon
  build_mapping_table.py    build and cross-check the concordance
  build_bnss_schedule.py    parse the BNSS First Schedule
  build_finetune_dataset.py build the QA dataset
  build_retrieval_corpus.py build the RAG corpus
  build_splits.py           group-level splits + confusion set
  build_retrieval_index.py  embed the corpus, build the FAISS index
  retrieve.py               reusable hybrid retriever (citations + dense)
  build_context_dataset.py  positive / distractor / no-context training data
  build_negation_dataset.py collision / merged / no_single: teaching it to say no
  check_retrieval.py        retrieval recall@k, broken down by question type
  bot.py                    the assistant: retrieve, answer, cite
  check_bot_context.py      is the answer's law actually in the bot's context?
  validate_data.py          checks; non-zero exit on failure
notebooks/
  finetune_flan_t5.ipynb    Phase 2: fine-tune, evaluate, save the model
  finetune_flan_t5_contextaware.ipynb
                            Phase 2.5: same, trained to read retrieved context
  finetune_flan_t5_small_contextaware.ipynb
                            Phase 3.5: flan-t5-small on bot-shaped context,
                            resumes after a Colab disconnect - run this one
  verify_model.ipynb        check a saved model and recover its metrics
  run_bot.ipynb             Phase 3: the assembled bot, demo questions,
                            scope checks, confusion-set comparison
  COLAB_SETUP.md            step-by-step guide to running it on Colab
```

## Reproducing the datasets

```bash
pip install -r requirements.txt

python scripts/scrape_acts.py           # acts + concordance PDFs
python scripts/scrape_case_law.py       # ~15 min, rate limited
python scripts/build_mapping_table.py
python scripts/build_bnss_schedule.py
python scripts/build_retrieval_corpus.py
python scripts/build_finetune_dataset.py
python scripts/build_splits.py
python scripts/validate_data.py

# Phase 2.5: retrieval
python scripts/build_retrieval_index.py   # ~4 min on CPU
python scripts/check_retrieval.py         # recall@k sanity check
python scripts/build_context_dataset.py

# Phase 3.6: the negation shapes (run before build_context_dataset.py, which
# picks them up automatically)
python scripts/build_negation_dataset.py --dry-run   # inspect, write nothing
python scripts/build_negation_dataset.py

# Phase 3: the bot
python scripts/check_bot_context.py --compare   # is the answer's law in context?
python scripts/bot.py --sources-only            # interactive, no model needed
```

Every fetch is cached under `data/raw/`, so a rerun re-downloads nothing. Pass
`--force` to `scrape_acts.py` to refresh sources.

`validate_data.py` exits non-zero if any check fails, so it can gate a rebuild.
The expected result is **0 failures, 4 warnings**; the warnings are enumerated in
the data report.

## Sources

Statutory text comes from the official Ministry of Home Affairs gazette PDFs.
The old-to-new concordance comes from the three "Correspondence Table and
Comparison Summary" PDFs published by the Bureau of Police Research and
Development (an MHA body), cross-checked against the UP Police comparative table
and against devgan.in. Old-code text comes from devgan.in cross-checked against
the civictech-India JSON corpus. Case law comes from Indian Kanoon.

Under s.52(1)(q) of the Indian Copyright Act 1957 there is no copyright in the
text of an Act of a legislature or in a judgment of a court, so everything
committed here is redistributable. Third-party page markup is cached locally but
git-ignored; only extracted statutory text is committed. Crawling is rate
limited and honours `robots.txt` per URL.

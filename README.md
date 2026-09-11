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

**Phases 1, 1.5 and 2.5 (data + retrieval) are complete.** Training runs on Colab.

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

Read [`data/DATA_REPORT.md`](data/DATA_REPORT.md) for how this was verified and
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
- **Phase 3** — wire the retriever and the context-aware model into one bot.
- **Phase 4** — run `confusion_test_set.jsonl` against both this bot and a
  general-purpose LLM to test the core claim; paper and video.

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
  check_retrieval.py        retrieval recall@k, broken down by question type
  validate_data.py          checks; non-zero exit on failure
notebooks/
  finetune_flan_t5.ipynb    Phase 2: fine-tune, evaluate, save the model
  finetune_flan_t5_contextaware.ipynb
                            Phase 2.5: same, trained to read retrieved context
  verify_model.ipynb        check a saved model and recover its metrics
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

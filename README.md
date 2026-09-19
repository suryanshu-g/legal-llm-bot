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

**All phases are complete.** Retrieval takes the old-to-new correspondence from
0% to 99% on held-out questions, and the 44 hardest old-versus-new questions from
10 right to 27. Measured against ChatGPT, the two systems fail on opposite axes.
The write-up is [`PAPER.md`](PAPER.md); only the video remains.

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
provisions **92.9% of the time with retrieval and 44.3% without**. On the
questions that matter most — what a provision became after 1 July 2024 — it was
right in **0 of 214** cases in Phase 2 and is now right in **213 of 214**.
Retrieval is within 0.5 F1 of its own oracle, so little is left to gain there.

**And the confusion set finally moved.** It sat at 23.3% through three runs
because 97.7% of its answers open with a negation against 0.3% of the training
answers — 33 of its 44 questions were shapes the training data did not contain at
all. Adding those shapes (Phase 3.6) took it to **60.5%**, and for the first time
the context assembly earns its keep on this set: 11 questions right with no
context, 13 with a single retrieved passage, **27 with the bot's context**. Full
numbers, the costs and the caveats in [`RESULTS.md`](RESULTS.md).

[`PAPER.md`](PAPER.md) is the write-up: problem, data provenance, method,
results, limitations and what I would do differently.
[`RESULTS.md`](RESULTS.md) has every measurement including the failed runs,
[`data/DATA_REPORT.md`](data/DATA_REPORT.md) covers how the data was verified and
what is wrong with it, and [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) holds the full
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
  Done: test citation accuracy 86.8% → 92.9%, and it left the confusion set
  untouched, which is what exposed the real gap.
- **Phase 3.6** — add the negation question shapes the training data lacked
  ([`scripts/build_negation_dataset.py`](scripts/build_negation_dataset.py)) and
  retrain. Done: the confusion set went from 10 to 27 of 44, at no cost to any
  other capability.
- **Phase 4** — the 44 confusion questions put to ChatGPT
  ([`scripts/build_phase4_prompts.py`](scripts/build_phase4_prompts.py),
  [`scripts/score_phase4.py`](scripts/score_phase4.py),
  [`scripts/compare_phase4.py`](scripts/compare_phase4.py)). **Done.** ChatGPT 25
  of 44, this bot 27 — and they disagree on 46.5% of questions, failing on opposite
  axes. Written up in [`PAPER.md`](PAPER.md); the video is the last deliverable.

### Using your own model

The trained model lives on Google Drive, not in this repository (weights are
deliberately never committed). To run it on your own machine:

**1. Set up an environment.** The blocker on a fresh Windows install is that
`transformers` 5.x refuses to use `torch` below 2.5, so models silently fail to
load. This makes an isolated environment that reuses the torch you already have:

In Windows PowerShell, from the project root. The `cd` matters — every
command below is run from there — and the quotes matter because the path
contains a space:

```powershell
cd "C:\Users\ACER\Desktop\Legal bot\legal-llm-bot"
python -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install "transformers<5" sentence-transformers faiss-cpu
```

**2. Copy the model down from Drive.** In Google Drive open
`MyDrive/legal-llm-bot`, right-click **`flan-t5-small-context-v3`** and choose
Download; Drive sends it as a zip. Extract it so the folder sits at
`models/flan-t5-small-context-v3` — about 300 MB. Do **not** download
`checkpoints-flan-t5-small-context-v3`, which is training scratch data and far
larger.

**3. Check it before trusting it.**

```powershell
cd "C:\Users\ACER\Desktop\Legal bot\legal-llm-bot"
.\.venv\Scripts\python.exe scripts\check_model.py models\flan-t5-small-context-v3
```

One check needs the stock Flan-T5 checkpoint, from the local cache or the hub. If
you are offline and it is not cached, that single comparison is skipped with a
note and everything else still runs; `--no-drift` skips it quietly.

This confirms the files are complete, that the weights really differ from the
stock Flan-T5 checkpoint (an untrained model loads and generates fluent English
perfectly happily), that five known-answer probes come out right, that the scope
refusals fire — and it tests specifically for the three defects recorded in
[`RESULTS.md`](RESULTS.md). Expect those three to fail; the point is to see which
ones on your copy.

**4. Ask it things.**

```powershell
cd "C:\Users\ACER\Desktop\Legal bot\legal-llm-bot"
.\.venv\Scripts\python.exe scripts\bot.py --model-dir models\flan-t5-small-context-v3
```

with no question, that opens a prompt you can type into. It runs on CPU — this is
a 77M-parameter model, so a GPU is not needed.

**Two version traps, both handled in `scripts/bot.py`, both worth knowing about
if you load this checkpoint any other way.** Colab trained it under transformers
5.17; loading it under 4.x hits these:

* the tokenizer config writes `extra_special_tokens` as a list where 4.x expects
  a dict, which raises `AttributeError: 'list' object has no attribute 'keys'`;
* more dangerously, the checkpoint stores `shared.weight` *and* `lm_head.weight`
  with different values while `config.json` says `tie_word_embeddings: true`.
  transformers 5.x noticed the conflict and left them untied; 4.x obeys the
  config, ties them and **silently discards the trained output layer**. The model
  loads with no warning, reports 60.5M parameters instead of 77.0M, and generates
  fluent nonsense. `load_model()` forces `tie_word_embeddings=False` and raises if
  tying happened anyway, and `check_model.py` asserts the parameter count.

### Asking it something

```powershell
cd "C:\Users\ACER\Desktop\Legal bot\legal-llm-bot"
.\.venv\Scripts\python.exe scripts\bot.py "Which BNS section replaced IPC Section 302?"
.\.venv\Scripts\python.exe scripts\bot.py --sources-only "Is an offence under BNS Section 303 bailable?"
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
citation accuracy from 86.8% to 92.9% — and left the confusion set untouched,
which is what Phase 3.6 addresses.

**The fine-tuned model's prose is not trustworthy, and the interface reflects
that.** Its section numbers drift to neighbouring provisions, it corrupts
statutory titles, and it invents counterparts for repealed sections. The retrieval
and concordance layer is verified; the 77M-parameter model on top of it is not.
See the limitations sections of [`PAPER.md`](PAPER.md) and
[`RESULTS.md`](RESULTS.md).

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
  metrics.py                EM / token F1 / citation metrics, one definition
  build_phase4_prompts.py   write the confusion questions for another chatbot
  score_phase4.py           score its replies against this bot, same metrics
  compare_phase4.py         question by question: who is right, and where
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

## The website

**https://suryanshu-g.github.io/legal-llm-bot/**

Served from [`docs/`](docs/) by GitHub Pages. Eight example questions, each with
its answer, the provisions it rests on, and a link to the gazette PDF or judgment
behind every one; the ChatGPT comparison; and the limitations.

## The interface

[`frontend/index.html`](frontend/index.html) is a self-contained page presenting
the assistant: eight example questions, each with the answer, the provisions it
rests on, and a link to the gazette PDF or judgment behind every one. Rebuild it
with `python scripts/build_frontend.py`, which regenerates both the page and
`frontend/data.json` from `mapping_table.csv`, `retrieval_corpus.jsonl` and
`bnss_schedule.csv`.

**Every answer on that page is assembled from those verified sources, not
generated by the fine-tuned model.** That is deliberate. Reading all 44 of the
model's confusion-set answers showed section numbers drifting to neighbouring
provisions, corrupted statutory titles, and invented counterparts for repealed
sections — see the limitations section of [`RESULTS.md`](RESULTS.md). The
retrieval and concordance layer is verified and sound; the 77M-parameter model's
prose is not publishable as legal information. The page says so, and labels the
two examples of model output it does show.

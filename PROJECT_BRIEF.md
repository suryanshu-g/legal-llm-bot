# PROJECT BRIEF — Industry-Specific LLM Bot (Legal Research Assistant)

This file is the durable record of the project's context. It exists so that any
future working session — human or model, with no access to earlier chat history —
can pick the project up from here alone.

## What we are building

A legal-research assistant bot, scoped narrowly and deliberately.

**Industry (for submission purposes):** Government and Public Administration.
The official industry list for the capstone has no "Legal Services" category, so
this is the closest fit and must be justified as such in documentation. The
subject matter — codified criminal law enacted by Parliament and administered by
police, prosecutors and courts — sits squarely inside public administration.

**Real framing:** a retrieval-grounded assistant for Indian criminal law,
specifically built around the July 2024 transition from the old British-era codes
to the new ones:

| Old code | Replaced by | Sections |
|---|---|---|
| IPC (Indian Penal Code, 1860) | **BNS** (Bharatiya Nyaya Sanhita, 2023) | 358 |
| CrPC (Code of Criminal Procedure, 1973) | **BNSS** (Bharatiya Nagarik Suraksha Sanhita, 2023) | 531 |
| Indian Evidence Act, 1872 | **BSA** (Bharatiya Sakshya Adhiniyam, 2023) | 170 |

All three came into force **1 July 2024**. FIRs registered before that date
continue under the old codes; offences on or after that date fall under the new
codes.

### Core differentiator thesis

This transition creates real, current confusion that generic LLMs — trained on
data spanning the changeover — often get wrong, mixing up old and new section
numbers. A bot that retrieves from authoritative current text should be more
accurate on "which section applies" and "what does section X say" questions than
a generic chatbot relying on parametric memory. That claim is the thesis the
project is built to demonstrate.

### Scope discipline — the bot must NOT

- give legal advice;
- suggest "loopholes" or ways to evade liability;
- claim to represent a licensed advocate.

It should describe what the law and precedent say, cite its source, and always
carry a disclaimer that it is informational only.

## Architecture (decided; do not deviate)

- Fine-tune **Flan-T5-base** (fallback `flan-t5-small` if compute-constrained)
  on an instruction/QA dataset built from public legal text.
- **Plus** a retrieval layer (FAISS + sentence embeddings) over a separate
  document corpus, so the model answers using retrieved context rather than pure
  memory.
- Training happens on Google Colab (T4 GPU, max 25 epochs). Phase 1 is data
  only — no training.

## Constraints that shape data collection

- Everything must be public-source. No real client or case files from any law
  firm.
- All files must end up shareable with "anyone with view access" on Google
  Drive, so nothing may be included that cannot legally be redistributed
  publicly.

Note on copyright: under s.52(1)(q) of the Indian Copyright Act 1957, there is no
copyright in the text of any Act of a legislature or in any judgment of a court.
The bare act text and case holdings in this repository are therefore
redistributable. Where a third-party site was used as a convenience source, we
store the extracted statutory text rather than redistributing that site's page
markup.

## Phases

- **Phase 1 (complete):** repo setup and two clean, documented datasets — a
  fine-tuning instruction/QA dataset and a retrieval corpus — plus an
  old-to-new section concordance.
- **Phase 1.5 (complete):** the BNSS First Schedule (classification of offences:
  cognizable, bailable, trying court), and leakage-safe train/validation/test
  splits plus a held-out confusion set for the Phase 4 comparison.
- **Phase 2:** fine-tuning Flan-T5 on Colab, from `train.jsonl` / `val.jsonl`.
- **Phase 3:** the FAISS retrieval layer over `retrieval_corpus.jsonl`.
- **Phase 4:** running `confusion_test_set.jsonl` against both this bot and a
  general-purpose LLM to test the differentiator claim; video and paper.

See `data/DATA_REPORT.md` for what was collected, how it was verified and what
is wrong with it.

## Phase 1 objective (as specified)

1. A **fine-tuning dataset** (instruction/QA pairs) for training Flan-T5.
2. A **retrieval corpus** (chunked reference documents with metadata) for RAG.

Get as much offence/topic coverage as data availability realistically allows —
don't artificially cap scope, but don't sacrifice quality or accuracy for
coverage either. Breadth is good; wrong or unverified legal text is not.

# Retrieval-grounded question answering across a statutory recodification

**An industry-specific LLM assistant for Indian criminal law after 1 July 2024**

Capstone project, Data Science / Agentic AI programme.
Industry: **Government and Public Administration**.
Code, datasets and full results: https://github.com/suryanshu-g/legal-llm-bot

---

## Abstract

On 1 July 2024 India replaced its three foundational criminal codes. The Indian
Penal Code, 1860 became the Bharatiya Nyaya Sanhita, 2023; the Code of Criminal
Procedure, 1973 became the Bharatiya Nagarik Suraksha Sanhita, 2023; and the
Indian Evidence Act, 1872 became the Bharatiya Sakshya Adhiniyam, 2023. Every
section was renumbered. Some provisions were merged, a few were split across
several new sections, and a number were dropped entirely.

This creates a failure mode that is specific, consequential and measurable: a
language model trained on text spanning the changeover will answer questions
about section numbers confidently and wrongly. This project builds a
retrieval-grounded assistant for that transition and measures the failure
directly.

Three findings. **First**, a fine-tuned Flan-T5-base answering from its own
parameters was correct on **0 of 214** held-out questions asking what a provision
became after the changeover; the same task with retrieval reaches **213 of 214**.
**Second**, the decisive retrieval component is not similarity search but a
deliberate lookup in the official correspondence table: nothing in the wording of
IPC §302 resembles the wording of BNS §103 more than any other homicide section,
so the counterpart must be looked up rather than searched for. **Third**, on 44
adversarial questions held out from the start, ChatGPT without retrieval scores 25
and this system 27 — a difference of no significance — but the two disagree on 20
of 43 and fail on opposite axes. ChatGPT scores 92.9% where a section number was
reused for a different offence and 7.1% where one new provision absorbed several
old ones. Where the correct answer is a *set* of provisions, ChatGPT was right on
1 of 19 and this system on 10 of 19, and ChatGPT's incomplete answers carry no
indication that they are incomplete.

The contribution is therefore narrower and more useful than "a small model beats
a large one". It is that a frontier model answering from memory returns
confidently partial answers about statutory correspondence, and that retrieval
over an official concordance addresses precisely that deficiency.

---

## 1. Industry and problem statement

### 1.1 Why Government and Public Administration

The programme's industry list contains no "Legal Services" category. Codified
criminal law is nonetheless squarely a public-administration subject: the BNS,
BNSS and BSA are administered by police forces, public prosecutors and the
subordinate judiciary, and the correspondence tables this project depends on were
published by the **Bureau of Police Research and Development**, a body under the
Ministry of Home Affairs, precisely to help state police forces retrain. The
intended user is an officer, prosecutor, clerk or student who must determine what
a provision has become — an administrative question, not an advisory one.

### 1.2 The specific problem

| Repealed code | Replacement | Sections |
|---|---|---|
| Indian Penal Code, 1860 | Bharatiya Nyaya Sanhita, 2023 | 358 |
| Code of Criminal Procedure, 1973 | Bharatiya Nagarik Suraksha Sanhita, 2023 | 531 |
| Indian Evidence Act, 1872 | Bharatiya Sakshya Adhiniyam, 2023 | 170 |

Four question types make this harder than a renumbering exercise:

1. **Collisions.** A section number survives into the new code attached to an
   unrelated provision. CrPC §482 was the High Court's inherent powers; BNSS §482
   is anticipatory bail. A model carrying the old numbering forward produces a
   confident answer about the wrong provision.
2. **Merged families.** BNS §324 absorbed IPC §§425, 426, 427 and 440. Naming one
   of them is not a complete answer.
3. **Splits.** IPC §171 was distributed across BNS §177 and §205. There is no
   single correct counterpart.
4. **Repeals.** Sedition (IPC §124A) has no BNS counterpart at all, so any
   section number offered in reply is wrong by construction.

---

## 2. Data

### 2.1 Sources and provenance

Everything is public-source. No client file, case file or other non-public
material from any law firm was used at any point.

| Content | Source |
|---|---|
| BNS, BNSS, BSA statutory text | Ministry of Home Affairs gazette PDFs |
| Old-to-new correspondence | BPRD "Correspondence Table and Comparison Summary" PDFs |
| Concordance cross-check | UP Police comparative table; devgan.in |
| IPC, CrPC, Evidence Act text | devgan.in, cross-checked against the civictech-India JSON corpus |
| Offence classification | BNSS First Schedule, parsed from the gazette |
| Case law | Indian Kanoon |

Every fetch is cached under `data/raw/` with a manifest recording URL, retrieval
date, byte count and SHA-256, so any figure in this paper can be traced to a
specific retrieved document.

Under **section 52(1)(q) of the Copyright Act 1957** there is no copyright in the
text of an Act of the legislature or in a court judgment, so the statutory corpus
is redistributable. Case summaries are short factual holdings rather than
reproduced judgments.

`robots.txt` was checked per URL before fetching, and requests were rate-limited.
India Code, the canonical portal, returned HTTP 403 to automated requests from
this environment; rather than work around that, the project pivoted to the MHA
gazette PDFs, which are the same authority in primary form. This is recorded in
`data/DATA_REPORT.md` rather than omitted.

### 2.2 Extraction

The gazette PDFs place section numbers and headings in a left margin whose side
alternates with page parity, set in small capitals, with no ruling lines around
the First Schedule's tabular matter. Coordinate-aware extraction was therefore
required: x-band column separation, margin/body separation, small-caps
reconstruction from font size and x-order, line clustering by vertical position,
and — for the concordance tables — per-entry column dividers inferred from the
widest inter-word gap, with row segmentation by vertical pitch (9.6pt within an
entry against roughly 14.5pt between entries).

Six extraction defects were found and fixed by comparing output against the
source documents rather than by inspection of aggregate counts. Two are worth
recording because they would have silently corrupted the data:

* pdfplumber dropped roughly one concordance row per page boundary. Supplying
  explicit horizontal lines derived from the vertical rules' extents recovered
  them, and took the BNSS concordance from 500 to its true 531 rows.
* IPC §489A was rendered in the source HTML with a displayed number of "498A".
  Anchor-first parsing with a collision fallback corrected it. Until that fix,
  IPC §498A carried the wrong heading throughout the dataset.

### 2.3 Resulting datasets

| Artifact | Size |
|---|---|
| Section coverage | BNS 358/358, BNSS 531/531, BSA 170/170 |
| Concordance (`mapping_table.csv`) | 1,324 rows, typed `direct` / `split` / `merged` / `new_provision` / `removed` |
| BNSS First Schedule | 465 rows over 288 BNS sections |
| Case summaries | 201 records over 174 judgments, 30 topics |
| Fine-tuning QA pairs | 14,524 |
| Retrieval corpus | 2,819 chunks, each with a source URL |

### 2.4 Verification

`scripts/validate_data.py` runs eight groups of checks and exits non-zero on
failure. Current state: **0 failures, 4 warnings**, the warnings being five
sections whose heading is absent from the gazette margin and two very short IPC
definition chunks — each enumerated in the data report.

The concordance was cross-checked across three independent sources over 413
comparisons: 100% coverage, 12 partial-overlap disagreements, **zero outright
conflicts**. Disagreements are recorded in `mapping_disagreements.csv` rather
than silently resolved.

### 2.5 Evaluation design

Two decisions here do more for the credibility of the results than anything in
the modelling.

**Splits are cut at group level, not row level.** A "group" is a *fact*: a
provision and everything the concordance connects it to, computed by union-find.
Splitting by row would put "What does BNS §103 cover?" in training and "Which BNS
section replaced IPC §302?" in test, which is leakage — the second is answerable
from the first. Groups are assigned 85/10/5 by whichever split is most short of
the question types that group carries, and disjointness is re-verified from the
written files.

**A 44-question confusion set was held out of all three splits from the
beginning**, built specifically from the four hard categories above, each entry
documenting the correct answer and why a general model is expected to fail. It was
never used for selection or tuning at any point.

**Metrics.** Token F1 is reported because it is conventional, but it is a poor
measure here and the project demonstrates why: an answer naming the wrong section
scores 0.909 token F1 against the correct answer. The reported metrics are
therefore citation-based — `allCites`, whether every provision the correct answer
cites appears in the prediction, and `citation_exact`, whether the two sets match.
The second exists because the first can be satisfied by naming the right section
alongside an invented one. `scripts/metrics.py` holds one definition, used by the
training notebooks and by the ChatGPT comparison alike.

---

## 3. Method

Architecture was fixed by the brief: fine-tune **Flan-T5-base**, with
`flan-t5-small` as the sanctioned fallback under compute constraint, plus a FAISS
retrieval layer, trained on a free Colab T4 within 25 epochs.

### 3.1 Retrieval

2,819 chunks — one per section, per First Schedule entry and per judgment summary
— embedded with `BAAI/bge-small-en-v1.5` into a FAISS `IndexFlatIP` over
L2-normalised vectors. The model's asymmetric query prefix is applied to queries
only, not to passages.

**Dense retrieval alone failed**: recall@3 of 26.3% overall and 7.1% on
section-text questions. The cause is that a query naming "BNS Section 303" has
little lexical or semantic purchase on a passage whose distinguishing feature is a
number. Adding exact statutory-citation matching, ranked above dense hits, took
recall@3 to **96.0%**, and top-1 accuracy to 80.3%.

### 3.2 Context assembly, and the counterpart lookup

Phase 2.5 improved the test set substantially but left the confusion set at
23.3%. The reason was mechanical rather than a model failure: **33 of the 44
confusion answers cite two or more provisions**, and a single retrieved passage
cannot contain both halves of a correspondence. Retrieving more passages barely
helped, because similarity ranking does not preferentially surface a provision's
*counterpart*.

The fix is a deliberate lookup. Provisions named in the question are resolved
first, then their counterparts from `mapping_table.csv`, then similarity hits,
into a 430-token budget shared across at most four passages — workable because
each chunk opens with its own header, so truncating a long passage preserves what
identifies the provision. Measured on whether the assembled context contains every
provision the correct answer needs:

| Context assembly | Confusion set |
|---|---|
| similarity only, 1 passage | 15 / 44 |
| similarity only, 5 passages | 38 / 44 |
| **citations + concordance counterparts** | **43 / 44** |

### 3.3 Training

Context-augmented training data in three conditions: 70% with the correct passage,
20% with a deliberately wrong but topically similar passage and the correct answer
still as the target, and 10% with no passage. The distractor condition exists so
the model does not learn to trust retrieved context unconditionally; it holds up,
at 76.7% citation accuracy when handed a wrong passage.

Early stopping selects on **citation accuracy, not validation loss**. This matters
for templated data: most of the token mass is boilerplate, so cross-entropy
flattens once the template is learned while the one or two tokens carrying the
section number are still wrong. The Phase 2 run stopped on a model that was fluent
and factually useless.

---

## 4. Results

### 4.1 The correspondence questions

676 held-out test questions, same model, same prompt, the only difference being
whether authoritative text was retrieved first:

| Context | token F1 | citation F1 | allCites | `citation_exact` |
|---|---|---|---|---|
| none | 58.9% | 61.3% | 44.3% | 43.2% |
| **retrieved** | **88.1%** | **95.6%** | **92.9%** | **91.8%** |
| oracle | 88.6% | 98.2% | 95.7% | 94.7% |

Per question type, the capability the project exists to provide:

| Question type | n | Phase 2, no retrieval | With retrieval |
|---|---|---|---|
| what a provision became (`old_to_new`) | 114 | **0.0%** | **99.1%** |
| what it was before (`new_to_old`) | 100 | **0.0%** | **100%** |
| which section covers a topic | 109 | 0.0% | 82.6% |
| bailable / cognizable / which court | 50 | 64.0% | 100% |
| case law | 20 | 0.0% | 20.0% |

Retrieval is within 0.5 token F1 of its own oracle, so almost nothing remains to
be gained from better retrieval; the residual error is in the model.

### 4.2 The 44 adversarial questions

These moved only when the model was taught the *shape* of the answer. Counting
the training data revealed the gap: **41 of 11,751 training answers open with a
negation (0.3%), against 43 of 44 confusion answers (97.7%)**, and the training
set contained no question of the collision, merged or split shapes at all. The
confusion set was not a harder version of a trained task; it was an untrained one.

Adding 1,006 generated examples of those shapes — from the concordance, for
provisions outside every held-out group, lifting negations to 6.4% of the mix:

| Context | token F1 | allCites | `citation_exact` | correct |
|---|---|---|---|---|
| none | 55.0% | 25.6% | 2.3% | 11 / 44 |
| single passage | 64.7% | 30.2% | 4.7% | 13 / 44 |
| **full context** | **77.0%** | **60.5%** | **34.9%** | **27 / 44** |

Before this, all three rows read 23.3% — retrieval made no difference whatever on
this set. It now separates cleanly, which is the evidence that the counterpart
lookup of §3.2 does real work once the model can use it.

### 4.3 Against ChatGPT

The same 44 questions, unaided, to **ChatGPT on 19 September 2026**, in two
batches of 22 in separate fresh chats, scored by the same code. It was told to
name sections, since that is what the metric measures, but not that the codes had
changed nor which answers were negative.

| System | token F1 | citation F1 | allCites | correct |
|---|---|---|---|---|
| ChatGPT, no retrieval | 50.2% | 83.9% | 55.8% | 25 / 44 |
| This system, full context | 77.0% | 79.6% | 60.5% | 27 / 44 |

27 against 25 is not a meaningful lead, and the token F1 column should be
disregarded in this comparison: the reference answers are written in this
project's own phrasing, which flatters a model trained on it.

The per-category split is the result:

| Question kind | n | ChatGPT | This system |
|---|---|---|---|
| collision — same number, different subject | 14 | **92.9%** | 35.7% |
| merged — several sections into one | 14 | 7.1% | **71.4%** |
| split — one section across several | 5 | 0.0% | 20.0% |
| repealed, no counterpart | 10 | *both wrong* | *both wrong* |

Question by question, over the 43 that cite a provision:

| | Questions |
|---|---|
| both correct | 15 |
| only this system | 11 |
| only ChatGPT | 9 |
| neither | 8 |

**They disagree on 20 of 43.** ChatGPT knows what provisions are *about*, so it
resolves collisions; it lacks the correspondence table, so on merged families it
names the headline correspondence and stops. Asked whether mischief is still dealt
with under IPC §425 it correctly identifies BNS §324 and says nothing of IPC §426,
§427 and §440, which the same provision absorbed. Of the 14 merged questions,
ChatGPT was right alone on **none**.

---

## 5. Limitations

Stated at length because several were found by reading outputs rather than by
reading scores, which is itself a finding.

**The generated prose is not trustworthy.** Reading all 44 answers turned up three
defects the metrics do not capture:

1. **Section numbers drift to neighbours.** All nine collision failures discuss an
   adjacent provision — BNSS 481 for 482, BSA 115 for 114, BNS 330 for 320 — with
   both provisions present in the context. This is generation, not retrieval: a
   77M-parameter model failing to copy a three-digit number.
2. **Statutory titles are corrupted**, including in answers scored correct: "BNS
   §324 (Mizachief)", "BNS §303 (Trash)" for theft.
3. **Correspondences are invented for repealed provisions**, ten times out of ten.

Consequently the retrieval and concordance layer is verified and sound while the
model's sentences are not publishable as legal information, and the project's
interface presents retrieved provisions and concordance results directly rather
than model output.

**One reported figure was a metric artifact.** The `removed` category scored 100%
for both systems, which reads as a tie. The correct answer — "there is none" —
cites only the old section, so naming it and then inventing a replacement passes a
subset test. `citation_exact` on that slice is 0 of 10 and both systems are simply
wrong. The 100% should not be quoted.

**The model is `flan-t5-small`, not `flan-t5-base`.** Base was the intention;
free Colab sessions repeatedly disconnected before a base run finished, and small
is the brief's sanctioned fallback under compute constraint. Most of the
limitations above are plausibly capacity-related and a base-sized run is the
first thing to try.

**The confusion set's question templates now appear in training**, on different
provisions. It remains held out at the level of fact — no provision in it, nor any
provision sharing a concordance group with one, contributes a training row — but
it is no longer novel in form. This is the correct trade, since a model cannot be
expected to answer a shape it has never seen, but it is a real qualification on
§4.2.

**Splits could not be taught.** All five split families in the concordance sit
inside held-out groups, so no split training example could be generated without
leaking. Five of the 44 questions remain shape-novel, and the achievable ceiling
on that set as trained is roughly 39/44.

**Case law is weak** at 20% citation accuracy, with 346 training rows against
3,824 for statutory text and far less templated targets.

**Subject-phrased questions can surface a neighbour.** "Is theft bailable?" ranks
BNS §304 (*Snatching*) above BNS §303 (*Theft*), because §304 opens "Theft is
snatching if…". Naming the section resolves it.

**Single evaluation run, single comparison model, one date.** No seed variance is
reported, and ChatGPT is a moving target.

---

## 6. Scope and ethics

The assistant is constrained by design, not only by training. Requests for legal
advice, for ways to evade liability, for loopholes, or for it to act as an
advocate are matched against a pattern list **before the model is invoked**, so a
refusal cannot be argued out of it. Verified: "be my lawyer", "how can I avoid
being convicted", "tell me a loophole" and "should I plead guilty" are refused,
while a legitimate question about a provision is answered.

Every answer carries its citations and an informational-only disclaimer. Every
interface makes clear that this is a student project, unaffiliated with and not
endorsed by any government body. No model weights are committed to the
repository, and no personal or client data exists anywhere in the pipeline.

The most significant ethical point is the one in §5: a system that produces fluent
but partially wrong statements about criminal law is dangerous in proportion to how
convincing it sounds. That is the argument for grounding every claim in a citation
the reader can follow, and for reporting the model's defects prominently rather
than in a footnote.

---

## 7. Reproducibility

```bash
pip install -r requirements.txt
python scripts/scrape_acts.py            # cached; --force to refetch
python scripts/scrape_case_law.py
python scripts/build_mapping_table.py
python scripts/build_bnss_schedule.py
python scripts/build_retrieval_corpus.py
python scripts/build_finetune_dataset.py
python scripts/build_splits.py
python scripts/build_retrieval_index.py
python scripts/build_negation_dataset.py
python scripts/build_context_dataset.py
python scripts/validate_data.py          # expects 0 failures
```

Training runs in `notebooks/finetune_flan_t5_small_contextaware.ipynb` (Colab T4,
about 100 minutes, resumable across disconnection). The comparison is
`build_phase4_prompts.py`, `score_phase4.py` and `compare_phase4.py`. Complete
results, including every failed run and what it turned out to be wrong about, are
in `RESULTS.md`.

---

## 8. What I would do differently

1. **Read model outputs from the first run, not the fourth.** Every real defect in
   this project — the wrong IPC heading, the collapsed notebook cells, the drifting
   section numbers, the vacuous `removed` score — was found by looking at actual
   output. None was visible in an aggregate.
2. **Design the training data from the evaluation set's shapes.** Three phases were
   spent on retrieval and formatting before anyone counted how many training
   answers begin with "No". That count, taken on day one, would have reordered the
   whole project.
3. **Distrust any metric that cannot fail.** `allCites` on the `removed` category
   cannot be failed by a wrong answer, and it reported 100% for two systems that
   were both entirely wrong.
4. **Budget compute for the intended model.** Four runs of `flan-t5-small` cost
   more in aggregate than one `flan-t5-base` run would have, and the fallback is
   now a caveat on every result.

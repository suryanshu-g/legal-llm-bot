# Results

**Phase 2.5 (retrieval) is the headline: the old-to-new correspondence went
from 0% to 93% correct on held-out questions. Phase 2, below, is the baseline
that makes that number mean something.**

## Phase 2 — fine-tuned Flan-T5-base, no retrieval

Recovered by `notebooks/verify_model.ipynb` from the saved weights after the
training runtime was lost. The training loss curve is not recoverable; every
other number below was regenerated from the checkpoint using the same prompt
template and metric code as the training run.

**Model:** `google/flan-t5-base`, 247.6M parameters, 990 MB checkpoint.
Weights differ from the stock checkpoint by 37.4% mean absolute change on a
sampled attention matrix, so this is genuinely a fine-tuned model.

### Headline

| Set | n | EM | token F1 | citation F1 | all gold citations present |
|---|---|---|---|---|---|
| test | 676 | 0.0% | 58.2% | 61.7% | 43.8% |
| confusion | 44 | 0.0% | 30.2% | 62.2% | 25.6% |

### By question type (test)

| qa_type | n | EM | F1 | citation F1 | allCites |
|---|---|---|---|---|---|
| section_text | 220 | 0.0% | 24.6% | 97.7% | 92.3% |
| section_lookup | 109 | 0.0% | **91.2%** | **0.9%** | 0.0% |
| old_to_new | 114 | 0.0% | 77.7% | 49.9% | **0.0%** |
| new_to_old | 100 | 0.0% | 69.7% | 49.8% | **0.0%** |
| offence_classification | 50 | 0.0% | 65.2% | 64.0% | 64.0% |
| punishment | 52 | 0.0% | 61.3% | 100.0% | 100.0% |
| case_law | 20 | 0.0% | 43.8% | **3.8%** | 0.0% |
| removed | 6 | 0.0% | 85.0% | 100.0% | 100.0% |
| new_provision | 2 | 0.0% | 77.4% | 100.0% | 100.0% |
| transition | 3 | 0.0% | 60.0% | n/a | n/a |

### What actually happened

**The model learned the answer format and almost none of the facts.** Every
output is fluent, correctly structured, and uses the right boilerplate. The
content is wrong.

| Question | Correct | Model said |
|---|---|---|
| What does BNS 103 cover? | Punishment for murder | "Punishment for gang rape" |
| Which BNS section replaced IPC 302? | BNS 103 | "BNS 290 (Punishment for bribery)" |
| Which CrPC section corresponds to BNSS 173? | CrPC 154 | "CrPC 179" |
| Is BNS 303 bailable? | Non-bailable (theft) | "Bailable", and called it assault |
| Offence on 15 August 2024 — IPC or BNS? | BNS | "The IPC applies" |
| Which BNS section corresponds to IPC 124A? | None; sedition was not carried forward | "BNS 127" |
| Can you be my lawyer? | Refusal + disclaimer | degenerate loop: "I'm not a witness." ×30 |

**The decisive number is `allCites` = 0.0% on `old_to_new` and `new_to_old`.**
Across 214 held-out questions asking what a provision became after 1 July 2024 —
the capability this entire project exists to provide — the model was right
**zero times**.

**The aggregate hides this, which is why the per-type breakdown was built in.**
Test F1 of 58.2% looks like a working model. It is not:

* `section_lookup` scores **F1 91.2% with citation F1 0.9%** — near-perfect
  prose naming the wrong section. This is the single clearest illustration that
  token overlap is not a measure of correctness here.
* `section_text` scores the reverse, **F1 24.6% with citation F1 97.7%**,
  because the model echoes the section number out of the question and then
  invents the text. The high citation score is vacuous.
* `punishment`, `removed` and `new_provision` show 100% citation scores for the
  same reason: their gold answers cite only the provision the question already
  names.

### Why it failed

The training data was checked first and is not the cause: 11,329/11,329 rows
have their index id present in their own text, and 1,736/1,736 `section_text`
answers match the gazette heading exactly. Questions and answers are correctly
paired.

Two causes, and they compound:

1. **The task is mostly arbitrary factual recall, which is what parameters are
   worst at.** ~1,059 section texts and ~1,300 concordance pairs, from 11,751
   examples, into 250M parameters. There is no generalisable rule connecting
   "IPC 302" to "BNS 103" — it simply has to be stored. The model learned the
   regularity that *was* learnable, which is the shape of the answer.

2. **Early stopping on validation loss was the wrong criterion for this data.**
   The answers are highly templated; most of the token mass is boilerplate like
   "The Bharatiya Nyaya Sanhita, 2023 replaced the Indian Penal Code, 1860 with
   effect from 1 July 2024." Cross-entropy is dominated by those tokens, so
   validation loss flattens once the template is learned while the one or two
   tokens carrying the section number are still wrong. The run can look
   converged and be factually useless.

   How much of the failure is (2) rather than (1) cannot be settled from the
   saved weights, because the epoch count and loss curve were lost with the
   runtime. The fix is cheap and applied regardless: the context-aware run stops
   on citation accuracy rather than on loss.

### What this means for the project

This is a genuine negative result on the Phase 2 approach, and it is also the
strongest available evidence for the project's central claim. The argument has
always been that a model answering Indian criminal law from parametric memory
will be confidently wrong about section numbers after the 2024 recodification.
Phase 2 demonstrates exactly that, measured, on held-out data — including a
model fine-tuned on the correct answers.

It is a baseline to beat, not a result to present as the system.

### Fixes carried into Phase 2.5

* **Early stopping on citation accuracy**, not validation loss.
* **`no_repeat_ngram_size=3` at generation**, which prevents the degenerate
  repetition seen on the scope question.
* **A stricter citation metric.** `allCites` only asks whether the gold
  citations are a subset of the predicted ones, so a reply that names the right
  old section *and invents a wrong new one* passes. `citation_exact`, which
  requires the sets to match, is now reported alongside it.
* **Retrieval.** The answer is in the retrieved passage, so it no longer has to
  be in the weights. This is the substantive change and the reason to expect a
  different outcome.

**Artifact:** `MyDrive/legal-llm-bot/flan-t5-base-finetuned` (not published to
the Hub — no `HF_TOKEN` was set for the run).

---

## Phase 2.5 — same model, but reading retrieved text

`google/flan-t5-small`, trained on the context-augmented data (70% correct
passage, 20% deliberately wrong passage, 10% none), selected on citation
accuracy rather than validation loss. 9 epochs, 133 minutes on a T4, best
validation citation accuracy 90.6%. Raw numbers in
`data/processed/phase2_5_small_results.json`.

### Headline — the same 676 test questions, three ways

| Context | token F1 | citation F1 | all gold citations present |
|---|---|---|---|
| none (no retrieval) | 58.4% | 61.6% | 43.8% |
| **retrieved (live)** | **91.9%** | **91.7%** | **86.8%** |
| gold (perfect retrieval) | 94.8% | 97.9% | 94.5% |

Retrieval is worth **+33.5 token F1** and **+43 points of citation accuracy**.
Of the remaining gap to oracle context, only 2.9 F1 is lost to retrieval error —
the retriever returns the right passage as its top hit for 80.3% of test
questions, and top-3 recall measured separately is 96%.

### The number this project was built to move

| qa_type | Phase 2 (no retrieval) | Phase 2.5 retrieved |
|---|---|---|
| `old_to_new` (n=114) | **0.0%** | **93.0%** |
| `new_to_old` (n=100) | **0.0%** | **71.0%** |

Those are "all gold citations present" — did the answer name the right
provisions. Phase 2 was right in 0 of 214 questions about what a section became
after 1 July 2024. With retrieval it is right in 179 of 214.

### Full per-type breakdown (token F1, and citation accuracy)

| qa_type | n | none | retrieved | gold | allCites none → retrieved |
|---|---|---|---|---|---|
| section_text | 220 | 28.2% | 88.4% | 93.0% | 92.3% → 96.8% |
| old_to_new | 114 | 75.5% | 94.4% | 94.4% | 0.0% → 93.0% |
| section_lookup | 109 | 90.7% | 96.1% | 98.3% | 0.0% → 75.2% |
| new_to_old | 100 | 67.7% | 93.1% | 94.4% | 0.0% → 71.0% |
| punishment | 52 | 62.5% | 93.0% | 96.9% | 100% → 100% |
| offence_classification | 50 | 59.9% | 92.6% | 94.2% | 64.0% → 92.0% |
| case_law | 20 | 45.1% | 85.3% | 96.1% | 0.0% → 30.0% |
| removed | 6 | 84.4% | 100% | 100% | 100% → 100% |
| transition | 3 | 75.2% | 55.6% | 83.0% | n/a |
| new_provision | 2 | 78.1% | 100% | 100% | 100% → 100% |

### Did it learn the three behaviours it was trained for?

Measured on validation, where each row's condition is known rather than left to
whatever retrieval returned:

| Condition | n | token F1 | all gold citations present |
|---|---|---|---|
| positive (correct passage) | 105 | 95.5% | 96.2% |
| distractor (wrong passage) | 30 | 43.9% | 73.3% |
| none (no passage) | 15 | 56.7% | 57.1% |

**The distractor row is the one that matters and it holds up.** Given a
plausible but wrong passage the model still names the right provisions 73% of
the time. It did not learn to believe context blindly, which was the risk the
20% distractor share was there to prevent. Token F1 drops to 43.9% because
without the right passage it cannot reproduce the statutory wording — it gets
the citation right and the prose approximate, which is the correct failure mode.

**Unaided performance did not regress.** The `none` column on test is 58.4 F1 /
43.8% citations against Phase 2's 58.2 / 43.8 — statistically the same, from a
model a third the size. Context-aware training cost nothing when no context is
supplied.

### What still does not work: the confusion set

| Context | token F1 | all gold citations present |
|---|---|---|
| none | 28.4% | 23.3% |
| retrieved | 38.4% | 23.3% |

Retrieval barely helped here, and the reason is mechanical rather than a model
failure. **33 of the 44 confusion questions have gold answers citing two or more
provisions** — the old section and its new counterpart, and for merged families
every constituent old section (one answer cites nine). A single retrieved chunk
cannot contain both sides of a correspondence. Measured directly:

| Retrieved chunks | Questions where every needed provision was available |
|---|---|
| k=1 (what this run used) | 10 / 44 |
| k=3 | 12 / 44 |
| k=5 | 15 / 44 |

So the model was asked to state a correspondence while being shown one half of
it. More chunks barely helps, because ranking by similarity does not
preferentially surface the *counterpart* provision.

**The fix is targeted rather than broader retrieval**, and Phase 3 should
implement it: parse the provision out of the question, look its counterparts up
in `mapping_table.csv`, and put *those* chunks in the context deliberately. The
concordance already exists and is complete; the retriever simply was not asked
to use it.

### Verdict

The project's central claim is now demonstrated rather than asserted. The same
questions, the same prompt, the same trained model — the only difference being
whether authoritative text was retrieved first — move from 0% to 93% on the
old-to-new correspondence. Phase 2 is the counterfactual that makes it
meaningful.

**Artifact:** `MyDrive/legal-llm-bot/flan-t5-small-context` (not published to the
Hub — no `HF_TOKEN` was set).

**Caveat for the write-up:** this is `flan-t5-small`, not `flan-t5-base`. The
base model was the intention; free Colab sessions kept timing out before the run
finished, and the brief names small as the sanctioned fallback when compute is
constrained. It should be reported as such, not glossed over.

---

## Phase 3 — the assembled bot

`scripts/bot.py` is the whole assistant: assemble context, build the prompt the
model was fine-tuned on, generate, cite. The one substantive addition over Phase
2.5 is the fix that section identified — the counterparts of a provision are
looked up deliberately in `mapping_table.csv` rather than hoped for from the
embedder.

Context is assembled in priority order, because it is truncated to a token
budget: provisions the question names outright, then those provisions'
counterparts, then similarity hits for questions phrased by subject rather than
by section number. Each chunk opens with its own header (act, section, heading,
counterpart), so truncating the tail of a long passage keeps the part that
identifies the provision — which is what makes sharing one 430-token budget
across up to four passages workable.

### Did it close the gap? (`scripts/check_bot_context.py`)

Whether every provision the gold answer cites is actually present in the
assembled context:

| Context assembly | Confusion set (44) | Test set (676) |
|---|---|---|
| similarity only, 1 passage | 15 / 44 — 34.1% | 93.6% |
| similarity only, 3 passages | 35 / 44 — 79.5% | — |
| similarity only, 5 passages | 38 / 44 — 86.4% | — |
| **citations + concordance counterparts** | **43 / 44 — 97.7%** | **95.1%** |

**The confusion set goes from 34% to 98% context completeness.** That was the
one measured blocker Phase 2.5 left open, and it is closed. Both halves of a
correspondence now appear together — for the BNSS 482 question the context
carries `bnss_482`, `crpc_482` and `bnss_528`, which no similarity ranking was
going to assemble on its own.

This measures *context*, not answers. Whether the model uses what it is now
shown is what `notebooks/run_bot.ipynb` reports.

### Scope enforcement

The project forbids the assistant from giving legal advice, suggesting ways to
evade liability, or posing as an advocate. The model was trained to refuse, but
a trained refusal is a tendency rather than a guarantee, so `bot.py` checks the
question against a pattern list before it ever reaches the model. Verified
locally: "be my lawyer", "how can I avoid being convicted", "tell me a
loophole" and "should I plead guilty" are all refused, while "what does BNS
Section 63 cover?" is answered normally. Every citation returned carries a
`source_url`.

### Known limitation: subject-phrased questions

Asking by subject rather than by section number can surface a neighbouring
provision. "Is theft bailable under the new law?" returns BNS 304 (*Snatching*),
BNS 309 (*Robbery*) and IPC 378 — but not the First Schedule row that actually
answers it. The cause is in the corpus rather than the code: BNS 304 opens
"**Theft** is snatching if, in order to commit theft, the offender suddenly …",
so on wording alone it is a better match for "theft" than BNS 303 (*Theft*),
which reaches rank 5.

Naming the provision fixes it completely — "Is an offence under BNS Section 303
bailable?" pulls both `bns_303` and `schedule_bns_303`, because a citation match
returns every chunk for that reference. All 50 `offence_classification`
questions in the test set are phrased with the section number, which is why this
does not show up in the table above.

Two things were tried and rejected. An intent-aware boost preferring First
Schedule chunks when a question mentions bailable/cognizable/triable does not
help, because the schedule chunk is not in the candidate set at all — the
mis-ranking happens one step earlier, between two section chunks. Widening to
five passages was measured and gains +0.8% on test (95.1% → 95.9%) and nothing
on the confusion set, while cutting each passage's share from ~107 to ~86
tokens — the wrong trade for a model whose `section_text` score depends on
reproducing statutory wording. The honest fix is a better embedder or a query
rewrite step, and neither belongs in Phase 3.

**Artifact:** `scripts/bot.py`, run from
[`notebooks/run_bot.ipynb`](notebooks/run_bot.ipynb) or the command line
(`python scripts/bot.py --sources-only "…"` works with no model at all, and
reports the law itself rather than composing an answer).

### What the bot run actually showed

Run on Colab against the Phase 2.5 model. The retrieval and safety halves hold
up; the model half does not use what it is given.

**The safety guard works.** All four out-of-scope questions were refused — "be my
lawyer", "how can I avoid being convicted", "tell me a loophole", "should I plead
guilty" — while "what does BNS Section 63 cover?" was answered normally. Every
citation carried a `source_url`.

**Five of the eight demo questions are right**, with correct sources: IPC 302 →
BNS 103, BNSS 173 → CrPC 154, BNS 303 is non-bailable, BNS 103 is triable by
Court of Session, and an offence on 15 August 2024 falls under the BNS.

**The confusion set did not move at all.**

| Context | token F1 | citation F1 | allCites |
|---|---|---|---|
| single passage (Phase 2.5 behaviour) | 38.6% | 63.5% | 23.3% |
| bot context, with counterparts | 39.2% | 59.4% | **23.3%** |

Context completeness went from 34% to 98% and the answers were unchanged. Two of
them are confidently false in instructive ways:

* *"IPC Section 124A corresponds to **BNS Section 105L** (Sedition)"* — **there
  is no BNS 105L**; the section does not exist in the Act. The correct answer is
  that sedition has no counterpart.
* *"BNSS Section 482 … is a new provision; it has no equivalent in the CrPC"* —
  wrong, **and the CrPC 482 passage was in the context it was handed**.
* The BNS 103 answer bled two passages into each other, producing non-words
  ("Cognizedable", "Punition") and a heading from an unrelated section.

Note also that the 100% `allCites` on `removed` questions is an artifact. The
metric asks only whether the gold citations are *present*, so the 124A answer
passed while inventing BNS 105L. `citation_exact` is the column to read there,
and it is reported from Phase 3.5 onward.

### Diagnosis: a training/serving mismatch, and it was mine

**Every one of the 10,571 context-augmented training examples held exactly one
passage. The bot serves up to four**, separated by rules and each truncated to a
quarter of the token budget. The model had never seen that shape.

That single fact explains all three failures at once: reading only the first
passage explains the BNSS 482 denial, blending explains the BNS 103 non-words,
and neither is fixed by putting *more* correct law in front of it.

The Phase 3 measurement was of context completeness, which was real and correct.
It was not a measurement of whether the model could consume the thing being
built, and I reported the first as though it implied the second.

---

## Phase 3.5 — training on the context the bot actually sends

The fix is the same one that worked in Phase 2.5: make the training data look
like what serving produces. `build_context_dataset.py` no longer assembles
context of its own — it imports `Bot` and calls the same `select_chunks` and
`build_context` methods that answer a live question, at the same
`MAX_CONTEXT_CHUNKS = 4` and `CONTEXT_TOKEN_BUDGET = 430`, with the same
tokenizer. Training and serving cannot drift apart again without the shared code
path changing.

### The regenerated data

| | Phase 2.5 data | Phase 3.5 data |
|---|---|---|
| Passages per context | always 1 | 4 (8,825 rows) · 3 (1,746) · 0 (1,180) |
| Assembled by | this script's own truncation | `bot.Bot.select_chunks` + `build_context` |
| Positives holding the grounding passage | 8,218 / 8,218 | 8,218 / 8,218 |
| Median prompt length | — | 389 tokens (p99 470, max 503) |

The condition mix is unchanged at 70% positive / 20% distractor / 10% none
(8,218 / 2,353 / 1,180 on train), and the seed is unchanged, so the split of
rows across conditions is identical to Phase 2.5 — only the shape of the context
differs.

Two details matter for honesty about what the model is being taught:

* **The grounding passage is not always first.** For 290 training rows the
  embedder did not surface it inside the top four, so it is inserted at a varying
  position rather than at the front. Otherwise the model could learn the shortcut
  "the answer is in passage one" — which is close to the failure being fixed.
* **Distractor contexts exclude every chunk about the provisions the gold answer
  cites**, so a distractor is genuinely unhelpful rather than a near-miss that
  happens to contain the answer.

Prompts fit the 512-token encoder that `bot.py` truncates to, with the longest
at 503, so nothing is silently dropped at serving time.

### Evaluation changes with it

`notebooks/finetune_flan_t5_small_contextaware.ipynb` now assembles its
evaluation context through `Bot` as well, instead of pasting the single top-1
passage, and reports the confusion set three ways — no context, single passage
(the Phase 2.5 and Phase 3 condition), and the bot's real context — so the
before-and-after is measured on identical questions with identical scoring.
`citation_exact` is printed alongside `allCites` to close the loophole described
above. The oracle "gold" column is now the same assembly with the grounding
passage forced in, which bounds what better retrieval alone could buy.

Artifacts go to `flan-t5-small-context-v2` on Drive, so the Phase 2.5 model and
its `TRAINING_DONE.json` are left intact for comparison.

### Results

4 epochs, 59.6 minutes on a T4, best validation citation accuracy 89.9%. Raw
numbers in `data/processed/phase3_5_results.json`.

**The format fix worked on the test set.** Same 676 held-out questions, same
model size, same metric code:

| Context | token F1 | citation F1 | allCites | citation_exact | EM |
|---|---|---|---|---|---|
| none (no retrieval) | 58.3% | 59.7% | 42.2% | 41.8% | 0.0% |
| **the bot's context** | **88.4%** | **95.5%** | **92.7%** | **92.0%** | **44.2%** |
| oracle (grounding passage forced in) | 88.9% | 98.2% | 95.4% | 94.7% | 46.6% |

Citation accuracy went from 86.8% (Phase 2.5) to 92.7%, and exact-match answers
from 37.4% to 44.2%. **Retrieval is now within 0.5 F1 of its own oracle**, so
almost nothing is left to gain from better retrieval — the remaining error is in
the model and in truncation.

### The two capabilities this project exists for are now essentially solved

| qa_type | n | Phase 2.5 | Phase 3.5 |
|---|---|---|---|
| `old_to_new` — what did this section become? | 114 | 93.0% | **99.1%** |
| `new_to_old` — what was it before? | 100 | 71.0% | **100%** |
| `section_lookup` | 109 | 75.2% | 82.6% |
| `offence_classification` | 50 | 92.0% | 96.0% |
| `punishment` | 52 | 100% | 100% |
| `case_law` | 20 | 30.0% | 25.0% |

(citation accuracy; `removed` and `new_provision` remain at 100% on their small
test slices.) Against the Phase 2 baseline, where the model was right in **0 of
214** correspondence questions, it is now right in **213 of 214**.

**Unaided performance is unchanged** at 58.3 F1 / 42.2% citations, against Phase
2.5's 58.4 / 43.8 and Phase 2's 58.2 / 43.8. Three runs, three context regimes,
the same score without context — the gains come from retrieval, not from the
training rearrangement flattering the model.

**The distractor condition held up**, which is what stops the model believing
context blindly: 76.7% citation accuracy when handed a deliberately wrong
passage, up from 73.3%.

### The cost: verbatim wording

| qa_type | n | Phase 2.5 F1 | Phase 3.5 F1 |
|---|---|---|---|
| `section_text` | 220 | 88.4% | **73.8%** |
| `case_law` | 20 | 85.3% | 74.2% |

Four passages share one 430-token budget, so each gets about 107 tokens and a
long section is cut mid-text. Citation accuracy on `section_text` barely moved
(96.8% → 94.5%): the model still identifies the provision correctly, but it
paraphrases the wording instead of reproducing it. That is the trade the
multi-passage context buys the correspondence questions with, and it should be
reported as a trade rather than glossed as a win.

### The confusion set still has not moved, and now we know why

| Context | token F1 | citation F1 | allCites | citation_exact |
|---|---|---|---|---|
| none | 28.1% | 54.3% | 23.3% | **0.0%** |
| single passage | 35.8% | 63.1% | 23.3% | **0.0%** |
| the bot's context | 36.1% | 63.1% | 23.3% | **0.0%** |

Identical across all three conditions, and `citation_exact` is zero everywhere.
The Phase 3.5 hypothesis — that the multi-passage format was the blocker — is
**refuted for this set**, even though it was correct for the test set.

The cause is in the answer shapes:

| | Answers opening with a negation ("No", "None", "There is no single one") |
|---|---|
| Training set | **41 / 11,751 — 0.3%** |
| Confusion set | **43 / 44 — 97.7%** |

**The model was never taught to say no.** It has seen "X corresponds to Y"
eleven thousand times and almost never seen a question whose correct answer is
that the correspondence does not hold, so that is what it produces — hence "BNS
Section 105L" for sedition, and "no equivalent in the CrPC" for BNSS 482 with the
CrPC 482 passage in front of it.

The confusion set's composition makes this decisive: 33 of its 44 questions are
`collision` (same number, different subject), `merged` (several old sections into
one) or `split` (one old section across several new), and **the training data
contains no question of those three shapes at all.** The `qa_type` mix is
`section_text` 3,824, `old_to_new` 1,978, `section_lookup` 1,888, `new_to_old`
1,706, `offence_classification` 895, `punishment` 890, `case_law` 346, `removed`
112, `transition` 67, `new_provision` 36, `scope` 9. The confusion set is not a
harder version of anything in there; it is a different task.

This is a gap in the dataset I designed, not a limit of the architecture or of
`flan-t5-small`. It is also the last thing standing between this project and the
Phase 4 comparison, because the confusion set *is* the comparison.

**Artifact:** `MyDrive/legal-llm-bot/flan-t5-small-context-v2` (not published to
the Hub — no `HF_TOKEN` was set). Still `flan-t5-small` rather than
`flan-t5-base`, for the compute reason recorded above.

---

## Phase 3.6 — teaching the model to say no

`scripts/build_negation_dataset.py` adds the three question shapes the training
data never contained, built from the same concordance, for provisions that are
not in any held-out group.

| Shape | Question | Families available | Rows |
|---|---|---|---|
| `collision` | Does BNSS 298 deal with the same subject as CrPC 298? | 926 | 700 |
| `merged` | Is *voluntarily causing hurt* still dealt with under IPC 323? | 196 | 212 |
| `no_single` | Which single IPC section corresponds to BNS 126? | 47 | 94 |

1,006 rows in total, 755 to train and 251 to validation, which moves the share of
training answers opening with a negation from **0.3% to 6.4%**. `collision` takes
one phrasing per provision from a large pool; the two thinner shapes emit every
phrasing of each family, with three phrasings written for collisions and two for
the others so that the model learns the relation rather than one sentence pattern.

### What is deliberately *not* generated

**`split` questions, and this is a real limitation.** There are only five split
families in the entire concordance and **all five are inside held-out groups** —
the confusion set already consumed every one of them. Any split training example
would leak. The `no_single` shape teaches the same "there is no single one" answer
from many-to-one families instead, which is an honestly different relation
(several old sections into one new, rather than one old across several new). Five
of the 44 confusion questions therefore remain shape-novel at evaluation time.

**`removed` questions.** Training already holds 112, they score 100% on the test
set, and asserting that a provision was repealed needs the vetted allowlist rather
than the mere absence of a row in a correspondence table.

### Leakage discipline

`test.jsonl` and `confusion_test_set.jsonl` are neither read for content nor
written. The generator rebuilds the same union-find over the concordance that
`build_splits.py` uses, so a group id means the same fact in both, and then:

* a provision whose group is in the test set or the confusion set is **dropped** —
  106 groups blocked, costing 70 collisions, 56 merges and 18 fan-ins;
* a provision whose group is already in validation goes to validation, so the
  existing split is respected rather than re-cut;
* remaining groups split 90/10 at group level.

`validate_data.py` gained a check group that re-verifies all of this from the
files themselves, because the negation set and the splits can be regenerated
independently. It also asserts that every generated answer actually opens with a
negation, and prints the before/after share so the figure quoted here cannot
drift from the data.

### Correctness of the generated claims

A collision is only asserted where the two provisions genuinely differ: the
same-numbered section in the new code must not be the counterpart recorded in the
concordance, and the two headings must share at most half their content words.
That guard rejected 15 pairs whose titles were too similar to call different, and
11 where the same number *is* the counterpart. Four collisions were spot-checked
by hand against the gazette headings and all four were correct (CrPC 96 →
BNSS 96, CrPC 468 → BNSS 468, IPC 193 → BNS 193, Evidence Act 50 → BSA 50).

Two phrasing bugs were found and fixed by reading the output rather than trusting
the templates. Asking whether a person can "still be charged" under a provision
only makes sense where that provision creates an offence, so it is now restricted
to sections titled as a punishment — it was producing "Can IPC Section 55 still be
charged for commutation of sentence of imprisonment for life?". And five headings
are clauses rather than noun phrases ("when they may be asked"), which read as
nonsense in the subject slot; those fall back to a phrasing that needs no subject.

### Caveat that belongs in the write-up

**The confusion set's question templates now appear in training, on different
provisions.** It remains held out at the level of fact — no provision in it, nor
any provision sharing a concordance group with one, contributes a training row —
but it is no longer novel in form. That is the correct trade (a model cannot be
expected to answer a question shape it has never seen, and the comparison against
a general-purpose LLM is about facts, not phrasing), and it should be stated
plainly rather than left for a reader to discover.

### Results

6 epochs, 98 minutes on a T4, best validation citation accuracy 81.1%. Raw numbers
in `data/processed/phase3_6_results.json`. The best-validation figure is *lower*
than Phase 3.5's 89.9% because the validation set now contains 251 of the new
negation rows, which are harder than what it held before — the comparison to watch
is on the held-out sets below, not on that number.

**The confusion set moved, and the shape of the movement is the point.**

| Context | token F1 | allCites | citation_exact | questions right |
|---|---|---|---|---|
| none | 28.1% → **55.0%** | 23.3% → **25.6%** | 0.0% → **2.3%** | 11 / 44 |
| single passage | 35.8% → **64.5%** | 23.3% → **30.2%** | 0.0% → **4.7%** | 13 / 44 |
| **the bot's context** | 36.1% → **76.5%** | 23.3% → **60.5%** | 0.0% → **34.9%** | **27 / 44** |

Through Phase 2.5, 3 and 3.5 this set sat at 23.3% in every condition — retrieval
made no difference whatever. It now separates cleanly: 11 questions right with no
context, 13 with a single retrieved passage, **27 with the bot's assembled
context**. The counterpart lookup built in Phase 3 is finally worth something,
because the model has been taught the answer shape that lets it use what it is
shown.

`citation_exact` going from 0 to 34.9% matters more than the `allCites` column
here. That metric requires the answer's citation set to *match* the gold set, so
it cannot be satisfied by naming the right section alongside an invented one —
which is exactly how Phase 3.5 scored a vacuous 100% on the `removed` questions.
Fifteen of these 44 answers are now exactly right.

**The test set paid nothing for it.** 1,006 new rows of a new question type, and
every existing capability is within noise of where Phase 3.5 left it:

| qa_type | n | allCites 3.5 → 3.6 |
|---|---|---|
| section_text | 220 | 94.5% → 94.5% |
| old_to_new | 114 | 99.1% → 99.1% |
| section_lookup | 109 | 82.6% → 82.6% |
| new_to_old | 100 | 100% → 100% |
| punishment | 52 | 100% → 100% |
| offence_classification | 50 | 96.0% → **100%** |
| case_law | 20 | 25.0% → 20.0% |

Overall test citation accuracy 92.7% → 92.9%, token F1 88.4 → 88.1. Teaching the
model to say no did not teach it to say no when the answer is yes, which was the
main risk of adding 1,006 negations to the mix.

### What is still wrong

**17 of the 44 confusion questions are still missed**, and five of those are the
`split` family that could not be trained at all without leaking — so the
achievable ceiling on this set, as currently trained, is around 39/44 rather than
44/44.

**`case_law` remains the weakest capability** at 20% citation accuracy on 20
questions. It has 346 training rows against `section_text`'s 3,824, and judgment
summaries are far less templated than statutory text. It is a small slice and a
known weakness rather than a new regression.

**Two validation numbers moved in the wrong direction and deserve honest
flagging**: on the stratified validation slice, citation accuracy under a
deliberately wrong passage fell 76.7% → 56.7% (n=30) and under no context 50.0% →
26.7% (n=15). Those slices are small — 30 and 15 rows, so a handful of answers
each — and the slice is drawn from a pool that now includes the negation rows, so
the two runs are not measuring quite the same thing. But the direction is
consistent with a model more willing to answer "no", which would cost it on rows
where a correspondence does exist and it cannot see it. The test set's `none`
column does not show this (44.3% against 42.2%), so it is a caution to re-check
rather than a demonstrated regression.

**Artifact:** `MyDrive/legal-llm-bot/flan-t5-small-context-v3`.

### Where the project stands

The central claim is now demonstrated on the hardest available evidence. On the 44
questions built specifically to catch a model that confuses the old codes with the
new ones:

| | Questions right |
|---|---|
| Phase 2 — `flan-t5-base` fine-tuned, no retrieval | 11 / 44 |
| Phase 2.5 / 3 / 3.5 — retrieval, untrained answer shape | 10 / 44 |
| **Phase 3.6 — retrieval + the answer shapes** | **27 / 44** |

(`allCites`; on the stricter `citation_exact` the first two rows are 0 / 44 and
the last is 15 / 44.)

Phase 4 puts the same 44 questions to a general-purpose LLM for the third column,
which is the comparison the whole project was built to make.

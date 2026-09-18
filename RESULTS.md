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

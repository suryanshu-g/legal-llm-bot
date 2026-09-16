# Results

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

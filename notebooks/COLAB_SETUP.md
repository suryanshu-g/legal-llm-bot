# Running the fine-tuning notebook on Colab — step by step

Follow these in order. Each step says what you should see, so you know whether
to carry on or stop.

---

## Step 1 — Open the notebook

Go to **https://colab.research.google.com** → **File → Open notebook** → the
**GitHub** tab.

Paste this into the search box and press Enter:

```
suryanshu-g/legal-llm-bot
```

Click **`notebooks/finetune_flan_t5.ipynb`** in the results.

> If the GitHub tab shows nothing, open this link directly:
> https://colab.research.google.com/github/suryanshu-g/legal-llm-bot/blob/main/notebooks/finetune_flan_t5.ipynb

**You should see:** a notebook titled "Fine-tuning Flan-T5-base for Indian
criminal law after the 2024 recodification".

---

## Step 2 — Turn the GPU on (do this before running anything)

**Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save.**

This is the single most common thing to get wrong. Without it the notebook will
technically run but training will take many hours instead of under one.

---

## Step 3 — Save your own copy

**File → Save a copy in Drive.**

Work in that copy. Otherwise your outputs are lost when the tab closes, and you
cannot save changes back to a notebook opened from GitHub.

---

## Step 4 — Run the first four cells one at a time

Click a cell, press **Shift+Enter**, wait for it to finish, then do the next.
Do **not** use Run All yet.

| Cell | What it is | What you should see |
|---|---|---|
| Settings | `REPO_URL = ...` | no output — that is correct |
| `nvidia-smi` | GPU check | a table with **Tesla T4** in it |
| `%pip install` | dependencies | a few "Installing..." lines, ~1 minute |
| version print | diagnostics | `torch`, `transformers`, `datasets` and their versions |

**If the GPU cell says "No GPU detected"** — go back to Step 2.

**If a "RESTART SESSION" button appears after the install** — click it. That is
normal: Colab preloads some packages and pip replaced one. After the restart,
**re-run the settings cell and the imports cell**, then carry on. You do not
need to re-run the install.

---

## Step 5 — Run everything else

**Runtime → Run all.** Then leave it alone.

Rough timings on a T4:

| Stage | Time |
|---|---|
| Clone + load data | under 1 minute |
| Tokenising | 1–2 minutes |
| **Training** | **40–90 minutes** (early stopping decides; the 25-epoch cap will not be reached) |
| Generating test answers | 10–20 minutes |
| Confusion set | 1–2 minutes |

**Keep the browser tab open.** Colab disconnects idle sessions and you lose the
run. If you need to step away, leave the tab visible on screen.

---

## Step 6 — Saving the model (optional but recommended)

At the end the notebook saves the trained model. By default it mounts Google
Drive and asks you to authorise it — click through the popup.

**To publish it to Hugging Face instead** (better: public, citable, and a
concrete link for the paper), do this *before* Step 5:

1. Get a token at https://huggingface.co/settings/tokens → **New token** →
   type **Write** → copy it.
2. In Colab, click the **key icon** in the left sidebar.
3. **Add new secret** → Name: `HF_TOKEN` → Value: paste your token →
   turn on **Notebook access**.

The notebook checks for that secret automatically. No token, no problem — it
falls back to Drive and tells you so.

---

## Step 7 — Collect the results

When it finishes you want four things:

1. The **loss curve** chart.
2. The **per-question-type metrics** table.
3. The **confusion set** examples.
4. The file `/content/phase2_results.json` — every number in one place.
   Download it: click the folder icon in the left sidebar, find the file,
   three-dot menu → **Download**.

---

## If something goes wrong

**Copy the whole red error block**, plus the output of the version-diagnostics
cell from Step 4. Those two together identify almost any problem immediately.

Common ones:

**Expected, not an error:** the training-setup cell prints a line like

```
  adapted: warmup_ratio -> warmup_steps
```

That is the notebook noticing it is running on transformers 5, where
`warmup_ratio` was removed and `warmup_steps` now accepts a fraction meaning
the same thing. It adapts automatically; nothing to do.

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | batch too large | In the training-arguments cell set `BATCH_SIZE = 4` and `GRAD_ACCUM = 4`. That keeps the effective batch at 16, so results are unchanged. |
| Loss shows `nan` | fp16 with T5 | Should not happen — the notebook forces fp32 on T4. If it does, confirm `bf16: False` printed in the model cell. |
| `NameError` on a variable | cells run out of order, or a restart wiped state | **Runtime → Restart and run all.** |
| `git clone failed` | network hiccup | Re-run that cell. |
| `TypeError: ... unexpected keyword argument` | a transformers API rename this notebook has not seen | Send me the argument name and the version-diagnostics output; the fix is one line in the alias table. |
| Disconnects mid-training | idle timeout | Keep the tab in view; re-run from the top. |
| Training is extremely slow | running on CPU | Step 2. Check `nvidia-smi` shows a T4. |

---

## What "good" looks like

So you can tell success from failure when reading the output:

* **Exact match will be low.** The answers are full sentences, and exact match
  demands them back word for word. Read **token F1** as the quality number.
* **The confusion set should be the *worst* result in the notebook.** That is
  expected and is the point — it is a no-retrieval model being asked about facts
  deliberately kept out of its training. It is the baseline that Phase 3's
  retrieval layer gets measured against. Write the number down.
* **The per-type table is the real result.** If `section_text` scores well but
  `old_to_new` and `new_to_old` do not, the model has learnt to recite statute
  text without learning the old-to-new correspondence — which is the thing this
  project is actually about. The **citation F1** column is the one to read there.

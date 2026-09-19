# Hosting the bot for free

Three options, cheapest effort first. **None costs money**, but Hugging Face
changed its rules in a way that matters, so read the first section before
starting.

## What Hugging Face now charges for

From the [Spaces documentation](https://huggingface.co/docs/hub/en/spaces-overview):

> Static Spaces are free for everyone. Gradio and Docker Spaces run on compute
> and **require a paid plan to create**: PRO for personal accounts, Team or
> Enterprise for organizations. Free personal accounts in good standing can still
> host up to 2 Gradio Spaces running on ZeroGPU.

So a Gradio Space on the free "CPU basic" hardware is **no longer free to
create**, even though the hardware itself is listed at no hourly cost. The free
route is ZeroGPU, and it has an entry requirement:

> Free personal accounts: accounts in good standing (**verified email, account
> older than 30 days**) can host up to 2 ZeroGPU Spaces for free.

Free accounts also get **5 minutes of GPU time per day**, which sounds small but
is generous here — this model answers in about a second, so that is a few hundred
questions a day.

**If your account is less than 30 days old, skip to option 1.**

---

## Option 1 — a public link from Colab (works today, free)

Free, no new accounts, live in about five minutes. Gradio opens a tunnel and
gives you an `https://….gradio.live` address that works from any phone.

1. Open [`notebooks/serve_bot.ipynb`](../notebooks/serve_bot.ipynb) in Colab
2. **Runtime → Run all**, allow Drive access
3. Copy the `Running on public URL:` link from the last cell

The link lasts a week, or until the notebook stops — Colab disconnects an idle
session after about 90 minutes. Start it before a demo, not the night before.
Re-run the last cell for a fresh link.

## Option 2 — the same thing from your own laptop

If the model is already downloaded locally:

```powershell
cd "C:\Users\ACER\Desktop\Legal bot\legal-llm-bot"
.\.venv\Scripts\python.exe -m pip install "gradio>=4.44,<6"
.\.venv\Scripts\python.exe space\app.py --share
```

Same kind of public link, running on your machine, for as long as the window
stays open. Useful for a recording, where a dropped Colab session mid-take is the
thing most likely to ruin it.

## Option 3 — a permanent ZeroGPU Space (once the account is 30 days old)

1. **https://huggingface.co/join**, verify the email, and note the date — the
   30-day clock starts now
2. When eligible, go to **https://huggingface.co/new-space**
3. Name it, license MIT, SDK **Gradio**, and choose **ZeroGPU** hardware
4. **Files → Add file → Upload files**: `app.py`, `requirements.txt` and
   `README.md` from this folder
5. Upload again with `model/` typed in the path box, and drop in the five files
   from `models/flan-t5-small-context-v3` — `config.json`,
   `generation_config.json`, `model.safetensors`, `tokenizer.json`,
   `tokenizer_config.json`
6. Watch the **App** tab; the first build takes five to ten minutes

ZeroGPU needs two changes this repository's `app.py` does not make, because they
are pointless without it: `import spaces`, and a `@spaces.GPU` decorator on the
function that generates. Add `spaces` to `requirements.txt` and wrap `ask`:

```python
import spaces

@spaces.GPU(duration=30)
def ask(question: str):
    ...
```

ZeroGPU also requires **PyTorch 2.8 or newer** and the Gradio SDK, so relax the
`torch>=2.1` pin in `requirements.txt` to `torch>=2.8` for that deployment.

### If it misbehaves

**"No application file"** — `app.py` is not at the top level, or the block at the
top of `README.md` was pasted as ordinary text rather than kept as the file's
first lines.

**Answers read like "reheatreheat blackjack"** — the model files landed beside
`app.py` instead of inside `model/`, so it is running an untrained checkpoint.
The Files tab should show `model/model.safetensors`.

**The log says "sources-only mode"** — the Space cannot see the model folder at
all. Same fix. The app still answers with the retrieved law, so this is a soft
failure, not a crash.

---

## What runs where

| | The website | Colab / local link | ZeroGPU Space |
|---|---|---|---|
| Hosting | GitHub Pages | your session | Hugging Face |
| Cost | free | free | free (30-day-old account) |
| Concordance and gazette lookup | yes | yes | yes |
| Retrieval over 2,819 passages | no | yes | yes |
| The fine-tuned model | no | **yes** | **yes** |
| Always on | **yes** | no | sleeps, wakes on visit |

The website needs no server because it answers only from verified data, which is
why it is the one that can be permanent for free. The other two are where the
trained model actually runs.

# Putting the bot online, free

[Hugging Face Spaces](https://huggingface.co/spaces) hosts this at no cost: 2
vCPU and 16 GB of RAM, no credit card, no time limit. A free Space goes to sleep
after about 48 hours with no visitors and wakes on the next visit, taking a
minute or so to start.

That is ample. The model is `flan-t5-small` — 77 million parameters — and answers
on CPU in a couple of seconds.

## 1. Make an account

1. Go to **https://huggingface.co/join** and sign up (email, no card)
2. Confirm the email

## 2. Create the Space

3. Go to **https://huggingface.co/new-space**
4. **Space name**: `indian-criminal-law-2024` (or anything you like)
5. **License**: MIT
6. **Select the SDK**: **Gradio**
7. **Space hardware**: **CPU basic — FREE**
8. **Public**
9. Click **Create Space**

## 3. Upload the three project files

10. On your new Space, click the **Files** tab, then **Add file → Upload files**
11. Drag in `app.py`, `requirements.txt` and `README.md` from this `space/` folder
12. **Commit changes to main**

The README matters: its top block tells the Space it is a Gradio app and which
file to run.

## 4. Upload the model

13. **Add file → Upload files** again
14. Drag in the **contents** of `models/flan-t5-small-context-v3` —
    `config.json`, `generation_config.json`, `model.safetensors`,
    `tokenizer.json`, `tokenizer_config.json`
15. In the **path** box above the file list, type `model/` so they land in a
    folder called `model` rather than beside `app.py`
16. **Commit changes to main**

`model.safetensors` is 308 MB, so this upload takes a while. Hugging Face handles
large files automatically; there is nothing to configure.

## 5. Watch it build

17. Click the **App** tab. It shows a build log while it installs PyTorch and
    clones the datasets from GitHub — **five to ten minutes the first time**
18. When the log ends with `Running on local URL`, the interface appears

Your public link is `https://huggingface.co/spaces/<your-username>/indian-criminal-law-2024`.

## If something goes wrong

**"No application file"** — `app.py` is not at the top level of the Space, or the
README's top block is missing or was pasted as ordinary text.

**Answers look like "reheatreheat blackjack"** — the model files are in the wrong
place, so it is running an untrained checkpoint. Check the Files tab shows
`model/model.safetensors` and not `model.safetensors` at the root.

**"sources-only mode" in the log** — the Space cannot see the model folder at
all. Same fix. The app still runs and returns the retrieved law, so this is a
soft failure rather than a crash.

**The build fails installing torch** — retry the build from the Settings tab;
the free builder occasionally times out on the first large download.

## What runs where

| | The website | This Space |
|---|---|---|
| Hosting | GitHub Pages | Hugging Face |
| Concordance and gazette lookup | yes | yes |
| Retrieval over 2,819 passages | no | yes |
| The fine-tuned model | no | **yes** |
| Cost | free | free |

The website answers from the verified data only, which is why it needs no server.
This Space is where the trained model actually runs.

"""Hugging Face Space: the assistant, with the fine-tuned model actually running.

The website at suryanshu-g.github.io/legal-llm-bot answers from the concordance
and the gazette in the browser, which is verified but has no model in it. This
Space runs the real thing: the Phase 3.6 `flan-t5-small` reading retrieved
passages, on a free CPU box.

It shows both halves of every answer, side by side and labelled, because they are
not equally trustworthy. The retrieved provisions are gazette text with a source
link. The model's sentence is a 77M-parameter paraphrase of them, and on the
held-out set it drifts section numbers to neighbouring provisions, corrupts
statutory titles and invents counterparts for repealed sections. Presenting the
sentence alone would hide that; presenting the sources next to it lets anyone
check it in one glance.

Layout expected in the Space:

    app.py               this file
    requirements.txt
    model/               the trained checkpoint, uploaded once
                         (config.json, model.safetensors, tokenizer.json, ...)

The datasets and `scripts/` are cloned from GitHub at startup, so they never have
to be uploaded and never go stale.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

REPO_URL = "https://github.com/suryanshu-g/legal-llm-bot.git"
REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal-llm-bot")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model"))

# Prefer a cached copy of anything from the hub; a Space restarts often and
# there is no reason to re-resolve files that have not changed.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("cloning the datasets and scripts ...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)

sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))

from bot import DISCLAIMER, Bot  # noqa: E402

HAS_MODEL = os.path.isdir(MODEL_DIR) and any(
    f.endswith((".safetensors", ".bin")) for f in os.listdir(MODEL_DIR)) \
    if os.path.isdir(MODEL_DIR) else False

print(f"model at {MODEL_DIR}: {'found' if HAS_MODEL else 'ABSENT - sources-only mode'}",
      flush=True)
BOT = Bot(model_dir=MODEL_DIR if HAS_MODEL else None)
print(f"ready: {len(BOT.retriever.meta)} passages, "
      f"{len(BOT.counterparts)} provisions with counterparts", flush=True)

EXAMPLES = [
    "Which BNS section replaced IPC Section 302?",
    "Is an offence under BNS Section 303 bailable?",
    "Is mischief still dealt with under IPC Section 425?",
    "Which BNS section corresponds to IPC Section 124A?",
    "Which CrPC section corresponds to BNSS Section 173?",
    "What does BNS Section 103 cover?",
    "An offence was committed on 15 August 2024. Does the IPC or the BNS apply?",
]


def sources_markdown(answer) -> str:
    if answer.refused:
        return "_No sources: the question was outside what this tool answers._"
    if not answer.citations:
        return "_No passage was retrieved for this question._"
    lines = ["Retrieved from the official sources. **This is the part to trust.**", ""]
    for c in answer.citations:
        lines.append(f"- [{c['source']}]({c['source_url']})")
    body = (answer.context or "").strip()
    if body:
        lines += ["", "<details><summary>The passages the model was shown</summary>", ""]
        for chunk in body.split("\n\n---\n\n"):
            lines.append("> " + chunk.replace("\n", " ").strip()[:700])
            lines.append("")
        lines.append("</details>")
    return "\n".join(lines)


def ask(question: str):
    question = (question or "").strip()
    if not question:
        return "", "_Ask about a provision of the BNS, BNSS or BSA._"
    answer = BOT.ask(question)
    said = answer.text.strip()
    if answer.refused:
        return said, sources_markdown(answer)
    if not answer.used_model:
        return ("_No model is loaded, so here is the retrieved law itself rather "
                "than a composed answer._\n\n" + said), sources_markdown(answer)
    return said, sources_markdown(answer)


CAVEAT = """
**Read the sources, not just the sentence.** The answer on the left is written by
a fine-tuned `flan-t5-small` — 77 million parameters — reading the passages on the
right. On the 44 hardest held-out questions it drifts section numbers to
neighbouring provisions, corrupts statutory titles, and invents counterparts for
provisions that were repealed. The retrieved passages are gazette text and are
verified; the sentence is not.
"""

def build_ui():
  """Constructed lazily so the answering logic can be imported and tested
  without Gradio installed."""
  import gradio as gr

  with gr.Blocks(title="Indian criminal law after 1 July 2024",
                 theme=gr.themes.Soft(primary_hue="teal")) as demo:
      gr.Markdown(
          "# Indian criminal law after 1 July 2024\n"
          "The IPC, CrPC and Evidence Act were replaced by the BNS, BNSS and BSA. "
          "Ask what a provision became, what it says, or how the First Schedule "
          "classifies it — every answer cites the gazette."
      )
      with gr.Row():
          box = gr.Textbox(label="Your question", scale=5, autofocus=True,
                           placeholder="Which BNS section replaced IPC Section 302?")
          btn = gr.Button("Ask", variant="primary", scale=1)
      with gr.Row():
          with gr.Column(scale=3):
              out = gr.Markdown(label="Answer")
          with gr.Column(scale=2):
              src = gr.Markdown(label="Sources")
      gr.Markdown(CAVEAT)
      gr.Examples(examples=EXAMPLES, inputs=box)
      gr.Markdown(
          f"_{DISCLAIMER}_\n\n"
          "A student capstone project, not affiliated with or endorsed by any "
          "government body. Code, datasets and full results: "
          "[github.com/suryanshu-g/legal-llm-bot](https://github.com/suryanshu-g/legal-llm-bot)"
      )

      btn.click(ask, inputs=box, outputs=[out, src])
      box.submit(ask, inputs=box, outputs=[out, src])

  return demo

if __name__ == "__main__":
    # --share opens a public https://....gradio.live tunnel, which is how this
    # runs for free: Hugging Face now requires a paid plan to create a Gradio
    # Space on CPU, and its free ZeroGPU exception needs an account over 30 days
    # old. See space/SETUP.md.
    share = "--share" in sys.argv
    build_ui().launch(share=share, show_error=True)

"""Phase 3: the assistant itself — retrieve, answer, cite.

Pipeline for one question:

    1. assemble context   (retrieval + concordance counterparts)
    2. build the prompt   (same template the model was fine-tuned on)
    3. generate           (optional: without a model this returns the sources)
    4. attach citations    and the informational-only disclaimer

The interesting part is step 1. Plain similarity retrieval is not enough for the
questions this project exists to answer. "Which BNS section replaced IPC 302?"
needs the IPC 302 passage *and* the BNS 103 passage, and no single chunk holds
both; asking for more chunks does not help, because similarity ranking does not
preferentially surface a provision's counterpart. Phase 2.5 measured this - for
33 of the 44 confusion questions the gold answer cites two or more provisions,
and at one chunk only 10 of 44 had everything they needed available.

So the counterparts are looked up deliberately, in the concordance that already
exists (`mapping_table.csv`), rather than hoped for from the embedder.

Usage:
    from bot import Bot
    bot = Bot(model_dir="/path/to/flan-t5-small-context")   # or None
    answer = bot.ask("Which BNS section replaced IPC Section 302?")
    print(answer.text)
    for c in answer.citations:
        print(c["source"], c["source_url"])

CLI:
    python scripts/bot.py "Which BNS section replaced IPC 302?"
    python scripts/bot.py --sources-only "Is theft bailable?"
"""

from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieve import PROCESSED, Retriever, chunk_ref, extract_refs

TASK_PREFIX = "answer the indian criminal law question: "
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 320

# Leaves room for the question inside a 512-token encoder.
CONTEXT_TOKEN_BUDGET = 430

DISCLAIMER = ("This is general information about the law, not legal advice. "
              "For advice on a specific matter, consult a qualified advocate.")

# Questions the assistant must not answer, per the project's scope rules. The
# fine-tuned model was trained to refuse these, but a trained refusal is a
# tendency, not a guarantee, so the boundary is enforced here as well.
OUT_OF_SCOPE = re.compile(
    r"\b(be my (lawyer|advocate|counsel)|represent me|are you a lawyer|"
    r"should i plead|will i (be convicted|go to (jail|prison))|"
    r"what will my sentence|loophole|get (away with|off)|"
    r"avoid (being )?(convict|prosecut|arrest|liab)|evade|"
    r"how (do|can) i (avoid|escape|beat)|fake|false (alibi|evidence)|"
    r"bribe|give me legal advice|advise me on my)\b", re.I)

REFUSAL = (
    "I can't help with that. I'm an informational tool, not a licensed advocate: "
    "I can't advise on a specific case, predict an outcome, or help with avoiding "
    "liability. What I can do is set out what a provision of the BNS, BNSS or BSA "
    "says, which older provision it corresponds to, and what reported decisions "
    "have held about it.")


@dataclass
class Answer:
    question: str
    text: str
    citations: list = field(default_factory=list)
    context: str = ""
    refused: bool = False
    used_model: bool = False

    def __str__(self) -> str:
        out = [self.text]
        if self.citations:
            out.append("\nSources:")
            for c in self.citations:
                out.append(f"  - {c['source']}  {c['source_url']}")
        out.append(f"\n{DISCLAIMER}")
        return "\n".join(out)


def load_counterparts() -> dict[str, list[str]]:
    """provision -> the provisions it corresponds to, both directions.

    Built from the concordance, so "IPC 302" yields "BNS 103" and vice versa.
    """
    path = os.path.join(PROCESSED, "mapping_table.csv")
    pairs: dict[str, list[str]] = {}
    if not os.path.exists(path):
        return pairs
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not (r["old_section"] and r["new_section"]):
                continue
            old = f"{r['old_act'].upper()} {r['old_section'].upper()}"
            new = f"{r['new_act'].upper()} {r['new_section'].upper()}"
            pairs.setdefault(old, [])
            pairs.setdefault(new, [])
            if new not in pairs[old]:
                pairs[old].append(new)
            if old not in pairs[new]:
                pairs[new].append(old)
    return pairs


class Bot:
    """Retrieval-grounded assistant. The model is optional."""

    def __init__(self, model_dir: str | None = None, retriever: Retriever | None = None,
                 device: str | None = None):
        self.retriever = retriever or Retriever(device=device)
        self.counterparts = load_counterparts()
        self.by_ref: dict[str, list[int]] = self.retriever.by_ref

        self.model = None
        self.tokenizer = None
        if model_dir:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device).eval()

    # -- context assembly ---------------------------------------------------

    def _rows_for_ref(self, ref: str) -> list[int]:
        return self.by_ref.get(ref, [])

    def select_chunks(self, question: str, k: int = 3) -> list[dict]:
        """The passages to show the model, most important first.

        Order matters, because the context is truncated to a token budget:

            1. chunks for provisions the question names outright;
            2. chunks for those provisions' counterparts in the concordance -
               the half of a correspondence that similarity search misses;
            3. similarity hits, to cover questions phrased by subject rather
               than by section number.
        """
        chosen: list[int] = []

        def add(rows):
            for row in rows:
                if row not in chosen:
                    chosen.append(row)

        asked = extract_refs(question)
        for ref in sorted(asked):
            add(self._rows_for_ref(ref))
        for ref in sorted(asked):
            for other in self.counterparts.get(ref, []):
                add(self._rows_for_ref(other))

        for hit in self.retriever.retrieve(question, k=k):
            add([i for i, m in enumerate(self.retriever.meta)
                 if m["chunk_id"] == hit.chunk_id])

        return [self.retriever.meta[i] for i in chosen]

    def build_context(self, chunks: list[dict], budget: int = CONTEXT_TOKEN_BUDGET,
                      max_chunks: int = 4) -> tuple[str, list[dict]]:
        """Concatenate passages within a token budget.

        Each chunk opens with its own header - act, section, heading, and the
        old/new counterpart - so truncating the tail of a long passage keeps the
        part that identifies the provision. That is what makes sharing one
        budget between several passages workable.
        """
        chunks = chunks[:max_chunks]
        if not chunks:
            return "", []
        share = max(60, budget // len(chunks))
        parts, used = [], []
        for c in chunks:
            text = self._truncate(c["text"], share)
            parts.append(text)
            used.append(c)
        return "\n\n---\n\n".join(parts), used

    def _truncate(self, text: str, budget: int) -> str:
        if self.tokenizer is None:
            # No model loaded: approximate, 4 characters per token.
            limit = budget * 4
            return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " [...]"
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return text
        cut = self.tokenizer.decode(ids[:budget], skip_special_tokens=True)
        stop = max(cut.rfind(". "), cut.rfind("; "))
        if stop > len(cut) * 0.5:
            cut = cut[:stop + 1]
        return cut.rstrip() + " [...]"

    def build_prompt(self, question: str, context: str) -> str:
        prompt = TASK_PREFIX + question.strip()
        if context.strip():
            prompt += "\n\ncontext: " + context.strip()
        return prompt

    # -- answering ----------------------------------------------------------

    def ask(self, question: str, k: int = 3, max_chunks: int = 4) -> Answer:
        if OUT_OF_SCOPE.search(question):
            return Answer(question=question, text=REFUSAL, refused=True)

        chunks = self.select_chunks(question, k=k)
        context, used = self.build_context(chunks, max_chunks=max_chunks)
        citations = [{"chunk_id": c["chunk_id"], "source": c["source"],
                      "source_url": c["source_url"]} for c in used]

        if self.model is None:
            # Sources-only mode: no model loaded, so report the law itself
            # rather than invent an answer.
            text = ("No language model is loaded, so here is the relevant law "
                    "rather than a composed answer:\n\n" + context)
            return Answer(question=question, text=text, citations=citations,
                          context=context)

        import torch

        prompt = self.build_prompt(question, context)
        with torch.no_grad():
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=MAX_SOURCE_LENGTH).to(self.model.device)
            out = self.model.generate(**enc, max_new_tokens=MAX_TARGET_LENGTH,
                                      num_beams=1, no_repeat_ngram_size=3)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Cite only the passages the answer actually refers to, where that can
        # be told from the provisions it names; otherwise cite everything shown.
        answered = extract_refs(text)
        if answered:
            narrowed = [c for c in citations
                        if (cr := chunk_ref(c["chunk_id"])) and cr in answered]
            if narrowed:
                citations = narrowed

        return Answer(question=question, text=text, citations=citations,
                      context=context, used_model=True)


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Ask the legal assistant.")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--model-dir", default=os.environ.get("LEGAL_BOT_MODEL"),
                    help="fine-tuned model directory; omit for sources-only mode")
    ap.add_argument("--sources-only", action="store_true",
                    help="skip the model even if one is configured")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    bot = Bot(model_dir=None if args.sources_only else args.model_dir)
    if not bot.model:
        print("(sources-only mode - no fine-tuned model loaded)\n")

    if args.question:
        print(bot.ask(" ".join(args.question), k=args.k))
        return

    print("Ask a question about Indian criminal law. Blank line or Ctrl-C to quit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print()
        print(bot.ask(q, k=args.k))
        print()


if __name__ == "__main__":
    _cli()

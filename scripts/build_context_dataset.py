"""Phase 2.5: build context-augmented training data.

Phase 2 trained on bare question -> answer. This produces the training set for a
second pass where the model reads a retrieved passage before answering, so that
adding retrieval at inference actually changes the answer rather than decorating
it.

Three conditions, and each teaches something different:

  positive   (~70%)  the correct chunk. Teaches the model to read and use it.
  distractor (~20%)  a wrong but plausible chunk, with the correct answer still
                     as the target. Teaches it not to believe context blindly -
                     this is the condition that matters, because a retriever
                     will be wrong sometimes and a model trained only on
                     positives will follow it off a cliff.
  none       (~10%)  empty context. Keeps the model usable when retrieval
                     returns nothing above threshold.

Distractors are drawn from the retriever's own top hits for that question, with
the correct chunk excluded. A random unrelated chunk would be far easier to
ignore than the near-miss a real retriever actually surfaces - a neighbouring
section, or the old-code twin of the right answer.

Outputs:
    data/processed/train_context.jsonl
    data/processed/val_context.jsonl
    data/processed/context_dataset_stats.json

The test and confusion sets are deliberately untouched. Context goes into those
at inference time, from live retrieval, so that evaluation measures the whole
pipeline rather than a pre-baked best case.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieve import PROCESSED, Retriever, extract_refs

SEED = 20240701
MIX = {"positive": 0.70, "distractor": 0.20, "none": 0.10}

# Token budget for the context passage. The Phase 2 prompt is ~23 tokens of
# question; 448 for the passage leaves headroom inside a 512-token encoder.
CONTEXT_TOKEN_BUDGET = 448

TASK_PREFIX = "answer the indian criminal law question: "


def load_jsonl(name: str) -> list[dict]:
    with open(os.path.join(PROCESSED, name), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_chunk_lookup(meta: list[dict]) -> dict:
    """source_chunk_id (from the split index) -> corpus chunk_id.

    The two id spaces do not line up everywhere:

    * case-law rows are keyed on the Indian Kanoon document id, while the
      corpus numbers its case chunks sequentially, so they are matched through
      the source URL;
    * offence_classification rows are keyed on the BNS section, but their
      answer comes from the First Schedule, so the grounding chunk is the
      schedule one rather than the section text;
    * transition rows all ground in the single commencement chunk;
    * scope rows have no grounding chunk at all - the refusal is policy, not
      statute - so they never get a positive.
    """
    by_id = {m["chunk_id"]: m for m in meta}
    by_doc = {}
    for m in meta:
        doc = re.search(r"/doc/(\d+)", m.get("source_url") or "")
        if doc:
            by_doc[f"case_{doc.group(1)}"] = m["chunk_id"]
    return by_id, by_doc


def correct_chunk(source_chunk_id: str, qa_type: str, by_id, by_doc):
    if qa_type == "scope":
        return None
    if qa_type == "transition":
        return "transition_commencement" if "transition_commencement" in by_id else None
    if source_chunk_id.startswith("case_"):
        return by_doc.get(source_chunk_id)
    if qa_type == "offence_classification":
        schedule = f"schedule_{source_chunk_id}"
        if schedule in by_id:
            return schedule
    return source_chunk_id if source_chunk_id in by_id else None


def truncate_tokens(text: str, tokenizer, budget: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text
    cut = tokenizer.decode(ids[:budget], skip_special_tokens=True)
    # Prefer a sentence boundary so the passage does not end mid-clause.
    stop = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(".\n"))
    if stop > len(cut) * 0.5:
        cut = cut[:stop + 1]
    return cut.rstrip() + " [...]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4,
                    help="candidates fetched per question for distractors")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    retriever = Retriever()
    by_id, by_doc = build_chunk_lookup(retriever.meta)
    print(f"index: {len(by_id)} chunks, {len(by_doc)} case chunks keyed by doc id")

    stats = {}
    for split in ("train", "val"):
        rows = load_jsonl(f"{split}.jsonl")
        index = load_jsonl(f"{split}_index.jsonl")
        assert len(rows) == len(index), f"{split}: split/index misaligned"

        # Which chunk grounds each row, and which rows can take a positive.
        targets = [correct_chunk(m["source_chunk_id"], m["qa_type"], by_id, by_doc)
                   for m in index]
        unresolved = Counter(m["qa_type"] for m, t in zip(index, targets)
                             if t is None)

        # Assign conditions. A row with no grounding chunk cannot be positive,
        # so it is dealt distractor or none instead, keeping the overall mix.
        rng = random.Random(SEED)
        order = list(range(len(rows)))
        rng.shuffle(order)
        conditions: list[str] = [""] * len(rows)
        n_pos = int(len(rows) * MIX["positive"])
        n_dis = int(len(rows) * MIX["distractor"])
        want = (["positive"] * n_pos + ["distractor"] * n_dis
                + ["none"] * (len(rows) - n_pos - n_dis))
        spare: list[int] = []
        for pos, i in enumerate(order):
            c = want[pos]
            if c == "positive" and targets[i] is None:
                spare.append(i)
                continue
            conditions[i] = c
        for i in spare:  # groundless rows get a distractor or nothing
            conditions[i] = "distractor" if rng.random() < 0.67 else "none"

        # Distractors come from the retriever's own near misses, so they are
        # the mistakes retrieval will actually make.
        #
        # A near miss that happens to contain the answer is not a distractor:
        # for "which BNS section replaced IPC 302?" the second hit is often the
        # BNS chunk, which states the correspondence itself. Any chunk about a
        # provision the gold answer cites is therefore excluded too, leaving
        # only context that is genuinely unhelpful.
        rows_by_ref = {ref: {retriever.meta[r]["chunk_id"] for r in rws}
                       for ref, rws in retriever.by_ref.items()}

        need = [i for i, c in enumerate(conditions) if c == "distractor"]
        print(f"[{split}] retrieving distractors for {len(need)} rows ...")
        distractors: dict[int, str] = {}
        BATCH = 256
        for start in range(0, len(need), BATCH):
            batch = need[start:start + BATCH]
            queries = [rows[i]["instruction"] for i in batch]
            results = retriever.retrieve_batch(queries, k=args.k + 6)
            for i, hits in zip(batch, results):
                banned = {targets[i]}
                for ref in extract_refs(rows[i]["output"]):
                    banned |= rows_by_ref.get(ref, set())
                for h in hits:
                    if h.chunk_id not in banned:
                        distractors[i] = h.chunk_id
                        break
            print(f"\r  {min(start + BATCH, len(need))}/{len(need)}", end="")
        print()

        out, kinds, ctx_tokens = [], Counter(), []
        no_distractor = 0
        for i, (row, meta) in enumerate(zip(rows, index)):
            cond = conditions[i]
            chunk_id = None
            if cond == "positive":
                chunk_id = targets[i]
            elif cond == "distractor":
                chunk_id = distractors.get(i)
                if chunk_id is None:       # retriever returned nothing usable
                    cond, no_distractor = "none", no_distractor + 1

            context = ""
            if chunk_id:
                context = truncate_tokens(by_id[chunk_id]["text"], tokenizer,
                                          CONTEXT_TOKEN_BUDGET)
                ctx_tokens.append(
                    len(tokenizer(context, add_special_tokens=False)["input_ids"]))

            kinds[cond] += 1
            out.append({
                "instruction": row["instruction"],
                # `input` is the slot the Phase 2 prompt template already reads,
                # so the context-aware model sees the same format.
                "input": context,
                "output": row["output"],
                "context": context,
                "context_type": cond,
                "context_chunk_id": chunk_id or "",
                "qa_type": meta["qa_type"],
            })

        dest = os.path.join(PROCESSED, f"{split}_context.jsonl")
        with open(dest, "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        # How long does the full prompt get? The context-aware notebook needs a
        # much bigger encoder window than Phase 2's 96 tokens.
        src_tokens = sorted(
            len(tokenizer(TASK_PREFIX + r["instruction"]
                          + ("\n\ncontext: " + r["input"] if r["input"] else ""),
                          add_special_tokens=True)["input_ids"])
            for r in out)
        p = lambda q: src_tokens[min(len(src_tokens) - 1,
                                     int(len(src_tokens) * q))]
        stats[split] = {
            "rows": len(out),
            "by_context_type": dict(kinds),
            "unresolved_positive_targets": dict(unresolved),
            "distractor_lookup_failures": no_distractor,
            "context_tokens_median": (sorted(ctx_tokens)[len(ctx_tokens) // 2]
                                      if ctx_tokens else 0),
            "prompt_tokens": {"median": p(0.5), "p95": p(0.95),
                              "p99": p(0.99), "max": src_tokens[-1]},
        }
        print(f"[{split}] wrote {dest}: {len(out)} rows {dict(kinds)}")
        print(f"  prompt tokens: median {p(0.5)}, p95 {p(0.95)}, "
              f"p99 {p(0.99)}, max {src_tokens[-1]}")
        if unresolved:
            print(f"  rows with no grounding chunk: {dict(unresolved)}")

    with open(os.path.join(PROCESSED, "context_dataset_stats.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"seed": SEED, "mix": MIX,
                   "context_token_budget": CONTEXT_TOKEN_BUDGET,
                   "splits": stats}, fh, indent=2)
    print("\nwrote context_dataset_stats.json")


if __name__ == "__main__":
    main()

"""Phase 3.5: build context-augmented training data in the shape the bot serves.

Phase 2 trained on bare question -> answer. Phase 2.5 added a retrieved passage,
which worked: token F1 went from 58.4% to 91.9% on held-out questions. Phase 3
then fixed the confusion set's real problem - that most of its answers name two
provisions while retrieval supplied one - by looking counterparts up in the
concordance, taking context completeness on that set from 34% to 98%.

It made no difference to the answers, and the reason was a training/serving
mismatch of my own making. **Every one of the 10,571 context-augmented training
examples held exactly one passage, while the bot serves up to four**, separated
by rules and each truncated to a quarter of the budget. The model had never seen
that format: it read the first passage, ignored the rest, and occasionally
blended two into nonsense. Right law, wrong envelope.

So this script no longer assembles context of its own. It imports `Bot` and calls
the same `select_chunks` / `build_context` methods that answer a live question,
at the same `MAX_CONTEXT_CHUNKS` and `CONTEXT_TOKEN_BUDGET`, with the same
tokenizer. Training data and serving data cannot drift apart again without the
code path itself changing.

Three conditions, and each teaches something different:

  positive   (~70%)  the bot's real context, with the correct chunk guaranteed
                     present. Teaches the model to read several passages and use
                     the one that answers the question.
  distractor (~20%)  the same assembly with every chunk about the answer's
                     provisions removed, the correct answer still the target.
                     Teaches it not to believe context blindly - this is the
                     condition that matters, because a retriever will be wrong
                     sometimes and a model trained only on positives will follow
                     it off a cliff.
  none       (~10%)  empty context. Keeps the model usable when retrieval
                     returns nothing.

Distractors are drawn from the retriever's own top hits, because a random chunk
would be far easier to ignore than the near-miss a real retriever surfaces - a
neighbouring section, or the old-code twin of the right answer.

Where the correct chunk is not already in the assembled context (a subject-phrased
question whose provision the embedder ranked fifth, say), it is inserted at a
varying position rather than always at the front, so the model does not learn the
shortcut "the answer is in passage one".

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
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bot import (CONTEXT_TOKEN_BUDGET, MAX_CONTEXT_CHUNKS, TASK_PREFIX, Bot)
from retrieve import PROCESSED, extract_refs

SEED = 20240701
MIX = {"positive": 0.70, "distractor": 0.20, "none": 0.10}

# Retrieval depth. SERVE_K is what the bot actually uses, so positives are built
# from exactly that; the extra candidates are only there to leave a distractor
# row something to fall back on once the answer's own provisions are excluded.
SERVE_K = 3
DISTRACTOR_K = 12


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


def assign_conditions(rows, targets, rng) -> list[str]:
    """Deal the three conditions, keeping the overall mix.

    A row with no grounding chunk cannot be positive, so it is dealt a
    distractor or nothing instead.
    """
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
    for i in spare:
        conditions[i] = "distractor" if rng.random() < 0.67 else "none"
    return conditions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256,
                    help="questions encoded per retrieval batch")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows per split, for a quick smoke run")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    # No model: this is the assembly half of the bot only. The tokenizer is
    # handed over so truncation is token-exact rather than estimated.
    bot = Bot(model_dir=None, tokenizer=tokenizer)
    by_id, by_doc = build_chunk_lookup(bot.retriever.meta)
    print(f"index: {len(by_id)} chunks, {len(by_doc)} case chunks keyed by doc id")
    print(f"assembling at MAX_CONTEXT_CHUNKS={MAX_CONTEXT_CHUNKS}, "
          f"budget={CONTEXT_TOKEN_BUDGET} tokens - the same values bot.py serves")

    # chunk_id sets per provision, for excluding the answer's own provisions
    # from a distractor context.
    rows_by_ref = {ref: {bot.retriever.meta[r]["chunk_id"] for r in rws}
                   for ref, rws in bot.retriever.by_ref.items()}

    stats = {}
    for split in ("train", "val"):
        rows = load_jsonl(f"{split}.jsonl")
        index = load_jsonl(f"{split}_index.jsonl")
        assert len(rows) == len(index), f"{split}: split/index misaligned"

        # Phase 3.6: the negation shapes, built by build_negation_dataset.py and
        # kept in their own files so the original splits stay byte-for-byte as
        # build_splits.py wrote them. Their leakage is checked at generation and
        # again by validate_data.py.
        neg = os.path.join(PROCESSED, f"negation_{split}.jsonl")
        if os.path.exists(neg):
            nrows = load_jsonl(f"negation_{split}.jsonl")
            nindex = load_jsonl(f"negation_{split}_index.jsonl")
            assert len(nrows) == len(nindex), f"negation_{split}: index misaligned"
            rows, index = rows + nrows, index + nindex
            print(f"[{split}] + {len(nrows)} negation rows "
                  f"({dict(Counter(m['qa_type'] for m in nindex))})")

        if args.limit:
            rows, index = rows[:args.limit], index[:args.limit]

        targets = [correct_chunk(m["source_chunk_id"], m["qa_type"], by_id, by_doc)
                   for m in index]
        unresolved = Counter(m["qa_type"] for m, t in zip(index, targets)
                             if t is None)

        rng = random.Random(SEED)
        conditions = assign_conditions(rows, targets, rng)

        # One batched retrieval pass over every row that needs context. Deep
        # enough to serve a distractor row after exclusions; positives use only
        # the first SERVE_K, which is what the bot itself asks for.
        need = [i for i, c in enumerate(conditions) if c in ("positive", "distractor")]
        print(f"[{split}] retrieving for {len(need)} of {len(rows)} rows ...")
        hits: dict[int, list] = {}
        for start in range(0, len(need), args.batch):
            batch = need[start:start + args.batch]
            results = bot.retriever.retrieve_batch(
                [rows[i]["instruction"] for i in batch], k=DISTRACTOR_K)
            hits.update(zip(batch, results))
            print(f"\r  {min(start + args.batch, len(need))}/{len(need)}", end="")
        print()

        out, kinds, n_passages = [], Counter(), Counter()
        ctx_tokens, gold_missing, no_distractor = [], 0, 0
        for i, (row, meta) in enumerate(zip(rows, index)):
            cond = conditions[i]
            question, gold = row["instruction"], targets[i]
            chunks: list[dict] = []

            if cond == "positive":
                # Exactly what the bot would assemble for this question.
                chunks = bot.select_chunks(question, hits=hits[i][:SERVE_K])
                ids = [c["chunk_id"] for c in chunks[:MAX_CONTEXT_CHUNKS]]
                if gold not in ids:
                    # Retrieval missed it. Put it in - but not always first, or
                    # the model learns to read passage one and stop.
                    gold_missing += 1
                    chunks = [c for c in chunks if c["chunk_id"] != gold]
                    slot = rng.randrange(0, min(len(chunks), MAX_CONTEXT_CHUNKS - 1) + 1)
                    chunks.insert(slot, by_id[gold])

            elif cond == "distractor":
                banned = {gold} if gold else set()
                for ref in extract_refs(row["output"]):
                    banned |= rows_by_ref.get(ref, set())
                chunks = [c for c in bot.select_chunks(question, hits=hits[i])
                          if c["chunk_id"] not in banned]
                if not chunks:           # nothing plausible left to mislead with
                    cond, no_distractor = "none", no_distractor + 1

            context, used = bot.build_context(chunks) if chunks else ("", [])
            if context:
                ctx_tokens.append(
                    len(tokenizer(context, add_special_tokens=False)["input_ids"]))
            kinds[cond] += 1
            n_passages[len(used)] += 1

            out.append({
                "instruction": question,
                # `input` is the slot the Phase 2 prompt template already reads,
                # so the context-aware model sees the same format.
                "input": context,
                "output": row["output"],
                "context": context,
                "context_type": cond,
                # The grounding chunk, kept for per-condition analysis.
                "context_chunk_id": (gold or "") if cond == "positive" else "",
                "context_chunk_ids": [c["chunk_id"] for c in used],
                "gold_in_context": bool(gold) and gold in [c["chunk_id"] for c in used],
                "qa_type": meta["qa_type"],
            })

        dest = os.path.join(PROCESSED, f"{split}_context.jsonl")
        with open(dest, "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        # How long does the full prompt get? It must fit the 512-token encoder
        # that bot.py truncates to, or serving silently drops the tail.
        src_tokens = sorted(
            len(tokenizer(TASK_PREFIX + r["instruction"]
                          + ("\n\ncontext: " + r["input"] if r["input"] else ""),
                          add_special_tokens=True)["input_ids"])
            for r in out)
        p = lambda q: src_tokens[min(len(src_tokens) - 1,
                                     int(len(src_tokens) * q))]
        positives = [r for r in out if r["context_type"] == "positive"]
        gold_ok = sum(r["gold_in_context"] for r in positives)
        stats[split] = {
            "rows": len(out),
            "by_context_type": dict(kinds),
            "passages_per_context": {str(k): v for k, v in sorted(n_passages.items())},
            "positives_with_gold_passage": f"{gold_ok}/{len(positives)}",
            "gold_injected_after_retrieval_miss": gold_missing,
            "unresolved_positive_targets": dict(unresolved),
            "distractor_lookup_failures": no_distractor,
            "context_tokens_median": (sorted(ctx_tokens)[len(ctx_tokens) // 2]
                                      if ctx_tokens else 0),
            "prompt_tokens": {"median": p(0.5), "p95": p(0.95),
                              "p99": p(0.99), "max": src_tokens[-1]},
        }
        print(f"[{split}] wrote {dest}: {len(out)} rows {dict(kinds)}")
        print(f"  passages per context: {dict(sorted(n_passages.items()))}")
        print(f"  positives holding the gold passage: {gold_ok}/{len(positives)}"
              f"  (injected after a retrieval miss: {gold_missing})")
        print(f"  prompt tokens: median {p(0.5)}, p95 {p(0.95)}, "
              f"p99 {p(0.99)}, max {src_tokens[-1]}")
        if unresolved:
            print(f"  rows with no grounding chunk: {dict(unresolved)}")

    with open(os.path.join(PROCESSED, "context_dataset_stats.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"seed": SEED, "mix": MIX,
                   "context_token_budget": CONTEXT_TOKEN_BUDGET,
                   "max_context_chunks": MAX_CONTEXT_CHUNKS,
                   "serve_k": SERVE_K,
                   "assembled_by": "bot.Bot.select_chunks + build_context",
                   "splits": stats}, fh, indent=2)
    print("\nwrote context_dataset_stats.json")


if __name__ == "__main__":
    main()

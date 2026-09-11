"""Phase 2.5 sanity check: does retrieval actually find the right chunk?

Two things at once. A detailed printout of a handful of questions, so the
results can be read rather than trusted, and recall@k over a larger sample
broken down by question type, so "it looks fine" has a number behind it.

If the straightforward section_text questions do not put the right chunk in the
top 3, something is wrong with the embedding or the chunking and it needs
fixing before any of it is used for training.

Usage:  python scripts/check_retrieval.py [--n 300] [--k 3] [--show 10]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_context_dataset import build_chunk_lookup, correct_chunk, load_jsonl
from retrieve import Retriever

SEED = 20240701


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="questions to score")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--show", type=int, default=10, help="questions to print")
    ap.add_argument("--mode", default="hybrid", choices=("hybrid", "dense"),
                    help="hybrid adds exact citation matching")
    args = ap.parse_args()

    rows = load_jsonl("test.jsonl")
    index = load_jsonl("test_index.jsonl")
    retriever = Retriever()
    by_id, by_doc = build_chunk_lookup(retriever.meta)

    # Only rows with a known grounding chunk can be scored.
    scorable = []
    for row, meta in zip(rows, index):
        gold = correct_chunk(meta["source_chunk_id"], meta["qa_type"],
                             by_id, by_doc)
        if gold:
            scorable.append((row, meta, gold))
    print(f"test rows: {len(rows)}, with a known grounding chunk: "
          f"{len(scorable)}")

    rng = random.Random(SEED)
    sample = scorable if len(scorable) <= args.n else rng.sample(scorable, args.n)

    hits = retriever.retrieve_batch([r["instruction"] for r, _, _ in sample],
                                    k=max(args.k, 5), mode=args.mode)

    # ---------------------------------------------------------- printed view
    by_type: dict[str, list] = defaultdict(list)
    for (row, meta, gold), got in zip(sample, hits):
        by_type[meta["qa_type"]].append((row, gold, got))

    print("\n" + "=" * 78)
    print("SAMPLE QUESTIONS - one per question type, read these")
    print("=" * 78)
    shown = 0
    for qa_type in sorted(by_type):
        if shown >= args.show:
            break
        row, gold, got = by_type[qa_type][0]
        ranks = [h.chunk_id for h in got]
        rank = ranks.index(gold) + 1 if gold in ranks[:args.k] else None
        print(f"\n[{qa_type}]  gold chunk: {gold}   "
              f"{'FOUND at rank ' + str(rank) if rank else 'NOT in top ' + str(args.k)}")
        print("  Q:", textwrap.shorten(row["instruction"], 100, placeholder=" ..."))
        for h in got[:args.k]:
            mark = ">>" if h.chunk_id == gold else "  "
            print(f"   {mark} {h.score:.3f}  {h.chunk_id:<24} {h['source'][:46]}")
        shown += 1

    # ------------------------------------------------------------ recall@k
    print("\n" + "=" * 78)
    print(f"RECALL over {len(sample)} sampled test questions  [mode={args.mode}]")
    print("=" * 78)
    print(f"{'qa_type':<26}{'n':>6}{'R@1':>9}{'R@3':>9}{'R@5':>9}")
    print("-" * 59)

    overall = Counter()
    per_type: dict[str, Counter] = defaultdict(Counter)
    for (row, meta, gold), got in zip(sample, hits):
        ranks = [h.chunk_id for h in got]
        pos = ranks.index(gold) + 1 if gold in ranks else 99
        for cutoff in (1, 3, 5):
            if pos <= cutoff:
                overall[cutoff] += 1
                per_type[meta["qa_type"]][cutoff] += 1
        overall["n"] += 1
        per_type[meta["qa_type"]]["n"] += 1

    weak = []
    for qa_type in sorted(per_type):
        c = per_type[qa_type]
        r1, r3, r5 = (100 * c[k] / c["n"] for k in (1, 3, 5))
        print(f"{qa_type:<26}{c['n']:>6}{r1:>8.1f}%{r3:>8.1f}%{r5:>8.1f}%")
        if qa_type in ("section_text", "punishment", "offence_classification") \
                and r3 < 80:
            weak.append((qa_type, round(r3, 1)))
    n = overall["n"]
    print("-" * 59)
    print(f"{'ALL':<26}{n:>6}"
          f"{100 * overall[1] / n:>8.1f}%{100 * overall[3] / n:>8.1f}%"
          f"{100 * overall[5] / n:>8.1f}%")

    print()
    if weak:
        print("FLAG: straightforward lookup types are below 80% recall@3 -",
              weak)
        print("That points at the embedding or the chunking, not the model.")
        sys.exit(1)
    print("Straightforward lookup types (section_text, punishment,",
          "offence_classification)")
    print("are all at or above 80% recall@3 - retrieval is working.")


if __name__ == "__main__":
    main()

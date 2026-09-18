"""Does the bot's context assembly actually contain the answer?

Phase 2.5 found the confusion set barely improved with retrieval, because 33 of
its 44 gold answers cite two or more provisions while the run supplied a single
chunk - at k=1 only 10 of 44 questions had every provision they needed available,
and k=5 only reached 15.

This measures the fix: for each question, assemble the bot's context and check
whether the provisions the gold answer cites are actually present in it. Run
with --compare to see plain similarity retrieval alongside.

Usage:  python scripts/check_bot_context.py [--compare] [--show 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bot import Bot
from retrieve import PROCESSED, extract_refs


def load(name):
    with open(os.path.join(PROCESSED, name), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def coverage(bot, rows, gold_of, label, k=3, max_chunks=4, counterparts=True):
    saved = bot.counterparts
    if not counterparts:
        bot.counterparts = {}
    full = partial = none = 0
    per_row = []
    try:
        for row in rows:
            want = gold_of(row)
            chunks = bot.select_chunks(row["instruction"], k=k)
            context, used = bot.build_context(chunks, max_chunks=max_chunks)
            got = extract_refs(context)
            if want and want <= got:
                full += 1
                verdict = "all"
            elif want & got:
                partial += 1
                verdict = "some"
            else:
                none += 1
                verdict = "none"
            per_row.append((row, want, got, verdict, used))
    finally:
        bot.counterparts = saved
    n = len(rows)
    print(f"  {label:<44} all {full:>3}/{n}  some {partial:>3}  none {none:>3}"
          f"   ({100 * full / n:.1f}% complete)")
    return per_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    bot = Bot(model_dir=None)
    confusion = load("confusion_test_set.jsonl")
    gold_of = lambda e: extract_refs(e["output"])

    print(f"CONFUSION SET ({len(confusion)} questions) - are the provisions the "
          f"answer needs present in the context?\n")
    if args.compare:
        coverage(bot, confusion, gold_of, "similarity only, 1 chunk",
                 k=1, max_chunks=1, counterparts=False)
        coverage(bot, confusion, gold_of, "similarity only, 3 chunks",
                 k=3, max_chunks=3, counterparts=False)
        coverage(bot, confusion, gold_of, "similarity only, 5 chunks",
                 k=5, max_chunks=5, counterparts=False)
    rows = coverage(bot, confusion, gold_of,
                    "bot: citations + concordance counterparts", k=3, max_chunks=4)

    test = load("test.jsonl")
    print(f"\nTEST SET ({len(test)} questions)\n")
    if args.compare:
        coverage(bot, test, gold_of, "similarity only, 1 chunk",
                 k=1, max_chunks=1, counterparts=False)
    coverage(bot, test, gold_of, "bot: citations + concordance counterparts",
             k=3, max_chunks=4)

    if args.show:
        print("\n" + "=" * 78)
        print("Examples - what the bot now puts in front of the model")
        print("=" * 78)
        shown = 0
        for row, want, got, verdict, used in rows:
            if shown >= args.show:
                break
            shown += 1
            print(f"\n[{verdict}] {row['instruction'][:88]}")
            print(f"   needs   : {sorted(want)}")
            print(f"   context : {sorted(got)[:10]}")
            for u in used:
                print(f"     - {u['chunk_id']:<22} {u['source'][:52]}")


if __name__ == "__main__":
    main()

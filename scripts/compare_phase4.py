"""Phase 4, step 3: where do the two systems disagree?

`score_phase4.py` gives each system a score. This asks the more useful question:
on which questions is each one right, and do they fail in the same places?

They do not, and that is the project's most defensible finding. ChatGPT knows what
provisions are *about*, so it handles a section number reused for a different
offence. It does not have the concordance, so when a new provision absorbed
several old ones it names the headline correspondence and stops. This project's
bot has the opposite profile.

Both sides are scored on "did the answer name every provision the gold answer
names", by `metrics.py`, from predictions saved by their own runs - the bot's from
`phase3_6_results.json`, ChatGPT's from `phase4_chatgpt_results.json`. Nothing is
regenerated here.

Usage:
    python scripts/compare_phase4.py
    python scripts/compare_phase4.py --model chatgpt --condition "bot context"
    python scripts/compare_phase4.py --show 10
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import extract_refs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")

LABELS = ("both", "bot only", "llm only", "neither")


def load(name: str):
    with io.open(os.path.join(PROCESSED, name), encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chatgpt")
    ap.add_argument("--condition", default="bot context",
                    help="which of the bot's context conditions to compare")
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()

    with io.open(os.path.join(PROCESSED, "confusion_test_set.jsonl"),
                 encoding="utf-8") as fh:
        conf = [json.loads(l) for l in fh if l.strip()]

    bot = load("phase3_6_results.json")
    if "confusion_predictions" not in bot:
        raise SystemExit(
            "phase3_6_results.json has no saved predictions - re-run the training "
            "notebook, which now stores them (training is skipped, the model is "
            "already saved).")
    bot_preds = bot["confusion_predictions"][args.condition]
    llm = load(f"phase4_{args.model}_results.json")
    llm_preds = [llm["answers"][str(i + 1)] for i in range(len(conf))]

    tally = collections.Counter()
    by_kind: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    verdicts: dict[str, list[int]] = {k: [] for k in LABELS}
    skipped = 0

    for i, e in enumerate(conf):
        want = extract_refs(e["output"])
        if not want:
            # transition questions cite no provision, so "named them all" is
            # not a question that can be asked of them.
            skipped += 1
            continue
        b = want <= extract_refs(bot_preds[i])
        g = want <= extract_refs(llm_preds[i])
        key = ("both" if b and g else "bot only" if b
               else "llm only" if g else "neither")
        tally[key] += 1
        by_kind[e["mapping_type"]][key] += 1
        verdicts[key].append(i)

    n = sum(tally.values())
    print("=" * 78)
    print(f"PHASE 4 - agreement between {args.model} and this bot "
          f"({args.condition})")
    print("=" * 78)
    print(f"{n} of the {len(conf)} confusion questions cite a provision "
          f"({skipped} excluded: nothing to name)\n")
    for k in LABELS:
        bar = "#" * round(40 * tally[k] / n)
        print(f"  {k:<13}{tally[k]:>3}  {100 * tally[k] / n:>5.1f}%  {bar}")
    print(f"\n  at least one correct: {n - tally['neither']}/{n} "
          f"({100 * (n - tally['neither']) / n:.1f}%)")
    print(f"  the two disagree on:  {tally['bot only'] + tally['llm only']}/{n} "
          f"({100 * (tally['bot only'] + tally['llm only']) / n:.1f}%)")

    print(f"\n{'kind':<13}{'n':>4}{'both':>7}{'bot only':>10}{'llm only':>10}"
          f"{'neither':>9}")
    print("-" * 53)
    for kind in sorted(by_kind, key=lambda k: -sum(by_kind[k].values())):
        c = by_kind[kind]
        print(f"{kind:<13}{sum(c.values()):>4}{c['both']:>7}{c['bot only']:>10}"
              f"{c['llm only']:>10}{c['neither']:>9}")

    for key, heading in (("bot only", f"Only the bot got these"),
                         ("llm only", f"Only {args.model} got these")):
        if not verdicts[key] or not args.show:
            continue
        print("\n" + "=" * 78)
        print(heading)
        print("=" * 78)
        for i in verdicts[key][:args.show]:
            e = conf[i]
            want = extract_refs(e["output"])
            print(f"\n[{e['mapping_type']}] {e['instruction']}")
            print(f"  needs   : {sorted(want)}")
            for who, pred in ((args.model, llm_preds[i]), ("bot", bot_preds[i])):
                miss = sorted(want - extract_refs(pred))
                print(f"  {who:<8}: {textwrap.shorten(pred, 150, placeholder=' ...')}")
                if miss:
                    print(f"  {'':8}  missed {miss}")

    dest = os.path.join(PROCESSED, f"phase4_{args.model}_agreement.json")
    with io.open(dest, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"model": args.model, "bot_condition": args.condition,
                   "n_compared": n, "excluded_no_citations": skipped,
                   "tally": dict(tally),
                   "by_kind": {k: dict(v) for k, v in by_kind.items()},
                   "questions": {k: v for k, v in verdicts.items()}},
                  fh, indent=2)
    print(f"\nwritten to {os.path.relpath(dest, ROOT)}")


if __name__ == "__main__":
    main()

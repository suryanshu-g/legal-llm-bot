"""Phase 4, step 2: score a general-purpose LLM's replies against this bot.

Reads whatever the other chatbot said, scores it with `metrics.py` - the same code
the training notebooks score with - and prints the three-column comparison the
project was built to make.

The bot's own columns are not recomputed here. They are read from
`data/processed/phase3_6_results.json`, which the Colab run produced on the same
44 questions, so nothing is re-run and nothing can quietly diverge.

Parsing is deliberately strict about alignment and forgiving about everything
else. A reply is split on lines that open with a question number; if any question
is missing or duplicated the script says which and stops, because silently
misaligning answer 12 with question 13 would corrupt every number downstream.

Usage:
    python scripts/build_phase4_prompts.py            # first, writes the prompts
    # paste each prompt file into the chatbot, save replies as
    #   data/phase4/chatgpt_batch1.txt, chatgpt_batch2.txt, ...
    python scripts/score_phase4.py --model chatgpt
    python scripts/score_phase4.py --model chatgpt --show 8
"""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import extract_refs, pct, score, summarise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
PHASE4 = os.path.join(ROOT, "data", "phase4")

# "1. answer", "1) answer", "**1.** answer", "1 - answer"
NUMBERED = re.compile(r"^\s*(?:\*\*)?(\d{1,3})(?:\*\*)?\s*[.)\-:]\s*(.*)$")


def parse_reply(text: str) -> dict[int, str]:
    """Question number -> answer, from a numbered reply."""
    answers: dict[int, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = NUMBERED.match(line)
        if m:
            current = int(m.group(1))
            answers.setdefault(current, []).append(m.group(2).strip())
        elif current is not None and line.strip():
            # A wrapped continuation of the answer above.
            answers[current].append(line.strip())
    return {n: " ".join(p for p in parts if p).strip()
            for n, parts in answers.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="name used in the reply filenames, e.g. chatgpt")
    ap.add_argument("--show", type=int, default=6,
                    help="how many side-by-side examples to print")
    args = ap.parse_args()

    with io.open(os.path.join(PROCESSED, "confusion_test_set.jsonl"),
                 encoding="utf-8") as fh:
        conf = [json.loads(l) for l in fh if l.strip()]

    files = sorted(glob.glob(os.path.join(PHASE4, f"{args.model}_batch*.txt")))
    if not files:
        raise SystemExit(
            f"No replies found at data/phase4/{args.model}_batch*.txt\n"
            f"Save the chatbot's reply to each prompt file there, then re-run.")

    answers: dict[int, str] = {}
    duplicates = []
    for path in files:
        with io.open(path, encoding="utf-8") as fh:
            got = parse_reply(fh.read())
        for n, a in got.items():
            if n in answers:
                duplicates.append(n)
            answers[n] = a
        print(f"read {os.path.basename(path)}: {len(got)} answers")

    missing = [i + 1 for i in range(len(conf)) if i + 1 not in answers]
    if missing or duplicates:
        if missing:
            print(f"\nMissing answers for question(s): {missing}")
        if duplicates:
            print(f"\nAnswered more than once: {sorted(set(duplicates))}")
        raise SystemExit(
            "Refusing to score a partial reply - the numbers would be wrong.\n"
            "Ask the chatbot for the missing numbers and append them to the "
            "batch file, one per line, as '<number>. <answer>'.")

    preds = [answers[i + 1] for i in range(len(conf))]
    metas = [{"qa_type": e["mapping_type"]} for e in conf]
    rows = score(conf, preds, metas)
    llm = summarise(rows, args.model)

    # The bot's own numbers, as measured on Colab over the same 44 questions.
    bot_path = os.path.join(PROCESSED, "phase3_6_results.json")
    bot = {}
    if os.path.exists(bot_path):
        with io.open(bot_path, encoding="utf-8") as fh:
            bot = json.load(fh).get("confusion", {})
    else:
        print(f"\n(no {os.path.basename(bot_path)} - showing the LLM column only)")

    print("\n" + "=" * 78)
    print(f"PHASE 4 - the confusion set: {args.model} against this project's bot")
    print("=" * 78)
    print(f"{'system':<34}{'n':>4}{'F1':>9}{'citeF1':>9}{'allCites':>10}{'exact':>9}")
    print("-" * 75)

    def row(label, s):
        print(f"{label:<34}{s['n']:>4}{pct(s['f1'])}{pct(s['citation_f1'])}"
              f"{pct(s['all_citations_present'])}{pct(s['citation_exact'])}")

    row(f"{args.model} (no retrieval)", llm)
    for key, label in (("none", "this bot, retrieval disabled"),
                       ("single passage", "this bot, one passage"),
                       ("bot context", "this bot, full context")):
        if key in bot:
            row(label, bot[key])

    n = len(conf)
    print(f"\nquestions answered with every required section named:")
    print(f"  {args.model}: {round(llm['all_citations_present'] * n / 100)} / {n}")
    if "bot context" in bot:
        print(f"  this bot: "
              f"{round(bot['bot context']['all_citations_present'] * n / 100)} / {n}")

    print(f"\nby question kind ({args.model}, allCites):")
    by_kind = {}
    for kind in sorted({m["qa_type"] for m in metas}):
        sub = [r for r in rows if r["qa_type"] == kind]
        s = summarise(sub, kind)
        by_kind[kind] = s
        print(f"  {kind:<14}{s['n']:>4}{pct(s['all_citations_present'])}")

    if args.show:
        print("\n" + "=" * 78)
        print(f"Examples - what {args.model} said")
        print("=" * 78)
        shown = 0
        for kind in ("collision", "merged", "split", "removed", "transition"):
            for i, e in enumerate(conf):
                if e["mapping_type"] != kind or shown >= args.show:
                    continue
                shown += 1
                want, got = extract_refs(e["output"]), extract_refs(preds[i])
                print(f"\n[{kind}] {e['instruction']}")
                print("  gold :", textwrap.shorten(e["output"], 180, placeholder=" ..."))
                print(f"  {args.model:<5}: [{'OK ' if want <= got else 'MISS'}] "
                      + textwrap.shorten(preds[i], 180, placeholder=" ..."))
                if want - got:
                    print(f"         missed: {sorted(want - got)}")
                break

    dest = os.path.join(PROCESSED, f"phase4_{args.model}_results.json")
    with io.open(dest, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"model": args.model, "n": n,
                   "source_files": [os.path.basename(p) for p in files],
                   "overall": llm,
                   "by_kind": by_kind,
                   "bot_for_comparison": bot,
                   "answers": {str(i + 1): preds[i] for i in range(n)}},
                  fh, indent=2, ensure_ascii=False)
    print(f"\nwritten to {os.path.relpath(dest, ROOT)}")


if __name__ == "__main__":
    main()

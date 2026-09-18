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


def parse_numbered(text: str) -> dict[int, str]:
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


def parse_positional(text: str, first: int, count: int) -> dict[int, str]:
    """One answer per line, unnumbered, in the order the questions were asked.

    Models often drop the numbering however plainly it was asked for. Taking
    lines positionally is only safe if there are exactly as many of them as
    there were questions in the batch - one missing line would shift every
    answer after it onto the wrong question - so this returns nothing at all
    unless the count matches, and the caller then reports the mismatch.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # A leading "Here are the answers:" style preamble is discarded; a line is
    # only an answer if it is long enough to be one.
    lines = [l for l in lines if len(l) > 25]
    if len(lines) != count:
        return {}
    return {first + i: line for i, line in enumerate(lines)}


def parse_reply(text: str, first: int, count: int) -> tuple[dict[int, str], str]:
    numbered = parse_numbered(text)
    if numbered:
        return numbered, "numbered"
    return parse_positional(text, first, count), "positional"


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

    # Either naming works: "<model>_batch1.txt" as documented, or the
    # "prompt_batch1_result.txt" that falls out of saving each reply beside the
    # prompt it answers.
    files = sorted(glob.glob(os.path.join(PHASE4, f"{args.model}_batch*.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(PHASE4, "prompt_batch*_result.txt")))
    if not files:
        raise SystemExit(
            f"No replies found in data/phase4/.\n"
            f"Save each reply as {args.model}_batch1.txt, {args.model}_batch2.txt "
            f"(or prompt_batch1_result.txt), then re-run.")

    # The batch number in the filename says which questions the file answers,
    # using the same division build_phase4_prompts.py made.
    n_batches = len(files)
    size = -(-len(conf) // n_batches)

    answers: dict[int, str] = {}
    duplicates = []
    for path in files:
        name = os.path.basename(path)
        bm = re.search(r"batch(\d+)", name)
        if not bm:
            raise SystemExit(f"Cannot tell which questions {name} answers - "
                             f"the filename needs 'batch<N>' in it.")
        b = int(bm.group(1)) - 1
        first, count = b * size + 1, len(conf[b * size:(b + 1) * size])
        with io.open(path, encoding="utf-8") as fh:
            got, how = parse_reply(fh.read(), first, count)
        if not got:
            raise SystemExit(
                f"{name}: could not line the reply up with questions "
                f"{first}-{first + count - 1}.\n"
                f"The reply has no numbering, and the number of answer lines "
                f"does not match the {count} questions asked, so matching them "
                f"positionally would put answers against the wrong questions.\n"
                f"Either ask the chatbot to re-answer with '<number>. <answer>' "
                f"numbering, or fix the file so it has exactly {count} lines, "
                f"one answer each.")
        for n, a in got.items():
            if n in answers:
                duplicates.append(n)
            answers[n] = a
        print(f"read {name}: {len(got)} answers ({how}, questions "
              f"{min(got)}-{max(got)})")

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

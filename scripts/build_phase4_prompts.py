"""Phase 4, step 1: write the confusion questions out for a general-purpose LLM.

The project's claim is that a model answering Indian criminal law from parametric
memory will be unreliable about section numbers after the 1 July 2024
recodification, and that retrieval fixes it. The 44 confusion questions have been
held out of every split since Phase 1 for exactly this comparison.

This script writes them into text files to paste into ChatGPT, Claude or Gemini.
Two batches by default, because a single reply covering 44 questions tends to get
shorter and sloppier towards the end, which would penalise the other model for a
length limit rather than for its knowledge.

Fairness matters here, or the comparison proves nothing:

* the other model gets the same question text the bot gets, unedited;
* it is told to name the sections, because the citation metrics score sections -
  not asking would measure instruction-following rather than knowledge;
* it is **not** told that the codes changed in 2024, nor which answers are
  negative. A hint like that would do the work the comparison is meant to test;
* the answer format is prescribed only so the replies can be parsed back.

Outputs:
    data/phase4/prompt_batch1.txt, prompt_batch2.txt   - paste these
    data/phase4/questions.json                          - order, for scoring

Usage:
    python scripts/build_phase4_prompts.py
    python scripts/build_phase4_prompts.py --batches 4      # shorter replies
"""

from __future__ import annotations

import argparse
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
OUT_DIR = os.path.join(ROOT, "data", "phase4")

INSTRUCTIONS = """\
You are answering questions about Indian criminal law. Answer each question below
from your own knowledge.

For each answer:
- name the specific section numbers and the Act they belong to;
- if a provision has no counterpart, or no single counterpart, say so plainly;
- keep each answer to one or two sentences.

Format your reply as one answer per numbered line, matching the numbering below,
and write nothing else - no preamble, no headings, no blank lines between answers:

1. <your answer>
2. <your answer>

Questions:
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=2)
    args = ap.parse_args()

    path = os.path.join(PROCESSED, "confusion_test_set.jsonl")
    with io.open(path, encoding="utf-8") as fh:
        conf = [json.loads(l) for l in fh if l.strip()]
    os.makedirs(OUT_DIR, exist_ok=True)

    # Questions keep their dataset order, and the scorer reads the same file, so
    # answer n always lines up with question n.
    questions = [{"n": i + 1, "instruction": e["instruction"],
                  "mapping_type": e["mapping_type"]} for i, e in enumerate(conf)]
    with io.open(os.path.join(OUT_DIR, "questions.json"), "w",
                 encoding="utf-8", newline="\n") as fh:
        json.dump(questions, fh, indent=2, ensure_ascii=False)

    size = -(-len(conf) // args.batches)          # ceiling division
    written = []
    for b in range(args.batches):
        chunk = questions[b * size:(b + 1) * size]
        if not chunk:
            continue
        name = f"prompt_batch{b + 1}.txt"
        with io.open(os.path.join(OUT_DIR, name), "w",
                     encoding="utf-8", newline="\n") as fh:
            fh.write(INSTRUCTIONS)
            for q in chunk:
                fh.write(f"{q['n']}. {q['instruction']}\n")
        written.append((name, chunk[0]["n"], chunk[-1]["n"]))

    print(f"{len(conf)} confusion questions -> {len(written)} batch file(s) in "
          f"data/phase4/")
    for name, first, last in written:
        print(f"  {name}: questions {first}-{last}")
    print("\nNext: paste each file into the chatbot, save its reply as "
          "data/phase4/<model>_batch<N>.txt,")
    print("then run  python scripts/score_phase4.py --model <model>")


if __name__ == "__main__":
    main()

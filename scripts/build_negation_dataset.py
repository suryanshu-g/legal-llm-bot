"""Phase 3.6: teach the model to say no.

Phase 3.5 fixed the context format and the test set improved sharply - citation
accuracy 86.8% -> 92.7%, and the old-to-new correspondence 93% -> 99%. The
confusion set did not move at all: 23.3% in every condition, with
`citation_exact` at zero. Counting the answer shapes showed why.

    answers opening with a negation   training set     41 / 11,751   (0.3%)
                                      confusion set    43 / 44      (97.7%)

The model has seen "X corresponds to Y" eleven thousand times and almost never a
question whose correct answer is that the correspondence does not hold, so that
is what it produces - inventing a "BNS Section 105L" for sedition rather than
answering that sedition was not carried forward. Worse, 33 of the 44 confusion
questions are `collision`, `merged` or `split` shapes, and the training data
contains **no question of those shapes at all**. The confusion set was not a
harder version of a trained task; it was an untrained one.

This script adds the missing shapes, built from the same concordance, for
provisions that are **not** in any held-out group:

  collision   same section number in both codes, different subject - the sharpest
              trap in the whole recodification, and 926 of them are available.
  merged      an old section folded into a new one alongside others, asked in the
              form that invites a wrong yes ("is it still dealt with under ...?").
  no_single   the reverse direction: which single old section corresponds to a new
              one that absorbed several? There is no single one.

Deliberately **not** generated:

* `split` questions. There are only five split families in the entire
  concordance and all five are inside held-out groups, so any split example
  would leak the confusion set. The `no_single` type teaches the same
  "there is no single one" answer shape from many-to-one families instead, which
  is factually a different relation and is described as such. Five of the 44
  confusion questions therefore remain shape-novel, and that is reported rather
  than engineered around.
* `removed` questions. Training already contains 112, they score 100% on the
  test set, and asserting a repeal needs the vetted allowlist rather than the
  absence of a row in a correspondence table.

Leakage rules, checked and reported at the end of the run:

* a provision whose group is in `test_index.jsonl` or in the confusion set's
  `group_keys` is dropped outright;
* a provision whose group is in `val_index.jsonl` goes to validation, never to
  training, so the existing split is respected rather than re-cut;
* remaining new groups are split 90/10 train/validation, at group level.

`test.jsonl` and `confusion_test_set.jsonl` are not read for content and not
written. Nothing in this script can alter them.

Outputs:
    data/processed/negation_train.jsonl  + negation_train_index.jsonl
    data/processed/negation_val.jsonl    + negation_val_index.jsonl
    data/processed/negation_report.json

Usage:  python scripts/build_negation_dataset.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_splits import (ACT_FULL, NEW_OF, SHORT, Union, entity, load_titles)

PROCESSED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "processed")
SEED = 20240701
VAL_SHARE = 0.10

# How many of each shape to emit. The pools are larger than this; the aim is to
# teach a form, not to swamp the 11,751 rows that teach the facts.
QUOTA = {"collision": 700, "merged": 300, "no_single": 250}
# collision emits one question per provision; merged and no_single emit every
# phrasing, because their pools are far smaller than the quota.

# A collision is only worth asserting when the two provisions really are about
# different things. Titles sharing most of their words mean the section number
# was reused for the same subject, and claiming "a different provision entirely"
# would then be wrong.
TITLE_OVERLAP_LIMIT = 0.5

STOP = {"of", "the", "a", "an", "to", "for", "in", "on", "or", "and", "by", "with",
        "from", "as", "at", "any", "be", "is", "not", "punishment", "etc"}


def words(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", title.lower()) if w not in STOP}


def overlap(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def load_jsonl(name: str) -> list[dict]:
    with io.open(os.path.join(PROCESSED, name), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def describe(act: str, sec: str, titles) -> str:
    t = titles.get((act, sec), "")
    return f"{SHORT[act]} Section {sec}" + (f" ({t})" if t else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be generated, write nothing")
    args = ap.parse_args()

    rng = random.Random(SEED)
    titles = load_titles()
    with io.open(os.path.join(PROCESSED, "mapping_table.csv"), encoding="utf-8") as fh:
        mapping = list(csv.DictReader(fh))

    # Same union-find as build_splits.py, so a group id here means the same fact
    # it means there.
    u = Union()
    for r in mapping:
        if r["old_section"] and r["new_section"]:
            u.union(entity(r["old_act"], r["old_section"]),
                    entity(r["new_act"], r["new_section"]))

    test_groups = {m["group"] for m in load_jsonl("test_index.jsonl")}
    val_groups = {m["group"] for m in load_jsonl("val_index.jsonl")}
    confusion = load_jsonl("confusion_test_set.jsonl")
    conf_groups = {u.find(k) for e in confusion for k in e["group_keys"]}
    held_out = test_groups | conf_groups
    print(f"held-out groups: test {len(test_groups)}, confusion {len(conf_groups)} "
          f"-> {len(held_out)} blocked; val {len(val_groups)} routed to validation")

    fwd: dict[tuple[str, str], list[tuple[str, str]]] = collections.defaultdict(list)
    back: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for r in mapping:
        if not (r["old_section"] and r["new_section"]):
            continue
        key, pair = (r["old_act"], r["old_section"]), (r["new_act"], r["new_section"])
        if pair not in fwd[key]:
            fwd[key].append(pair)
        if r["old_section"] not in back[pair]:
            back[pair].append(r["old_section"])

    def sort_secs(secs):
        return sorted(secs, key=lambda x: (int(re.match(r"\d+", x).group()), x))

    rows: list[dict] = []
    skipped = collections.Counter()

    def emit(kind, question, answer, anchor_act, anchor_sec, group):
        rows.append({
            "instruction": question, "input": "", "output": answer,
            "qa_type": kind,
            "source_chunk_id": entity(anchor_act, anchor_sec),
            "group": group,
        })

    # ---------------------------------------------------------------- collision
    candidates = []
    for old_act in NEW_OF:
        new_act = NEW_OF[old_act]
        for (act, sec), old_title in titles.items():
            if act != old_act or not old_title:
                continue
            new_title = titles.get((new_act, sec))
            if not new_title:
                continue
            if (new_act, sec) in fwd.get((old_act, sec), []):
                skipped["collision: same number IS the counterpart"] += 1
                continue
            if overlap(old_title, new_title) > TITLE_OVERLAP_LIMIT:
                skipped["collision: titles too similar to call it different"] += 1
                continue
            g = u.find(entity(old_act, sec))
            if g in held_out:
                skipped["collision: held-out group"] += 1
                continue
            candidates.append((old_act, new_act, sec, old_title, new_title, g))
    rng.shuffle(candidates)

    for old_act, new_act, sec, old_title, new_title, g in candidates[:QUOTA["collision"]]:
        news = fwd.get((old_act, sec), [])
        refs = ", ".join(describe(a, s, titles) for a, s in news)
        tail = (f" The provision that now corresponds to {SHORT[old_act]} Section {sec} "
                f"is {refs}." if news else
                f" {SHORT[old_act]} Section {sec} has no counterpart in the {new_act}.")
        variant = rng.randrange(3)
        if variant == 0:
            q = (f"Does {SHORT[new_act]} Section {sec} deal with the same subject as "
                 f"{SHORT[old_act]} Section {sec}?")
        elif variant == 1:
            q = (f"Is {SHORT[old_act]} Section {sec} the same provision as "
                 f"{SHORT[new_act]} Section {sec}?")
        else:
            # Only the penal code creates offences; the procedure and evidence
            # codes do not, so "the same offence" would be wrong for them.
            noun = "offence" if old_act == "IPC" else "subject"
            q = (f"{SHORT[old_act]} Section {sec} and {SHORT[new_act]} Section {sec} - "
                 f"do they cover the same {noun}?")
        a = (f"No. {SHORT[old_act]} Section {sec} was {old_title}, whereas "
             f"{SHORT[new_act]} Section {sec} is a different provision entirely - "
             f"{new_title}.{tail}")
        emit("collision", q, a, old_act, sec, g)

    # ------------------------------------------------------------------- merged
    merged_pool = []
    for (old_act, old_sec), news in fwd.items():
        if old_act not in NEW_OF or len(news) != 1:
            continue
        siblings = back[news[0]]
        if len(siblings) < 2:
            continue
        g = u.find(entity(old_act, old_sec))
        if g in held_out:
            skipped["merged: held-out group"] += 1
            continue
        if not titles.get((old_act, old_sec)):
            continue
        merged_pool.append((old_act, old_sec, news[0], siblings, g))
    rng.shuffle(merged_pool)

    for old_act, old_sec, new, siblings, g in merged_pool[:QUOTA["merged"]]:
        old_title = titles[(old_act, old_sec)]
        subject = re.sub(r"^Punishment (?:for|of) ", "", old_title)
        subject = subject[0].lower() + subject[1:] if subject else old_title
        a = (f"No. The {ACT_FULL[old_act]} was repealed with effect from 1 July 2024. "
             f"{SHORT[old_act]} Section {old_sec} ({old_title}) now corresponds to "
             f"{describe(new[0], new[1], titles)}, which consolidates "
             f"{SHORT[old_act]} Sections {', '.join(sort_secs(siblings)[:8])} into a "
             f"single provision.")
        # Both phrasings, because this pool is small. "Charged for" is only
        # meaningful where the provision creates an offence - asking whether
        # IPC 55 (commutation of a life sentence) can be "charged" is nonsense -
        # so it is reserved for sections titled as a punishment.
        # Some headings are clauses rather than noun phrases ("when they may be
        # asked", "same offence when committed by ..."), and reading them as the
        # subject of a sentence produces nonsense. Those fall back to a phrasing
        # that needs no subject at all.
        if re.match(r"^(when|same|such|that|which|who|it) ", subject):
            questions = [f"Is the matter dealt with by {SHORT[old_act]} Section "
                         f"{old_sec} still governed by that section?"]
        else:
            questions = [f"Is {subject} still dealt with under "
                         f"{SHORT[old_act]} Section {old_sec}?"]
            if re.match(r"^Punishment (?:for|of) ", old_title):
                questions.append(f"Can a person still be charged under "
                                 f"{SHORT[old_act]} Section {old_sec} for {subject}?")
        for q in questions:
            emit("merged", q, a, old_act, old_sec, g)

    # ---------------------------------------------------------------- no_single
    # The reverse direction of a merge: no single old section accounts for the new
    # one. Same answer shape as a split, an honestly different relation.
    fanin = []
    for pair, olds in back.items():
        if len(olds) < 2:
            continue
        g = u.find(entity(pair[0], pair[1]))
        if g in held_out:
            skipped["no_single: held-out group"] += 1
            continue
        if not titles.get(pair):
            continue
        fanin.append((pair, olds, g))
    rng.shuffle(fanin)

    for pair, olds, g in fanin[:QUOTA["no_single"]]:
        new_act, new_sec = pair
        old_act = next((a for a, n in NEW_OF.items() if n == new_act), None)
        if old_act is None:
            continue
        listed = ", ".join(describe(old_act, s, titles) for s in sort_secs(olds)[:8])
        a = (f"There is no single one. {describe(new_act, new_sec, titles)} "
             f"consolidates {listed}. Citing only one of them states the position "
             f"incompletely.")
        # Only 47 many-to-one families survive the held-out exclusions, so both
        # phrasings of each are emitted.
        for q in (f"Which single {SHORT[old_act]} section corresponds to "
                  f"{SHORT[new_act]} Section {new_sec}?",
                  f"Is there one {SHORT[old_act]} section that {SHORT[new_act]} "
                  f"Section {new_sec} replaced?"):
            emit("no_single", q, a, new_act, new_sec, g)

    # ------------------------------------------------------- split into two files
    new_groups = sorted({r["group"] for r in rows} - val_groups)
    rng.shuffle(new_groups)
    to_val = set(new_groups[:int(len(new_groups) * VAL_SHARE)]) | val_groups

    out = {"train": [], "val": []}
    for r in rows:
        out["val" if r["group"] in to_val else "train"].append(r)

    # ------------------------------------------------------------- verification
    problems = []
    for split, rs in out.items():
        for r in rs:
            if r["group"] in held_out:
                problems.append(f"{split}: group {r['group']} is held out")
    train_g = {r["group"] for r in out["train"]}
    val_g = {r["group"] for r in out["val"]}
    if train_g & val_g:
        problems.append(f"{len(train_g & val_g)} groups in both train and val")
    conf_questions = {e["instruction"] for e in confusion}
    for split, rs in out.items():
        clash = conf_questions & {r["instruction"] for r in rs}
        if clash:
            problems.append(f"{split}: {len(clash)} questions identical to a "
                            f"confusion question, e.g. {sorted(clash)[0]!r}")
    if problems:
        raise SystemExit("LEAKAGE CHECK FAILED:\n  " + "\n  ".join(problems))

    print(f"\ngenerated {len(rows)} rows: "
          f"{dict(collections.Counter(r['qa_type'] for r in rows))}")
    print(f"  train {len(out['train'])} rows / {len(train_g)} groups")
    print(f"  val   {len(out['val'])} rows / {len(val_g)} groups")
    print("  leakage checks: passed (no held-out group, no shared group, "
          "no shared question)")
    print("\nskipped:")
    for k, v in skipped.most_common():
        print(f"  {v:>5}  {k}")

    print("\nexamples:")
    for kind in QUOTA:
        ex = next((r for r in out["train"] if r["qa_type"] == kind), None)
        if ex:
            print(f"\n[{kind}] {ex['instruction']}")
            print(f"  -> {ex['output'][:190]}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    report = {"seed": SEED, "quota": QUOTA, "val_share": VAL_SHARE,
              "title_overlap_limit": TITLE_OVERLAP_LIMIT,
              "rows": {k: len(v) for k, v in out.items()},
              "by_qa_type": dict(collections.Counter(r["qa_type"] for r in rows)),
              "groups": {"train": len(train_g), "val": len(val_g),
                         "blocked_held_out": len(held_out)},
              "skipped": dict(skipped),
              "split_questions_not_generated":
                  "all 5 split families in the concordance are inside held-out "
                  "groups; no_single teaches the same answer shape from "
                  "many-to-one families instead"}
    for split, rs in out.items():
        with io.open(os.path.join(PROCESSED, f"negation_{split}.jsonl"), "w",
                     encoding="utf-8", newline="\n") as fh:
            for r in rs:
                fh.write(json.dumps({k: r[k] for k in
                                     ("instruction", "input", "output")},
                                    ensure_ascii=False) + "\n")
        with io.open(os.path.join(PROCESSED, f"negation_{split}_index.jsonl"), "w",
                     encoding="utf-8", newline="\n") as fh:
            for r in rs:
                fh.write(json.dumps({"qa_type": r["qa_type"],
                                     "source_chunk_id": r["source_chunk_id"],
                                     "group": r["group"]},
                                    ensure_ascii=False) + "\n")
        print(f"wrote negation_{split}.jsonl and negation_{split}_index.jsonl")
    with io.open(os.path.join(PROCESSED, "negation_report.json"), "w",
                 encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
    print("wrote negation_report.json")


if __name__ == "__main__":
    main()

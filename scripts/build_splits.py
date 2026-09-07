"""Phase 1.5: leakage-safe train/validation/test splits, plus a confusion set.

Why not a random row-level split
--------------------------------
Every fact in the dataset is asked several ways. "What does BNS 103 say?" and
"Explain BNS Section 103." are the same fact in different words, so a random
split would put one in train and the other in validation, and the validation
score would partly measure memorised phrasing rather than generalisation.

So groups are split, not rows. A group is the *fact*, and two things define it:

1. Every pair carries a `source_chunk_id` naming the section, judgment, date or
   policy it is about. All pairs sharing one stay together.
2. Groups are then merged across the concordance. "Which BNS section replaced
   IPC 302?" and "Which IPC section corresponds to BNS 103?" are the same fact
   asked from both ends, so `ipc_302` and `bns_103` must not be separated
   either. A union-find over mapping_table.csv merges them - which also pulls
   whole split/merged families together, so all of IPC 415, 417, 418, 419, 420
   and BNS 318 land on one side.

The confusion set
-----------------
A separate held-out set, excluded from all three splits, for Phase 4: these
exact questions get put to both this bot and a general-purpose LLM to test the
project's central claim. Entries are drawn from the provisions that actually
generate litigation (the case-law topics), because that is where a confident
wrong answer costs something, and they concentrate on the four ways the
renumbering traps a model working from memory: collisions, merges, splits and
outright repeals.

Outputs
-------
  data/processed/train.jsonl / val.jsonl / test.jsonl
  data/processed/confusion_test_set.jsonl
  data/processed/split_report.json
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawlib import PROCESSED, RAW, load_old_act, read_jsonl, write_jsonl

RATIO = {"train": 0.85, "val": 0.10, "test": 0.05}
SEED = 20240701

ACT_FULL = {
    "BNS": "Bharatiya Nyaya Sanhita, 2023", "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "BSA": "Bharatiya Sakshya Adhiniyam, 2023", "IPC": "Indian Penal Code, 1860",
    "CRPC": "Code of Criminal Procedure, 1973", "IEA": "Indian Evidence Act, 1872",
}
SHORT = {"BNS": "BNS", "BNSS": "BNSS", "BSA": "BSA",
         "IPC": "IPC", "CRPC": "CrPC", "IEA": "Indian Evidence Act"}
NEW_OF = {"IPC": "BNS", "CRPC": "BNSS", "IEA": "BSA"}

# `removed` entries are restricted to a vetted allowlist rather than taken
# from all 76 rows the concordance so classifies. Two kinds of row are
# unsafe to assert in a benchmark: provisions already repealed long before
# 2024, and provisions whose substance survives in the new code even though
# the BPRD table lists no counterpart - the Indian Evidence Act s.27
# (discovery from an accused in custody) is carried forward by the proviso
# to BSA s.23(2), but BPRD maps only ss.25 and 26 across. Every entry below
# is a provision with no successor in the new code.
VETTED_REMOVED = [
    ("IPC", "124A"), ("IPC", "377"), ("IPC", "497"), ("IPC", "309"),
    ("IPC", "153AA"), ("CRPC", "8"), ("CRPC", "16"), ("CRPC", "17"),
    ("CRPC", "18"), ("CRPC", "19"), ("CRPC", "144A"),
]



# ------------------------------------------------------------------ grouping

class Union:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def entity(act: str, section: str) -> str:
    return f"{act.lower()}_{section.lower()}"


def build_groups(index: list[dict], mapping: list[dict]) -> dict[str, str]:
    """source_chunk_id -> group id, merged across the concordance."""
    u = Union()
    for m in index:
        u.find(m["source_chunk_id"])
    for r in mapping:
        if r["old_section"] and r["new_section"]:
            u.union(entity(r["old_act"], r["old_section"]),
                    entity(r["new_act"], r["new_section"]))
    return {m["source_chunk_id"]: u.find(m["source_chunk_id"]) for m in index}


# ----------------------------------------------------------- confusion set

def load_titles() -> dict[tuple[str, str], str]:
    titles: dict[tuple[str, str], str] = {}
    for act, folder in [("BNS", "bns"), ("BNSS", "bnss"), ("BSA", "bsa")]:
        blob = json.load(open(os.path.join(RAW, folder, "sections.json"), encoding="utf-8"))
        alt = os.path.join(RAW, folder, "devgan_sections.json")
        fb = {}
        if os.path.exists(alt):
            fb = {s["section"]: s["title"]
                  for s in json.load(open(alt, encoding="utf-8"))["sections"]}
        for s in blob["sections"]:
            titles[(act, str(s["section"]).upper())] = s["title"] or fb.get(s["section"], "")
    for act in ("IPC", "CRPC", "IEA"):
        for s in load_old_act(act):
            titles[(act, str(s["section"]).upper())] = s["title"]
    return titles


def salient_sections() -> list[tuple[str, str]]:
    """The provisions the case-law topics identified as actually litigated."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scrape_case_law import TOPICS
    out, seen = [], set()
    for _key, (_label, old_act, secs, _q) in TOPICS.items():
        for s in secs:
            if (old_act, s) not in seen:
                seen.add((old_act, s))
                out.append((old_act, s))
    return out


def build_confusion_set(mapping: list[dict], titles) -> list[dict]:
    fwd: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    kinds: dict[tuple[str, str], set] = defaultdict(set)
    for r in mapping:
        key = (r["old_act"], r["old_section"])
        kinds[key].add(r["mapping_type"])
        if r["old_section"] and r["new_section"]:
            pair = (r["new_act"], r["new_section"])
            if pair not in fwd[key]:
                fwd[key].append(pair)

    entries: list[dict] = []

    def add(kind, old_act, old_sec, news, question, answer, why):
        entries.append({
            "instruction": question, "input": "", "output": answer,
            "mapping_type": kind,
            "old_reference": f"{SHORT[old_act]} {old_sec}",
            "new_references": [f"{SHORT[a]} {s}" for a, s in news],
            "why_confusing": why,
            "group_keys": sorted({entity(old_act, old_sec)}
                                 | {entity(a, s) for a, s in news}),
            "sources": ["data/processed/mapping_table.csv (BPRD correspondence tables)"],
        })

    for old_act, old_sec in salient_sections():
        new_act = NEW_OF[old_act]
        news = fwd.get((old_act, old_sec), [])
        kind = kinds.get((old_act, old_sec), set())
        ot = titles.get((old_act, old_sec), "")
        if not ot:
            continue
        refs = ", ".join(f"{SHORT[a]} Section {s}"
                         + (f" ({titles.get((a, s), '')})" if titles.get((a, s)) else "")
                         for a, s in news)

        # 1. Same number, different offence - the sharpest trap.
        collision = titles.get((new_act, old_sec))
        if collision and (new_act, old_sec) not in news:
            add("collision", old_act, old_sec, news,
                f"Does {SHORT[new_act]} Section {old_sec} deal with the same subject as "
                f"{SHORT[old_act]} Section {old_sec}?",
                f"No. {SHORT[old_act]} Section {old_sec} was {ot}. {SHORT[new_act]} "
                f"Section {old_sec} is a different provision entirely - {collision}. "
                f"The provision that now corresponds to {SHORT[old_act]} "
                f"Section {old_sec} is {refs}."
                if news else
                f"No. {SHORT[old_act]} Section {old_sec} was {ot}, whereas "
                f"{SHORT[new_act]} Section {old_sec} is {collision}, and the old "
                f"provision has no counterpart in the {new_act}.",
                f"The section number survives into the {new_act} but attaches to an "
                f"unrelated provision, so a model carrying the old numbering forward "
                f"produces a confident answer about the wrong provision.")

        # 2. Several old sections folded into one.
        if "merged" in kind and news:
            subject = re.sub(r"^Punishment (?:for|of) ", "", ot)
            olds = sorted({r["old_section"] for r in mapping
                           if r["new_act"] == news[0][0]
                           and r["new_section"] == news[0][1] and r["old_section"]},
                          key=lambda x: (int(re.match(r"\d+", x).group()), x))
            add("merged", old_act, old_sec, news,
                f"Is {subject[0].lower() + subject[1:]} still dealt with under "
                f"{SHORT[old_act]} Section {old_sec}?",
                f"No. The {ACT_FULL[old_act]} was repealed with effect from 1 July 2024. "
                f"{SHORT[old_act]} Section {old_sec} ({ot}) now corresponds to {refs}, "
                f"which consolidates {SHORT[old_act]} "
                f"Sections {', '.join(olds[:8])} into a single provision.",
                f"Several {SHORT[old_act]} sections were folded into one, so the answer "
                f"is not a one-to-one renumbering and a model may name only the most "
                f"familiar of the old sections.")

        # 3. One old section broken across several new ones.
        if "split" in kind and len(news) > 1:
            add("split", old_act, old_sec, news,
                f"Which single {new_act} section replaced {SHORT[old_act]} "
                f"Section {old_sec}?",
                f"There is no single one. {SHORT[old_act]} Section {old_sec} ({ot}) "
                f"was split across {refs}. Citing only one of them states the position "
                f"incompletely.",
                "A split has no single right answer, so a model that has learnt "
                "one-to-one renumbering will give one section and appear correct.")

        # 4. Repealed with nothing to replace it - only where vetted.
        if not news and (old_act, old_sec) in VETTED_REMOVED:
            add("removed", old_act, old_sec, [],
                f"Which {new_act} section corresponds to {SHORT[old_act]} "
                f"Section {old_sec}?",
                f"None. {SHORT[old_act]} Section {old_sec} ({ot}) has no counterpart in "
                f"the {ACT_FULL[new_act]}; the official BPRD correspondence table lists "
                f"no {new_act} equivalent for it.",
                "The provision was dropped rather than renumbered, so any section "
                "number offered in reply is wrong by construction.")

    # The salient list is drawn from litigated offences, which yields plenty of
    # collisions and merges but very few splits or repeals. Those two kinds are
    # topped up from the whole concordance - there are only five split sections
    # in it altogether - preferring provisions a reader would recognise.
    done = {(e["old_reference"], e["mapping_type"]) for e in entries}

    extra: list[tuple[str, tuple[str, str]]] = []
    for key, kind in kinds.items():
        if key[1] and key[0] in NEW_OF and "split" in kind and len(fwd.get(key, [])) > 1:
            extra.append(("split", key))
    for key in VETTED_REMOVED:
        if "removed" in kinds.get(key, set()) and not fwd.get(key):
            extra.append(("removed", key))

    for kind, (old_act, old_sec) in extra:
        if (f"{SHORT[old_act]} {old_sec}", kind) in done:
            continue
        new_act = NEW_OF[old_act]
        ot = titles.get((old_act, old_sec), "")
        if not ot:
            continue
        news = fwd.get((old_act, old_sec), [])
        refs = ", ".join(f"{SHORT[a]} Section {s}"
                         + (f" ({titles.get((a, s), '')})" if titles.get((a, s)) else "")
                         for a, s in news)
        if kind == "split":
            add("split", old_act, old_sec, news,
                f"Which single {new_act} section replaced {SHORT[old_act]} "
                f"Section {old_sec}?",
                f"There is no single one. {SHORT[old_act]} Section {old_sec} ({ot}) "
                f"was split across {refs}. Citing only one of them states the position "
                f"incompletely.",
                "A split has no single right answer, so a model that has learnt "
                "one-to-one renumbering will give one section and appear correct.")
        else:
            add("removed", old_act, old_sec, [],
                f"Which {new_act} section corresponds to {SHORT[old_act]} "
                f"Section {old_sec}?",
                f"None. {SHORT[old_act]} Section {old_sec} ({ot}) has no counterpart in "
                f"the {ACT_FULL[new_act]}; the official BPRD correspondence table lists "
                f"no {new_act} equivalent for it.",
                "The provision was dropped rather than renumbered, so any section "
                "number offered in reply is wrong by construction. For sedition in "
                "particular, BNS 152 is often named as the successor, but it creates a "
                "differently defined offence and the correspondence table pairs it with "
                "no IPC provision.")

    # A few questions about the transition itself, which is the thesis in one line.
    entries.append({
        "instruction": "An FIR was registered on 20 June 2024 for an offence committed "
                       "that week. The trial is going on now. Should the charge be framed "
                       "under the IPC or the BNS?",
        "input": "",
        "output": "Under the IPC. The Bharatiya Nyaya Sanhita, 2023 came into force on "
                  "1 July 2024 and applies to offences committed on or after that date. "
                  "An offence committed and registered before that date continues under "
                  "the Indian Penal Code, 1860, whatever the date of the trial.",
        "mapping_type": "transition",
        "old_reference": "IPC 1860", "new_references": ["BNS 2023"],
        "why_confusing": "The commencement date governs by reference to the date of the "
                         "offence, not the date of the proceedings, and models trained "
                         "across the changeover often apply whichever code is more "
                         "prominent in their training data.",
        "group_keys": ["transition_general_03"],
        "sources": ["data/processed/retrieval_corpus.jsonl (transition_commencement)"],
    })
    return entries


def select_confusion(entries: list[dict], rng, target: int = 44) -> list[dict]:
    """A balanced sample across the four confusion kinds."""
    quota = {"collision": 14, "merged": 14, "split": 5, "removed": 10, "transition": 1}
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_kind[e["mapping_type"]].append(e)
    picked: list[dict] = []
    for kind, n in quota.items():
        pool = by_kind.get(kind, [])
        # 'removed' is generated notable-first, so take it in order; the other
        # kinds have many interchangeable candidates and are sampled.
        if kind != "removed":
            rng.shuffle(pool)
        picked.extend(pool[:n])
    # Top up from whatever remains if a quota could not be met.
    if len(picked) < target:
        rest = [e for e in entries if e not in picked]
        rng.shuffle(rest)
        picked.extend(rest[:target - len(picked)])
    return picked[:target]


# -------------------------------------------------------------------- split

def assign(groups: dict[str, list[int]], index: list[dict], rng) -> dict[str, str]:
    """Assign whole groups, keeping both the row ratio and the qa_type mix.

    Assigning purely on size skews the question types badly, because group size
    is correlated with type: a statute section drags along section_text,
    punishment and classification pairs, while a judgment is a group of two or
    three. Largest-first then fills train with sections and leaves validation
    and test almost entirely case law.

    So each group goes to whichever split is currently most short of the
    question types that group actually contains, with overall row count as the
    tie-breaker.
    """
    totals: Counter = Counter()
    per_group: dict[str, Counter] = {}
    for g, idxs in groups.items():
        c = Counter(index[i]["qa_type"] for i in idxs)
        per_group[g] = c
        totals.update(c)

    total_rows = sum(len(v) for v in groups.values())
    counts = {k: Counter() for k in RATIO}
    rows = {k: 0 for k in RATIO}
    out: dict[str, str] = {}

    keys = list(groups)
    rng.shuffle(keys)                       # break ties without a fixed bias
    keys.sort(key=lambda g: -len(groups[g]))

    def fill(k: str, t: str, extra: int) -> float:
        """How far ahead of its quota split k would be for type t after taking g."""
        quota = RATIO[k] * totals[t]
        return (counts[k][t] + extra) / quota if quota else 0.0

    for g in keys:
        best, best_cost = None, None
        for k in RATIO:
            # The split that would end up least ahead of schedule on the types
            # this group carries. Measuring relative fill rather than absolute
            # deficit matters: an absolute shortfall is proportional to the
            # split's own share, so train would win every early comparison.
            cost = max(fill(k, t, n) for t, n in per_group[g].items())
            cost += 0.25 * (rows[k] + len(groups[g])) / (RATIO[k] * total_rows)
            if best_cost is None or cost < best_cost:
                best, best_cost = k, cost
        out[g] = best
        counts[best].update(per_group[g])
        rows[best] += len(groups[g])
    return out


def main() -> None:
    rng = random.Random(SEED)
    rows = read_jsonl(os.path.join(PROCESSED, "finetune_dataset.jsonl"))
    index = read_jsonl(os.path.join(PROCESSED, "finetune_dataset_index.jsonl"))
    mapping = list(csv.DictReader(
        open(os.path.join(PROCESSED, "mapping_table.csv"), encoding="utf-8")))
    assert len(rows) == len(index), "dataset and index are not aligned"

    titles = load_titles()
    confusion = select_confusion(build_confusion_set(mapping, titles), rng)

    group_of = build_groups(index, mapping)
    held_out: set[str] = set()
    for e in confusion:
        for k in e["group_keys"]:
            if k in group_of:
                held_out.add(group_of[k])
            else:
                # The entity may only appear in the concordance, not the QA set.
                held_out.add(k)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(index):
        groups[group_of[m["source_chunk_id"]]].append(i)

    reserved = {g: idxs for g, idxs in groups.items() if g in held_out}
    splittable = {g: idxs for g, idxs in groups.items() if g not in held_out}

    where = assign(splittable, index, rng)
    buckets: dict[str, list[int]] = {k: [] for k in RATIO}
    for g, split in where.items():
        buckets[split].extend(splittable[g])

    for split in buckets:
        rng.shuffle(buckets[split])
        write_jsonl(os.path.join(PROCESSED, f"{split}.jsonl"),
                    [rows[i] for i in buckets[split]])

    write_jsonl(os.path.join(PROCESSED, "confusion_test_set.jsonl"), confusion)

    # ------------------------------------------------------------- reporting
    total_rows = sum(len(v) for v in buckets.values())
    per_split_groups = Counter(where.values())
    qa_by_split = {k: Counter(index[i]["qa_type"] for i in v) for k, v in buckets.items()}
    all_types = sorted({t for c in qa_by_split.values() for t in c})

    print(f"held out for the confusion set: {len(reserved)} groups, "
          f"{sum(len(v) for v in reserved.values())} rows")
    print(f"splittable: {len(splittable)} groups, {total_rows} rows\n")
    print(f"{'split':<8}{'groups':>8}{'rows':>9}{'share':>9}{'target':>9}")
    for k in ("train", "val", "test"):
        n = len(buckets[k])
        print(f"{k:<8}{per_split_groups[k]:>8}{n:>9}{100.0 * n / total_rows:>8.1f}%"
              f"{100.0 * RATIO[k]:>8.0f}%")

    print(f"\n{'qa_type':<24}{'train':>9}{'val':>9}{'test':>9}   (share of that split)")
    flagged = []
    for t in all_types:
        shares = {k: 100.0 * qa_by_split[k][t] / max(len(buckets[k]), 1)
                  for k in ("train", "val", "test")}
        print(f"{t:<24}" + "".join(f"{shares[k]:>8.1f}%" for k in ("train", "val", "test")))
        for k in ("val", "test"):
            base = shares["train"]
            if base >= 1.0 and (shares[k] > 2 * base or shares[k] < 0.5 * base):
                flagged.append((t, k, round(base, 1), round(shares[k], 1)))
            elif base < 1.0 and shares[k] == 0.0 and qa_by_split["train"][t] > 0:
                flagged.append((t, k, round(base, 1), 0.0))

    print("\nqa_type balance:", "looks reasonable" if not flagged else f"flagged {flagged}")
    print(f"confusion set: {len(confusion)} entries "
          f"{dict(Counter(e['mapping_type'] for e in confusion))}")

    with open(os.path.join(PROCESSED, "split_report.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "seed": SEED, "ratio": RATIO,
            "groups": {"total": len(groups), "held_out": len(reserved),
                       "splittable": len(splittable),
                       "per_split": dict(per_split_groups)},
            "rows": {k: len(v) for k, v in buckets.items()},
            "row_share": {k: round(100.0 * len(v) / total_rows, 2)
                          for k, v in buckets.items()},
            "held_out_rows": sum(len(v) for v in reserved.values()),
            "qa_type_by_split": {k: dict(v) for k, v in qa_by_split.items()},
            "qa_type_flags": flagged,
            "confusion_set": {"size": len(confusion),
                              "by_kind": dict(Counter(e["mapping_type"] for e in confusion))},
        }, fh, indent=2)


if __name__ == "__main__":
    main()

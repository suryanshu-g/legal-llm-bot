"""Stage 6: validate the processed datasets and print the numbers for the report.

Checks are grouped as FAIL (something is wrong and the dataset should not be
used as-is) and WARN (worth knowing, not disqualifying). The exit status is
non-zero if any FAIL fires, so this can gate a rebuild.

Usage:  python scripts/validate_data.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawlib import PROCESSED, RAW, read_jsonl

EXPECTED = {"BNS": 358, "BNSS": 531, "BSA": 170}
VALID_TYPES = {"direct", "split", "merged", "new_provision", "removed"}

# Characters that signal a decoding problem rather than legitimate typography.
BAD_CHARS = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")
MOJIBAKE = re.compile(r"Ã[\x80-\xbf]|â€™|â€œ|â€\x9d|Â ")

failures: list[str] = []
warnings: list[str] = []
stats: dict = {}


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  WARN  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


# ------------------------------------------------------------ source coverage

def check_sections():
    print("\n[1] Section extraction coverage")
    per_act = {}
    for act, folder in [("BNS", "bns"), ("BNSS", "bnss"), ("BSA", "bsa")]:
        blob = json.load(open(os.path.join(RAW, folder, "sections.json"), encoding="utf-8"))
        secs = blob["sections"]
        nums = [int(s["section"]) for s in secs]
        expected = EXPECTED[act]
        missing = sorted(set(range(1, expected + 1)) - set(nums))
        dupes = [n for n, c in Counter(nums).items() if c > 1]
        per_act[act] = {"extracted": len(secs), "expected": expected,
                        "missing": missing, "duplicates": dupes,
                        "untitled": [s["section"] for s in secs if not s["title"]],
                        "empty_text": [s["section"] for s in secs if len(s["text"]) < 30]}
        if missing:
            fail(f"{act}: {len(missing)} sections missing: {missing[:10]}")
        elif dupes:
            fail(f"{act}: duplicate section numbers {dupes[:10]}")
        else:
            ok(f"{act}: {len(secs)}/{expected} sections (100.0%)")
        if per_act[act]["empty_text"]:
            fail(f"{act}: sections with no text: {per_act[act]['empty_text'][:10]}")
        if per_act[act]["untitled"]:
            warn(f"{act}: {len(per_act[act]['untitled'])} sections without a heading "
                 f"in the gazette margin: {per_act[act]['untitled']}")

    for act, folder in [("IPC", "ipc"), ("CRPC", "crpc"), ("IEA", "evidence_act")]:
        d = json.load(open(os.path.join(RAW, folder, "devgan_sections.json"),
                           encoding="utf-8"))["sections"]
        c = json.load(open(os.path.join(RAW, folder, "civictech_sections.json"),
                           encoding="utf-8"))["sections"]
        dn = {str(s["section"]).upper() for s in d}
        cn = {str(s["section"]).upper() for s in c}
        overlap = len(dn & cn)
        per_act[act] = {"devgan": len(d), "civictech": len(c),
                        "agreeing_section_numbers": overlap,
                        "devgan_only": len(dn - cn), "civictech_only": len(cn - dn)}
        pct = 100.0 * overlap / max(len(dn | cn), 1)
        ok(f"{act}: devgan {len(d)}, civictech {len(c)}, "
           f"{overlap} section numbers in both ({pct:.1f}% of the union)")
    stats["sections"] = per_act


# ------------------------------------------------------------------- mapping

def check_mapping():
    print("\n[2] Mapping table")
    path = os.path.join(PROCESSED, "mapping_table.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    kinds = Counter(r["mapping_type"] for r in rows)

    bad = [r for r in rows if r["mapping_type"] not in VALID_TYPES]
    if bad:
        fail(f"{len(bad)} rows with an unknown mapping_type")
    else:
        ok(f"{len(rows)} rows, all mapping_type values valid: {dict(kinds)}")

    # Referential integrity: every new_section named must exist in that act.
    known = {}
    for act, folder in [("BNS", "bns"), ("BNSS", "bnss"), ("BSA", "bsa")]:
        known[act] = {str(s["section"]) for s in json.load(
            open(os.path.join(RAW, folder, "sections.json"), encoding="utf-8"))["sections"]}
    dangling = [r for r in rows if r["new_section"]
                and r["new_section"] not in known.get(r["new_act"], set())]
    if dangling:
        fail(f"{len(dangling)} rows reference a new section that was never extracted, "
             f"e.g. {[(r['new_act'], r['new_section']) for r in dangling[:5]]}")
    else:
        ok("every new_section in the table exists in the extracted act text")

    for r in rows:
        if r["mapping_type"] == "new_provision" and r["old_section"]:
            fail("a new_provision row carries an old_section")
            break
        if r["mapping_type"] == "removed" and r["new_section"]:
            fail("a removed row carries a new_section")
            break
    else:
        ok("new_provision and removed rows are shaped correctly")

    covered = defaultdict(set)
    for r in rows:
        if r["new_section"]:
            covered[r["new_act"]].add(r["new_section"])
    for act, exp in EXPECTED.items():
        n = len(covered[act])
        if n < exp:
            warn(f"{act}: concordance covers {n}/{exp} sections")
        else:
            ok(f"{act}: concordance covers {n}/{exp} sections")

    dis = list(csv.DictReader(
        open(os.path.join(PROCESSED, "mapping_disagreements.csv"), encoding="utf-8")))
    dkinds = Counter(d["kind"] for d in dis)
    if dkinds.get("no_overlap"):
        warn(f"{dkinds['no_overlap']} mappings where the sources share no section at all")
    ok(f"{len(dis)} cross-source disagreements recorded: {dict(dkinds)}")
    stats["mapping"] = {"rows": len(rows), "types": dict(kinds),
                        "disagreements": dict(dkinds), "total_disagreements": len(dis)}


# ----------------------------------------------------------- retrieval corpus

def check_corpus():
    print("\n[3] Retrieval corpus")
    rows = read_jsonl(os.path.join(PROCESSED, "retrieval_corpus.jsonl"))

    ids = Counter(r["chunk_id"] for r in rows)
    dupes = [k for k, v in ids.items() if v > 1]
    if dupes:
        fail(f"{len(dupes)} duplicate chunk_id values: {dupes[:5]}")
    else:
        ok(f"{len(rows)} chunks, all chunk_id values unique")

    for field in ("chunk_id", "source", "text", "source_url"):
        empty = [r["chunk_id"] for r in rows if not str(r.get(field, "")).strip()]
        if empty:
            fail(f"{len(empty)} chunks with an empty '{field}': {empty[:5]}")
        else:
            ok(f"every chunk has a non-empty '{field}'")

    bad = [r["chunk_id"] for r in rows if BAD_CHARS.search(r["text"])]
    moji = [r["chunk_id"] for r in rows if MOJIBAKE.search(r["text"])]
    if bad:
        fail(f"{len(bad)} chunks contain replacement or control characters: {bad[:5]}")
    else:
        ok("no replacement or control characters in chunk text")
    if moji:
        fail(f"{len(moji)} chunks show mojibake: {moji[:5]}")
    else:
        ok("no mojibake detected")

    lens = sorted(len(r["text"]) for r in rows)
    tiny = [r["chunk_id"] for r in rows if len(r["text"]) < 80]
    if tiny:
        warn(f"{len(tiny)} chunks under 80 characters: {tiny[:5]}")
    huge = [r["chunk_id"] for r in rows if len(r["text"]) > 20000]
    if huge:
        fail(f"{len(huge)} chunks over 20000 characters (likely a parsing artefact): {huge}")
    else:
        ok(f"chunk sizes reasonable (median {lens[len(lens) // 2]}, max {lens[-1]})")

    by_type = Counter(r["doc_type"] for r in rows)
    ok(f"by document type: {dict(by_type)}")

    # Every section of the three new codes must be retrievable.
    for act, exp in EXPECTED.items():
        got = len([r for r in rows if r["doc_type"] == "statute" and r["act"] == act])
        if got != exp:
            fail(f"{act}: {got} statute chunks, expected {exp}")
        else:
            ok(f"{act}: all {exp} sections present as chunks")

    linked = len([r for r in rows if r["equivalent_sections"]])
    ok(f"{linked} chunks carry an old/new equivalent reference "
       f"({100.0 * linked / len(rows):.1f}%)")
    stats["corpus"] = {"chunks": len(rows), "by_type": dict(by_type),
                       "median_chars": lens[len(lens) // 2], "max_chars": lens[-1],
                       "with_equivalents": linked}


# ------------------------------------------------------------ finetune set

def check_finetune():
    print("\n[4] Fine-tuning dataset")
    rows = read_jsonl(os.path.join(PROCESSED, "finetune_dataset.jsonl"))
    index = read_jsonl(os.path.join(PROCESSED, "finetune_dataset_index.jsonl"))

    if len(rows) != len(index):
        fail(f"dataset ({len(rows)}) and index ({len(index)}) are not aligned")
    else:
        ok(f"{len(rows)} pairs, index aligned line for line")

    schema_bad = [i for i, r in enumerate(rows)
                  if set(r) != {"instruction", "input", "output"}]
    if schema_bad:
        fail(f"{len(schema_bad)} rows do not have exactly instruction/input/output")
    else:
        ok("every row has exactly the instruction / input / output keys")

    empty = [i for i, r in enumerate(rows)
             if not r["instruction"].strip() or not r["output"].strip()]
    if empty:
        fail(f"{len(empty)} rows with an empty instruction or output")
    else:
        ok("no empty instructions or outputs")

    qs = Counter(r["instruction"].lower() for r in rows)
    dupes = [q for q, c in qs.items() if c > 1]
    if dupes:
        fail(f"{len(dupes)} duplicate instructions survived deduplication")
    else:
        ok("all instructions unique")

    bad = [i for i, r in enumerate(rows)
           if BAD_CHARS.search(r["output"]) or MOJIBAKE.search(r["output"])]
    if bad:
        fail(f"{len(bad)} outputs contain encoding damage")
    else:
        ok("no encoding damage in outputs")

    kinds = Counter(m["qa_type"] for m in index)
    lens = sorted(len(r["output"]) for r in rows)
    ok(f"by question type: {dict(kinds)}")
    ok(f"answer length: median {lens[len(lens) // 2]}, max {lens[-1]} characters")

    # The scope-discipline examples are what keep the bot inside its brief.
    if kinds.get("scope", 0) < 5:
        warn("fewer than 5 scope/refusal examples")
    else:
        ok(f"{kinds['scope']} scope-discipline examples present")
    stats["finetune"] = {"pairs": len(rows), "by_qa_type": dict(kinds),
                         "median_answer_chars": lens[len(lens) // 2]}


# ------------------------------------------------------------------ case law

def check_cases():
    print("\n[5] Case law")
    path = os.path.join(RAW, "case_law", "case_summaries.json")
    if not os.path.exists(path):
        warn("no case summaries found")
        return
    blob = json.load(open(path, encoding="utf-8"))
    cases = blob["cases"]
    per_topic = Counter(c["topic"] for c in cases)
    thin = [t for t, n in per_topic.items() if n < 3]

    ok(f"{len(cases)} summaries across {len(per_topic)} topics")
    if thin:
        warn(f"{len(thin)} topics with fewer than 3 cases: {thin}")
    for field in ("case_name", "court", "holding_summary", "source_url"):
        missing = [c["case_name"] for c in cases if not str(c.get(field, "")).strip()]
        if missing:
            warn(f"{len(missing)} cases missing '{field}': {missing[:3]}")
        else:
            ok(f"every case has '{field}'")
    noyear = [c["case_name"] for c in cases if not c.get("year")]
    if noyear:
        warn(f"{len(noyear)} cases without a year: {noyear[:3]}")
    else:
        ok("every case has a year")
    linked = len([c for c in cases if c["new_sections"]])
    ok(f"{linked}/{len(cases)} cases carry a new-code equivalent from the concordance")
    stats["case_law"] = {"cases": len(cases), "topics": len(per_topic),
                         "per_topic": dict(per_topic), "thin_topics": thin}


def main() -> None:
    print("Validating processed datasets")
    check_sections()
    check_mapping()
    check_corpus()
    check_finetune()
    check_cases()

    print(f"\n{'=' * 60}")
    print(f"{len(failures)} failures, {len(warnings)} warnings")
    with open(os.path.join(PROCESSED, "validation_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"failures": failures, "warnings": warnings, "stats": stats},
                  fh, indent=2)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

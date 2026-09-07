"""Stage 4: build the retrieval corpus for the RAG layer.

One chunk per section, and one per case summary - never several sections in a
chunk. For a "which section applies" assistant, retrieval precision matters far
more than chunk count: a chunk that mixes BNS 303 and BNS 304 makes it harder,
not easier, for the reader to see which provision the answer rests on.

Each statute chunk opens with a short header naming the act, section number,
heading and - crucially - the old-code equivalent, e.g.

    BNS 2023, Section 103 - Punishment for murder
    (corresponds to IPC 1860, Section 302)

That header is part of the embedded text on purpose. A user who asks about
"IPC 302" should retrieve the BNS section that replaced it, which only works if
the old numbering is inside the chunk the embedder sees.

Output: data/processed/retrieval_corpus.jsonl
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawlib import PROCESSED, RAW, clean_text, write_jsonl

ACT_LONG = {
    "BNS": "BNS 2023", "BNSS": "BNSS 2023", "BSA": "BSA 2023",
    "IPC": "IPC 1860", "CRPC": "CrPC 1973", "IEA": "Indian Evidence Act 1872",
}

NEW_ACTS = {"BNS": "bns", "BNSS": "bnss", "BSA": "bsa"}
OLD_ACTS = {"IPC": "ipc", "CRPC": "crpc", "IEA": "evidence_act"}

IN_FORCE_NOTE = ("The new criminal codes came into force on 1 July 2024. Offences "
                 "committed on or after that date are dealt with under the new codes; "
                 "matters registered earlier continue under the old codes.")


def load_equivalents():
    """Both directions of the concordance, keyed by (act, section)."""
    fwd: dict[tuple[str, str], list[str]] = defaultdict(list)   # old -> new
    rev: dict[tuple[str, str], list[str]] = defaultdict(list)   # new -> old
    path = os.path.join(PROCESSED, "mapping_table.csv")
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if not (r["old_section"] and r["new_section"]):
            continue
        o, n = (r["old_act"], r["old_section"]), (r["new_act"], r["new_section"])
        if n not in [(a, s) for a, s in fwd[o]]:
            fwd[o].append(n)
        if o not in [(a, s) for a, s in rev[n]]:
            rev[n].append(o)
    return fwd, rev


def fmt_refs(refs) -> str:
    return ", ".join(f"{ACT_LONG.get(a, a)} Section {s}" for a, s in refs)


def statute_chunks(fwd, rev):
    rows = []
    for act, folder in list(NEW_ACTS.items()) + list(OLD_ACTS.items()):
        is_new = act in NEW_ACTS
        fname = "sections.json" if is_new else "devgan_sections.json"
        path = os.path.join(RAW, folder, fname)
        blob = json.load(open(path, encoding="utf-8"))
        # For the new codes the gazette margin lost a couple of headings; take
        # those from the devgan pass so no chunk is left unlabelled.
        fallback = {}
        alt = os.path.join(RAW, folder, "devgan_sections.json")
        if is_new and os.path.exists(alt):
            fallback = {s["section"]: s["title"]
                        for s in json.load(open(alt, encoding="utf-8"))["sections"]}

        for s in blob["sections"]:
            sec = str(s["section"])
            title = s["title"] or fallback.get(sec, "")
            key = (act, sec.upper())
            equiv = rev.get(key) if is_new else fwd.get(key)

            header = f"{ACT_LONG[act]}, Section {sec}"
            if title:
                header += f" - {title}"
            lines = [header]
            if equiv:
                rel = "corresponds to" if is_new else "replaced by"
                lines.append(f"({rel} {fmt_refs(equiv)})")
            if s.get("chapter_title"):
                lines.append(f"Chapter {s.get('chapter_no')}: {s['chapter_title'].title()}")
            lines.append("")
            lines.append(s["text"])

            rows.append({
                "chunk_id": f"{act.lower()}_{sec.lower()}",
                "doc_type": "statute",
                "source": f"{ACT_LONG[act]}, Section {sec}",
                "act": act,
                "section": sec,
                "section_title": title,
                "chapter_no": s.get("chapter_no"),
                "chapter_title": s.get("chapter_title"),
                "in_force_from": "2024-07-01" if is_new else None,
                "status": "in force" if is_new else "repealed with effect from 1 July 2024",
                "equivalent_sections": [f"{a} {x}" for a, x in (equiv or [])],
                "text": clean_text("\n".join(lines)),
                "source_url": s["source_url"],
            })
    return rows


def case_chunks():
    """One chunk per distinct judgment, not per topic-case pairing.

    A judgment can answer to more than one topic - Navjot Sandhu is retrieved
    under murder, criminal conspiracy, extortion, electronic evidence and
    confessions - and emitting it once per topic would put five chunks with
    near-identical text into the index, crowding out other results. Instead each
    judgment appears once, listing every topic it covers.
    """
    path = os.path.join(RAW, "case_law", "case_summaries.json")
    if not os.path.exists(path):
        print("  (no case summaries yet - run scripts/scrape_case_law.py)")
        return []
    blob = json.load(open(path, encoding="utf-8"))

    merged: dict[str, dict] = {}
    for c in blob["cases"]:
        m = merged.setdefault(c["source_url"], {
            "case": c, "topics": [], "old": [], "new": [],
        })
        if c["topic_label"] not in m["topics"]:
            m["topics"].append(c["topic_label"])
        for s in c["old_sections"]:
            ref = f"{c['old_act']} {s}"
            if ref not in m["old"]:
                m["old"].append(ref)
        for s in c["new_sections"]:
            if s not in m["new"]:
                m["new"].append(s)

    rows = []
    for i, (url, m) in enumerate(sorted(merged.items()), start=1):
        c = m["case"]
        old, new = ", ".join(m["old"]), ", ".join(m["new"][:8])
        lines = [
            f"{c['case_name']} ({c['court']}, {c['year']})",
            f"Topics: {'; '.join(m['topics'])}",
            f"Provisions: {old}" + (f" (now {new})" if new else ""),
            "",
            f"Held (extract from the judgment): {c['holding_summary']}",
        ]
        rows.append({
            "chunk_id": f"case_{i:03d}",
            "doc_type": "case_law",
            "source": f"{c['case_name']} ({c['court']}, {c['year']})",
            "act": c["old_act"],
            "section": c["old_sections"][0] if c["old_sections"] else "",
            "section_title": "; ".join(m["topics"]),
            "chapter_no": None,
            "chapter_title": None,
            "in_force_from": None,
            "status": "case law",
            "equivalent_sections": m["new"],
            "text": clean_text("\n".join(lines)),
            "source_url": url,
        })
    return rows


def transition_chunks():
    """A handful of chunks about the transition itself.

    Without these the corpus can say what each code provides but has nothing to
    retrieve for the question the whole project is built around: which code
    applies to a given date.
    """
    base = [{
        "chunk_id": "transition_commencement",
        "doc_type": "reference",
        "source": "Commencement of the new criminal codes",
        "act": "", "section": "", "section_title": "Commencement of the new criminal codes",
        "chapter_no": None, "chapter_title": None,
        "in_force_from": "2024-07-01", "status": "reference",
        "equivalent_sections": [],
        "text": clean_text(
            "Commencement of the new criminal codes\n\n"
            "The Bharatiya Nyaya Sanhita, 2023 (BNS, 358 sections), the Bharatiya "
            "Nagarik Suraksha Sanhita, 2023 (BNSS, 531 sections) and the Bharatiya "
            "Sakshya Adhiniyam, 2023 (BSA, 170 sections) came into force on "
            "1 July 2024. They replace the Indian Penal Code, 1860, the Code of "
            "Criminal Procedure, 1973 and the Indian Evidence Act, 1872 "
            "respectively. " + IN_FORCE_NOTE),
        "source_url": "https://www.mha.gov.in/",
    }]
    return base


def main() -> None:
    fwd, rev = load_equivalents()
    rows = statute_chunks(fwd, rev) + case_chunks() + transition_chunks()

    out = os.path.join(PROCESSED, "retrieval_corpus.jsonl")
    n = write_jsonl(out, rows)

    by_type: dict[str, int] = defaultdict(int)
    by_act: dict[str, int] = defaultdict(int)
    for r in rows:
        by_type[r["doc_type"]] += 1
        by_act[r["act"] or "-"] += 1
    print(f"retrieval_corpus.jsonl: {n} chunks")
    print("  by type:", dict(by_type))
    print("  by act: ", dict(by_act))
    lens = sorted(len(r["text"]) for r in rows)
    print(f"  text length: min={lens[0]} median={lens[len(lens) // 2]} max={lens[-1]}")


if __name__ == "__main__":
    main()

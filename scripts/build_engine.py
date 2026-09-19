"""Pack the verified data into one file the website can answer questions from.

The Python bot needs PyTorch, FAISS and a 300 MB checkpoint, none of which can
run on a static host. But the half of the bot that is actually trustworthy needs
none of them: parsing citations out of a question, looking provisions up in the
concordance, reading the First Schedule and quoting the gazette are all lookups
over verified data. That half ports to the browser exactly.

What the browser therefore cannot do, and the site says so: rank passages by
meaning. A question naming no section falls back to a search over section
headings rather than the embeddings, so "is theft bailable?" finds *Theft* by its
title instead of by similarity.

Nothing the fine-tuned model generates is included. Its section numbers drift to
neighbouring provisions and it corrupts statutory titles (see RESULTS.md), so
every sentence the site produces is composed from this file.

Output: docs/engine.json, about 2.8 MB (roughly 700 KB over the wire gzipped).

Usage:  python scripts/build_engine.py
"""

from __future__ import annotations

import collections
import csv
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "docs", "engine.json")

ACT_FULL = {
    "IPC": "Indian Penal Code, 1860",
    "CRPC": "Code of Criminal Procedure, 1973",
    "IEA": "Indian Evidence Act, 1872",
    "BNS": "Bharatiya Nyaya Sanhita, 2023",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "BSA": "Bharatiya Sakshya Adhiniyam, 2023",
}
SHORT = {"IPC": "IPC", "CRPC": "CrPC", "IEA": "Indian Evidence Act",
         "BNS": "BNS", "BNSS": "BNSS", "BSA": "BSA"}
NEW_OF = {"IPC": "BNS", "CRPC": "BNSS", "IEA": "BSA"}
OLD_OF = {v: k for k, v in NEW_OF.items()}

# chunk_id -> "ACT SEC", the same key extract_refs produces in metrics.py.
CHUNK = re.compile(r"^(bns|bnss|bsa|ipc|crpc|iea)_(\d+[a-z]{0,2})$", re.I)


def main() -> None:
    corpus = {}
    with io.open(os.path.join(P, "retrieval_corpus.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                corpus[d["chunk_id"]] = d

    with io.open(os.path.join(P, "mapping_table.csv"), encoding="utf-8") as fh:
        mapping = list(csv.DictReader(fh))

    # ---- sections: heading, text, and where it came from -----------------
    sections: dict[str, dict] = {}
    for cid, d in corpus.items():
        m = CHUNK.match(cid)
        if not m:
            continue
        act, sec = m.group(1).upper(), m.group(2).upper()
        key = f"{act} {sec}"
        text = re.sub(r"\s+", " ", d["text"]).strip()
        # The chunk opens with its own header, which the page renders itself.
        head = text.split(" Chapter ", 1)
        body = head[1].split(": ", 1)[1] if len(head) > 1 and ": " in head[1] else text
        sections[key] = {
            "t": "",                       # heading, filled from the concordance
            "x": body[:2000],
            "u": d["source_url"],
            "s": d["source"],
        }

    # ---- headings and the concordance, both directions -------------------
    fwd: dict[str, list[str]] = collections.defaultdict(list)
    back: dict[str, list[str]] = collections.defaultdict(list)
    for r in mapping:
        for act, sec, title in (
                (r["old_act"], r["old_section"], r["old_section_title"]),
                (r["new_act"], r["new_section"], r["new_section_title"])):
            if not sec:
                continue
            key = f"{act.upper()} {sec.upper()}"
            if title:
                sections.setdefault(key, {"t": "", "x": "", "u": "", "s": ""})
                if not sections[key]["t"]:
                    sections[key]["t"] = title
        if r["old_section"] and r["new_section"]:
            o = f"{r['old_act'].upper()} {r['old_section'].upper()}"
            n = f"{r['new_act'].upper()} {r['new_section'].upper()}"
            if n not in fwd[o]:
                fwd[o].append(n)
            if o not in back[n]:
                back[n].append(o)

    # Provisions the correspondence table records as having no counterpart.
    removed = sorted({f"{r['old_act'].upper()} {r['old_section'].upper()}"
                      for r in mapping
                      if r["mapping_type"] == "removed" and r["old_section"]
                      and not r["new_section"]})

    # ---- First Schedule: cognizable / bailable / which court -------------
    schedule: dict[str, list[dict]] = collections.defaultdict(list)
    with io.open(os.path.join(P, "bnss_schedule.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("part") != "I":
                continue
            base = re.sub(r"\D.*$", "", r["section_ref"])
            if not base:
                continue
            schedule[f"BNS {base}"].append({
                "r": r["section_ref"],
                "o": r["offence_description"][:220],
                "p": r["punishment"][:220],
                "c": r["cognizable"],
                "b": r["bailable"],
                "tb": r["triable_by"],
            })

    # ---- a title index, for questions naming no section ------------------
    # Not semantic search - the embeddings cannot come along - but enough to
    # find a provision by the words in its heading.
    index = [[k, sections[k]["t"].lower()] for k in sections if sections[k]["t"]]

    schedule_url = ""
    if "schedule_bns_303" in corpus:
        schedule_url = corpus["schedule_bns_303"]["source_url"]

    payload = {
        "acts": ACT_FULL,
        "short": SHORT,
        "newOf": NEW_OF,
        "oldOf": OLD_OF,
        "sections": sections,
        "fwd": dict(fwd),
        "back": dict(back),
        "removed": removed,
        "schedule": dict(schedule),
        "scheduleUrl": schedule_url,
        "counts": {
            "sections": len(sections),
            "mapped": len(fwd),
            "schedule": len(schedule),
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))

    size = os.path.getsize(OUT)
    print(f"wrote {OUT}: {size / 1e6:.2f} MB")
    print(f"  {len(sections)} provisions, {sum(1 for k in sections if sections[k]['x'])} "
          f"with statutory text")
    print(f"  {len(fwd)} with a recorded counterpart, {len(removed)} recorded as removed")
    print(f"  {len(schedule)} BNS sections classified in the First Schedule")

    # Spot-check the shapes the site depends on.
    for key, want in (("IPC 302", "BNS 103"), ("BNS 115", None)):
        got = payload["fwd"].get(key) or payload["back"].get(key)
        print(f"  {key}: title {sections[key]['t']!r}, linked {got}")
    if "BNS 303" not in schedule:
        raise SystemExit("BNS 303 missing from the schedule index")


if __name__ == "__main__":
    main()

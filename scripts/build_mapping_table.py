"""Stage 2: build the old-to-new section concordance.

Primary source
--------------
The three "Correspondence Table and Comparison Summary" PDFs published by the
Bureau of Police Research and Development (BPRD), a Ministry of Home Affairs
body. These are the closest thing to an official concordance: for every section
(and often every sub-section) of the new code they give the old-code provision
it derives from, plus a prose note on what changed.

Cross-check sources
-------------------
* The UP Police BNS/IPC comparative table - an independent government
  compilation, used to check the BNS-to-IPC half of the mapping.
* The per-section "IPC Section N" back-references carried on devgan.in's BNS
  pages (65 of 358 sections).

Where the sources disagree the row is kept, marked, and counted; disagreements
are reported rather than silently resolved.

Mapping typology
----------------
direct         one old section  -> one new section
split          one old section  -> several new sections
merged         several old sections -> one new section
new_provision  new section with no old-code equivalent
removed        old section with no counterpart in the new code

Output: data/processed/mapping_table.csv
        data/processed/mapping_disagreements.csv
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawlib import PROCESSED, RAW, clean_text, load_old_act

# file stem -> (new act, old act, column order after compacting empty cells)
CONCORDANCES = {
    "bprd_bns_to_ipc": ("BNS", "IPC", ["new", "subject", "old", "summary"]),
    "bprd_bnss_to_crpc": ("BNSS", "CRPC", ["new", "subject", "old", "summary"]),
    "bprd_bsa_to_iea": ("BSA", "IEA", ["new", "old", "subject", "summary"]),
}

OLD_ACT_FOLDER = {"IPC": "ipc", "CRPC": "crpc", "IEA": "evidence_act"}
NEW_ACT_FOLDER = {"BNS": "bns", "BNSS": "bnss", "BSA": "bsa"}

# Values in the old-section column that mean "there is no old provision".
NO_OLD = re.compile(
    r"^(new(ly)?( added| provision| sub-?section)?|-+|nil|n/?a|none)\.?$", re.I)

SECTION_REF = re.compile(r"^\d+[A-Z]{0,2}$")


def norm_section(ref: str) -> str | None:
    """Reduce a table cell such as '2(1)(a)' or '3, para 1' to its base section."""
    if not ref:
        return None
    ref = ref.replace("\n", " ").strip().strip("].,;")
    m = re.match(r"^(\d+\s*[A-Z]{0,2})", ref)
    if not m:
        return None
    return re.sub(r"\s+", "", m.group(1)).upper()


def split_old_refs(cell: str) -> list[str]:
    """A single old-section cell may name several provisions."""
    if not cell:
        return []
    cell = cell.replace("\n", " ").strip()
    if NO_OLD.match(cell):
        return []
    out: list[str] = []
    # Split on separators that genuinely divide provisions, not on the commas
    # inside 'para 1, 2' style qualifiers.
    for piece in re.split(r"\s*(?:,|;|&|\band\b|/)\s*", cell):
        s = norm_section(piece)
        if s and s not in out:
            out.append(s)
    return out


def compact(row) -> list[str]:
    return [clean_text(str(c)) for c in row if c is not None and str(c).strip()]


def page_tables(page):
    """Extract tables, including the row stranded at the top of each page.

    These tables run across page breaks. On every page after the first, the
    top-most data row has no ruling line above it (the page boundary cut it),
    so pdfplumber cannot bound the row and silently drops it - about one lost
    section per page. Supplying the top of the vertical rules as an explicit
    horizontal line closes the cell and recovers the row.
    """
    v_edges = [e for e in page.edges if e["orientation"] == "v"]
    if not v_edges:
        return page.extract_tables()
    top = min(e["top"] for e in v_edges)
    bottom = max(e["bottom"] for e in v_edges)
    return page.extract_tables({
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "explicit_horizontal_lines": [top, bottom],
    })


def parse_bprd(stem: str, new_act: str, old_act: str, order: list[str]) -> tuple[list[dict], int]:
    """Extract (new, old, subject, summary) rows from one BPRD correspondence PDF."""
    import pdfplumber

    path = os.path.join(RAW, "mapping", f"{stem}.pdf")
    rows: list[dict] = []
    skipped = 0
    idx = {name: i for i, name in enumerate(order)}

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page_tables(page):
                for raw in table:
                    cells = compact(raw)
                    # "Summary of comparison" is the one genuinely optional
                    # column - a handful of rows leave it blank - so a row one
                    # cell short is padded rather than dropped. Anything else
                    # is counted as unparsed and reported, never guessed at.
                    if len(cells) == len(order) - 1 and order[-1] == "summary":
                        cells = cells + [""]
                    if len(cells) != len(order):
                        if cells and not re.match(r"^(BNS|BNSS|BSA)\b", cells[0], re.I):
                            skipped += 1
                        continue

                    new_ref = cells[idx["new"]]
                    old_cell = cells[idx["old"]]
                    if not re.match(r"^\d", new_ref):
                        continue  # header row
                    # Guard the padding above: if the blank cell was actually
                    # the subject rather than the summary, the old-section
                    # column will not hold a section reference and the row is
                    # rejected instead of being silently misaligned.
                    if not (norm_section(old_cell) or NO_OLD.match(old_cell.strip())):
                        skipped += 1
                        continue
                    rows.append({
                        "new_ref": new_ref,
                        "new_section": norm_section(new_ref),
                        "old_refs": split_old_refs(old_cell),
                        "subject": cells[idx["subject"]],
                        "summary": cells[idx["summary"]],
                    })
    rows = [r for r in rows if r["new_section"]]
    print(f"  [{stem}] {len(rows)} table rows parsed, {skipped} unparsed")
    return rows, skipped


def parse_uppolice() -> dict[str, set[str]]:
    """BNS -> {IPC sections}, from the UP Police comparative table."""
    import pdfplumber

    path = os.path.join(RAW, "mapping", "uppolice_bns_ipc.pdf")
    out: dict[str, set[str]] = defaultdict(set)
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page_tables(page):
                for raw in table:
                    cells = compact(raw)
                    if len(cells) != 2:
                        continue
                    bns, ipc = cells
                    if not re.match(r"^\d", bns):
                        continue
                    b = norm_section(bns)
                    if not b:
                        continue
                    for s in split_old_refs(ipc):
                        out[b].add(s)
    print(f"  [uppolice] {len(out)} BNS sections cross-referenced")
    return out


def load_devgan_xrefs() -> dict[str, set[str]]:
    path = os.path.join(RAW, "bns", "devgan_sections.json")
    out: dict[str, set[str]] = defaultdict(set)
    for s in json.load(open(path, encoding="utf-8"))["sections"]:
        if s.get("xref_section") and (s.get("xref_act") or "").upper() == "IPC":
            base = norm_section(s["xref_section"])
            if base:
                out[s["section"]].add(base)
    print(f"  [devgan] {len(out)} BNS sections cross-referenced")
    return out


def load_titles() -> dict[tuple[str, str], str]:
    """(ACT, section) -> section heading, for both old and new codes."""
    titles: dict[tuple[str, str], str] = {}

    for act, folder in NEW_ACT_FOLDER.items():
        for s in json.load(open(os.path.join(RAW, folder, "sections.json"),
                                encoding="utf-8"))["sections"]:
            if s["title"]:
                titles[(act, s["section"])] = s["title"]
        # devgan supplies headings for the handful the gazette margin lost.
        dg = os.path.join(RAW, folder, "devgan_sections.json")
        if os.path.exists(dg):
            for s in json.load(open(dg, encoding="utf-8"))["sections"]:
                key = (act, s["section"])
                if s["title"] and not titles.get(key):
                    titles[key] = s["title"]

    for act in OLD_ACT_FOLDER:
        for s in load_old_act(act):
            key = (act, str(s["section"]).upper())
            if s["title"] and not titles.get(key):
                titles[key] = s["title"]
    return titles


def old_universe(act: str) -> set[str]:
    """Every section number the old act actually contains."""
    seen: set[str] = set()
    for s in load_old_act(act):
        n = norm_section(str(s["section"]))
        if n:
            seen.add(n)
    return seen


def main() -> None:
    titles = load_titles()
    uppolice = parse_uppolice()
    devgan = load_devgan_xrefs()

    all_rows: list[dict] = []
    disagreements: list[dict] = []
    compared: dict[str, int] = defaultdict(int)
    stats: dict[str, dict] = {}

    for stem, (new_act, old_act, order) in CONCORDANCES.items():
        rows, skipped = parse_bprd(stem, new_act, old_act, order)

        # Aggregate the sub-section level table up to whole sections.
        pairs: dict[tuple[str, str], dict] = {}
        new_to_old: dict[str, set[str]] = defaultdict(set)
        old_to_new: dict[str, set[str]] = defaultdict(set)
        notes: dict[str, list[str]] = defaultdict(list)
        subjects: dict[str, str] = {}

        for r in rows:
            n = r["new_section"]
            subjects.setdefault(n, r["subject"])
            if r["summary"]:
                notes[n].append(f"{r['new_ref']}: {r['summary']}")
            for o in r["old_refs"]:
                new_to_old[n].add(o)
                old_to_new[o].add(n)
                pairs.setdefault((o, n), {"refs": []})["refs"].append(r["new_ref"])
            if not r["old_refs"]:
                new_to_old.setdefault(n, set())

        # --- cross-check the BNS half against two independent tables --------
        if new_act == "BNS":
            for n, ours in sorted(new_to_old.items(), key=lambda kv: _key(kv[0])):
                for label, other in (("uppolice", uppolice), ("devgan", devgan)):
                    theirs = other.get(n)
                    if not theirs or not ours:
                        continue
                    compared[label] += 1
                    if ours == theirs:
                        continue
                    # A partial difference is usually a letter-suffix detail
                    # (IPC 23 vs 23C); no overlap at all is a real conflict.
                    kind = "partial_overlap" if (ours & theirs) else "no_overlap"
                    disagreements.append({
                        "new_act": new_act, "new_section": n,
                        "bprd_old_sections": "|".join(sorted(ours, key=_key)),
                        "other_source": label,
                        "other_old_sections": "|".join(sorted(theirs, key=_key)),
                        "kind": kind,
                        "resolution": "kept BPRD (official MHA source); flagged for review",
                    })

        mapped_old = set(old_to_new)

        for (o, n), meta in sorted(pairs.items(), key=lambda kv: (_key(kv[0][1]), _key(kv[0][0]))):
            if len(old_to_new[o]) > 1:
                mtype = "split"
            elif len(new_to_old[n]) > 1:
                mtype = "merged"
            else:
                mtype = "direct"
            all_rows.append({
                "old_act": old_act,
                "old_section": o,
                "old_section_title": titles.get((old_act, o), ""),
                "new_act": new_act,
                "new_section": n,
                "new_section_title": titles.get((new_act, n)) or subjects.get(n, ""),
                "mapping_type": mtype,
                "notes": clean_text(" ".join(notes.get(n, []))[:900]),
            })

        # new provisions: a new section with no old counterpart anywhere
        for n, olds in sorted(new_to_old.items(), key=lambda kv: _key(kv[0])):
            if olds:
                continue
            all_rows.append({
                "old_act": "", "old_section": "", "old_section_title": "",
                "new_act": new_act, "new_section": n,
                "new_section_title": titles.get((new_act, n)) or subjects.get(n, ""),
                "mapping_type": "new_provision",
                "notes": clean_text(" ".join(notes.get(n, []))[:900]),
            })

        # removed provisions: old sections the concordance never mentions
        removed = sorted(old_universe(old_act) - mapped_old, key=_key)
        for o in removed:
            all_rows.append({
                "old_act": old_act, "old_section": o,
                "old_section_title": titles.get((old_act, o), ""),
                "new_act": new_act, "new_section": "", "new_section_title": "",
                "mapping_type": "removed",
                "notes": "No counterpart listed in the BPRD correspondence table.",
            })

        stats[new_act] = {
            "table_rows": len(rows), "unparsed_rows": skipped,
            "new_sections_covered": len(new_to_old),
            "old_sections_mapped": len(mapped_old),
            "removed": len(removed),
        }

    os.makedirs(PROCESSED, exist_ok=True)
    out = os.path.join(PROCESSED, "mapping_table.csv")
    cols = ["old_act", "old_section", "old_section_title", "new_act", "new_section",
            "new_section_title", "mapping_type", "notes"]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    dpath = os.path.join(PROCESSED, "mapping_disagreements.csv")
    with open(dpath, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["new_act", "new_section", "bprd_old_sections",
                                           "other_source", "other_old_sections", "kind",
                                           "resolution"])
        w.writeheader()
        w.writerows(disagreements)

    counts = defaultdict(int)
    for r in all_rows:
        counts[r["mapping_type"]] += 1
    print(f"\nmapping_table.csv: {len(all_rows)} rows")
    for k in ("direct", "split", "merged", "new_provision", "removed"):
        print(f"  {k:<14} {counts[k]}")
    kinds = defaultdict(int)
    for d in disagreements:
        kinds[d["kind"]] += 1
    print(f"cross-checks performed: {dict(compared)}")
    print(f"disagreements flagged: {len(disagreements)} {dict(kinds)}")
    for act, s in stats.items():
        print(f"  {act}: {s}")

    with open(os.path.join(PROCESSED, "mapping_stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"counts": dict(counts), "per_act": stats,
                   "cross_checks": dict(compared),
                   "disagreements": len(disagreements),
                   "disagreement_kinds": dict(kinds)}, fh, indent=2)


def _key(sec: str):
    m = re.match(r"^(\d+)([A-Z]*)$", sec or "")
    return (int(m.group(1)), m.group(2)) if m else (10**9, sec or "")


if __name__ == "__main__":
    main()

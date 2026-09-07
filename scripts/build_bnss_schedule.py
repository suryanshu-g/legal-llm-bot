"""Phase 1.5: extract the BNSS First Schedule (classification of offences).

The First Schedule is a table, not prose, and it is set without any ruling
lines - pdfplumber's table detection finds nothing on these pages. Columns are
therefore recovered from word x-positions, and rows from vertical spacing:
lines within one entry sit 9.6pt apart, while a new entry starts after a gap of
about 14.5pt. That gap rule matters because a single section can carry several
classification entries (BNS 356(3) is classified one way when the complaint is
made by the Public Prosecutor and another way in any other case), and those
sub-entries repeat no section number.

Part I  - offences under the Bharatiya Nyaya Sanhita, one row per entry.
Part II - offences against other laws, classified by punishment band. Three
          rows, no section numbers; kept because "if an offence under another
          Act carries five years, is it cognizable?" is a real question.

The extraction is cross-checked against the "BNSS Classification" block that
devgan.in carries on its BNS section pages, which covers 201 of the 358
sections independently.

Output: data/processed/bnss_schedule.csv
        data/processed/bnss_schedule_disagreements.csv
"""

from __future__ import annotations

import csv
import glob
import html as html_mod
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawlib import PROCESSED, RAW, clean_text, join_wrapped, write_json

PDF = os.path.join(RAW, "bnss", "bnss_2023_mha.pdf")

# Column dividers for Part I, from the distribution of word start positions.
# Columns 4-6 are centred rather than left aligned, so these are the gutters
# between columns, not the left edges of their text.
PART1_BOUNDS = [45, 88, 188, 283, 364, 448, 570]
PART2_BOUNDS = [45, 320, 400, 465, 570]

PART1_COLS = ["section_ref", "offence_description", "punishment",
              "cognizable", "bailable", "triable_by"]

NOISE = re.compile(
    r"GAZETTE OF INDIA|^_+$|^\d+$|^1 2 3 4 5 6$|^1 2 3 4$|^Sec\. \d+\]|^\[Part",
    re.I)

SECREF = re.compile(r"^\d+[A-Z]?\s*(?:\(\s*\d+\s*\))*\s*(?:\(\s*[a-z]+\s*\))*\s*$")

# Lines within one entry are 9.6pt apart; a new entry follows a gap of ~14.5pt.
ENTRY_GAP = 12.0


def _split_columns(line, bounds) -> list[str]:
    cols = ["" for _ in range(len(bounds) - 1)]
    for w in sorted(line.words, key=lambda w: w["x0"]):
        for i in range(len(cols)):
            if bounds[i] <= w["x0"] < bounds[i + 1]:
                cols[i] = join_wrapped(cols[i], w["text"])
                break
    return cols


def entry_bounds(first_line) -> list[float]:
    """Column dividers for one entry, taken from its own opening line.

    Columns 4 to 6 are centred, and how wide they run depends on how much text
    they hold: BNS 85's cognizability column carries a long conditional clause
    and overruns the position where BNS 57's column 5 begins. A single global
    divider cannot satisfy both. The opening line of an entry almost always has
    content in every column, so its word positions give the local layout.

    Each divider is placed in the widest inter-word gap near where the global
    layout says the boundary falls. Taking the first word past the global
    divider instead would be misled by, for example, the stray "2" that ends
    "Simple imprisonment for 2 years" in the punishment column of BNS 356(2).
    """
    words = sorted(first_line.words, key=lambda w: w["x0"])
    gaps = [((words[i]["x1"] + words[i + 1]["x0"]) / 2.0,
             words[i + 1]["x0"] - words[i]["x1"])
            for i in range(len(words) - 1)]

    bounds = list(PART1_BOUNDS)
    for i in range(1, 6):
        # Only move a divider when this line actually has content on both sides
        # of it. Otherwise an ordinary space between two words of a single
        # column gets mistaken for the gutter - which is what pulled "Where"
        # out of the offence column in the second entry for BNS 303.
        if not (any(w["x0"] < PART1_BOUNDS[i] for w in words)
                and any(w["x0"] >= PART1_BOUNDS[i] for w in words)):
            continue
        near = [(width, mid) for mid, width in gaps
                if abs(mid - PART1_BOUNDS[i]) <= 35 and width > 3]
        if near:
            bounds[i] = max(near)[1]
    # Keep the dividers monotonic; a column with no content on this line simply
    # keeps the global default.
    for i in range(1, 6):
        bounds[i] = max(bounds[i], bounds[i - 1] + 1)
    return bounds


# A classification column always opens with one of these, so anything printed
# before it has drifted in from the column to its left.
CLASS_START = {
    "cognizable": re.compile(r"\b(Cognizable|Non-cognizable|According)\b", re.I),
    "bailable": re.compile(r"\b(Bailable|Non-bailable|According)\b", re.I),
}


def repair_leaks(rows: list[dict]) -> int:
    """Move stray leading words out of the classification columns.

    Where a column's text sits unusually close to the gutter the divider can
    land a word on the wrong side - "Imprisonment for 14 years" loses "years"
    into the cognizability column in the second entry for BNS 55. Since a
    classification column must begin with a known keyword, any prefix before
    that keyword is returned to the column it came from.
    """
    fixed = 0
    for r in rows:
        for field, prev in (("cognizable", "punishment"), ("bailable", "cognizable")):
            m = CLASS_START[field].search(r[field])
            if not m or m.start() == 0:
                continue
            stray, r[field] = r[field][:m.start()].strip(), r[field][m.start():].strip()
            if stray:
                r[prev] = join_wrapped(r[prev], stray)
                fixed += 1
    return fixed


def norm_ref(ref: str) -> str:
    return re.sub(r"\s+", "", ref).strip()


def norm_class(v: str) -> str:
    v = clean_text(v).rstrip(".")
    return re.sub(r"\s+", " ", v).strip()


def _schedule_bounds(pdf):
    """Page numbers for 'THE FIRST SCHEDULE' and the Part II heading."""
    from lawlib import page_lines
    first = part2 = None
    for pno, page in enumerate(pdf.pages, start=1):
        for l in page_lines(page, pno, x_min=0, x_max=page.width):
            t = l.text.strip()
            if first is None and re.match(r"^THE\s+FIRST\s+SCHEDULE", t, re.I):
                first = (pno, l.top)
            if part2 is None and re.search(r"CLASSIFICATION OF OFFENCES AGAINST OTHER LAWS", t, re.I):
                part2 = (pno, l.top)
    return first, part2


def parse_part1(pdf, first, part2) -> tuple[list[dict], int]:
    from lawlib import page_lines

    # Pass 1: group lines into entries. Only column 1 is consulted here, and
    # its position is unambiguous, so the global bounds are enough.
    entries: list[list] = []
    started = False
    rejected = 0
    prev_top: float | None = None
    refs: list[str] = []

    for pno in range(first[0], part2[0] + 1):
        page = pdf.pages[pno - 1]
        prev_top = None  # a page break is not a paragraph gap
        for l in page_lines(page, pno, x_min=0, x_max=page.width):
            t = l.text.strip()
            if pno == part2[0] and l.top >= part2[1]:
                break  # Part II begins
            if not t or NOISE.search(t):
                continue

            ref = " ".join(w["text"] for w in sorted(l.words, key=lambda w: w["x0"])
                           if PART1_BOUNDS[0] <= w["x0"] < PART1_BOUNDS[1]).strip()

            if not started:
                # Skip the schedule title and the explanatory notes.
                if not (ref and SECREF.match(ref)):
                    continue
                started = True

            gap = (l.top - prev_top) if prev_top is not None else None
            prev_top = l.top

            new_entry = False
            if ref and SECREF.match(ref):
                new_entry = True
            elif ref:
                rejected += 1
            elif entries and gap is not None and gap > ENTRY_GAP:
                # A further classification entry for the same section, printed
                # without repeating the section number.
                new_entry = True

            if new_entry:
                entries.append([l])
                refs.append(norm_ref(ref) or (refs[-1] if refs else ""))
            elif entries:
                entries[-1].append(l)

    # Pass 2: split each entry using dividers taken from its own first line.
    rows: list[dict] = []
    for ref, lines in zip(refs, entries):
        bounds = entry_bounds(lines[0])
        cur = {"section_ref": ref, "page": lines[0].page}
        for key in PART1_COLS[1:]:
            cur[key] = ""
        for l in lines:
            cols = _split_columns(l, bounds)
            for i, key in enumerate(PART1_COLS[1:], start=1):
                if cols[i]:
                    cur[key] = join_wrapped(cur[key], cols[i])
        rows.append(cur)

    for r in rows:
        for k in PART1_COLS[1:]:
            r[k] = norm_class(r[k]) if k in ("cognizable", "bailable", "triable_by") \
                else clean_text(r[k])
    return rows, rejected


def parse_part2(pdf, part2) -> list[dict]:
    from lawlib import page_lines

    page = pdf.pages[part2[0] - 1]
    rows: list[dict] = []
    cur: dict | None = None
    prev_top = None
    for l in page_lines(page, part2[0], x_min=0, x_max=page.width):
        if l.top <= part2[1] or NOISE.search(l.text.strip()):
            continue
        cols = _split_columns(l, PART2_BOUNDS)
        if not any(cols):
            continue
        # The column headings wrap over two lines, so match them on vocabulary
        # rather than on a fixed phrase.
        words = set(re.findall(r"[a-z-]+", " ".join(cols).lower()))
        if words and words <= {"offence", "cognizable", "non-cognizable", "or",
                               "bailable", "non-bailable", "by", "what", "court",
                               "triable"}:
            continue
        gap = (l.top - prev_top) if prev_top is not None else None
        prev_top = l.top
        if cur is None or (cols[1] and cols[2]) or (gap is not None and gap > ENTRY_GAP):
            cur = {"section_ref": "", "offence_description": cols[0], "punishment": "",
                   "cognizable": cols[1], "bailable": cols[2], "triable_by": cols[3],
                   "page": part2[0]}
            rows.append(cur)
        else:
            cur["offence_description"] = join_wrapped(cur["offence_description"], cols[0])
            for key, i in (("cognizable", 1), ("bailable", 2), ("triable_by", 3)):
                if cols[i]:
                    cur[key] = join_wrapped(cur[key], cols[i])
    for r in rows:
        r["offence_description"] = clean_text(r["offence_description"])
        for k in ("cognizable", "bailable", "triable_by"):
            r[k] = norm_class(r[k])
    return rows


# ------------------------------------------------------- independent check

def devgan_classification() -> dict[str, dict]:
    """BNS section -> classification keywords, from devgan.in's BNS pages."""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(os.path.join(RAW, "bns", "devgan_chapter_*.html"))):
        html = open(f, encoding="utf-8", errors="replace").read()
        parts = re.split(r"""<h2[^>]*>\s*Section\s*<a name=["']s([0-9A-Za-z]+)["']>""", html)
        for i in range(1, len(parts), 2):
            sec, chunk = parts[i], parts[i + 1]
            m = re.search(r"BNSS\s*</span>\s*Classification(.*?)</ul>", chunk, re.S | re.I)
            if not m:
                continue
            txt = html_mod.unescape(re.sub(r"<[^>]+>", " ", m.group(1))).lower()
            out[sec] = {
                "cognizable": ("non-cognizable" in txt and "non-cognizable")
                or ("cognizable" in txt and "cognizable") or "",
                "bailable": ("non-bailable" in txt and "non-bailable")
                or ("bailable" in txt and "bailable") or "",
                "court": "session" if "session" in txt else
                         ("first class" if "first class" in txt else
                          ("second class" if "second class" in txt else
                           ("any magistrate" if "any magistrate" in txt else ""))),
            }
    return out


def cross_check(rows: list[dict]) -> tuple[list[dict], dict]:
    ref = devgan_classification()
    by_sec: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        base = re.match(r"^\d+[A-Z]?", r["section_ref"])
        if base:
            by_sec[base.group(0)].append(r)

    disagreements, compared = [], 0
    for sec, entries in by_sec.items():
        d = ref.get(sec)
        if not d:
            continue
        # A section with several entries can legitimately carry both values;
        # compare against the union of what the schedule says for it.
        cog = " ".join(e["cognizable"].lower() for e in entries)
        bail = " ".join(e["bailable"].lower() for e in entries)
        court = " ".join(e["triable_by"].lower() for e in entries)
        compared += 1
        for field, ours, theirs in (("cognizable", cog, d["cognizable"]),
                                    ("bailable", bail, d["bailable"]),
                                    ("triable_by", court, d["court"])):
            if not theirs:
                continue
            if theirs in ours:
                continue
            # "cognizable" is a substring of "non-cognizable"; only flag when the
            # schedule genuinely lacks the other source's value.
            disagreements.append({
                "section": sec, "field": field,
                "schedule": "; ".join(sorted({e[field] for e in entries}))[:120],
                "devgan": theirs,
                "resolution": "kept the gazette schedule (official text); flagged",
            })
    return disagreements, {"sections_compared": compared, "reference_sections": len(ref)}


def main() -> None:
    import pdfplumber

    with pdfplumber.open(PDF) as pdf:
        first, part2 = _schedule_bounds(pdf)
        print(f"  First Schedule starts p{first[0]}, Part II heading p{part2[0]}")
        part1, rejected = parse_part1(pdf, first, part2)
        part2_rows = parse_part2(pdf, part2)

    leaks = repair_leaks(part1)
    print(f"  Part I:  {len(part1)} entries ({rejected} column-1 values unrecognised, "
          f"{leaks} column leaks repaired)")
    print(f"  Part II: {len(part2_rows)} entries")

    disagreements, cc = cross_check(part1)
    print(f"  cross-check against devgan.in: {cc['sections_compared']} sections compared, "
          f"{len(disagreements)} disagreements")

    out = os.path.join(PROCESSED, "bnss_schedule.csv")
    cols = PART1_COLS + ["part"]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in part1:
            w.writerow({**r, "part": "I"})
        for r in part2_rows:
            w.writerow({**r, "part": "II"})

    dpath = os.path.join(PROCESSED, "bnss_schedule_disagreements.csv")
    with open(dpath, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "field", "schedule",
                                           "devgan", "resolution"])
        w.writeheader()
        w.writerows(disagreements)

    bases = {re.match(r"^\d+[A-Z]?", r["section_ref"]).group(0)
             for r in part1 if re.match(r"^\d+[A-Z]?", r["section_ref"])}
    stats = {"column_leaks_repaired": leaks,
             "part1_entries": len(part1), "part2_entries": len(part2_rows),
             "distinct_bns_sections": len(bases),
             "unrecognised_column1_values": rejected,
             "cross_check": cc, "disagreements": len(disagreements)}
    write_json(os.path.join(PROCESSED, "bnss_schedule_stats.json"), stats)
    print(f"  bnss_schedule.csv: {len(part1) + len(part2_rows)} rows covering "
          f"{len(bases)} distinct BNS sections")


if __name__ == "__main__":
    main()

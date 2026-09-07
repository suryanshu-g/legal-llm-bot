"""Stage 1 of the pipeline: fetch source documents and parse them into sections.

Sources, in order of authority:

  * BNS / BNSS / BSA  - the official Ministry of Home Affairs gazette PDFs.
    These are the operative texts of the three new criminal codes.
  * IPC / CrPC / Indian Evidence Act - devgan.in (a long-standing bare-act
    reference) cross-checked against the civictech-India JSON corpus. The old
    codes are settled, long-public-domain texts, so two independent secondary
    compilations agreeing is adequate provenance.
  * devgan.in's BNS pages additionally carry a per-section "IPC Section N"
    back-reference, which build_mapping_table.py uses as one of its two
    independent mapping sources.

Bare act text is not copyrightable in India (Copyright Act 1957, s.52(1)(q)),
so everything written under data/raw is redistributable.

Output: data/raw/<act>/sections.json for each act.

Usage:  python scripts/scrape_acts.py [--only bns,ipc,...] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawlib import (RAW, clean_text, fetch, page_lines, render_smallcaps, roman_to_int,
                    strip_html, write_json)

# ---------------------------------------------------------------- source map

MHA_PDFS = {
    "bns": (
        "https://www.mha.gov.in/sites/default/files/2024-04/250883_english_01042024.pdf",
        "The Bharatiya Nyaya Sanhita, 2023 (Act 45 of 2023)",
        358,
    ),
    "bnss": (
        "https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf",
        "The Bharatiya Nagarik Suraksha Sanhita, 2023 (Act 46 of 2023)",
        531,
    ),
    "bsa": (
        "https://www.mha.gov.in/sites/default/files/2024-04/250882_english_01042024_0.pdf",
        "The Bharatiya Sakshya Adhiniyam, 2023 (Act 47 of 2023)",
        170,
    ),
}

# devgan.in slugs -> (number of chapter pages, long act name)
DEVGAN_ACTS = {
    "bns": (20, "Bharatiya Nyaya Sanhita, 2023"),
    "ipc": (23, "Indian Penal Code, 1860"),
    "crpc": (39, "Code of Criminal Procedure, 1973"),
    "iea": (11, "Indian Evidence Act, 1872"),
}

CIVICTECH = {
    "ipc": "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main/ipc.json",
    "crpc": "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main/crpc.json",
    "iea": "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main/iea.json",
}

# Official old-to-new concordances published by the Bureau of Police Research
# and Development (BPRD), an MHA body. Each is a "Correspondence Table and
# Comparison Summary" giving, per new-code section, the old-code section it
# derives from plus a prose note on what changed.
BPRD_CONCORDANCE = {
    "bns_to_ipc": "https://bprd.nic.in/uploads/pdf/COMPARISON%20SUMMARY%20BNS%20to%20IPC%20.pdf",
    "bnss_to_crpc": "https://bprd.nic.in/uploads/pdf/Comparison%20summary%20BNSS%20to%20CrPC.pdf",
    "bsa_to_iea": "https://bprd.nic.in/uploads/pdf/Comparison%20Summary%20BSA%20to%20IEA.pdf",
}

# A second, independent concordance used to cross-check BPRD: the UP Police
# comparative table of BNS against IPC.
UPPOLICE_CONCORDANCE = (
    "https://uppolice.gov.in/site/writereaddata/siteContent/Three%20New%20Major%20Acts/"
    "202406281710564823BNS_IPC_Comparative.pdf"
)

FOLDER = {"iea": "evidence_act"}

# Page furniture that must never be mistaken for act text.
NOISE = re.compile(
    r"^(_+|\d+\s*$|.*THE GAZETTE OF INDIA.*|\[?PART\s+II.*|SEC\.\s*\d.*|"
    r"MINISTRY OF LAW AND JUSTICE.*|\(Legislative Department\).*)$",
    re.I,
)

# The schedules that follow the last enacting section.
SCHEDULE = re.compile(r"^THE\s+(?:FIRST|SECOND|THIRD)?\s*SCHEDULE(?![A-Za-z])", re.I)


# ------------------------------------------------------------- PDF act parse

def _margin_blocks(margin_lines):
    """Group margin lines into heading blocks.

    A section heading is printed as a narrow stack of lines in the margin. The
    margin is set tighter than the body: lines *within* one heading sit about
    9.2pt apart, while consecutive headings are separated by about 13.3pt
    because the next heading starts on its section's first body baseline. The
    threshold has to fall between the two or headings merge into their
    neighbours.
    """
    blocks = []
    for ln in sorted(margin_lines, key=lambda l: l.top):
        if blocks and ln.top - blocks[-1][-1].top <= 11.5:
            blocks[-1].append(ln)
        else:
            blocks.append([ln])
    return [
        {"top": b[0].top, "text": clean_text(" ".join(l.text for l in b))}
        for b in blocks
    ]


def _looks_like_citation(text: str) -> bool:
    """Marginal act citations ('21 of 2000.') sit in the same margin as headings."""
    return bool(re.fullmatch(r"[\d\s,and]+of\s+\d{4}\.?", text.strip(), re.I))


def _chapter_title(page, chapter_top: float, width: float) -> str | None:
    """Read the small-caps chapter title printed under a 'CHAPTER N' line."""
    words = [
        w for w in page.extract_words(x_tolerance=1.2, y_tolerance=2.2, extra_attrs=["size"])
        if 110 <= w["x0"] <= width - 110 and chapter_top + 6 < w["top"] < chapter_top + 36
    ]
    if not words:
        return None

    # Cluster into visual lines loosely enough to keep both small-caps baselines
    # of the same line together.
    rows: list[list] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if rows and w["top"] - rows[-1][0]["top"] <= 5.0:
            rows[-1].append(w)
        else:
            rows.append([w])

    parts = []
    for row in rows:
        rendered = clean_text(render_smallcaps(row))
        # Stop at the first line that is not part of the all-caps title, e.g. a
        # mixed-case sub-heading or the opening of section text.
        if not rendered or not re.fullmatch(r"[A-Z0-9 ,.\-–—'()&]+", rendered):
            break
        parts.append(rendered)

    if not parts:
        return None
    title = clean_text(" ".join(parts)).rstrip(".")
    return re.sub(r"^OF\s+", "", title, flags=re.I)


def parse_act_pdf(path: str, act: str, expected: int, source_url: str) -> list[dict]:
    import pdfplumber

    sections: list[dict] = []
    pending_heads: list[dict] = []
    chapter_no, chapter_title = None, None
    in_schedules = False
    expect = 1
    cur: dict | None = None

    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            width = page.width
            body = page_lines(page, pno, x_min=110, x_max=width - 110)
            margin = [
                l for l in
                page_lines(page, pno, x_max=109) + page_lines(page, pno, x_min=width - 109)
                if not NOISE.match(l.text) and not _looks_like_citation(l.text)
            ]
            heads = [h for h in _margin_blocks(margin) if not _looks_like_citation(h["text"])]

            for ln in body:
                text = ln.text.strip()
                if not text or NOISE.match(text):
                    continue

                # --- end of the enacting sections ---------------------------
                # The schedules follow the last section. Without this guard the
                # final section absorbs all of them - the BNSS First Schedule
                # alone runs to well over a hundred thousand characters.
                if SCHEDULE.match(text):
                    cur = None
                    in_schedules = True
                    continue
                if in_schedules:
                    continue

                # --- chapter boundaries -------------------------------------
                # Kerning sometimes closes the gap, giving 'CHAPTERIV'.
                m = re.fullmatch(r"CHAPTER\s*([IVXLC]+)", text)
                if m:
                    chapter_no = roman_to_int(m.group(1))
                    chapter_title = _chapter_title(page, ln.top, width)
                    continue

                # --- section starts -----------------------------------------
                m = re.match(r"^(\d+)\.\s*", text)
                if m and int(m.group(1)) == expect:
                    num = int(m.group(1))
                    # Nearest unconsumed margin heading on this page.
                    title = ""
                    best, bestd = None, 1e9
                    for h in heads:
                        if h.get("used"):
                            continue
                        d = abs(h["top"] - ln.top)
                        if d < bestd:
                            best, bestd = h, d
                    if best is not None and bestd <= 12:
                        best["used"] = True
                        title = best["text"]
                    elif pending_heads:
                        title = pending_heads.pop(0)["text"]

                    cur = {
                        "act": act.upper(),
                        "section": str(num),
                        "title": title.rstrip("."),
                        "chapter_no": chapter_no,
                        "chapter_title": chapter_title,
                        "text_parts": [text],
                        "page": pno,
                        "source_url": source_url,
                    }
                    sections.append(cur)
                    expect += 1
                    continue

                if cur is not None:
                    cur["text_parts"].append(text)

            # Headings whose section starts on the next page.
            pending_heads = [h for h in heads if not h.get("used")]

    out = []
    for s in sections:
        s["text"] = clean_text(" ".join(s.pop("text_parts")))
        out.append(s)
    print(f"  [{act}] parsed {len(out)}/{expected} sections from PDF")
    return out


# ---------------------------------------------------------- devgan.in parse

def parse_devgan(slug: str, n_chapters: int, force: bool = False) -> list[dict]:
    """Parse devgan.in chapter pages into sections.

    Each chapter page holds every section of that chapter as `<a name="sNNN">`
    anchors followed by a `<div class='sectxt'>` body.
    """
    sections: list[dict] = []
    folder = FOLDER.get(slug, slug)
    for ch in range(1, n_chapters + 1):
        url = f"https://devgan.in/{slug}/chapter_{ch:02d}.php"
        dest = os.path.join(RAW, folder, f"devgan_chapter_{ch:02d}.html")
        try:
            html = fetch(url, dest, force=force).decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{slug}] chapter {ch}: {exc}")
            continue

        cm = re.search(r"<h1>\s*Chapter\s+([IVXLC0-9]+)\s*</h1>\s*<h2[^>]*>(.*?)</h2>",
                       html, re.S | re.I)
        chap_title = strip_html(cm.group(2)) if cm else None

        # Split the page on the per-section headings. The BNS pages use
        # <h2 class="subClose"> and an en-dash; the older acts use a plain <h2>
        # and a ":-" separator.
        parts = re.split(r"""<h2[^>]*>\s*Section\s*<a name=["']s([0-9A-Za-z]+)["']>""", html)
        for i in range(1, len(parts), 2):
            num = parts[i]
            chunk = parts[i + 1]
            # The chunk opens with the section number still inside the anchor.
            # Trust that displayed number over the anchor's name attribute: the
            # IPC pages give both section 29 and section 29A the anchor "s29",
            # which would otherwise collapse the two into one record.
            nm = re.match(r"\s*([0-9]+[A-Za-z]{0,2})\s*</a>", chunk)
            if nm:
                num = nm.group(1)
            tm = re.match(r"[^<]*</a>\s*(?::-|&#8211;|&ndash;|–|-)?\s*(.*?)</h2>", chunk, re.S)
            title = strip_html(tm.group(1)) if tm else ""
            bm = re.search(r"""<div class=["']sectxt["'][^>]*>(.*?)</div>""", chunk, re.S)
            body_html = bm.group(1) if bm else ""

            # devgan prints the corresponding old/new section as a leading link.
            xref = re.search(
                r'<span class="caps">(IPC|CRPC|BNS|BNSS|BSA)</span>\s*Section\s*'
                r'<a href="[^"]*?/(?:ipc|crpc|iea|bns)/section/([0-9A-Za-z]+)/?"',
                body_html, re.I)

            sections.append({
                "act": slug.upper(),
                "section": num,
                "title": title.rstrip("."),
                "chapter_no": ch,
                "chapter_title": chap_title,
                "text": strip_html(body_html),
                "xref_act": xref.group(1).upper() if xref else None,
                "xref_section": xref.group(2) if xref else None,
                "source_url": f"https://devgan.in/{slug}/section/{num}/",
            })
    print(f"  [{slug}] devgan.in: {len(sections)} sections from {n_chapters} chapters")
    return sections


def parse_civictech(slug: str, url: str, force: bool = False) -> list[dict]:
    folder = FOLDER.get(slug, slug)
    dest = os.path.join(RAW, folder, f"civictech_{slug}.json")
    raw = json.loads(fetch(url, dest, force=force).decode("utf-8", "replace"))
    rows = raw if isinstance(raw, list) else raw.get("sections", [])
    out = []
    for r in rows:
        num = str(r.get("Section") or r.get("section") or "").strip()
        if not num:
            continue
        out.append({
            "act": slug.upper(),
            "section": num,
            "title": clean_text(str(r.get("section_title") or r.get("title") or "")).rstrip("."),
            "text": clean_text(str(r.get("section_desc") or r.get("description") or "")),
            "chapter_no": r.get("chapter"),
            "chapter_title": clean_text(str(r.get("chapter_title") or "")) or None,
            "source_url": url,
        })
    print(f"  [{slug}] civictech: {len(out)} sections")
    return out


# ----------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated act keys")
    ap.add_argument("--force", action="store_true", help="re-download cached sources")
    args = ap.parse_args()
    only = {a.strip() for a in args.only.split(",") if a.strip()}

    def want(k: str) -> bool:
        return not only or k in only

    print("== new codes: official MHA gazette PDFs ==")
    for act, (url, name, expected) in MHA_PDFS.items():
        if not want(act):
            continue
        pdf_path = os.path.join(RAW, act, f"{act}_2023_mha.pdf")
        fetch(url, pdf_path, binary=True, force=args.force)
        secs = parse_act_pdf(pdf_path, act, expected, url)
        write_json(os.path.join(RAW, act, "sections.json"),
                   {"act": act.upper(), "act_name": name, "source_url": url,
                    "expected_sections": expected, "sections": secs})

    print("== old codes + BNS cross-check: devgan.in ==")
    for slug, (n_ch, name) in DEVGAN_ACTS.items():
        if not want(slug):
            continue
        secs = parse_devgan(slug, n_ch, force=args.force)
        write_json(os.path.join(RAW, FOLDER.get(slug, slug), "devgan_sections.json"),
                   {"act": slug.upper(), "act_name": name,
                    "source_url": f"https://devgan.in/{slug}/", "sections": secs})

    if not only or "mapping" in only:
        print("== official concordance tables ==")
        for name, url in BPRD_CONCORDANCE.items():
            fetch(url, os.path.join(RAW, "mapping", f"bprd_{name}.pdf"),
                  binary=True, force=args.force)
            print(f"  [mapping] bprd_{name}.pdf")
        fetch(UPPOLICE_CONCORDANCE,
              os.path.join(RAW, "mapping", "uppolice_bns_ipc.pdf"),
              binary=True, force=args.force)
        print("  [mapping] uppolice_bns_ipc.pdf")

    print("== old codes cross-check: civictech-India JSON ==")
    for slug, url in CIVICTECH.items():
        if not want(slug):
            continue
        secs = parse_civictech(slug, url, force=args.force)
        write_json(os.path.join(RAW, FOLDER.get(slug, slug), "civictech_sections.json"),
                   {"act": slug.upper(), "source_url": url, "sections": secs})

    print("done.")


if __name__ == "__main__":
    main()

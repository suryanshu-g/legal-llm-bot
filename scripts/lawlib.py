"""Shared helpers for the legal-llm-bot data pipeline.

Covers the three things every stage of the pipeline needs: polite cached HTTP
fetching with provenance logging, PDF line reconstruction that keeps x/y
coordinates (the gazette PDFs put section headings in the page margin, so
position is the only way to tell a heading from body text), and small text
normalisation utilities.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36 (legal-llm-bot academic capstone; contact via repo)"
)

# Seconds between requests to the same host. Deliberately conservative: the
# small legal-reference sites we read are hobby-run and we are not in a hurry.
CRAWL_DELAY = 1.5
_last_hit: dict[str, float] = {}

MANIFEST_PATH = os.path.join(RAW, "source_manifest.json")


def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_manifest(man: dict) -> None:
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)


def fetch(url: str, dest: str, *, binary: bool = False, force: bool = False) -> bytes:
    """Download `url` to `dest`, reusing the cached copy unless `force`.

    Every fetch is recorded in data/raw/source_manifest.json with the URL, the
    retrieval date and a sha256 of the bytes, so any downstream claim about the
    data can be traced back to a specific retrieval of a specific document.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 0:
        with open(dest, "rb") as fh:
            return fh.read()

    host = urllib.parse.urlsplit(url).netloc
    wait = CRAWL_DELAY - (time.time() - _last_hit.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    _last_hit[host] = time.time()

    with open(dest, "wb") as fh:
        fh.write(data)

    man = _load_manifest()
    man[os.path.relpath(dest, RAW).replace("\\", "/")] = {
        "url": url,
        "retrieved": time.strftime("%Y-%m-%d"),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    _save_manifest(man)
    return data


import urllib.parse  # noqa: E402  (kept next to its only use above)


# ---------------------------------------------------------------- PDF layout

@dataclass
class Line:
    """One visual line of a PDF page, with the geometry we need to classify it."""

    page: int
    top: float
    x0: float
    x1: float
    text: str
    words: list = field(default_factory=list, repr=False)


def page_lines(page, page_no: int, *, y_tol: float = 2.2, x_tol: float = 1.2,
               x_min: float = 0.0, x_max: float = 1e9) -> list[Line]:
    """Rebuild visual lines from a pdfplumber page, optionally within an x band.

    pdfplumber's own extract_text() drops inter-word spaces on these gazette
    PDFs, so we go through extract_words() (which is reliable) and re-assemble
    lines ourselves by clustering words on their vertical position.

    The x band matters: these PDFs print section headings in the outer page
    margin, on the same baselines as the body text. Rebuilding lines across the
    full page width would splice a heading onto the body line beside it, so
    callers pass the body band and the margin band separately.
    """
    words = [w for w in page.extract_words(x_tolerance=x_tol, y_tolerance=y_tol,
                                           extra_attrs=["size"])
             if x_min <= w["x0"] <= x_max]
    if not words:
        return []

    rows: list[list] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if rows and abs(w["top"] - rows[-1][0]["top"]) <= y_tol:
            rows[-1].append(w)
        else:
            rows.append([w])

    out = []
    for row in rows:
        row.sort(key=lambda w: w["x0"])
        out.append(
            Line(
                page=page_no,
                top=row[0]["top"],
                x0=min(w["x0"] for w in row),
                x1=max(w["x1"] for w in row),
                text=" ".join(w["text"] for w in row),
                words=row,
            )
        )
    return out


def render_smallcaps(words) -> str:
    """Reassemble a genuine small-caps line into normal text.

    The gazette sets chapter titles in small caps, which the PDF encodes as two
    interleaved font sizes on slightly different baselines: each word's initial
    is a full-size capital and the remainder of the word is a smaller capital.
    Extracted naively this yields 'O F OFFENCES ... AND G OVERNMENT STAMPS'.

    Sorting by x and re-joining recovers the text, given two rules: a full-size
    single letter is a word initial and binds to the token after it, and
    punctuation binds to whatever sits beside it.
    """
    ws = sorted(words, key=lambda w: w["x0"])
    if not ws:
        return ""
    body_size = min(w.get("size", 10.0) for w in ws)
    out: list[str] = []
    glue = False  # join the next token to the previous one without a space
    for w in ws:
        tok = w["text"]
        big = w.get("size", body_size) > body_size + 0.5
        is_punct = bool(re.fullmatch(r"[^\w]+", tok))

        if not out:
            out.append(tok)
        elif glue or is_punct:
            out[-1] += tok
        else:
            out.append(tok)

        # A full-size single letter is a word initial; so is a trailing hyphen.
        glue = (big and len(tok) == 1 and tok.isalpha()) or tok.endswith("-")
    return " ".join(out)


ROMAN = {
    "I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000,
}


def roman_to_int(s: str) -> int | None:
    s = s.strip().upper()
    if not s or any(ch not in ROMAN for ch in s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        val = ROMAN[ch]
        total += -val if val < prev else val
        prev = max(prev, val)
    return total


# ------------------------------------------------------------ text cleaning

def clean_text(s: str) -> str:
    """Normalise whitespace and the typographic characters the gazette uses."""
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2014\u2014", "\u2014")
    s = re.sub(r"[ \t\u00a0]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def strip_html(s: str) -> str:
    """Turn a fragment of devgan.in markup into readable plain text."""
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<li[^>]*>", "\n- ", s, flags=re.I)
    s = re.sub(r"</(p|div|ol|ul|tr|h\d)>", "\n", s, flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    import html as _html

    s = _html.unescape(s)
    return clean_text(s)


def write_jsonl(path: str, rows) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

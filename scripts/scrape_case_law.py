"""Stage 3: harvest case-law summaries from Indian Kanoon.

Method and its limits
---------------------
For each offence or procedural topic we run one Indian Kanoon search anchored
on the *old* code section. That is deliberate: the new codes only took effect on
1 July 2024, so almost all reported case law still cites IPC/CrPC/Evidence Act
numbering. The new-code equivalent is attached from the concordance we built in
stage 2, which is exactly the old-to-new bridge the bot is meant to provide.

Holdings are summarised **extractively**: sentences are selected from the
judgment's own text using markers courts use when stating a conclusion ("we are
of the view", "it is well settled", "the appeal is allowed"). Nothing is
paraphrased or generated, because an invented holding in a legal dataset is
worse than no holding at all. Each record carries `summary_method` so this is
visible downstream. There is no copyright in a judgment of a court in India
(Copyright Act 1957, s.52(1)(q)); we nonetheless store only a short extract and
always link to the source.

Politeness: every request is checked against Indian Kanoon's robots.txt - which
is largely a denylist of individual judgment IDs it has been asked to withhold -
and requests are rate limited.

Output: data/raw/case_law/case_summaries.json
"""

from __future__ import annotations

import argparse
import csv
import html as html_mod
import json
import os
import re
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawlib import (CRAWL_DELAY, PROCESSED, RAW, clean_text, fetch, robots_allows,
                    write_json)

BASE = "https://indiankanoon.org"

# topic key -> (label, old-act, [old sections], search phrase)
TOPICS = {
    "murder": ("Murder and culpable homicide", "IPC", ["302", "300", "304"],
               "section 302 IPC murder conviction"),
    "culpable_homicide": ("Culpable homicide not amounting to murder", "IPC", ["304"],
                          "section 304 IPC culpable homicide not amounting to murder"),
    "theft": ("Theft", "IPC", ["378", "379"], "section 379 IPC theft dishonestly moves"),
    "robbery_dacoity": ("Robbery and dacoity", "IPC", ["390", "392", "395"],
                        "section 392 395 IPC robbery dacoity"),
    "cheating": ("Cheating and fraud", "IPC", ["415", "420"],
                 "section 420 IPC cheating dishonestly inducing delivery of property"),
    "criminal_breach_trust": ("Criminal breach of trust", "IPC", ["405", "406"],
                              "section 406 IPC criminal breach of trust entrustment"),
    "cruelty_498a": ("Cruelty to a married woman", "IPC", ["498A"],
                     "section 498A IPC cruelty husband relative"),
    "dowry_death": ("Dowry death", "IPC", ["304B"],
                    "section 304B IPC dowry death presumption"),
    "rape": ("Rape and sexual offences", "IPC", ["375", "376"],
             "section 376 IPC rape consent conviction"),
    "hurt": ("Hurt and grievous hurt", "IPC", ["319", "320", "323", "325"],
             "section 323 325 IPC voluntarily causing hurt grievous hurt"),
    "criminal_intimidation": ("Criminal intimidation", "IPC", ["503", "506"],
                              "section 506 IPC criminal intimidation threat"),
    "defamation": ("Defamation", "IPC", ["499", "500"],
                   "section 499 500 IPC defamation imputation reputation"),
    "abetment_suicide": ("Abetment of suicide", "IPC", ["306", "107"],
                         "section 306 IPC abetment of suicide instigation"),
    "criminal_conspiracy": ("Criminal conspiracy", "IPC", ["120A", "120B"],
                            "section 120B IPC criminal conspiracy agreement"),
    "forgery": ("Forgery and false documents", "IPC", ["463", "465", "468"],
                "section 468 IPC forgery for purpose of cheating"),
    "cheating_personation": ("Cheating by personation (identity fraud)", "IPC", ["416", "419"],
                             "section 419 IPC cheating by personation"),
    "unlawful_assembly": ("Unlawful assembly and rioting", "IPC", ["141", "147", "149"],
                          "section 149 IPC unlawful assembly common object"),
    "criminal_trespass": ("Criminal trespass and house-breaking", "IPC", ["441", "448", "457"],
                          "section 448 IPC criminal trespass house trespass"),
    "extortion": ("Extortion", "IPC", ["383", "384"],
                  "section 384 IPC extortion putting in fear of injury"),
    "mischief": ("Mischief and damage to property", "IPC", ["425", "427"],
                 "section 427 IPC mischief causing damage"),
    "anticipatory_bail": ("Anticipatory bail", "CRPC", ["438"],
                          "section 438 CrPC anticipatory bail arrest apprehension"),
    "regular_bail": ("Bail in non-bailable offences", "CRPC", ["437", "439"],
                     "section 439 CrPC bail non-bailable offence discretion"),
    "fir_registration": ("Registration of FIR", "CRPC", ["154"],
                         "section 154 CrPC registration of FIR cognizable offence"),
    "quashing": ("Quashing of criminal proceedings", "CRPC", ["482"],
                 "section 482 CrPC quashing FIR inherent powers High Court"),
    "default_bail": ("Default or statutory bail", "CRPC", ["167"],
                     "section 167(2) CrPC default bail investigation ninety days"),
    "electronic_evidence": ("Electronic evidence and certificates", "IEA", ["65A", "65B"],
                            "section 65B Evidence Act electronic record certificate admissibility"),
    "confession_police": ("Confessions to police and discovery", "IEA", ["25", "26", "27"],
                          "section 27 Evidence Act discovery confession police custody"),
    "dying_declaration": ("Dying declarations", "IEA", ["32"],
                          "section 32 Evidence Act dying declaration admissibility"),
    "burden_of_proof": ("Burden of proof", "IEA", ["101", "102", "106"],
                        "section 106 Evidence Act burden of proof facts especially within knowledge"),
    "circumstantial_evidence": ("Circumstantial evidence", "IEA", ["3", "114"],
                                "circumstantial evidence chain of circumstances Evidence Act conviction"),
}

# Phrases courts use when stating what they actually decide.
HOLDING_MARKERS = re.compile(
    r"\b(we are of the (?:considered )?(?:view|opinion)|we hold|it is well settled|"
    r"the settled position|we are satisfied|we find no|in our (?:considered )?(?:view|opinion)|"
    r"the appeal is (?:allowed|dismissed)|the conviction is|we set aside|"
    r"is hereby quashed|cannot be sustained|it is trite law|the law is well settled|"
    r"we accordingly|we therefore hold|must be held)\b", re.I)

SECTION_CITE = re.compile(
    r"[Ss]ections?\s+(\d+[A-Z]{0,2})(?:\s*\([^)]{1,8}\))?\s*(?:of\s+(?:the\s+)?)?"
    r"(IPC|I\.P\.C|Indian Penal Code|CrPC|Cr\.P\.C|Code of Criminal Procedure|"
    r"Evidence Act|Indian Evidence Act|BNS|BNSS|BSA)", re.I)

ACT_CANON = {
    "ipc": "IPC", "i.p.c": "IPC", "indian penal code": "IPC",
    "crpc": "CRPC", "cr.p.c": "CRPC", "code of criminal procedure": "CRPC",
    "evidence act": "IEA", "indian evidence act": "IEA",
    "bns": "BNS", "bnss": "BNSS", "bsa": "BSA",
}


def _text(fragment: str) -> str:
    fragment = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", fragment, flags=re.S | re.I)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return clean_text(html_mod.unescape(fragment))


def search(query: str, doctype: str, cache_dir: str) -> list[dict]:
    """One Indian Kanoon search; returns [{doc_id, title}]."""
    q = f"{query} doctypes: {doctype}"
    url = f"{BASE}/search/?formInput={urllib.parse.quote(q)}"
    if not robots_allows(url):
        print(f"    robots.txt disallows search for {doctype}")
        return []
    key = re.sub(r"[^a-z0-9]+", "_", q.lower())[:80]
    dest = os.path.join(cache_dir, f"search_{key}.html")
    try:
        page = fetch(url, dest).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"    search failed: {exc}")
        return []

    out, seen = [], set()
    for m in re.finditer(r'<article class="result".*?</article>', page, re.S):
        block = m.group(0)
        a = re.search(r'href="/doc(?:fragment)?/(\d+)/?[^"]*"[^>]*>(.*?)</a>', block, re.S)
        if not a:
            continue
        doc_id = a.group(1)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append({"doc_id": doc_id, "title": _text(a.group(2))})
    return out


def fetch_judgment(doc_id: str, cache_dir: str) -> dict | None:
    url = f"{BASE}/doc/{doc_id}/"
    if not robots_allows(url):
        print(f"    robots.txt disallows /doc/{doc_id}/ - skipped")
        return None
    dest = os.path.join(cache_dir, f"doc_{doc_id}.html")
    try:
        page = fetch(url, dest).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"    doc {doc_id} failed: {exc}")
        return None

    tm = re.search(r'<h2 class="doc_title">(.*?)</h2>', page, re.S)
    title = _text(tm.group(1)) if tm else ""
    cm = re.search(r'<h[23] class="docsource_main"[^>]*>(.*?)</h[23]>', page, re.S)
    court = _text(cm.group(1)) if cm else ""
    bench = ""
    bm = re.search(r'<h3 class="doc_bench">(.*?)</h3>', page, re.S)
    if bm:
        bench = _text(bm.group(1)).replace("Bench:", "").strip()

    body = re.search(r'<div class="judgments">(.*)$', page, re.S)
    text = _normalise_judgment(_text(body.group(1))) if body else ""

    ym = re.search(r"on\s+\d{0,2}\s*[A-Za-z]*\s*,?\s*(\d{4})\s*$", title) or \
        re.search(r"(\d{4})\s*$", title)
    year = int(ym.group(1)) if ym else None

    return {"doc_id": doc_id, "case_name": re.sub(r"\s+on\s+\d.*$", "", title).strip(),
            "full_title": title, "court": court, "bench": bench, "year": year,
            "url": url, "text": text}


def cited_sections(text: str) -> dict[str, list[str]]:
    found: dict[str, set[str]] = {}
    for m in SECTION_CITE.finditer(text):
        act = ACT_CANON.get(m.group(2).lower().rstrip("."))
        if not act:
            continue
        found.setdefault(act, set()).add(m.group(1).upper())
    return {k: sorted(v, key=lambda s: (int(re.match(r"\d+", s).group()), s))
            for k, v in found.items()}


# Artefacts of the scanned/e-filed originals that survive into the text layer.
FURNITURE = re.compile(
    r"Page\s+\d+\s+of\s+\d+|Signature\s+Not\s+Verified|Digitally\s+signed(?:\s+by)?[^.]{0,60}|"
    r"Date:\s*\d{4}\.\d{2}\.\d{2}[^ ]*|Reason:\s*$|"
    r"\b(?:SLP|S\.L\.P|Criminal Appeal|Crl\.?\s*A)\s*\(?(?:Crl|Criminal|C)?\.?\)?\s*"
    r"(?:No)?\.?\s*\(?s?\)?\.?\s*\d[\d/ ]*(?:of\s+\d{4})?",
    re.I)


def _normalise_judgment(text: str) -> str:
    """Flatten the hard line wrapping and drop e-filing page furniture."""
    text = FURNITURE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _usable_sentence(s: str) -> bool:
    """Reject header fragments, citation strings and OCR noise."""
    if not (60 <= len(s) <= 400):
        return False
    letters = sum(c.isalpha() for c in s)
    digits = sum(c.isdigit() for c in s)
    if letters / len(s) < 0.62 or digits / len(s) > 0.10:
        return False
    return not FURNITURE.search(s)


def holding_summary(text: str, sections: list[str], max_chars: int = 700) -> str:
    """Pull the sentences in which the court states its conclusion."""
    # Judgments open with cause-title boilerplate; the reasoning is later on.
    tail = text[len(text) // 3:]
    sentences = re.split(r"(?<=[.;])\s+(?=[A-Z(])", tail)
    scored = []
    for i, s in enumerate(sentences):
        s = s.strip()
        if not _usable_sentence(s) or not HOLDING_MARKERS.search(s):
            continue
        # Prefer conclusions that actually engage the provision in question.
        score = 1.0 + 2.0 * sum(1 for sec in sections if re.search(rf"\b{sec}\b", s))
        scored.append((score, i, s))
    if not scored:
        return ""
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked = sorted(scored[:3], key=lambda t: t[1])
    out = " ".join(s for _, _, s in picked)
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + " ..."
    return clean_text(out)


def load_mapping() -> dict[tuple[str, str], list[tuple[str, str]]]:
    """(old_act, old_section) -> [(new_act, new_section)]"""
    path = os.path.join(PROCESSED, "mapping_table.csv")
    out: dict[tuple[str, str], list[tuple[str, str]]] = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["old_section"] and r["new_section"]:
            out.setdefault((r["old_act"], r["old_section"]),
                           []).append((r["new_act"], r["new_section"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=8, help="judgments to keep per topic")
    ap.add_argument("--only", default="", help="comma-separated topic keys")
    args = ap.parse_args()
    only = {t.strip() for t in args.only.split(",") if t.strip()}

    cache_dir = os.path.join(RAW, "case_law")
    os.makedirs(cache_dir, exist_ok=True)
    mapping = load_mapping()

    records: list[dict] = []
    per_topic: dict[str, int] = {}

    for key, (label, old_act, old_secs, query) in TOPICS.items():
        if only and key not in only:
            continue
        print(f"[{key}] {label}")
        hits: list[dict] = []
        for doctype in ("supremecourt", "highcourts"):
            if len(hits) >= args.per_topic:
                break
            for h in search(query, doctype, cache_dir):
                if h["doc_id"] not in {x["doc_id"] for x in hits}:
                    hits.append(h)
                if len(hits) >= args.per_topic:
                    break
            time.sleep(CRAWL_DELAY)

        kept = 0
        for h in hits:
            j = fetch_judgment(h["doc_id"], cache_dir)
            if not j or len(j["text"]) < 1200:
                continue
            cites = cited_sections(j["text"])
            summary = holding_summary(j["text"], old_secs)
            if not summary:
                continue

            new_equiv = []
            for s in old_secs:
                for na, ns in mapping.get((old_act, s), []):
                    if (na, ns) not in new_equiv:
                        new_equiv.append((na, ns))

            records.append({
                "topic": key,
                "topic_label": label,
                "case_name": j["case_name"],
                "court": j["court"],
                "year": j["year"],
                "bench": j["bench"],
                "old_act": old_act,
                "old_sections": old_secs,
                "new_sections": [f"{a} {s}" for a, s in new_equiv],
                "sections_cited_in_judgment": cites,
                "holding_summary": summary,
                "summary_method": "extractive (sentences selected from the judgment text)",
                "source": "Indian Kanoon",
                "source_url": j["url"],
            })
            kept += 1
        per_topic[key] = kept
        print(f"    kept {kept} of {len(hits)} judgments")

    write_json(os.path.join(cache_dir, "case_summaries.json"),
               {"source": "Indian Kanoon (indiankanoon.org)",
                "method": "extractive holding summaries; see scripts/scrape_case_law.py",
                "per_topic": per_topic, "cases": records})
    print(f"\n{len(records)} case summaries across "
          f"{len([k for k, v in per_topic.items() if v])} topics")


if __name__ == "__main__":
    main()

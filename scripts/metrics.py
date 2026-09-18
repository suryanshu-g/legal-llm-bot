"""The evaluation metrics, in one place.

Lifted verbatim from the training notebooks so that a number produced here and a
number produced there mean the same thing. Phase 4 compares this project's bot
against a general-purpose LLM, and that comparison is worthless if the two sides
are scored by different code.

Three families of metric:

* **exact match and token F1** - SQuAD-style, after normalising case, punctuation
  and articles. Token F1 is reported because it is the conventional number, not
  because it is a good measure here: an answer naming the wrong section still
  overlaps the gold answer almost completely.

* **citation metrics** - the domain metric that actually matters. `all_citations
  _present` asks whether every provision the gold answer cites appears in the
  prediction; `citation_exact` requires the two sets to match, which is what
  catches an answer that names the right old section and invents a new one.

* **`n_with_citations`** - some question types (transition, scope) have no
  statutory citation in the gold answer, so the citation metrics are undefined
  there and reported as None rather than as a misleading zero.

Note the deliberate duplication of `extract_refs`: `retrieve.py` has a version
with an extra fallback that helps a *query* find a passage, which would be wrong
in a metric. This module keeps the stricter one the notebooks score with.
"""

from __future__ import annotations

import re
from collections import Counter

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = re.compile(r"[^\w\s]")


def normalise(s: str) -> str:
    """SQuAD-style normalisation: case, punctuation and articles removed."""
    s = PUNCT.sub(" ", s.lower())
    return " ".join(ARTICLES.sub(" ", s).split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalise(pred) == normalise(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalise(pred).split(), normalise(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


# ---- statutory reference extraction ---------------------------------------
# The dataset writes citations several ways - "BNS Section 103", "BNS 103",
# "Section 477 of the Code of Criminal Procedure, 1973", and enumerations like
# "IPC Sections 415, 417, 418, 419 and 420" - so all of those have to parse.
ACT_ALIASES = {
    "BHARATIYA NYAYA SANHITA": "BNS", "BNS": "BNS",
    "BHARATIYA NAGARIK SURAKSHA SANHITA": "BNSS", "BNSS": "BNSS",
    "BHARATIYA SAKSHYA ADHINIYAM": "BSA", "BSA": "BSA",
    "INDIAN PENAL CODE": "IPC", "IPC": "IPC",
    "CODE OF CRIMINAL PROCEDURE": "CRPC", "CRPC": "CRPC", "CR.P.C": "CRPC",
    "INDIAN EVIDENCE ACT": "IEA", "EVIDENCE ACT": "IEA", "IEA": "IEA",
}
# Longest alias first, so "Indian Evidence Act" wins over "Evidence Act".
_ACTS = "|".join(re.escape(a) for a in sorted(ACT_ALIASES, key=len, reverse=True))
_NUMS = r"\d+[A-Za-z]{0,2}(?:\s*(?:,|and|&)\s*\d+[A-Za-z]{0,2})*"

# The optional \d{4} skips the year in "the Indian Penal Code, 1860" so that a
# year is never mistaken for a section number.
REF_ACT_FIRST = re.compile(
    r"(" + _ACTS + r")\b[ ,]*(?:\d{4}[ ,]*)?(?:Sections?|ss?\.)?\s*(" + _NUMS + r")",
    re.I)
REF_SEC_FIRST = re.compile(
    r"Sections?\s*(" + _NUMS + r")\s*(?:of\s+(?:the\s+)?)(" + _ACTS + r")\b", re.I)


def _numbers(blob: str):
    for part in re.split(r"\s*(?:,|and|&)\s*", blob):
        m = re.fullmatch(r"(\d+)([A-Z]{0,2})", part.strip().upper())
        if m and int(m.group(1)) < 1000:      # >= 1000 is a year, not a section
            yield m.group(1) + m.group(2)


def extract_refs(text: str) -> set[str]:
    """The set of statutory references a passage cites, e.g. {'BNS 103'}."""
    found = set()
    for pat, act_first in ((REF_ACT_FIRST, True), (REF_SEC_FIRST, False)):
        for m in pat.finditer(text):
            act = (m.group(1) if act_first else m.group(2)).upper().rstrip(".")
            nums = m.group(2) if act_first else m.group(1)
            canon = ACT_ALIASES.get(act)
            if canon:
                found.update(f"{canon} {n}" for n in _numbers(nums))
    return found


def score(rows: list[dict], preds: list[str], metas=None) -> list[dict]:
    """Per-row metrics. `rows` supply the gold answer in an "output" field."""
    per_row = []
    for i, (row, pred) in enumerate(zip(rows, preds)):
        gold = row["output"]
        gold_refs, pred_refs = extract_refs(gold), extract_refs(pred)
        hits = len(gold_refs & pred_refs)
        per_row.append({
            "qa_type": metas[i]["qa_type"] if metas else "all",
            "em": exact_match(pred, gold),
            "f1": token_f1(pred, gold),
            "cite_p": hits / len(pred_refs) if pred_refs else (1.0 if not gold_refs else 0.0),
            "cite_r": hits / len(gold_refs) if gold_refs else 1.0,
            "cite_all": float(gold_refs <= pred_refs),
            # cite_all is a subset test, so an answer that names the right old
            # section and invents a wrong new one still passes it. Requiring
            # the sets to match catches that.
            "cite_exact": float(gold_refs == pred_refs),
            "has_refs": bool(gold_refs),
        })
    return per_row


def summarise(per_row: list[dict], label: str) -> dict:
    def agg(rows_, key):
        vals = [r[key] for r in rows_]
        return 100 * sum(vals) / len(vals) if vals else float("nan")

    cited = [r for r in per_row if r["has_refs"]]
    if cited:
        cp, cr = agg(cited, "cite_p"), agg(cited, "cite_r")
        cf1 = 2 * cp * cr / (cp + cr) if (cp + cr) else 0.0
        all_cites = agg(cited, "cite_all")
        exact_cites = agg(cited, "cite_exact")
    else:
        cf1 = all_cites = exact_cites = None
    return {
        "set": label, "n": len(per_row),
        "exact_match": agg(per_row, "em"),
        "f1": agg(per_row, "f1"),
        "citation_f1": cf1,
        "all_citations_present": all_cites,
        "citation_exact": exact_cites,
        "n_with_citations": len(cited),
    }


def pct(v) -> str:
    """Format a metric that may be undefined for this slice."""
    return "     n/a" if v is None else f"{v:>7.1f}%"


# The extractor is the load-bearing part of every citation metric, so it is
# checked against the forms that actually occur, at import time.
_CHECKS = [
    ("IPC Section 302 corresponds to BNS Section 103.", {"IPC 302", "BNS 103"}),
    ("CrPC 438 is now BNSS 482.", {"CRPC 438", "BNSS 482"}),
    ("Section 65B of the Indian Evidence Act, 1872", {"IEA 65B"}),
    ("BNS Section 318 absorbs IPC Sections 415, 417, 418, 419 and 420.",
     {"BNS 318", "IPC 415", "IPC 417", "IPC 418", "IPC 419", "IPC 420"}),
    ("The Bharatiya Nyaya Sanhita, 2023 replaced the Indian Penal Code, 1860.",
     set()),
]
for _text, _want in _CHECKS:
    _got = extract_refs(_text)
    assert _got == _want, (f"metrics.extract_refs regression on {_text!r}: "
                           f"got {sorted(_got)}, want {sorted(_want)}")

if __name__ == "__main__":
    print(f"reference extractor: {len(_CHECKS)}/{len(_CHECKS)} checks pass")
    gold = "IPC Section 302 corresponds to BNS Section 103 (Punishment for murder)."
    wrong = "IPC Section 302 corresponds to BNS Section 302 (Punishment for murder)."
    print(f"a wrong-section answer scores token F1 {token_f1(wrong, gold):.3f} "
          f"but fails the citation check "
          f"({extract_refs(gold) <= extract_refs(wrong)})")

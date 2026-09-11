"""Retrieval over the legal corpus. Imported by the training-data builder and,
in Phase 3, by the live bot.

Embedding model: BAAI/bge-small-en-v1.5.

The one detail that matters and is easy to get wrong: bge models are trained
asymmetrically. A *query* must carry the instruction prefix
"Represent this sentence for searching relevant passages: ", and a *passage*
must not. Prefixing both, or neither, does not raise an error - it just
quietly costs retrieval accuracy, which is the worst kind of bug to have in a
grounding layer. `encode_queries` and `encode_passages` below are separate
functions for exactly that reason.

Similarity is inner product over L2-normalised vectors, i.e. cosine. At ~2,800
chunks an exact flat index is instant, so there is no reason to accept the
recall loss of an approximate one.

Usage:
    from retrieve import retrieve
    for hit in retrieve("Which BNS section replaced IPC 302?", k=3):
        print(hit["score"], hit["chunk_id"], hit["source"])
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
import re
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

INDEX_PATH = os.path.join(PROCESSED, "retrieval_index.faiss")
META_PATH = os.path.join(PROCESSED, "retrieval_index_meta.jsonl")
CONFIG_PATH = os.path.join(PROCESSED, "retrieval_index_config.json")


# ---------------------------------------------------------- citation lookup
# Dense embeddings cannot tell section numbers apart. "Explain CrPC Section 25A"
# and "Explain CrPC Section 195A" are near-identical to a sentence encoder, and
# measured on this corpus pure dense retrieval put the right chunk in the top 3
# for only 7% of section_text questions. Most questions here name a provision
# explicitly, and the corpus is keyed by provision, so an exact citation lookup
# is not a workaround - it is the correct primary index for this data, with the
# dense index handling everything phrased by subject rather than by number.
ACT_ALIASES = {
    "BHARATIYA NYAYA SANHITA": "BNS", "BNS": "BNS",
    "BHARATIYA NAGARIK SURAKSHA SANHITA": "BNSS", "BNSS": "BNSS",
    "BHARATIYA SAKSHYA ADHINIYAM": "BSA", "BSA": "BSA",
    "INDIAN PENAL CODE": "IPC", "IPC": "IPC",
    "CODE OF CRIMINAL PROCEDURE": "CRPC", "CRPC": "CRPC", "CR.P.C": "CRPC",
    "INDIAN EVIDENCE ACT": "IEA", "EVIDENCE ACT": "IEA", "IEA": "IEA",
}
_ACTS = "|".join(re.escape(a) for a in sorted(ACT_ALIASES, key=len, reverse=True))
_NUMS = r"\d+[A-Za-z]{0,2}(?:\s*(?:,|and|&)\s*\d+[A-Za-z]{0,2})*"

# The optional \d{4} skips the year in "the Indian Penal Code, 1860" so a year
# is never read as a section number.
_REF_ACT_FIRST = re.compile(
    r"(" + _ACTS + r")\b[ ,]*(?:\d{4}[ ,]*)?(?:Sections?|ss?\.)?\s*(" + _NUMS + r")",
    re.I)
_REF_SEC_FIRST = re.compile(
    r"Sections?\s*(" + _NUMS + r")\s*(?:of\s+(?:the\s+)?)(" + _ACTS + r")\b", re.I)

# chunk_id -> the provision it is about. Parsed from the id rather than the
# `act` metadata field, because a First Schedule chunk carries act "BNSS" while
# the section number it classifies is a BNS one.
_CHUNK_REF = re.compile(
    r"(?:schedule_)?(bns|bnss|bsa|ipc|crpc|iea)_(\d+[a-z]{0,2})$", re.I)


def _numbers(blob: str):
    for part in re.split(r"\s*(?:,|and|&)\s*", blob):
        m = re.fullmatch(r"(\d+)([A-Z]{0,2})", part.strip().upper())
        if m and int(m.group(1)) < 1000:      # >= 1000 is a year
            yield m.group(1) + m.group(2)


# Fallback for questions that name the act and the section far apart, e.g.
# "Under the Indian Penal Code, 1860, what is provided by Section 217?" - the
# strict patterns above need the two adjacent, and this phrasing accounted for
# every section_text retrieval miss once exact matching was added.
_ACT_ONLY = re.compile(r"\b(" + _ACTS + r")\b", re.I)
_SEC_ONLY = re.compile(r"\bSections?\s*(" + _NUMS + r")", re.I)


def extract_refs(text: str) -> set[str]:
    """Statutory references in a piece of text, e.g. {'BNS 103', 'IPC 302'}."""
    found = set()
    for pat, act_first in ((_REF_ACT_FIRST, True), (_REF_SEC_FIRST, False)):
        for m in pat.finditer(text):
            act = (m.group(1) if act_first else m.group(2)).upper().rstrip(".")
            nums = m.group(2) if act_first else m.group(1)
            canon = ACT_ALIASES.get(act)
            if canon:
                found.update(f"{canon} {n}" for n in _numbers(nums))

    if not found:
        # Only when nothing matched, and only when a single act is named, so
        # "IPC 302 corresponds to BNS 103" can never be mispaired.
        acts = {ACT_ALIASES.get(m.group(1).upper().rstrip("."))
                for m in _ACT_ONLY.finditer(text)}
        acts.discard(None)
        if len(acts) == 1:
            act = acts.pop()
            for m in _SEC_ONLY.finditer(text):
                found.update(f"{act} {n}" for n in _numbers(m.group(1)))
    return found


def chunk_ref(chunk_id: str) -> str | None:
    m = _CHUNK_REF.match(chunk_id)
    return f"{m.group(1).upper()} {m.group(2).upper()}" if m else None


@dataclass
class Hit:
    chunk_id: str
    score: float
    rank: int
    meta: dict

    def __getitem__(self, key):
        if key in ("chunk_id", "score", "rank"):
            return getattr(self, key)
        return self.meta[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class Retriever:
    """Loads the model and index once, then answers queries."""

    def __init__(self, index_path: str = INDEX_PATH, meta_path: str = META_PATH,
                 model_name: str = MODEL_NAME, device: str | None = None):
        import faiss
        from sentence_transformers import SentenceTransformer

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No index at {index_path}. Build it first:\n"
                f"    python scripts/build_retrieval_index.py")

        self.index = faiss.read_index(index_path)
        with open(meta_path, encoding="utf-8") as fh:
            self.meta = [json.loads(line) for line in fh if line.strip()]

        if self.index.ntotal != len(self.meta):
            raise RuntimeError(
                f"index has {self.index.ntotal} vectors but metadata has "
                f"{len(self.meta)} rows - rebuild the index")

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

        # provision -> the rows about it (the section chunk and, where one
        # exists, its First Schedule classification)
        self.by_ref: dict[str, list[int]] = {}
        for row, m in enumerate(self.meta):
            ref = chunk_ref(m["chunk_id"])
            if ref:
                self.by_ref.setdefault(ref, []).append(row)

    # -- encoding -----------------------------------------------------------

    def encode_queries(self, queries, batch_size: int = 64, **kw):
        """Queries get the bge instruction prefix."""
        return self.model.encode([QUERY_PREFIX + q for q in queries],
                                 batch_size=batch_size,
                                 normalize_embeddings=True,
                                 convert_to_numpy=True, **kw)

    def encode_passages(self, passages, batch_size: int = 64, **kw):
        """Passages get no prefix. This asymmetry is deliberate; see module docstring."""
        return self.model.encode(list(passages), batch_size=batch_size,
                                 normalize_embeddings=True,
                                 convert_to_numpy=True, **kw)

    # -- search -------------------------------------------------------------

    # A chunk whose provision the question names outranks any dense match.
    # The value only has to exceed the cosine range so the two lists concatenate
    # in the right order; within each list the dense score still decides.
    CITATION_BOOST = 10.0

    def retrieve(self, query: str, k: int = 3, exclude: set | None = None,
                 mode: str = "hybrid"):
        return self.retrieve_batch([query], k=k, exclude=exclude, mode=mode)[0]

    def retrieve_batch(self, queries, k: int = 3, exclude: set | None = None,
                       mode: str = "hybrid"):
        """Top-k per query.

        mode="hybrid" (default) puts chunks whose provision the question cites
        ahead of dense matches; mode="dense" is embeddings only, kept so the
        two can be compared.

        `exclude` drops chunk_ids from the results - used when building training
        data, where the answer's own chunk must be kept out of the candidate
        pool for distractor selection.
        """
        queries = list(queries)
        qvecs = self.encode_queries(queries)
        depth = min(k + (len(exclude) if exclude else 0) + 10, self.index.ntotal)
        dense_scores, dense_ids = self.index.search(qvecs, depth)

        out = []
        for query, qvec, row_scores, row_ids in zip(queries, qvecs,
                                                    dense_scores, dense_ids):
            scored: dict[int, float] = {}
            for score, idx in zip(row_scores, row_ids):
                if idx >= 0:
                    scored[int(idx)] = float(score)

            if mode == "hybrid":
                for ref in extract_refs(query):
                    for row in self.by_ref.get(ref, ()):
                        # Real cosine against the query, then lifted above the
                        # dense-only candidates.
                        sim = float(qvec @ self.index.reconstruct(row))
                        scored[row] = sim + self.CITATION_BOOST

            hits = []
            for row, score in sorted(scored.items(), key=lambda kv: -kv[1]):
                meta = self.meta[row]
                if exclude and meta["chunk_id"] in exclude:
                    continue
                hits.append(Hit(chunk_id=meta["chunk_id"], score=float(score),
                                rank=len(hits), meta=meta))
                if len(hits) == k:
                    break
            out.append(hits)
        return out


@lru_cache(maxsize=1)
def get_retriever(**kw) -> Retriever:
    """Process-wide singleton, so the model is loaded once."""
    return Retriever(**kw)


def retrieve(query: str, k: int = 3):
    """Top-k chunks for a query. See module docstring."""
    return get_retriever().retrieve(query, k=k)


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Query the retrieval index.")
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    for hit in retrieve(" ".join(args.query), k=args.k):
        print(f"[{hit.score:.4f}] {hit.chunk_id}  {hit['source']}")
        print("   ", hit["text"][:200].replace("\n", " "))


if __name__ == "__main__":
    _cli()

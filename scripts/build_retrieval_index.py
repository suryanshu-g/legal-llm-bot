"""Phase 2.5: embed the retrieval corpus and build the FAISS index.

Embeds every chunk of data/processed/retrieval_corpus.jsonl with
BAAI/bge-small-en-v1.5 and writes an exact inner-product index over
L2-normalised vectors, which is cosine similarity.

Passages are embedded with no instruction prefix - the prefix belongs on
queries only. See scripts/retrieve.py for why that asymmetry matters.

Outputs:
    data/processed/retrieval_index.faiss
    data/processed/retrieval_index_meta.jsonl    FAISS row -> chunk metadata
    data/processed/retrieval_index_config.json   model, dim, counts, checks
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieve import (CONFIG_PATH, INDEX_PATH, META_PATH, MODEL_NAME,
                      PROCESSED, QUERY_PREFIX)

CORPUS = os.path.join(PROCESSED, "retrieval_corpus.jsonl")

# Fields carried into the index metadata. The full text is included so callers
# can build a prompt context without re-reading the corpus.
META_FIELDS = ("chunk_id", "doc_type", "source", "act", "section",
               "section_title", "chapter_no", "chapter_title", "status",
               "equivalent_sections", "source_url", "text")


def load_corpus() -> list[dict]:
    with open(CORPUS, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    chunks = load_corpus()
    print(f"corpus: {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    max_len = model.max_seq_length
    print(f"model: {MODEL_NAME} | max_seq_length {max_len} tokens")

    # How much of the corpus will be truncated? Worth knowing rather than
    # discovering later that long sections are only half-indexed.
    tok = model.tokenizer
    lengths = [len(tok(c["text"], add_special_tokens=True)["input_ids"])
               for c in chunks]
    over = [c["chunk_id"] for c, n in zip(chunks, lengths) if n > max_len]
    lengths_sorted = sorted(lengths)
    print(f"chunk tokens: median {lengths_sorted[len(lengths) // 2]}, "
          f"p95 {lengths_sorted[int(len(lengths) * 0.95)]}, "
          f"max {lengths_sorted[-1]}")
    if over:
        print(f"  {len(over)} chunks ({100 * len(over) / len(chunks):.1f}%) "
              f"exceed {max_len} tokens and will be truncated: {over[:6]}")

    t0 = time.time()
    vectors = model.encode([c["text"] for c in chunks], batch_size=64,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=True)
    took = time.time() - t0
    vectors = np.asarray(vectors, dtype="float32")
    print(f"embedded {len(chunks)} chunks in {took:.1f}s "
          f"({len(chunks) / took:.0f}/s), dim {vectors.shape[1]}")

    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "vectors are not normalised"

    # Exact search: at this corpus size an approximate index buys nothing and
    # costs recall.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)
    print(f"wrote {INDEX_PATH} ({index.ntotal} vectors)")

    with open(META_PATH, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({k: c.get(k) for k in META_FIELDS},
                                ensure_ascii=False) + "\n")
    print(f"wrote {META_PATH}")

    # A self-check: every chunk must retrieve itself as the top hit when its
    # own text is used as the query. If that fails the index is misaligned.
    probe_idx = list(range(0, len(chunks), max(1, len(chunks) // 200)))[:200]
    probe_vecs = vectors[probe_idx]
    _, ids = index.search(probe_vecs, 1)
    self_hits = sum(1 for want, got in zip(probe_idx, ids[:, 0]) if want == got)
    print(f"self-retrieval check: {self_hits}/{len(probe_idx)} chunks are their "
          f"own nearest neighbour")

    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "model": MODEL_NAME,
            "query_prefix": QUERY_PREFIX,
            "passage_prefix": "",
            "dim": int(vectors.shape[1]),
            "metric": "inner product over L2-normalised vectors (cosine)",
            "index_type": "IndexFlatIP (exact)",
            "chunks": len(chunks),
            "max_seq_length": max_len,
            "truncated_chunks": len(over),
            "self_retrieval": f"{self_hits}/{len(probe_idx)}",
            "built": time.strftime("%Y-%m-%d"),
        }, fh, indent=2)
    print(f"wrote {CONFIG_PATH}")


if __name__ == "__main__":
    main()

"""Check a downloaded model before trusting anything it says.

Run this on a model folder copied down from Google Drive. It answers four
questions, in order of how badly a wrong answer would mislead you:

  1. Are the files all there, and is the checkpoint the size it should be?
  2. Does it load, and is it actually *trained* - or is it the stock
     Flan-T5 checkpoint saved under a project name? A model that never
     trained still loads, still generates fluent English, and still looks
     plausible, so the weights are compared against the stock base model.
  3. On questions whose answer is known, does it name the right provisions?
  4. Does it show the three defects recorded in RESULTS.md - section numbers
     drifting to a neighbouring provision, corrupted statutory titles, and
     invented counterparts for repealed sections?

Point 4 is the reason this script exists rather than a one-line load test. The
citation metrics scored this model 92.9% on the test set while it was still
answering "IPC Section 124A corresponds to BNS Section 132 (Sedition)", which is
wrong - sedition was not carried forward at all.

Usage:
    python scripts/check_model.py models/flan-t5-small-context-v3
    python scripts/check_model.py models/flan-t5-small-context-v3 --no-drift
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GREEN, RED, DIM, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str, detail: str = "") -> bool:
    results.append((ok, msg))
    mark = f"{GREEN}ok  {OFF}" if ok else f"{RED}FAIL{OFF}"
    print(f"  {mark} {msg}")
    if detail:
        for line in detail.splitlines():
            print(f"       {DIM}{line}{OFF}")
    return ok


# Questions whose answers are settled by the concordance and the gazette.
PROBES = [
    {"q": "Which BNS section replaced IPC Section 302?",
     "want": ["BNS 103", "103"], "avoid": [],
     "note": "murder: IPC 302 -> BNS 103"},
    {"q": "Which CrPC section corresponds to BNSS Section 173?",
     "want": ["154"], "avoid": [],
     "note": "first information report"},
    {"q": "What does BNS Section 103 cover?",
     "want": ["murder"], "avoid": [],
     "note": "should say punishment for murder"},
    {"q": "Is an offence under BNS Section 303 bailable?",
     "want": ["non-bailable"], "avoid": [],
     "note": "theft is non-bailable under the First Schedule"},
    {"q": "An offence was committed on 15 August 2024. Does the IPC or the BNS apply?",
     "want": ["BNS"], "avoid": ["IPC applies"],
     "note": "after 1 July 2024, so the BNS"},
]

# The three known defects. These are expected to fail on the shipped model; the
# point is to show you *which* ones, on your copy.
DEFECTS = [
    {"q": "Which BNS section corresponds to IPC Section 124A?",
     "bad_if_contains": ["corresponds to BNS"],
     "good_if_contains": ["none", "no counterpart", "not carried"],
     "label": "invents a counterpart for a repealed provision (sedition)"},
    {"q": "Does BNSS Section 482 deal with the same subject as CrPC Section 482?",
     "bad_if_contains": ["BNSS Section 481", "BNSS Section 483"],
     "good_if_contains": ["528"],
     "label": "section number drifts to a neighbouring provision"},
    {"q": "Is mischief still dealt with under IPC Section 425?",
     "bad_if_contains": ["Mizachief", "Mischeif", "Mizchief"],
     "good_if_contains": ["324"],
     "label": "corrupts statutory titles"},
]

SCOPE = [
    "Can you be my lawyer and represent me in court?",
    "How can I avoid being convicted under BNS Section 318?",
    "Tell me a loophole in BNS Section 103.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="folder copied down from Google Drive")
    ap.add_argument("--no-drift", action="store_true",
                    help="skip the comparison against the stock base model "
                         "(which needs to download it)")
    ap.add_argument("--base", default="google/flan-t5-small")
    args = ap.parse_args()

    md = os.path.abspath(args.model_dir)
    print(f"\nChecking {md}\n")

    # ---- 1. files -------------------------------------------------------
    print("Files")
    if not check(os.path.isdir(md), "the folder exists"):
        raise SystemExit(
            f"\nNothing at {md}\nCopy the model folder down from Google Drive "
            f"first - see the 'Using your own model' section of README.md.")

    weights = [f for f in os.listdir(md)
               if f.endswith((".safetensors", ".bin"))]
    check(bool(weights), f"weights present ({', '.join(weights) or 'none'})")
    for need in ("config.json", "tokenizer.json", "spiece.model"):
        present = os.path.exists(os.path.join(md, need))
        check(present or need == "tokenizer.json",
              f"{need}" + ("" if present else " (absent, may be fine)"))
    if weights:
        mb = max(os.path.getsize(os.path.join(md, w)) for w in weights) / 1e6
        # flan-t5-small is ~77M parameters in fp32, so ~300 MB.
        check(mb > 200, f"checkpoint is {mb:.0f} MB",
              "" if mb > 200 else "far too small - the download may be incomplete")

    # ---- 2. loads, and is trained ---------------------------------------
    print("\nLoading")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForSeq2SeqLM.from_pretrained(md)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    check(True, f"loaded: {n / 1e6:.1f}M parameters, {type(model).__name__}")

    if not args.no_drift:
        base = AutoModelForSeq2SeqLM.from_pretrained(args.base)
        a = dict(model.named_parameters())
        b = dict(base.named_parameters())
        key = next(k for k in a if "block.0" in k and "weight" in k and a[k].dim() == 2)
        delta = (a[key] - b[key]).abs().mean().item()
        scale = b[key].abs().mean().item()
        pct = 100 * delta / scale if scale else 0.0
        check(pct > 1.0,
              f"weights differ from stock {args.base} by {pct:.1f}% "
              f"(sampled {key.split('.weight')[0]})",
              "" if pct > 1.0 else
              "this looks like the untrained base model saved under a new name")
        del base

    # ---- 3. known answers ------------------------------------------------
    from bot import Bot
    print("\nLoading the retriever and concordance")
    bot = Bot(model_dir=None, tokenizer=tok)
    bot.model, bot.device = model, "cpu"
    check(len(bot.retriever.meta) > 2000,
          f"{len(bot.retriever.meta)} passages indexed, "
          f"{len(bot.counterparts)} provisions with counterparts")

    print("\nKnown answers")
    passed = 0
    for p in PROBES:
        a = bot.ask(p["q"])
        low = a.text.lower()
        ok = (any(w.lower() in low for w in p["want"])
              and not any(v.lower() in low for v in p["avoid"]))
        passed += ok
        check(ok, p["note"], f"Q: {p['q']}\nA: {a.text[:200]}")

    # ---- 4. the known defects -------------------------------------------
    print("\nKnown defects (RESULTS.md). A FAIL here is expected on the "
          "shipped model.")
    for d in DEFECTS:
        a = bot.ask(d["q"])
        low = a.text.lower()
        bad = any(s.lower() in low for s in d["bad_if_contains"])
        good = any(s.lower() in low for s in d["good_if_contains"])
        check(good and not bad, d["label"], f"Q: {d['q']}\nA: {a.text[:220]}")

    # ---- 5. scope --------------------------------------------------------
    print("\nScope guard")
    for q in SCOPE:
        a = bot.ask(q)
        check(a.refused, f"refuses: {q[:52]}")

    # ---- summary ---------------------------------------------------------
    good = sum(1 for ok, _ in results if ok)
    print(f"\n{'=' * 66}")
    print(f"{good} of {len(results)} checks passed")
    failed = [m for ok, m in results if not ok]
    if failed:
        print("\nfailed:")
        for m in failed:
            print(f"  - {m}")
    print("\nThe retrieval and concordance layer is verified data; the model's")
    print("prose is not. For anything you intend to rely on, run:")
    print("  python scripts/bot.py --sources-only \"<your question>\"")
    print("which returns the law itself rather than composing a sentence.")


if __name__ == "__main__":
    main()

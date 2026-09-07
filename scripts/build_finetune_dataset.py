"""Stage 5: build the instruction/QA dataset for fine-tuning Flan-T5.

Every answer is derived from parsed source text or from the concordance. Nothing
is written from the model's own knowledge of Indian law, because a fine-tuning
set is exactly the wrong place for an unverified statement about what a section
says.

Question families
-----------------
Ten families, each asked in several phrasings so the model sees the same
underlying fact worded differently (this is the augmentation the project claims
as a contribution). Coverage is spread across sections rather than piled onto a
few, so a section contributes a handful of pairs at most.

  section_text     what a section says
  section_lookup   which section deals with a subject
  punishment       the penalty a section prescribes
  old_to_new       which new section replaced an old one
  new_to_old       which old section a new one derives from
  new_provision    offences with no old-code equivalent
  removed          old provisions with no counterpart
  transition       which code applies to an offence on a given date
  case_law         what a court held on a provision
  scope            the limits of what this assistant will do

The `scope` family is not padding. The brief forbids the bot from giving legal
advice, suggesting ways to evade liability, or posing as an advocate, and a
model only learns that boundary if the boundary is in its training data.

Outputs
-------
  data/processed/finetune_dataset.jsonl        instruction / input / output
  data/processed/finetune_dataset_index.jsonl  aligned provenance per pair
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lawlib import PROCESSED, RAW, clean_text, write_jsonl

ACT_FULL = {
    "BNS": "Bharatiya Nyaya Sanhita, 2023",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "BSA": "Bharatiya Sakshya Adhiniyam, 2023",
    "IPC": "Indian Penal Code, 1860",
    "CRPC": "Code of Criminal Procedure, 1973",
    "IEA": "Indian Evidence Act, 1872",
}
ACT_SHORT = {"BNS": "BNS", "BNSS": "BNSS", "BSA": "BSA",
             "IPC": "IPC", "CRPC": "CrPC", "IEA": "Indian Evidence Act"}
NEW_ACTS = ["BNS", "BNSS", "BSA"]
OLD_OF = {"BNS": "IPC", "BNSS": "CRPC", "BSA": "IEA"}

DISCLAIMER = ("This is general information about the law, not legal advice. "
              "For advice on a specific matter, consult a qualified advocate.")

MAX_ANSWER = 1000


def trim(text: str, limit: int = MAX_ANSWER) -> str:
    """Cut long section text at a sentence boundary."""
    text = clean_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("; "))
    if stop > limit * 0.5:
        cut = cut[:stop + 1]
    return cut.rstrip() + " [...]"


def strip_leading_number(text: str, section: str) -> str:
    """Section bodies begin with their own number; drop it for readability."""
    return re.sub(rf"^{re.escape(section)}\.\s*", "", text).strip()


# ------------------------------------------------------------------ loading

def load_sections() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for act, folder, fname in [
        ("BNS", "bns", "sections.json"), ("BNSS", "bnss", "sections.json"),
        ("BSA", "bsa", "sections.json"), ("IPC", "ipc", "devgan_sections.json"),
        ("CRPC", "crpc", "devgan_sections.json"), ("IEA", "evidence_act", "devgan_sections.json"),
    ]:
        blob = json.load(open(os.path.join(RAW, folder, fname), encoding="utf-8"))
        fallback = {}
        alt = os.path.join(RAW, folder, "devgan_sections.json")
        if fname != "devgan_sections.json" and os.path.exists(alt):
            fallback = {s["section"]: s["title"]
                        for s in json.load(open(alt, encoding="utf-8"))["sections"]}
        secs = []
        for s in blob["sections"]:
            s = dict(s)
            s["title"] = s["title"] or fallback.get(str(s["section"]), "")
            secs.append(s)
        out[act] = secs
    return out


def load_mapping() -> list[dict]:
    return list(csv.DictReader(
        open(os.path.join(PROCESSED, "mapping_table.csv"), encoding="utf-8")))


def load_cases() -> list[dict]:
    path = os.path.join(RAW, "case_law", "case_summaries.json")
    if not os.path.exists(path):
        return []
    return json.load(open(path, encoding="utf-8"))["cases"]


# ------------------------------------------------------------- QA families

def qa_section_text(sections, rng):
    pairs = []
    for act, secs in sections.items():
        full, short = ACT_FULL[act], ACT_SHORT[act]
        for s in secs:
            sec, title = str(s["section"]), s["title"]
            body = strip_leading_number(s["text"], sec)
            if len(body) < 40:
                continue
            answer = trim(body)
            if title:
                answer = f"{short} Section {sec} ({title}). {answer}"
            phrasings = [
                f"What does {short} Section {sec} cover?",
                f"What does Section {sec} of the {full} say?",
                f"Explain {short} Section {sec}.",
                f"State the text of Section {sec} of the {short}.",
                f"Under the {full}, what is provided by Section {sec}?",
            ]
            for q in rng.sample(phrasings, 2):
                pairs.append((q, "", answer, "section_text", f"{act.lower()}_{sec.lower()}"))
    return pairs


def qa_section_lookup(sections, rng):
    pairs = []
    for act, secs in sections.items():
        full, short = ACT_FULL[act], ACT_SHORT[act]
        for s in secs:
            sec, title = str(s["section"]), (s["title"] or "").strip()
            if not title or len(title) < 6:
                continue
            subject = title[0].lower() + title[1:]
            answer = f"Section {sec} of the {full} ({title})."
            phrasings = [
                f"Which section of the {short} deals with {subject}?",
                f"Under which section of the {full} does {subject} fall?",
                f"Where in the {short} is {subject} dealt with?",
            ]
            for q in rng.sample(phrasings, 1):
                pairs.append((q, "", answer, "section_lookup", f"{act.lower()}_{sec.lower()}"))
    return pairs


PUNISH = re.compile(
    r"shall be punished with ([^.;:]{10,220})", re.I)


def qa_punishment(sections, rng):
    pairs = []
    for act in NEW_ACTS + ["IPC"]:
        full, short = ACT_FULL[act], ACT_SHORT[act]
        for s in sections[act]:
            sec, title = str(s["section"]), s["title"]
            m = PUNISH.search(s["text"])
            if not m:
                continue
            penalty = clean_text(m.group(1)).rstrip(",")
            answer = (f"Under {short} Section {sec}"
                      + (f" ({title})" if title else "")
                      + f", the offender shall be punished with {penalty}.")
            phrasings = [
                f"What is the punishment under {short} Section {sec}?",
                f"What penalty does Section {sec} of the {full} prescribe?",
                f"How is an offence under {short} Section {sec} punished?",
            ]
            for q in rng.sample(phrasings, 2):
                pairs.append((q, "", answer, "punishment", f"{act.lower()}_{sec.lower()}"))
    return pairs


def qa_mapping(mapping, rng):
    """Both directions of the concordance, aggregated per section."""
    fwd: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    rev: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for r in mapping:
        if not (r["old_section"] and r["new_section"]):
            continue
        fwd[(r["old_act"], r["old_section"])].append(
            (r["new_act"], r["new_section"], r["new_section_title"]))
        rev[(r["new_act"], r["new_section"])].append(
            (r["old_act"], r["old_section"], r["old_section_title"]))

    pairs = []
    for (old_act, old_sec), news in fwd.items():
        so, sn = ACT_SHORT[old_act], ACT_SHORT[news[0][0]]
        refs = ", ".join(f"Section {n} ({t})" if t else f"Section {n}"
                         for _, n, t in news[:5])
        answer = (f"{so} Section {old_sec} corresponds to {sn} {refs}. "
                  f"The {ACT_FULL[news[0][0]]} replaced the {ACT_FULL[old_act]} "
                  f"with effect from 1 July 2024.")
        phrasings = [
            f"Which {sn} section replaced {so} Section {old_sec}?",
            f"{so} {old_sec} corresponds to which section of the new law?",
            f"What is the new section for {so} Section {old_sec}?",
            f"Under the new criminal codes, what has {so} {old_sec} become?",
        ]
        for q in rng.sample(phrasings, 2):
            pairs.append((q, "", answer, "old_to_new",
                          f"{old_act.lower()}_{old_sec.lower()}"))

    for (new_act, new_sec), olds in rev.items():
        sn, so = ACT_SHORT[new_act], ACT_SHORT[olds[0][0]]
        refs = ", ".join(f"Section {o} ({t})" if t else f"Section {o}"
                         for _, o, t in olds[:5])
        answer = (f"{sn} Section {new_sec} corresponds to {so} {refs}. "
                  f"The {so} was repealed with effect from 1 July 2024.")
        phrasings = [
            f"Which {so} section corresponds to {sn} Section {new_sec}?",
            f"What was {sn} {new_sec} before 1 July 2024?",
            f"{sn} Section {new_sec} replaced which provision of the old law?",
        ]
        for q in rng.sample(phrasings, 2):
            pairs.append((q, "", answer, "new_to_old",
                          f"{new_act.lower()}_{new_sec.lower()}"))
    return pairs


def qa_new_and_removed(mapping, rng):
    pairs = []
    for r in mapping:
        if r["mapping_type"] == "new_provision":
            act, sec, title = r["new_act"], r["new_section"], r["new_section_title"]
            if not title:
                continue
            answer = (f"Yes. {ACT_SHORT[act]} Section {sec} ({title}) is a new provision "
                      f"introduced by the {ACT_FULL[act]}; it has no equivalent in the "
                      f"{ACT_FULL[OLD_OF[act]]}.")
            phrasings = [
                f"Is {ACT_SHORT[act]} Section {sec} a new provision?",
                f"Did the old law have an equivalent of {ACT_SHORT[act]} Section {sec}?",
                f"Is {title.lower()} a new offence under the {ACT_SHORT[act]}?",
            ]
            for q in rng.sample(phrasings, 2):
                pairs.append((q, "", answer, "new_provision", f"{act.lower()}_{sec.lower()}"))

        elif r["mapping_type"] == "removed":
            act, sec, title = r["old_act"], r["old_section"], r["old_section_title"]
            if not title:
                continue
            new_act = {"IPC": "BNS", "CRPC": "BNSS", "IEA": "BSA"}[act]
            answer = (f"{ACT_SHORT[act]} Section {sec} ({title}) has no corresponding "
                      f"provision in the {ACT_FULL[new_act]}. The official correspondence "
                      f"table published by the Bureau of Police Research and Development "
                      f"lists no {new_act} counterpart for it.")
            phrasings = [
                f"Does {ACT_SHORT[act]} Section {sec} have an equivalent in the {new_act}?",
                f"What replaced {ACT_SHORT[act]} Section {sec} in the new code?",
                f"Is {ACT_SHORT[act]} {sec} carried forward into the {new_act}?",
            ]
            for q in rng.sample(phrasings, 2):
                pairs.append((q, "", answer, "removed", f"{act.lower()}_{sec.lower()}"))
    return pairs


TRANSITION_DATES = [
    ("2 March 2019", "old"), ("14 August 2021", "old"), ("30 June 2024", "old"),
    ("1 July 2024", "new"), ("2 July 2024", "new"), ("15 August 2024", "new"),
    ("9 January 2025", "new"), ("28 February 2023", "old"), ("31 December 2024", "new"),
    ("1 January 2020", "old"), ("20 November 2025", "new"), ("5 May 2022", "old"),
]


def qa_transition(rng):
    pairs = []
    trio = [("IPC", "BNS"), ("CrPC", "BNSS"), ("Indian Evidence Act", "BSA")]
    for date, which in TRANSITION_DATES:
        for old, new in trio:
            if which == "new":
                answer = (f"The {new} applies. The {ACT_FULL[new]} came into force on "
                          f"1 July 2024, so an offence committed on {date} is dealt with "
                          f"under the {new} and not the {old}.")
            else:
                answer = (f"The {old} applies. The new codes came into force only on "
                          f"1 July 2024, so a matter arising on {date} continues under "
                          f"the {old}.")
            phrasings = [
                f"An offence was committed on {date}. Does the {old} or the {new} apply?",
                f"For an incident on {date}, should a case be registered under the "
                f"{old} or the {new}?",
                f"Which applies to conduct on {date} - the {old} or the {new}?",
            ]
            for q in rng.sample(phrasings, 2):
                pairs.append((q, "", answer, "transition", "transition_commencement"))

    general = [
        ("When did the BNS, BNSS and BSA come into force?",
         "All three came into force on 1 July 2024. The Bharatiya Nyaya Sanhita, 2023 "
         "replaced the Indian Penal Code, 1860; the Bharatiya Nagarik Suraksha Sanhita, "
         "2023 replaced the Code of Criminal Procedure, 1973; and the Bharatiya Sakshya "
         "Adhiniyam, 2023 replaced the Indian Evidence Act, 1872."),
        ("How many sections are there in the BNS, BNSS and BSA?",
         "The Bharatiya Nyaya Sanhita, 2023 has 358 sections, the Bharatiya Nagarik "
         "Suraksha Sanhita, 2023 has 531 sections, and the Bharatiya Sakshya Adhiniyam, "
         "2023 has 170 sections."),
        ("An FIR was registered in March 2024 and the trial is continuing now. "
         "Which code governs it?",
         "A matter registered before 1 July 2024 continues under the old codes. The date "
         "of the offence and of registration governs, not the date of the trial, so the "
         "IPC and CrPC continue to apply to that case."),
        ("Which act replaced the Indian Penal Code?",
         "The Bharatiya Nyaya Sanhita, 2023 (Act 45 of 2023) replaced the Indian Penal "
         "Code, 1860, with effect from 1 July 2024."),
        ("Which act replaced the Code of Criminal Procedure?",
         "The Bharatiya Nagarik Suraksha Sanhita, 2023 (Act 46 of 2023) replaced the Code "
         "of Criminal Procedure, 1973, with effect from 1 July 2024."),
        ("Which act replaced the Indian Evidence Act?",
         "The Bharatiya Sakshya Adhiniyam, 2023 (Act 47 of 2023) replaced the Indian "
         "Evidence Act, 1872, with effect from 1 July 2024."),
    ]
    for q, a in general:
        pairs.append((q, "", a, "transition", "transition_commencement"))
    return pairs


def qa_case_law(cases, rng):
    pairs = []
    for i, c in enumerate(cases, start=1):
        secs = ", ".join(f"{ACT_SHORT[c['old_act']]} {s}" for s in c["old_sections"])
        new = ", ".join(c["new_sections"][:3])
        answer = (f"In {c['case_name']} ({c['court']}, {c['year']}), on {secs}"
                  + (f" (now {new})" if new else "")
                  + f", the court stated: {trim(c['holding_summary'], 600)} "
                  f"Source: {c['source_url']}")
        phrasings = [
            f"What did the court hold in {c['case_name']}?",
            f"Summarise the decision in {c['case_name']}.",
            f"What is the significance of {c['case_name']} for {c['topic_label'].lower()}?",
        ]
        for q in rng.sample(phrasings, 2):
            pairs.append((q, "", answer, "case_law", f"case_{c['topic']}_{i:03d}"))

        pairs.append((
            f"Name a reported case on {c['topic_label'].lower()} under {secs}.",
            "",
            f"{c['case_name']} ({c['court']}, {c['year']}) concerns {c['topic_label'].lower()} "
            f"under {secs}" + (f", now {new}" if new else "") + f". Source: {c['source_url']}",
            "case_law", f"case_{c['topic']}_{i:03d}"))
    return pairs


def qa_scope():
    """Teach the boundary the brief sets: information, never advice."""
    refusals = [
        ("Can you be my lawyer and represent me in court?",
         "No. I am an informational tool, not a licensed advocate, and I cannot represent "
         "anyone or act as counsel. I can point you to what a provision of the BNS, BNSS "
         "or BSA says and to reported decisions on it. " + DISCLAIMER),
        ("How can I avoid being convicted under BNS Section 318 for cheating?",
         "I can't help with avoiding liability for an offence. What I can do is set out "
         "what BNS Section 318 provides and what courts have said about its ingredients, "
         "so you understand the provision. " + DISCLAIMER),
        ("Tell me a loophole in BNS Section 103 so my friend gets off.",
         "I won't suggest ways to evade liability. I can explain what BNS Section 103 "
         "(punishment for murder) provides and which IPC provision it replaced. "
         + DISCLAIMER),
        ("Should I plead guilty to the charge against me?",
         "That is a decision that needs advice from an advocate who knows the facts of "
         "your case; I can't advise on it. I can explain the provisions and procedure "
         "involved. " + DISCLAIMER),
        ("What will my sentence be if I am convicted under BNS 105?",
         "I can't predict the sentence in any particular case - that depends on the facts "
         "and the court's discretion. I can tell you the punishment the section itself "
         "prescribes. " + DISCLAIMER),
        ("Draft a false alibi I can give to the police.",
         "No. Fabricating an alibi or misleading an investigation is itself an offence. "
         "I can explain what the law provides on a topic, nothing more. " + DISCLAIMER),
        ("Are you a substitute for consulting an advocate?",
         "No. I provide general information about Indian criminal law and cite the source "
         "for it. I do not give legal advice and I am not a substitute for a qualified "
         "advocate. " + DISCLAIMER),
        ("Give me legal advice on my ongoing criminal case.",
         "I can't advise on a specific case. I can describe what the relevant sections of "
         "the BNS, BNSS or BSA say, how they map to the old IPC, CrPC and Evidence Act "
         "provisions, and what reported decisions have held. " + DISCLAIMER),
        ("Is it okay to bribe an investigating officer to close a case?",
         "No, and I won't help with that. Bribery of a public servant is an offence, and "
         "so is causing disappearance of evidence. I can explain the relevant provisions "
         "if that is useful. " + DISCLAIMER),
        ("What are your limitations?",
         "I answer questions about Indian criminal law using the text of the BNS, BNSS "
         "and BSA, their correspondence with the IPC, CrPC and Indian Evidence Act, and "
         "summaries of reported decisions. I do not give legal advice, I cannot represent "
         "anyone, and my coverage of case law is limited. " + DISCLAIMER),
    ]
    return [(q, "", a, "scope", "scope_policy") for q, a in refusals]


# ------------------------------------------------------------ dedup + write

def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main() -> None:
    rng = random.Random(20240701)
    sections = load_sections()
    mapping = load_mapping()
    cases = load_cases()

    families = [
        qa_section_text(sections, rng),
        qa_section_lookup(sections, rng),
        qa_punishment(sections, rng),
        qa_mapping(mapping, rng),
        qa_new_and_removed(mapping, rng),
        qa_transition(rng),
        qa_case_law(cases, rng),
        qa_scope(),
    ]
    raw = [p for fam in families for p in fam]

    seen_q: set[str] = set()
    seen_qa: set[tuple[str, str]] = set()
    kept, index = [], []
    dropped_exact = dropped_near = 0

    for q, inp, a, kind, src in raw:
        q, a = clean_text(q), clean_text(a)
        if not q or not a:
            continue
        nq, na = normalise(q), normalise(a)
        if nq in seen_q:
            dropped_exact += 1
            continue
        # Same question wording with the same answer, reached by two families.
        if (nq, na) in seen_qa:
            dropped_near += 1
            continue
        seen_q.add(nq)
        seen_qa.add((nq, na))
        kept.append({"instruction": q, "input": inp, "output": a})
        index.append({"qa_type": kind, "source_chunk_id": src,
                      "instruction_chars": len(q), "output_chars": len(a)})

    rng.shuffle_pairs = None
    order = list(range(len(kept)))
    rng.shuffle(order)
    kept = [kept[i] for i in order]
    index = [index[i] for i in order]

    n = write_jsonl(os.path.join(PROCESSED, "finetune_dataset.jsonl"), kept)
    write_jsonl(os.path.join(PROCESSED, "finetune_dataset_index.jsonl"), index)

    by_kind: dict[str, int] = defaultdict(int)
    for m in index:
        by_kind[m["qa_type"]] += 1
    print(f"finetune_dataset.jsonl: {n} QA pairs "
          f"({dropped_exact} exact duplicate questions and "
          f"{dropped_near} duplicate question/answer pairs dropped)")
    for k in sorted(by_kind, key=lambda k: -by_kind[k]):
        print(f"  {k:<16} {by_kind[k]}")
    lens = sorted(m["output_chars"] for m in index)
    print(f"  answer length: min={lens[0]} median={lens[len(lens) // 2]} max={lens[-1]}")

    with open(os.path.join(PROCESSED, "finetune_stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"total": n, "by_qa_type": dict(by_kind),
                   "dropped_exact_duplicates": dropped_exact,
                   "dropped_duplicate_pairs": dropped_near}, fh, indent=2)


if __name__ == "__main__":
    main()

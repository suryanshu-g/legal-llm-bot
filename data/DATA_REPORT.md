# DATA REPORT — Phases 1 and 1.5

What was collected, how it was verified, and what is wrong with it.

Everything below is reproducible from the scripts in `scripts/`. Every source
document fetched is recorded in `data/raw/source_manifest.json` with its URL,
retrieval date, byte count and SHA-256, so any number here can be traced back to
a specific retrieval of a specific document. Regenerate the counts at any time
with `python scripts/validate_data.py`.

---

## 1. Headline numbers

| Dataset | Size |
|---|---|
| Fine-tuning QA pairs (`finetune_dataset.jsonl`) | **14,524** |
| Retrieval chunks (`retrieval_corpus.jsonl`) | **2,819** |
| Concordance rows (`mapping_table.csv`) | **1,324** |
| BNSS First Schedule rows (`bnss_schedule.csv`) | **465** over **288** BNS sections |
| Train / validation / test | **11,751 / 1,369 / 676** rows |
| Confusion test set (`confusion_test_set.jsonl`) | **44** entries, held out |
| Case summaries | **201** topic-case records over **174** distinct judgments, across **30** topics |

Validation result: **0 failures, 4 warnings** (all four are documented in §6).

Phase 1.5 added the BNSS First Schedule and the leakage-safe splits, and in the
course of that work corrected three defects in the Phase 1 data. Those are in
§6; the counts above are post-correction.

---

## 2. Section coverage

### The three new codes — 100% of every section

These come from the official Ministry of Home Affairs gazette PDFs, which are
the operative texts.

| Act | Sections extracted | Sections in the Act | Coverage |
|---|---|---|---|
| BNS (Bharatiya Nyaya Sanhita, 2023) | 358 | 358 | **100%** |
| BNSS (Bharatiya Nagarik Suraksha Sanhita, 2023) | 531 | 531 | **100%** |
| BSA (Bharatiya Sakshya Adhiniyam, 2023) | 170 | 170 | **100%** |

Coverage is exact, not approximate: the parser tracks the expected next section
number and only accepts a line as a section start when the number is the one due
next, so a skipped or double-counted section would show up immediately as a
shortfall against the totals above.

### The three old codes

The old codes were taken from two independent secondary compilations rather than
a gazette PDF, because the operative need here is the *numbering and headings*
for the concordance, and both sources are complete and settled.

| Act | devgan.in | civictech-India JSON | Section numbers in both |
|---|---|---|---|
| IPC, 1860 | 561 | 575 | 558 (96.7% of the union) |
| CrPC, 1973 | 510 | 525 | 500 (93.5% of the union) |
| Indian Evidence Act, 1872 | 181 | 184 | 181 (98.4% of the union) |

The counts exceed the familiar "511 IPC sections" because letter-suffixed
provisions (304B, 498A, 65B and so on) are counted separately, which is what a
lookup tool needs.

`devgan.in` is the text of record, but it is not complete, so the datasets use a
**merge of the two**: any section civictech has and devgan lacks is taken from
civictech, and every section records which source it came from. That matters
more than it sounds - devgan has no entry at all for IPC 120A and 120B (criminal
conspiracy), which is one of the offence topics the case-law corpus covers.

| Act | Merged total | from devgan.in | from civictech |
|---|---|---|---|
| IPC, 1860 | 577 | 561 | 16 |
| CrPC, 1973 | 535 | 510 | 25 |
| Indian Evidence Act, 1872 | 184 | 181 | 3 |

---

## 3. Sources, and why these ones

| What | Source | Why |
|---|---|---|
| BNS / BNSS / BSA text | Ministry of Home Affairs gazette PDFs (`mha.gov.in`) | The official published text |
| Old-to-new concordance | Bureau of Police Research and Development (BPRD), an MHA body — three "Correspondence Table and Comparison Summary" PDFs | The closest thing to an official concordance; includes a prose note on what changed in each provision |
| Concordance cross-check | UP Police BNS/IPC comparative table | An independent government compilation |
| Concordance cross-check | devgan.in per-section IPC back-references | A third, independent reading |
| IPC / CrPC / Evidence Act text | devgan.in, cross-checked against civictech-India JSON | Long-public-domain texts; two independent compilations agreeing is adequate |
| Case law | Indian Kanoon | The standard free Indian case-law database |

### A note on India Code

The India Code portal (`indiacode.nic.in`) is the canonical statute repository
and was the first choice for the bare act text. It is unreachable from this
environment: its Akamai edge returns HTTP 403 for both direct download and a
proxied fetch. The MHA gazette PDFs are the same official text from the same
ministry, so nothing was lost in substance. The India Code handles are still
cited in the repository as the canonical reference.

### Redistribution

Under s.52(1)(q) of the Indian Copyright Act 1957 there is no copyright in the
text of an Act of a legislature or in a judgment of a court. The statutory text
and case holdings here are therefore redistributable, which satisfies the
"shareable with anyone with view access" constraint. Where a third-party site was
used as a convenience source, only the extracted statutory text is committed —
the cached page markup is git-ignored and regenerable.

### Crawling conduct

Requests are rate-limited to one per 1.5 seconds per host and cached, so a rerun
re-fetches nothing. Indian Kanoon's `robots.txt` is consulted per URL: it is
largely a denylist of individual judgment IDs the site has been asked to
withhold, and those are skipped rather than merely avoided in bulk.

---

## 4. The mapping table

`mapping_table.csv` has 1,324 rows across the typology the brief specifies.

| `mapping_type` | Rows | Meaning |
|---|---|---|
| `direct` | 964 | One old section → one new section |
| `merged` | 253 | Several old sections consolidated into one new section |
| `split` | 10 | One old section split across several new sections |
| `new_provision` | 21 | New section with no old-code equivalent |
| `removed` | 76 | Old provision with no counterpart in the new code |

Concordance coverage is complete for all three new codes: BNS 358/358,
BNSS 531/531, BSA 170/170.

The heavy skew towards `merged` over `split` is real, not an artefact. The BNS
reduces 511 IPC sections to 358, largely by folding related provisions together —
BNS 318 alone absorbs IPC 415, 417, 418, 419 and 420. `split` is also
undercounted by a deliberate choice: the BPRD tables map at sub-section
granularity, and rows are aggregated to whole sections, so IPC 302 → BNS 103(1)
and 103(2) is recorded as one `direct` mapping rather than a split. The
sub-section detail survives in the `notes` column.

### Spot checks against well-known mappings

Every one of these came out correct:

| Old | New | Type |
|---|---|---|
| IPC 302 (murder) | BNS 103 | direct |
| IPC 378 (theft) | BNS 303 | merged |
| IPC 420 (cheating) | BNS 318 | merged |
| IPC 498A (cruelty) | BNS 85 **and** BNS 86 | split |
| IPC 124A (sedition) | — | **removed** |
| CrPC 154 (FIR) | BNSS 173 | direct |
| CrPC 438 (anticipatory bail) | BNSS 482 | direct |
| CrPC 482 (inherent powers) | BNSS 528 | direct |
| IEA 65B (electronic records) | BSA 63 | direct |
| IEA 32 (dying declaration) | BSA 26 | direct |

### Disagreements between sources

413 cross-checks were run (348 BNS sections against the UP Police table, 65
against devgan.in). **12 disagreements** were found, all recorded in
`mapping_disagreements.csv`.

Every one is a `partial_overlap` — the sources agree on some of the old sections
but not all of them. **None** is a `no_overlap` case, i.e. there is no BNS
section for which two sources name completely different IPC provisions. The
disagreements fall into two kinds:

1. **Letter-suffix detail.** BNS 70 → BPRD says IPC 376D and 376DB; UP Police
   says 376D and 376DA. BNS 2 → BPRD says IPC 23C and 29A where UP Police says 23
   and (nothing).
2. **Depth of listing.** For consolidating sections, BPRD lists every constituent
   old provision where UP Police lists only the principal one. BNS 179, for
   example, is BPRD's IPC 237, 250, 251, 254, 258, 260 and 489B against UP
   Police's IPC 237 alone.

**Resolution:** BPRD is kept in every case, because it is the more authoritative
source (an MHA body) and the more complete listing. Nothing was silently
overwritten — each disagreement is a row in `mapping_disagreements.csv` with both
readings and the resolution recorded, so a reviewer can second-guess the call.

---

## 5. Case law

201 topic-case records across all 30 topics attempted, drawn from 174 distinct
judgments (22 judgments answer to more than one topic). No topic was dropped for thin
coverage; the weakest, bail in non-bailable offences, still returned 4 cases.

| Topic | Cases | | Topic | Cases |
|---|---|---|---|---|
| Murder and culpable homicide | 8 | | Registration of FIR | 7 |
| Theft | 8 | | Electronic evidence and certificates | 7 |
| Criminal breach of trust | 8 | | Confessions to police and discovery | 7 |
| Hurt and grievous hurt | 8 | | Circumstantial evidence | 7 |
| Abetment of suicide | 8 | | Robbery and dacoity | 6 |
| Criminal conspiracy | 8 | | Cruelty to a married woman (498A) | 6 |
| Extortion | 8 | | Dowry death | 6 |
| Default or statutory bail | 8 | | Mischief and damage to property | 6 |
| Dying declarations | 8 | | Anticipatory bail | 6 |
| Culpable homicide (not murder) | 7 | | Quashing of criminal proceedings | 6 |
| Cheating and fraud | 7 | | Burden of proof | 6 |
| Criminal intimidation | 7 | | Rape and sexual offences | 5 |
| Defamation | 7 | | Forgery and false documents | 5 |
| Unlawful assembly and rioting | 7 | | Cheating by personation | 5 |
| | | | Criminal trespass and house-breaking | 5 |
| | | | Bail in non-bailable offences | 4 |

**Why these topics.** They were chosen to span the three codes rather than to
maximise raw count: substantive offences (BNS/IPC), procedure and bail
(BNSS/CrPC), and evidence (BSA/IEA). Within each, the sections picked are the
ones that actually generate litigation, which is why coverage came out even.

**All 201 are Supreme Court decisions.** The harvester searches the Supreme
Court first and only falls back to High Courts if a topic comes up short; no
topic did. That is a good outcome for authority, and a limitation for breadth —
there is no High Court divergence in the corpus.

**Searches are anchored on the old section numbers, deliberately.** The new codes
took effect on 1 July 2024, so nearly all reported case law still cites IPC, CrPC
and Evidence Act numbering. Each case record carries the new-code equivalent
attached from the concordance — 201 of 201 cases have one — which is exactly the
old-to-new bridge the bot is meant to provide.

Year range: 1965–2026.

### How holdings were summarised, and why it matters

Holdings are **extractive**. Sentences are selected from the judgment's own text
using phrases courts use when stating a conclusion ("we are of the view", "it is
well settled", "the appeal is allowed"), scored by whether they engage the
provision in question, and joined. Nothing is paraphrased or generated.

This was a deliberate trade-off. Generated summaries would read better, but an
invented holding in a legal dataset is worse than no holding at all, and there is
no way to verify 201 generated summaries at this scale. Every record carries
`summary_method` recording that it is extractive, so the limitation travels with
the data.

The cost is real and should be stated plainly: an extracted conclusion is the
court's *operative outcome in that case*, not a curated statement of the *ratio
decidendi*. For a long judgment covering several issues, the extract may land on
a subsidiary point rather than the one the case is famous for. See §6.

---

## 6. Cleaning, deduplication, and the bugs that were found

Nine extraction defects were found and fixed - six in Phase 1, three more that
Phase 1.5 turned up in the Phase 1 output. They are listed because each one
would have silently corrupted the dataset, and because finding them is the
substance of the data-validation work.

1. **Section headings live in the page margin.** The gazette PDFs print each
   section's heading in the outer margin, alternating left and right by page
   parity, on the same baselines as the body text. Flat text extraction drops
   them. The parser works from word coordinates instead, splitting each page into
   a body band and margin bands.

2. **Headings were merging into their neighbours.** Lines *within* one margin
   heading sit about 9.2pt apart; consecutive headings are about 13.3pt apart.
   An initial threshold of 13.5pt merged adjacent headings and left 10 BNS
   sections untitled. Tightening it to 11.5pt reduced that to 2.

3. **Chapter titles are set in small caps.** The PDF encodes these as two
   interleaved font sizes on slightly different baselines — word initials at
   10pt, the rest at 7pt — so naive extraction yields
   `O F OFFENCES ... AND G OVERNMENT STAMPS`. Sorting by x-position and binding
   full-size single letters to the token after them recovers the text. Separately,
   kerning closes the space in `CHAPTER IV`, giving `CHAPTERIV`, which defeated
   the chapter regex and mis-assigned every section in chapters IV, V and VI.

4. **The concordance tables were losing a row per page.** These tables run across
   page breaks, and the top row of each page has no ruling line above it, so
   pdfplumber could not bound the cell and dropped the row silently — 31 BNSS and
   10 BSA sections missing. Supplying the top of the vertical rules as an explicit
   horizontal line recovers them. Coverage went from 500/531 to 531/531.

5. **The last section absorbed the schedules.** Nothing terminates the final
   section, so BNSS 531 ran to 169,729 characters — the whole First and Second
   Schedules. A schedule-heading guard cuts it at 1,840.

6. **devgan.in labels two IPC sections wrongly, in opposite ways.** The page
   markup gives each section both an HTML anchor and a displayed number, and
   the two disagree twice - each field being the wrong one once. Sections 29
   and 29A share the anchor `s29`, so trusting anchors collapses them into a
   single record; section 489A is *displayed* as "498A", so trusting the
   displayed number files counterfeiting-currency text under the
   cruelty-to-a-wife section. The rule that satisfies both is to take the
   anchor and fall back to the displayed number only when the anchor is
   already in use. Found in Phase 1.5; it had put the wrong heading on IPC
   498A - one of the most frequently cited provisions in Indian criminal law -
   throughout the Phase 1 mapping table.

7. **Words hyphenated across a line break were losing their hyphen.** Both the
   margin headings and the First Schedule columns wrap, and a word broken at
   the wrap keeps its hyphen, so joining lines with a space produced
   "non- treatment" and "non- cognizable". 22 section headings were affected,
   and headings propagate into hundreds of QA answers and chunks.

8. **Footnote markers rode along on section headings**, giving titles like
   "Rape1" and "Punishment for rape1".

9. **Two genuine data defects in the sources.** devgan.in gives both IPC 29 and
   IPC 29A the HTML anchor `s29`, which collapsed them into one record; the fix
   reads the displayed section number instead of the anchor. And some Indian
   Kanoon judgment text carries Windows-1252 punctuation with the high bit lost,
   so an em dash arrives as byte 0x17 and a left double quote as 0x13; these are
   repaired by restoring the bit and decoding as cp1252.

### Deduplication

- **Fine-tuning set:** 210 exact-duplicate questions were dropped. Deduplication
  is on a normalised form (lowercased, punctuation stripped) of the instruction,
  and separately on the normalised instruction/answer pair, so the same question
  reached by two different question families is caught. 14,524 pairs survive,
  all with unique instructions.
- **Retrieval corpus:** `chunk_id` uniqueness is enforced and verified; the
  IPC 29/29A collision above was found by exactly this check.
- **Encoding:** every chunk and every answer is scanned for replacement
  characters, C0 control characters and mojibake. Zero remain.
- **Empty fields:** every chunk is verified to have a non-empty `chunk_id`,
  `source`, `text` and `source_url`; every QA pair a non-empty instruction and
  output.

### Remaining warnings (all four, in full)

1. **BNS 191 and 340 have no heading.** Their margin headings could not be
   resolved; the section text is complete.
2. **BNSS 169 and 489 have no heading.** Same cause.
3. **BSA 162 has no heading.** Same cause.
4. **IPC 13 and 15 are under 80 characters.** These are the repealed definitions
   of "Queen" and "British India". They are genuinely that short.

Five untitled sections out of 1,059 is a 99.5% heading recovery rate.

---

## 7. The two datasets

### `finetune_dataset.jsonl` — 14,524 pairs

One JSON object per line with exactly `instruction`, `input` and `output`.
A parallel `finetune_dataset_index.jsonl`, aligned line for line, carries
`qa_type` and `source_chunk_id` for stratified splitting and analysis in Phase 2,
so the training file itself stays exactly to the specified schema.

| Question type | Pairs | What it teaches |
|---|---|---|
| `section_text` | 4,702 | What a section says |
| `old_to_new` | 2,444 | Which new section replaced an old one |
| `section_lookup` | 2,322 | Which section deals with a subject |
| `new_to_old` | 2,076 | Which old section a new one derives from |
| `punishment` | 1,170 | The penalty a section prescribes |
| `offence_classification` | 1,122 | Cognizable, bailable, and which court tries it |
| `case_law` | 406 | What a court held on a provision |
| `removed` | 152 | Old provisions with no counterpart |
| `transition` | 78 | Which code applies to a given date |
| `new_provision` | 42 | Offences with no old-code equivalent |
| `scope` | 10 | The limits of what the assistant will do |

Median answer 214 characters, maximum 1,411 — sized for Flan-T5.

`offence_classification` is the family the BNSS First Schedule made possible
(see §8). It is not redundant with `punishment`: the penalty is in the BNS
section text, but whether the offence is cognizable, whether it is bailable
and which court may try it are only in the Schedule.

**Augmentation.** Each underlying fact is phrased 3–5 different ways and 1–2
phrasings are sampled per fact, so the model sees varied wording without the set
collapsing into near-duplicates on a few popular sections. Coverage is spread
deliberately: a section contributes a handful of pairs at most, rather than
dozens being piled onto BNS 103.

**Every answer is derived from parsed source text or the concordance.** None is
written from the model's own knowledge of Indian law. A fine-tuning set is
exactly the wrong place for an unverified claim about what a section says.

**The `scope` family is not padding.** The brief forbids the bot from giving
legal advice, suggesting ways to evade liability, or posing as an advocate. A
model only learns that boundary if the boundary is in its training data, so
these 10 pairs cover refusals to advise, refusals to help evade liability,
refusals to impersonate counsel, and an honest statement of limitations — each
carrying the informational-only disclaimer.

### `retrieval_corpus.jsonl` — 2,819 chunks

| Type | Chunks |
|---|---|
| Statute sections | 2,355 |
| First Schedule classifications | 289 |
| Case summaries | 174 |
| Transition reference | 1 |

By act: BNS 358, BNSS 531, BSA 170, IPC 577, CrPC 535, Evidence Act 184, plus
289 First Schedule chunks, 174 case chunks and 1 reference chunk.

One chunk per section, never several sections merged — for a "which section
applies" assistant, retrieval precision matters more than chunk count, and a
chunk mixing BNS 303 and 304 makes it *harder* to see which provision an answer
rests on. Median chunk 682 characters; the largest, 13,433, is BNS 2
(definitions), which is genuinely that long.

**Each chunk carries its old/new counterpart in the embedded text**, not only in
metadata:

```
BNS 2023, Section 103 - Punishment for murder
(corresponds to IPC 1860 Section 302)
Chapter 6: Offences Affecting The Human Body

103. (1) Whoever commits murder shall be punished with death or ...
```

This is a deliberate design decision. A user who asks about "IPC 302" should
retrieve the BNS section that replaced it, and that only works if the old
numbering is inside the text the embedder actually sees. 2,432 of 2,819 chunks
(86.3%) carry such a cross-reference. Every chunk has a `source_url` so the bot
can cite it.

---

## 8. The BNSS First Schedule

The First Schedule classifies every offence under the BNS: what it is, what it
carries, whether it is cognizable, whether it is bailable, and which court may
try it. It answers "is theft bailable?" — a question none of the rest of the
corpus could answer, because the classification lives in the BNSS rather than in
the BNS section text.

| | |
|---|---|
| Part I entries (offences under the BNS) | **462** |
| Distinct BNS sections covered | **288** of 358 |
| Part II entries (offences against other laws) | **3** |
| Column-1 values that failed to parse | **0** |

**Why 288 and not 358.** The Schedule classifies offences, and 70 BNS sections
do not create one. They are the preliminary and definition sections (Chapter I),
the punishments chapter (Chapter II), the general exceptions (Chapter III), and
a handful of purely definitional provisions elsewhere — BNS 227 defines giving
false evidence while BNS 229 punishes it, and it is 229 that the Schedule lists.
The omission is the Schedule's own, correctly reproduced, not a gap in
extraction.

**Extraction.** pdfplumber's table detection finds nothing on these pages: the
Schedule is set in six columns with no ruling lines at all. Columns were
recovered from word x-positions and rows from vertical spacing, which needed
three things the Phase 1 machinery did not:

* **Columns 4 to 6 are centred**, so how far they run depends on how much text
  they hold. BNS 85's cognizability column carries a long conditional clause and
  overruns the point where BNS 57's column 5 begins; no single set of dividers
  works for both. Each entry's dividers are therefore taken from its own opening
  line, placed in the widest inter-word gap near the nominal boundary.
* **Rows are found by spacing, not by section number.** Lines within an entry sit
  9.6pt apart and a new entry follows a gap of about 14.5pt. This matters because
  one section can carry several classification entries that repeat no section
  number: BNS 356(3) is triable by a Court of Session when the complaint is made
  by the Public Prosecutor, and by a Magistrate of the first class in any other
  case. Keying on the section number alone merged the two and lost 32 entries.
* **A residual guard.** A classification column must open with a known term
  ("Cognizable", "Non-cognizable", "According as..."). One row in 462 had a word
  drift across a divider; the guard moves any prefix before that keyword back to
  the column it came from, and reports how often it fires.

**Cross-check.** devgan.in publishes a "BNSS Classification" block on its BNS
section pages, covering 201 of the 358 sections independently. Comparing against
it gives **4 disagreements**, recorded in `bnss_schedule_disagreements.csv`. All
four were checked against the raw gazette line and the extraction is faithful in
every case, so devgan is the source in error:

| Section | Field | Gazette (kept) | devgan |
|---|---|---|---|
| BNS 87 | cognizable | Cognizable | non-cognizable |
| BNS 90 | cognizable | Cognizable | non-cognizable |
| BNS 269 | triable by | Any Magistrate | first class |
| BNS 299 | triable by | Magistrate of the first class | any magistrate |

**In the datasets.** The Schedule contributes 1,122 `offence_classification` QA
pairs and 289 retrieval chunks. Chunks are grouped one per BNS section rather
than one per row: 462 entries over 288 sections would otherwise put near
identical chunks in the index and split a single answer across them — the same
reasoning that merged multi-topic judgments into one chunk each in Phase 1.

---

## 9. Train / validation / test splits, and the confusion set

### Why the split is not random

Every fact in this dataset is asked several ways — that is the point of the
augmentation. "What does BNS 103 say?" and "Explain BNS Section 103." are one
fact in two wordings. A random row-level split would put one in train and the
other in validation, and the validation score would then partly measure
memorised phrasing rather than generalisation.

So **groups are split, not rows**, and a group is defined in two steps:

1. Every pair carries a `source_chunk_id` naming the section, judgment, date or
   policy it is about. All pairs sharing one stay together.
2. Groups are then **merged across the concordance**. "Which BNS section replaced
   IPC 302?" and "Which IPC section corresponds to BNS 103?" are the same fact
   asked from both ends, so `ipc_302` and `bns_103` must not be separated either.
   A union-find over `mapping_table.csv` merges them, which also pulls whole
   split and merged families together — all of IPC 415, 417, 418, 419, 420 and
   BNS 318 land on one side of the line.

The second step goes beyond keying on the section alone, and it is the one that
matters most here: the old-to-new correspondence *is* the project's core content,
so letting it straddle the split would flatter exactly the capability the project
claims.

Two group keys had to be repaired for this to hold. Case-law pairs were keyed on
the topic-case pairing, but 22 judgments answer to more than one topic, so the
same judgment could have landed on both sides; they are now keyed on the Indian
Kanoon document id. Transition pairs were all keyed to a single bucket, which
would have put all 78 in one split; they are now keyed per (date, code pair).

### Result

| Split | Groups | Rows | Share | Target |
|---|---|---|---|---|
| train | 1,118 | 11,751 | 85.2% | 85% |
| val | 132 | 1,369 | 9.9% | 10% |
| test | 65 | 676 | 4.9% | 5% |

41 groups (728 rows) are held out entirely for the confusion set and appear in
none of the three.

Group sizes vary a lot, so hitting the ratio takes care. Assigning groups
largest-first by row count alone produced 88/8/4 **and** wrecked the
question-type balance — group size correlates with type, because a statute
section drags along section-text, punishment and classification pairs while a
judgment is a group of two or three. Largest-first filled train with sections and
left validation and test almost entirely case law: `case_law` was 1.2% of train
but 18% of test, and `offence_classification` was absent from val and test
altogether. Each group is now assigned to whichever split is least ahead of its
quota *for the question types that group actually contains*, with row count as a
tie-breaker.

### qa_type distribution across splits

| Question type | train | val | test |
|---|---|---|---|
| section_text | 32.6% | 32.6% | 32.5% |
| old_to_new | 16.7% | 16.6% | 16.2% |
| section_lookup | 16.1% | 16.2% | 16.1% |
| new_to_old | 14.5% | 14.6% | 15.0% |
| offence_classification | 7.6% | 7.7% | 7.5% |
| punishment | 7.5% | 7.6% | 7.8% |
| case_law | 2.9% | 2.9% | 3.0% |
| removed | 1.1% | 1.0% | 0.9% |
| transition | 0.6% | 0.5% | 0.4% |
| new_provision | 0.3% | 0.3% | 0.3% |
| scope | 0.1% | 0.1% | 0.0% |

**Balanced, with one flag:** `scope` has no example in test. There are only 10
scope pairs in the whole dataset, so 5% of them rounds to zero. That is worth
knowing rather than worth fixing by moving rows — scope behaviour needs its own
qualitative check in Phase 2, not a single test row.

Leakage is verified rather than assumed. `validate_data.py` recomputes the groups
and checks that no group and no instruction appears in more than one split, and
that no confusion-set group or question appears in any of them.

### The confusion set

`confusion_test_set.jsonl` holds **44** questions, held out of all three splits,
for Phase 4: these exact questions get put to both this bot and a
general-purpose LLM side by side. Each entry carries the correct answer, the
`mapping_type` it was drawn from, and a `why_confusing` note.

| Kind | Entries | The trap |
|---|---|---|
| `collision` | 14 | The number survives into the new code but attaches to an unrelated provision — CrPC 482 was the High Court's inherent powers, BNSS 482 is anticipatory bail |
| `merged` | 14 | Several old sections folded into one, so the answer is not a renumbering — BNS 318 absorbs IPC 415, 417, 418, 419 and 420 |
| `split` | 5 | One old section broken across several, so any single answer is incomplete — IPC 498A became BNS 85 *and* 86 |
| `removed` | 10 | Repealed outright, so any section number offered in reply is wrong by construction — IPC 124A, 377, 497, 309 |
| `transition` | 1 | Which code applies turns on the date of the offence, not the date of the trial |

Candidates come from the provisions that actually generate litigation — the same
sections the case-law topics identified — because that is where a confident wrong
answer costs something. `split` takes all five that exist in the entire
concordance.

**`removed` entries are restricted to a vetted allowlist**, not drawn from all 76
rows the concordance classifies that way, because two kinds of row are unsafe to
assert in a benchmark. Some were repealed long before 2024 and make dull items.
More importantly, some have a counterpart in substance even though the BPRD table
lists none: **Indian Evidence Act s.27**, on how much of the information received
from an accused in custody may be proved, is carried forward by the proviso to
**BSA s.23(2)**, but the BPRD table maps only ss.25 and 26 across. Our extraction
faithfully reproduces BPRD — the raw table was re-read to confirm it — but a
benchmark answer saying s.27 "has no counterpart" would be misleading, so it was
excluded. This is a limitation of the BPRD-derived `removed` class generally, and
is recorded in §10.

---

## 10. Known gaps and limitations


Stated honestly, because these feed the "challenges" section of the paper and the
video.

1. **Extractive case summaries capture the outcome, not always the ratio.** For a
   long judgment covering several issues the extract may land on a subsidiary
   point. The clearest example in the corpus: *Arjun Panditrao Khotkar* is the
   leading authority on s.65B certificates, but the extracted sentences land on a
   passage about when the certificate may be produced rather than on the
   three-judge ruling itself. Fixing this properly needs either headnote data or
   human curation.

2. **Case law is Supreme Court only.** No High Court decisions were needed to
   fill the topics, so there is no High Court divergence represented.

3. **Topic assignment is only as good as the keyword search.** Cases are
   collected by searching Indian Kanoon for a section number plus subject words,
   which returns judgments that *cite* the provision as well as judgments *about*
   it. *Vijay Madanlal Choudhary v. Union of India* — a money-laundering case —
   is filed under theft, robbery and hurt because it cites those sections in
   passing. The section-level metadata on each record is accurate; the topic
   label is a retrieval heuristic and should be treated as such.

4. **Almost no case law under the new codes.** The BNS/BNSS/BSA took effect on
   1 July 2024, so reported decisions citing the new numbering barely exist yet.
   The corpus bridges this with the concordance, but it is a bridge, not
   authority: a court's construction of IPC 420 is *persuasive* on BNS 318, not
   binding, and BNS 318 is not word-identical to IPC 420.

5. **The BNSS Second Schedule is still excluded.** The First Schedule is now
   parsed (§8); the Second Schedule, which is forms rather than substantive law,
   is not. Low priority.

6. **The `removed` classification inherits BPRD's omissions.** A section the BPRD
   table pairs with nothing is classified `removed`, but that is not always the
   same as having no successor: Indian Evidence Act s.27 is carried forward by
   the proviso to BSA s.23(2) while BPRD maps only ss.25 and 26 across. The
   confusion set works around this with a vetted allowlist (§9), but the 76
   `removed` rows in `mapping_table.csv` are not individually verified and the
   152 `removed` QA pairs inherit the caveat.

7. **Sub-section granularity is aggregated away in the CSV.** The BPRD tables map
   sub-section to sub-section; the deliverable CSV is section-to-section. The
   detail survives in `notes` but is not separately queryable.

8. **53 concordance rows could not be parsed** (13 BNS, 37 BNSS, 3 BSA) out of
   ~1,300. These are continuation fragments and repeated page headers whose cells
   wrapped unusually. They are counted and reported rather than guessed at, and
   because section-level coverage is nonetheless 100% for all three codes, no
   section is missing — at worst an individual old-section reference is.

9. **Old-code text is from secondary compilations**, not a gazette PDF. Two
   independent sources agree on 93–98% of section numbers, and the residue is
   repealed and sub-lettered entries. Adequate for numbering and headings; a
   gazette PDF would be better for verbatim text.

10. **The IPC/CrPC/IEA section counts include letter-suffixed provisions**, so
   they do not match the headline "511 sections" figure. This is correct for a
   lookup tool but will confuse a reader who expects 511.

11. **The splits protect against paraphrase leakage, not against topical
   overlap.** Groups are separated by section, judgment and concordance family
   (§9), so no fact appears on both sides. Two *different* sections about
   related subject matter can still fall on opposite sides, and a model that
   has learnt the shape of BNS theft provisions will do better on a held-out
   BNS property offence than on an unseen area of law. That is generalisation,
   not leakage, but it means the test score is not a measure of performance on
   genuinely unfamiliar material.

12. **No baseline comparison yet.** The project's central claim — that retrieval
    grounding beats a generic LLM's parametric memory on old/new section
    questions — is untested. The `old_to_new` and `new_to_old` families are the
    natural evaluation set for it in Phase 3.

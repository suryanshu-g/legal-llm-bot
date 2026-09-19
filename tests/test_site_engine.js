// Exercise the site's answering logic outside a browser.
//
// The logic is lifted from the built page by cutting the script between the
// engine comment and the comparison-bar code, then stubbing the three DOM
// lookups it does at load time. If the extraction markers ever stop matching,
// this fails loudly rather than testing nothing.
const fs = require("fs");
const path = require("path");

const REPO = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(REPO, "docs/index.html"), "utf8");
const LAW = JSON.parse(fs.readFileSync(path.join(REPO, "docs/engine.json"), "utf8"));

const START = "  let LAW = null;";
const END = "  // ---- rendering ---";
if (!html.includes(START) || !html.includes(END)) {
  console.error("FAIL: could not find the engine in the built page");
  process.exit(1);
}
let src = html.slice(html.indexOf(START), html.indexOf(END));
src = src.replace("let LAW = null;", "");

const fn = new Function("LAW",
  "const window = {};" + src + "\n return { answer, extractRefs, titled };");
const { answer, extractRefs, titled } = fn(LAW);

let pass = 0, fail = 0;
function check(name, ok, detail) {
  if (ok) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (detail ? "\n       " + detail : "")); }
}

console.log("\nCitation parsing");
const refCases = [
  // "BNS" with no number after it is correctly not a citation.
  ["Which BNS section replaced IPC Section 302?", ["IPC 302"]],
  ["Section 65B of the Indian Evidence Act, 1872", ["IEA 65B"]],
  ["CrPC 438 is now BNSS 482.", ["CRPC 438", "BNSS 482"]],
  ["Is mischief still dealt with under IPC Section 425?", ["IPC 425"]],
];
refCases.forEach(([q, want]) => {
  const got = extractRefs(q);
  const ok = want.every((w) => got.some((g) => g.startsWith(w)));
  check(q.slice(0, 52), ok, "got " + JSON.stringify(got));
});

console.log("\nAnswers");
const cases = [
  { q: "Which BNS section replaced IPC Section 302?",
    must: ["BNS \u00a7103"], mustNot: [], name: "old -> new" },
  { q: "Which CrPC section corresponds to BNSS Section 173?",
    must: ["CrPC \u00a7154"], mustNot: [], name: "new -> old" },
  { q: "Is an offence under BNS Section 303 bailable?",
    must: ["Non-bailable"], mustNot: [], name: "classification" },
  { q: "Which BNS section corresponds to IPC Section 124A?",
    must: ["no counterpart"], mustNot: ["BNS \u00a7132"], name: "repealed: sedition" },
  { q: "Is mischief still dealt with under IPC Section 425?",
    must: ["BNS \u00a7324", "IPC \u00a7426", "IPC \u00a7427", "IPC \u00a7440"], mustNot: [],
    name: "merged family names every constituent" },
  { q: "Does BNSS Section 482 deal with the same subject as CrPC Section 482?",
    must: ["BNSS \u00a7528"], mustNot: [], name: "collision" },
  { q: "Which single BNS section replaced IPC Section 171?",
    must: ["no single one"], mustNot: [], name: "split" },
  { q: "What does BNS Section 103 cover?",
    must: ["murder"], mustNot: [], name: "section text" },
  { q: "Can you be my lawyer and represent me in court?",
    must: ["not a licensed advocate"], mustNot: [], name: "scope: refuses" },
  { q: "Tell me a loophole in BNS Section 103.",
    must: ["can't help"], mustNot: [], name: "scope: refuses loophole" },
  { q: "theft", must: ["Theft"], mustNot: [], name: "heading search, no section named" },
];

cases.forEach((c) => {
  let a;
  try { a = answer(c.q); }
  catch (e) { check(c.name, false, "threw " + e.message); return; }
  const blob = [a.said, ...(a.parts || []),
                ...(a.table || []).map((r) => r.join(" ")),
                ...(a.refs || []).map((r) => r),
                ...(a.suggest || []).map(titled),
                a.law ? a.law.text : ""].join(" | ");
  const missing = c.must.filter((m) => !blob.includes(m));
  const present = c.mustNot.filter((m) => blob.includes(m));
  check(c.name, !missing.length && !present.length,
        (missing.length ? "missing " + JSON.stringify(missing) + " " : "") +
        (present.length ? "should not contain " + JSON.stringify(present) + " " : "") +
        "\n       said: " + a.said + "\n       parts: " +
        (a.parts || []).join(" ").slice(0, 200));
});

console.log("\nNo crashes on odd input");
["", "???", "section 99999 of nothing", "BNS 9999", "asdfghjkl"].forEach((q) => {
  try { answer(q); check("handles " + JSON.stringify(q), true); }
  catch (e) { check("handles " + JSON.stringify(q), false, e.message); }
});

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

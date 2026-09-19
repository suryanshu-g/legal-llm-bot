---
title: Indian Criminal Law After 1 July 2024
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Ask what a provision became after the 2024 recodification
---

# Indian criminal law after 1 July 2024

On 1 July 2024 India replaced the Indian Penal Code, 1860, the Code of Criminal
Procedure, 1973 and the Indian Evidence Act, 1872 with the BNS, BNSS and BSA.
Every section was renumbered; some provisions were merged, a few were split, and
a number were dropped.

This Space runs a `flan-t5-small` fine-tuned on that transition, reading passages
retrieved from the official gazette text and the Bureau of Police Research and
Development correspondence tables. Every answer shows the sources it rests on.

**Read the sources, not just the sentence.** The model is 77 million parameters.
On the held-out set it drifts section numbers to neighbouring provisions,
corrupts statutory titles and invents counterparts for repealed sections. The
retrieved passages are verified; the generated sentence is not.

This is general information about the law, not legal advice. It is a student
capstone project, not affiliated with or endorsed by any government body.

Code, datasets and the full results, including the runs that failed:
https://github.com/suryanshu-g/legal-llm-bot

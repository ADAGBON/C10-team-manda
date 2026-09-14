# Customer Support Chatbot — Submission Notes

## Architecture note

The project rubric provided to me describes a Bedrock **Flow**-based architecture
(a classifier node, Condition nodes, and separate Output nodes per path). However,
every project page I was given — Project Overview, Environment Setup, Instructions,
and Testing Framework — specifies the **Amazon Bedrock AgentCore managed harness**
architecture instead, and explicitly states:

> "Bedrock Agents Classic was closed to new customers on July 30, 2026, so this
> course uses its successor, the AgentCore managed harness."

> "There are no condition nodes or separate classifiers... your prompt supplies
> the behavior."

I built the project according to the Instructions and Environment Setup pages
(AgentCore harness + Gateway + Lambda + DynamoDB), since those are the pages that
describe the actual starter files, scripts, and CLI commands provided in
`project/starter/`. The rubric appears to predate this architecture migration.

Below is how each piece of rubric evidence maps to what this AgentCore-based
submission actually contains.

## Evidence mapping

| Rubric asks for (Flow-based) | What's in this submission (AgentCore-based) |
|---|---|
| Screenshot of full flow diagram | No visual flow exists in AgentCore — routing logic lives entirely in `system_prompt.txt` |
| Screenshot of classifier prompt configuration | `system_prompt.txt`, Category 1/2/3 definitions (the "classifier" is prose, not a node) |
| Screenshot of Condition node expressions | Not applicable — no Condition nodes; see the routing rules in `system_prompt.txt` |
| `flow-tests.json` | `harness-tests.json` (same purpose: one test per route, plus edge cases) |
| FAQ Prompt node template screenshot | The `{{FAQ}}` placeholder in `system_prompt.txt`, populated from `online_shop_faq.md` by `create_harness.py` |
| Flow test response screenshots | `chat.py` terminal transcripts (see `/evidence`) for a covered FAQ question, an uncovered question, and an other-request message |

## Deliverables included

- `system_prompt.txt` — main deliverable; defines all three routing behaviors, the
  bug-report collection checklist (description / stepsToReproduce / environment),
  the FAQ-only answering rule with a hand-off fallback, the redirect behavior, and
  prompt-injection hardening (stand-out item).
- `online_shop_faq.md` — fictional NimbusMart FAQ, substituted into `{{FAQ}}`.
- `harness-tests.json` — 12 single-turn test cases covering all three routes plus
  edge cases: an ambiguous/very-short message, a direct prompt-injection attempt,
  and an injection embedded inside a legitimate bug report (stand-out items).
- `output_eval_dataset.jsonl` — produced by `generate-eval-dataset.py`.
- `/evidence/` — DynamoDB screenshot, chat.py transcripts, Bedrock Evaluations
  results screenshot.

## Written observations (Bedrock Evaluations)

_Fill in after running the evaluation job:_

- Overall correctness score:
- Bug-report route observations:
- FAQ route observations:
- Redirect route observations:
- Any misrouted or low-scoring cases and how the prompt was adjusted:

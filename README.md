# Bridging the Tokenization Gap in African LLMs via SuperBPE

**Team Manda — TRI AI Saturdays Cohort 10, Project 3**
CodaBench competition [17580](https://www.codabench.org/competitions/17580/).

Standard tokenizers are trained mostly on English. Applied to morphologically
rich African languages they over-fragment words, inflating token counts, cost
and latency. We train a SuperBPE tokenizer on a multilingual African corpus so
the same text costs fewer tokens, under a hard losslessness constraint.

**Result:** 23,826 tokens on the development task against the baseline's
50,051 — a 52% reduction. 2.839 chars/token over a 25.5M-character corpus,
lossless, 0.17s runtime against a 20-minute limit.

## Dataset

The competition provides no dataset, so data engineering is part of the task.

`scripts/fetch_corpus.py` assembles a corpus from African-language Wikipedia
editions via Hugging Face, capped at 4,000 paragraphs per language. Selection
was driven by **breadth over depth**: the evaluation corpus is random samples
across a range of African datasets, so family coverage beats volume in a few
languages.

29 languages were collected across Bantu (15), Afro-Asiatic (5), Volta-Niger
(3), Senegambian (2), Mande, Ubangian, Austronesian and Germanic. Akan failed
to load. Total: 94,504 paragraphs, 25.5M characters.

We do not redistribute source text. Wikipedia is CC-BY-SA and licences differ
across sources, so the repository ships the fetch script and the derived
vocabulary instead; the corpus is reconstructible under each source's own
terms. Full provenance, bias and privacy analysis: `docs/data_card.pdf`.

## Training pipeline

**Stage 1 — subword BPE.** Byte-pair encoding where merges may not cross a word
boundary, so learned tokens stay inside words. Runs to 70% of the vocabulary.

**Stage 2 — superword BPE.** The whitespace constraint is lifted; merges may
span a space, producing superword tokens for frequent multi-word expressions.

**Refinement.** Because inference uses optimal segmentation (below), only the
token *set* matters, not merge order. So we segment the corpus optimally, drop
tokens nothing used, and spend the freed slots on new merges. Rounds scoring
worse on holdout are discarded; both improved here (3.650 → 3.669).

**Final hyperparameters:** `--vocab-size 20000 --transition 0.7
--max-token-len 32 --refine-rounds 2 --max-chars 8000000`. Training is CPU-only
and took about 11 minutes on 7.6M characters.

**Key design choices.**

- *Optimal segmentation instead of merge replay.* Score depends only on token
  count and losslessness is the only constraint, so `encode()` runs a dynamic
  program returning the fewest-token segmentation for the vocabulary. Never
  worse than replaying merges, and it enables the refinement loop.
- *Byte-level internals.* All 256 single bytes are in the vocabulary, so a
  segmentation always exists and `decode(encode(x)) == x` for any input.

## Evaluation

`scripts/validate_submission.py` reproduces the evaluator's calling convention
(`encode(list[str]) -> list[list[int]]`, then a *separate* instance to decode)
and checks losslessness, total tokens, unique ids against the 20,000 ceiling,
and timing projected onto the 2M-character final corpus.

Edge cases verified: empty strings, empty corpus, space-only lines, the full
printable ASCII range, Unicode markers, and non-ASCII input.

`scripts/evaluate.py` reports characters per token per language **and per
language family**, worst first, with the aggregate printed last — an average
hides which languages lost.

Both candidates were re-scored on the same 25.5M-character corpus for a fair
comparison: 2.651 (3.8M chars training) versus 2.839 (7.6M). The larger run
won and was submitted.

## Known limitations

Stated plainly, per the commitments in our Data Card and Impact Statement:

- **Training text was not preprocessed with the starter kit's
  `preporocess_text.py`.** The evaluator supplies ASCII transliterations with
  Unicode markers; we trained on raw UTF-8. Part of the vocabulary therefore
  learned byte patterns that cannot appear at evaluation. We believe this is
  the single largest remaining source of lost score.
- **No hyperparameter sweep.** Only `--transition 0.7` was run, under time
  pressure. The sweep is implemented but was not executed.
- **No morphological rules.** The brief mentions combining frequency with
  morphological rules; ours is purely statistical, as the SuperBPE paper
  itself is. Hand-written morphology for 29 languages without native-speaker
  validation would have been guesswork.
- **ASCII flattening degrades the languages we most want to serve** (see
  `docs/impact_statement_card.pdf`).

## Reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install datasets

python3 scripts/fetch_corpus.py --out corpus.txt --per-lang 4000

python3 scripts/train_tokenizer.py --corpus corpus.txt --vocab-size 20000 \
    --transition 0.7 --max-token-len 32 --refine-rounds 2 \
    --max-chars 8000000 --out tokenizer.json

python3 scripts/validate_submission.py --dir . --corpus corpus.txt
zip -X submission.zip tokenizer.py tokenizer.json
```

`tokenizer.py` is the submission entry point and loads `tokenizer.json` from
alongside itself. Nothing is trained, downloaded or fetched at evaluation time.
Requires ~4GB RAM at 8M characters.

## Repository layout

```
tokenizer.py                     submission entry point
tokenizer.json                   learned vocabulary (20,000 tokens)
scripts/fetch_corpus.py          corpus assembly
scripts/train_tokenizer.py       two-stage trainer + refinement
scripts/validate_submission.py   pre-submission checks
scripts/evaluate.py              per-language and per-family results
docs/                            the four cohort challenges
```

## Appendix

**Contributor:** Law Adagbon — sole contributor, working under the team name
Team Manda. No additional members and no assigned mentor.

**Cohort challenges** are in `docs/`.

## References

Liu, A., Hayase, J., Hofmann, V., Oh, S., Smith, N. A. and Choi, Y. (2025).
*SuperBPE: Space Travel for Language Models.* arXiv:2503.13423.

Oladipo, A. et al. (2023). *Better Quality Pre-training Data and T5 Models for
African Languages.* EMNLP 2023.

Kreutzer, J. et al. (2022). *Quality at a Glance: An Audit of Web-Crawled
Multilingual Datasets.* TACL 10:50–72.

Pushkarna, M., Zaldivar, A. and Kjartansson, O. (2022). *Data Cards: Purposeful
and Transparent Dataset Documentation for Responsible AI.* FAccT '22.

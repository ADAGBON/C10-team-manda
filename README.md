# Bridging the Tokenization Gap in African LLMs via SuperBPE

**Team Manda — TRI AI Saturdays Cohort 10, Project 3**
CodaBench competition [17580](https://www.codabench.org/competitions/17580/).

Standard tokenizers are trained mostly on English, so they over-fragment
morphologically rich African languages, inflating token counts, cost and
latency. We train a SuperBPE tokenizer on a multilingual African corpus so the
same text costs fewer tokens, under a hard losslessness constraint.

**Result:** 23,826 tokens on the development task against the baseline's
50,051 — a 52% reduction. 2.839 chars/token over a 25.5M-character corpus,
lossless, and the fastest runtime on the final leaderboard (1.38s of 1200s).

## Dataset

The competition provides no dataset, so data engineering is part of the task.
`scripts/fetch_corpus.py` assembles a corpus from African-language Wikipedia
editions via Hugging Face, capped at 4,000 paragraphs per language, chosen for
**breadth over depth**: the evaluation corpus is random samples across a range
of African datasets, so family coverage beats volume in a few languages.

29 languages across Bantu (15), Afro-Asiatic (5), Volta-Niger (3), Senegambian
(2), Mande, Ubangian, Austronesian and Germanic; Akan failed to load. 94,504
paragraphs, 25.5M characters.

Source text is not redistributed: Wikipedia is CC-BY-SA and licences differ, so
`data/` ships rebuild instructions instead. Provenance, bias and privacy
analysis: `docs/data_card.pdf`.

## Training Pipeline

**Data collection:** `scripts/fetch_corpus.py` (see Dataset).
**Preprocessing.** Paragraphs are whitespace-normalised, length-filtered and
written one per line. The fetch script applies the starter kit's
`preporocess_text.py` when present; it was unavailable for our run (see Known
limitations).


**Stage 1 — subword BPE.** Merges may not cross a word boundary, so tokens stay
inside words. Runs to 70% of the vocabulary.

**Stage 2 — superword BPE.** The whitespace constraint is lifted; merges may
span a space, producing superword tokens for frequent multi-word expressions.

**Refinement.** Because inference uses optimal segmentation (below), only the
token *set* matters, not merge order. We segment optimally, drop unused tokens
and spend the freed slots on new merges. Rounds scoring worse on holdout are
discarded; both improved (3.650 → 3.669).

**Hyperparameter search.** The trainer prints holdout chars/token. We varied
corpus size only (3.8M vs 7.6M characters); no transition sweep was run. Final:
`--vocab-size 20000 --transition 0.7 --max-token-len 32 --refine-rounds 2
--max-chars 8000000`. CPU-only, ~11 minutes.

**Key design choices.**

- *Optimal segmentation instead of merge replay.* Score depends only on token
  count and losslessness is the only constraint, so `encode()` runs a dynamic
  program returning the fewest-token segmentation. Never worse than replaying
  merges, and it enables the refinement loop.
- *Byte-level internals.* All 256 single bytes are in the vocabulary, so a
  segmentation always exists and `decode(encode(x)) == x` for any input.
- *Paragraph-bounded merges.* No token spans a paragraph boundary.

## Evaluation

`scripts/validate_submission.py` reproduces the evaluator's calling convention
(`encode(list[str]) -> list[list[int]]`, then a *separate* instance to decode)
and checks losslessness, total tokens, unique ids against the 20,000 ceiling,
and projected runtime on the 2M-character final corpus.

Edge cases verified: empty strings, empty corpus, space-only lines, full
printable ASCII, Unicode markers, non-ASCII input.
`scripts/evaluate.py` reports chars/token per language **and per language
family**, worst first, aggregate last — an average hides which languages lost.

Both candidates were re-scored on the same 25.5M-character corpus: 2.651 (3.8M
chars training) versus 2.839 (7.6M). The larger run won and was submitted.

## Known limitations

Stated plainly, per our Data Card and Impact Statement commitments:

- **Training text was not preprocessed with `preporocess_text.py`.** The
  evaluator supplies ASCII transliterations with Unicode markers; we trained on
  raw UTF-8, so part of the vocabulary learned byte patterns that cannot appear
  at evaluation. Likely the largest remaining source of lost score.
- **No transition sweep.** Only `--transition 0.7` was run.
- **No morphological rules.** The brief mentions combining frequency with
  morphological rules; ours is purely statistical, as the SuperBPE paper is.
  Hand-written morphology for 29 languages without native-speaker validation
  would have been guesswork.
- **ASCII flattening degrades the languages we most want to serve.**

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

`tokenizer.py` loads `tokenizer.json` from alongside itself. Nothing is trained
or fetched at evaluation time. Needs ~4GB RAM at 8M characters.

## Repository layout

```
tokenizer.py / tokenizer.json    submission entry point + 20,000-token vocab
scripts/                         fetch_corpus, train_tokenizer,
                                 validate_submission, evaluate
data/                            corpus rebuild instructions
docs/ (and doc/)                 the four cohort challenges
```

## Appendix

**Team Manda:** Abdurrazaq Khidir Olalekan · Adetayo Tella · Arturo Espinosa
Vargas · Ayomide Adeduro · Boluwatife Odunlami · Chitom Uzokwe · Lawrence
Adagbon · Surajo Nuhu Umar · Zakaria Tibtiba

**Mentor:** none assigned.

**Implementation and documentation:** Lawrence Adagbon.

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

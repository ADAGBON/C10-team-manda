# Data

The training corpus is not committed here. It is ~25 MB of text assembled from
African-language Wikipedia editions, which are CC-BY-SA; redistributing the
text would make this repository a licensed redistributor, so we ship the
script that rebuilds it instead.

## Rebuild the corpus

```bash
pip install datasets
python3 ../scripts/fetch_corpus.py --out corpus.txt --per-lang 4000
```

This produces `corpus.txt`, one paragraph per line: 94,504 paragraphs,
25,469,281 characters, across 29 languages spanning Bantu, Afro-Asiatic,
Volta-Niger, Senegambian, Mande, Ubangian, Austronesian and Germanic families.
Akan (ak) fails to load and is skipped.

The run is deterministic given the same Hugging Face dataset revision.

## What was actually trained on

The submitted tokenizer was trained on the first 8,000,000 characters of that
corpus (`--max-chars 8000000`), which is 29,901 paragraphs. See the repository
README for the full command and hyperparameters.

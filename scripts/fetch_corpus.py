"""
Fetch a broad African-language corpus and write corpus.txt for training.

Run this on your own machine (it needs network). It writes one paragraph per
line, already passed through the competition's own preprocessing function if
preporocess_text.py is on the path.

    pip install datasets
    python fetch_corpus.py --out corpus.txt --per-lang 4000

Breadth matters more than depth here: the evaluation corpus is described as
random samples from a *range* of African datasets, so a vocabulary trained on
many languages beats one trained deeply on a few.
"""

import argparse
import random
import sys

# Spread across language families, not just the big four.
WIKI_LANGS = [
    "sw", "yo", "ha", "ig", "am", "zu", "xh", "sn", "so", "af",
    "rw", "lg", "ti", "om", "mg", "st", "tn", "nso", "wo", "ak",
    "ny", "ts", "ve", "ss", "kg", "ln", "bm", "ff", "sg", "ee",
]

try:
    from preporocess_text import preprocess_text          # starter kit
except Exception:
    try:
        from preprocess_text import preprocess_text       # in case it's spelled right
    except Exception:
        preprocess_text = None


def clean(line):
    line = " ".join(line.split())
    if preprocess_text is not None:
        line = preprocess_text(line)
        line = " ".join(line.split())
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus.txt")
    ap.add_argument("--per-lang", type=int, default=4000,
                    help="paragraphs to keep per language")
    ap.add_argument("--min-len", type=int, default=60)
    args = ap.parse_args()

    if preprocess_text is None:
        print("WARNING: preporocess_text.py not found. Put the starter kit's "
              "copy next to this script so your training text matches what "
              "the evaluator will pass you.\n", file=sys.stderr)

    from datasets import load_dataset

    random.seed(0)
    kept_total = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for lang in WIKI_LANGS:
            try:
                ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                                  split="train", streaming=True)
            except Exception as e:
                print(f"  skip {lang}: {type(e).__name__}", file=sys.stderr)
                continue

            kept = 0
            try:
                for row in ds:
                    for para in row["text"].split("\n"):
                        para = clean(para)
                        if len(para) < args.min_len:
                            continue
                        out.write(para + "\n")
                        kept += 1
                        if kept >= args.per_lang:
                            break
                    if kept >= args.per_lang:
                        break
            except Exception as e:
                print(f"  {lang} stopped early: {type(e).__name__}",
                      file=sys.stderr)

            kept_total += kept
            print(f"  {lang}: {kept} paragraphs", flush=True)

    print(f"\nwrote {args.out}: {kept_total} paragraphs")
    print("If a language you care about was skipped, its Wikipedia config name "
          "may differ — check on Hugging Face and add it to WIKI_LANGS.")


if __name__ == "__main__":
    main()

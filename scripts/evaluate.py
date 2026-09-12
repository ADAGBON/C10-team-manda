"""
Evaluate the tokenizer per language and per language family.

The project brief asks for evaluation "across major African language families",
and a single average is exactly the statistic that lets a gain on one language
hide a loss on another. This reports both, against the competition's own
baseline (one token per character plus a terminator).

Expects one file per language, named <lang>.txt, one paragraph per line:

    python evaluate.py --corpus-dir heldout/ --out results.md
"""

import argparse
import glob
import os
import sys

# Rough family grouping for the languages fetch_corpus.py collects.
FAMILY = {
    "sw": "Bantu", "zu": "Bantu", "xh": "Bantu", "sn": "Bantu", "rw": "Bantu",
    "lg": "Bantu", "ny": "Bantu", "ts": "Bantu", "ve": "Bantu", "ss": "Bantu",
    "kg": "Bantu", "ln": "Bantu", "st": "Bantu", "tn": "Bantu", "nso": "Bantu",
    "yo": "Volta-Niger", "ig": "Volta-Niger", "ee": "Volta-Niger",
    "ak": "Kwa",
    "ha": "Afro-Asiatic", "am": "Afro-Asiatic", "ti": "Afro-Asiatic",
    "so": "Afro-Asiatic", "om": "Afro-Asiatic",
    "wo": "Senegambian", "ff": "Senegambian",
    "bm": "Mande",
    "sg": "Ubangian",
    "mg": "Austronesian",
    "af": "Germanic",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True,
                    help="directory of <lang>.txt held-out files")
    ap.add_argument("--tokenizer", default="tokenizer.py")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--out", default="results.md")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.tokenizer)) or ".")
    import importlib.util
    spec = importlib.util.spec_from_file_location("sub_tok", args.tokenizer)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok = mod.Tokenizer(args.vocab) if args.vocab else mod.Tokenizer()

    rows = []
    fam_chars, fam_tokens = {}, {}
    tot_chars = tot_tokens = tot_base = 0

    for path in sorted(glob.glob(os.path.join(args.corpus_dir, "*.txt"))):
        lang = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            paras = [l.rstrip("\n") for l in f if l.strip()]
        if not paras:
            continue

        chars = sum(len(p) for p in paras)
        encoded = tok.encode(paras)
        ntok = sum(len(x) for x in encoded)
        assert tok.decode(encoded) == paras, f"LOSSLESSNESS FAILED on {lang}"
        base = sum(len(p) + 1 for p in paras)

        fam = FAMILY.get(lang, "Other")
        fam_chars[fam] = fam_chars.get(fam, 0) + chars
        fam_tokens[fam] = fam_tokens.get(fam, 0) + ntok
        tot_chars += chars
        tot_tokens += ntok
        tot_base += base

        rows.append((lang, fam, chars, ntok, chars / ntok, base / ntok))

    rows.sort(key=lambda r: r[4])          # worst fertility first

    lines = ["# Evaluation", "",
             "Characters per token — higher is better. "
             "`vs baseline` is the factor fewer tokens than the competition's "
             "per-character baseline.", "",
             "## Per language", "",
             "| Language | Family | Chars | Tokens | Chars/token | vs baseline |",
             "|---|---|---|---|---|---|"]
    for lang, fam, chars, ntok, cpt, ratio in rows:
        lines.append(f"| {lang} | {fam} | {chars:,} | {ntok:,} | "
                     f"{cpt:.2f} | {ratio:.1f}x |")

    lines += ["", "## Per family", "",
              "| Family | Chars | Tokens | Chars/token |", "|---|---|---|---|"]
    for fam in sorted(fam_chars, key=lambda f: fam_chars[f] / fam_tokens[f]):
        lines.append(f"| {fam} | {fam_chars[fam]:,} | {fam_tokens[fam]:,} | "
                     f"{fam_chars[fam] / fam_tokens[fam]:.2f} |")

    if tot_tokens:
        lines += ["", f"**Overall:** {tot_chars:,} characters, "
                      f"{tot_tokens:,} tokens, "
                      f"{tot_chars / tot_tokens:.2f} chars/token, "
                      f"{tot_base / tot_tokens:.1f}x fewer tokens than baseline.",
                  "", "The overall figure is reported last deliberately. The "
                      "per-language table above is the one that matters — an "
                      "average hides which languages lost."]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""
Check a submission before uploading it.

Unzips (or reads) a submission directory, calls the tokenizer exactly the way
the evaluator does, verifies losslessness, counts unique tokens, and reports
the timing split so you can see whether you fit inside the 20-minute budget.

    python validate_submission.py --dir . --corpus heldout.txt
    python validate_submission.py --zip submission.zip --corpus heldout.txt

Scale the timing yourself: the final corpus is 2,000,000 post-processed
characters, and the number printed below is for whatever you fed it.
"""

import argparse
import importlib.util
import os
import sys
import tempfile
import time
import zipfile

FINAL_CORPUS_CHARS = 2_000_000
TIME_LIMIT_S = 20 * 60
MAX_UNIQUE_TOKENS = 20_000


def load_tokenizer_class(directory):
    path = os.path.join(directory, "tokenizer.py")
    if not os.path.exists(path):
        sys.exit(f"no tokenizer.py at the root of {directory}")
    spec = importlib.util.spec_from_file_location("submission_tokenizer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_tokenizer"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Tokenizer"):
        sys.exit("tokenizer.py does not define class Tokenizer")
    return module.Tokenizer


def flatten(encoded):
    if not encoded:
        return []
    if isinstance(encoded[0], int):
        return list(encoded)
    out = []
    for part in encoded:
        out.extend(flatten(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--zip")
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()

    workdir = args.dir
    tmp = None
    if args.zip:
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(args.zip) as z:
            names = z.namelist()
            if "tokenizer.py" not in names:
                print("WARNING: tokenizer.py is not at the archive root -- "
                      "the evaluator will not find it")
            z.extractall(tmp)
        workdir = tmp

    Tokenizer = load_tokenizer_class(workdir)

    with open(args.corpus, "r", encoding="utf-8") as f:
        paras = [l.rstrip("\n") for l in f if l.strip()]
    corpus = paras
    n_chars = sum(len(p) for p in paras)

    t0 = time.time()
    encode_tokenizer = Tokenizer()
    t1 = time.time()
    encoded = encode_tokenizer.encode(corpus)
    t2 = time.time()
    decode_tokenizer = Tokenizer()
    t3 = time.time()
    decoded = decode_tokenizer.decode(encoded)
    t4 = time.time()

    ok = decoded == corpus
    ids = flatten(encoded)
    unique = len(set(ids))
    total = t4 - t0
    projected = total * FINAL_CORPUS_CHARS / max(n_chars, 1)

    print(f"corpus              {n_chars:,} chars, {len(paras):,} paragraphs")
    print(f"lossless            {'YES' if ok else 'NO -- SUBMISSION INVALID'}")
    print(f"tokens emitted      {len(ids):,}")
    print(f"chars per token     {n_chars / max(len(ids), 1):.3f}")
    print(f"unique token ids    {unique:,} "
          f"({'ok' if unique <= MAX_UNIQUE_TOKENS else 'OVER THE 20,000 LIMIT'})")
    print()
    print(f"encoder init        {t1 - t0:6.1f}s")
    print(f"encode              {t2 - t1:6.1f}s")
    print(f"decoder init        {t3 - t2:6.1f}s")
    print(f"decode              {t4 - t3:6.1f}s")
    print(f"total               {total:6.1f}s")
    print(f"projected on 2M ch  {projected:6.1f}s of {TIME_LIMIT_S}s "
          f"({'ok' if projected < TIME_LIMIT_S * 0.6 else 'TOO CLOSE'})")

    if not ok:
        # Find the first paragraph that fails, to make debugging quick.
        for i, (a, b) in enumerate(zip(corpus, decoded)):
            if a != b:
                print(f"\nfirst mismatch at paragraph {i}:")
                print(f"  in : {a[:120]!r}")
                print(f"  out: {b[:120]!r}")
                break
        sys.exit(1)

    if tmp:
        print(f"\n(extracted to {tmp})")


if __name__ == "__main__":
    main()

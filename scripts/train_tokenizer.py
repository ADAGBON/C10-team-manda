"""
Train a SuperBPE tokenizer.

Two-stage training, following the SuperBPE idea (Liu et al., 2025):

  Stage 1  -- ordinary BPE.  Merges are forbidden from crossing a word
              boundary, so learned tokens are subwords.
  Stage 2  -- the whitespace constraint is lifted.  Merges may now span a
              space, producing "superword" tokens for frequent multi-word
              expressions.

After the two stages we run an optional refinement loop.  The inference-time
encoder does not replay the merge list -- it computes the segmentation with
the fewest tokens for the learned vocabulary (see tokenizer.py).  That means
only the *set* of tokens matters, not the order they were learned in, so we
can measure what each token actually earns, throw away the dead weight and
spend the freed slots on better candidates.

Everything is done over UTF-8 bytes.  The competition corpus is pure ASCII,
where this is identical to working on characters, but it makes encode/decode
lossless for any input at no cost.

Usage:
    python train_tokenizer.py --corpus data/*.txt --vocab-size 20000
"""

import argparse
import glob
import heapq
import json
import os
import sys
import time
from collections import defaultdict

SPACE = 0x20


# --------------------------------------------------------------------------
# core BPE trainer over linked-list sequences
# --------------------------------------------------------------------------

def train_merges(seqs, vocab, target_size, allow_cross_word, max_token_len,
                 min_pair_freq=2, log_every=1000, label=""):
    """Grow `vocab` up to `target_size` by merging frequent adjacent pairs.

    seqs   : list of lists of token ids (each list is one paragraph)
    vocab  : list of bytes objects, index == token id
    allow_cross_word : if False, a merge whose right half begins with a space
                       is rejected, which keeps tokens inside word boundaries
    Returns (new_seqs, vocab).
    """
    # Flatten into one array with -1 sentinels between paragraphs so that no
    # merge can ever span two paragraphs.
    sym = []
    for s in seqs:
        sym.extend(s)
        sym.append(-1)
    n = len(sym)

    prv = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    nxt[n - 1] = -1

    counts = defaultdict(int)
    where = defaultdict(set)

    def pair_at(i):
        j = nxt[i]
        if j == -1:
            return None
        a, b = sym[i], sym[j]
        if a == -1 or b == -1:
            return None
        return (a, b)

    for i in range(n - 1):
        p = pair_at(i)
        if p is not None:
            counts[p] += 1
            where[p].add(i)

    heap = [(-c, p) for p, c in counts.items()]
    heapq.heapify(heap)

    def bump(p, delta, pos, add):
        counts[p] += delta
        if add:
            where[p].add(pos)
        else:
            where[p].discard(pos)
        heapq.heappush(heap, (-counts[p], p))

    start_size = len(vocab)
    t0 = time.time()
    rejected = set()

    while len(vocab) < target_size and heap:
        negc, pair = heapq.heappop(heap)
        c = -negc
        if pair in rejected:
            continue
        if counts.get(pair, 0) != c:
            continue          # stale heap entry
        if c < min_pair_freq:
            break

        a, b = pair
        merged = vocab[a] + vocab[b]
        if len(merged) > max_token_len:
            rejected.add(pair)
            continue
        if not allow_cross_word and vocab[b][:1] == b" ":
            rejected.add(pair)
            continue

        new_id = len(vocab)
        vocab.append(merged)

        # Apply the merge everywhere it occurs.  Positions are consumed in
        # sorted order so that overlapping occurrences (aa -> a) behave.
        for i in sorted(where[pair]):
            if sym[i] != a:
                continue
            j = nxt[i]
            if j == -1 or sym[j] != b:
                continue

            left = prv[i]
            right = nxt[j]

            if left != -1 and sym[left] != -1:
                bump((sym[left], a), -1, left, False)
            if right != -1 and sym[right] != -1:
                bump((b, sym[right]), -1, j, False)

            sym[i] = new_id
            sym[j] = -1
            nxt[i] = right
            if right != -1:
                prv[right] = i

            if left != -1 and sym[left] != -1:
                bump((sym[left], new_id), 1, left, True)
            if right != -1 and sym[right] != -1:
                bump((new_id, sym[right]), 1, i, True)

        counts[pair] = 0
        where.pop(pair, None)

        done = len(vocab) - start_size
        if log_every and done % log_every == 0:
            print(f"  [{label}] {len(vocab):>6} tokens  "
                  f"last merge {merged[:24]!r} x{c}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    # Rebuild paragraph sequences.
    out, cur = [], []
    for s in sym:
        if s == -1:
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(s)
    if cur:
        out.append(cur)
    return out, vocab


# --------------------------------------------------------------------------
# optimal (fewest-token) segmentation, used for refinement + scoring
# --------------------------------------------------------------------------

def build_trie(vocab):
    root = {}
    for tid, tok in enumerate(vocab):
        node = root
        for byte in tok:
            node = node.setdefault(byte, {})
        node[-1] = tid
    return root


def segment(data, trie):
    """Fewest-token segmentation of `data` (bytes) via backward DP."""
    n = len(data)
    INF = 1 << 30
    dp = [INF] * (n + 1)
    dp[n] = 0
    pick_id = [0] * n
    pick_len = [0] * n

    for i in range(n - 1, -1, -1):
        node = trie
        best = INF
        bid = -1
        blen = 0
        j = i
        while j < n:
            node = node.get(data[j])
            if node is None:
                break
            j += 1
            tid = node.get(-1)
            if tid is not None:
                cost = dp[j] + 1
                if cost < best:
                    best, bid, blen = cost, tid, j - i
        dp[i] = best
        pick_id[i] = bid
        pick_len[i] = blen

    ids = []
    i = 0
    while i < n:
        ids.append(pick_id[i])
        i += pick_len[i]
    return ids


# --------------------------------------------------------------------------
# corpus loading
# --------------------------------------------------------------------------

def load_corpus(patterns, max_chars):
    paras = []
    total = 0
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    paras.append(line)
                    total += len(line)
                    if max_chars and total >= max_chars:
                        return paras
    return paras


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True,
                    help="text files, one paragraph per line")
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=20000)
    ap.add_argument("--transition", type=float, default=0.7,
                    help="fraction of the vocabulary learned before "
                         "cross-word merges are switched on")
    ap.add_argument("--max-token-len", type=int, default=32)
    ap.add_argument("--max-chars", type=int, default=0,
                    help="cap on training characters (0 = no cap)")
    ap.add_argument("--refine-rounds", type=int, default=2)
    ap.add_argument("--holdout", type=float, default=0.05,
                    help="fraction of paragraphs held out for scoring")
    args = ap.parse_args()

    paras = load_corpus(args.corpus, args.max_chars)
    if not paras:
        sys.exit("no text found -- check --corpus")

    n_hold = max(1, int(len(paras) * args.holdout)) if args.holdout > 0 else 0
    holdout = paras[:n_hold]
    train = paras[n_hold:] or paras
    print(f"corpus: {len(train)} training paragraphs, "
          f"{sum(len(p) for p in train):,} chars; "
          f"{len(holdout)} held out", flush=True)

    vocab = [bytes([i]) for i in range(256)]
    seqs = [list(p.encode("utf-8")) for p in train]

    t_stage1 = max(256, int(args.vocab_size * args.transition))

    print(f"\nstage 1: subword BPE -> {t_stage1} tokens", flush=True)
    seqs, vocab = train_merges(seqs, vocab, t_stage1,
                               allow_cross_word=False,
                               max_token_len=args.max_token_len,
                               label="stage1")

    print(f"\nstage 2: superword BPE -> {args.vocab_size} tokens", flush=True)
    seqs, vocab = train_merges(seqs, vocab, args.vocab_size,
                               allow_cross_word=True,
                               max_token_len=args.max_token_len,
                               label="stage2")

    def score(v):
        trie = build_trie(v)
        chars = sum(len(p) for p in holdout) or 1
        toks = sum(len(segment(p.encode("utf-8"), trie)) for p in holdout)
        return toks, chars / toks

    best_vocab = list(vocab)
    best_toks = None
    if holdout:
        best_toks, ratio = score(vocab)
        print(f"\nholdout: {best_toks:,} tokens, {ratio:.3f} chars/token",
              flush=True)

    # ---- refinement -------------------------------------------------------
    train_bytes = [p.encode("utf-8") for p in train]

    for r in range(args.refine_rounds):
        print(f"\nrefinement round {r + 1}", flush=True)
        trie = build_trie(vocab)
        usage = defaultdict(int)
        segs = []
        for b in train_bytes:
            ids = segment(b, trie)
            segs.append(ids)
            for t in ids:
                usage[t] += 1

        # A token earns its slot only if using it beats spelling the same
        # bytes with whatever else remains.  Zero-use tokens are pure waste.
        dead = [t for t in range(256, len(vocab)) if usage[t] == 0]
        print(f"  pruning {len(dead)} unused tokens", flush=True)
        if not dead:
            print("  vocabulary is saturated, stopping", flush=True)
            break

        keep = [t for t in range(len(vocab)) if t < 256 or usage[t] > 0]
        new_vocab = [vocab[t] for t in keep]
        remap = {old: new for new, old in enumerate(keep)}
        seqs = [[remap[t] for t in ids] for ids in segs]

        seqs, new_vocab = train_merges(seqs, new_vocab, args.vocab_size,
                                       allow_cross_word=True,
                                       max_token_len=args.max_token_len,
                                       label=f"refill{r + 1}")
        vocab = new_vocab
        if holdout:
            toks, ratio = score(vocab)
            better = best_toks is None or toks < best_toks
            print(f"  holdout: {toks:,} tokens, {ratio:.3f} chars/token"
                  f"  {'(best so far)' if better else '(worse, discarded)'}",
                  flush=True)
            if better:
                best_toks = toks
                best_vocab = list(vocab)
        else:
            best_vocab = list(vocab)

    vocab = best_vocab
    if best_toks is not None:
        print(f"\nkeeping best vocabulary: {best_toks:,} holdout tokens",
              flush=True)

    # ---- save -------------------------------------------------------------
    payload = {
        "format": "superbpe-v1",
        "tokens": [tok.hex() for tok in vocab],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    size = os.path.getsize(args.out)
    print(f"\nwrote {args.out}: {len(vocab)} tokens, {size / 1024:.0f} KB",
          flush=True)


if __name__ == "__main__":
    main()

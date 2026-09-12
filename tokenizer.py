"""
SuperBPE tokenizer -- competition submission entry point.

Interface matches the starter kit baseline:

    encode(texts: list[str]) -> list[list[int]]
    decode(encoded_texts: list[list[int]]) -> list[str]

The vocabulary is learned offline by train_tokenizer.py and shipped in
tokenizer.json next to this file.  Nothing is trained, downloaded, or
fetched at evaluation time, and the two calls are independent: the
evaluator builds one Tokenizer to encode and a separate one to decode, so
all state lives in the vocabulary file.

Encoding does not replay a merge list.  The score depends only on how many
tokens come out, so encode() computes the segmentation with the fewest
tokens for the shipped vocabulary, by dynamic programming over the input.
That is never worse than replaying merges in order.

All work happens on UTF-8 bytes.  Every single byte 0x00-0xFF is in the
vocabulary, so a segmentation always exists and decode(encode(x)) == x for
any input, not just the preprocessed ASCII the evaluator promises.
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_VOCAB = os.path.join(_HERE, "tokenizer.json")


class Tokenizer:

    def __init__(self, vocab_path: str | None = None) -> None:
        path = vocab_path or _DEFAULT_VOCAB
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.tokens: list[bytes] = [bytes.fromhex(h) for h in payload["tokens"]]

        # Trie of byte -> child dict; the key -1 holds the id ending here.
        root: dict = {}
        for tid, tok in enumerate(self.tokens):
            node = root
            for byte in tok:
                child = node.get(byte)
                if child is None:
                    child = {}
                    node[byte] = child
                node = child
            node[-1] = tid
        self._trie = root

    # -- encoding ----------------------------------------------------------

    def _encode_one(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        n = len(data)
        if n == 0:
            return []

        trie = self._trie
        INF = 1 << 30
        dp = [INF] * (n + 1)
        dp[n] = 0
        pick_id = [0] * n
        pick_len = [0] * n

        # Backward pass: dp[i] is the fewest tokens covering data[i:].
        for i in range(n - 1, -1, -1):
            node = trie
            best = INF
            best_id = -1
            best_len = 0
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
                        best = cost
                        best_id = tid
                        best_len = j - i
            dp[i] = best
            pick_id[i] = best_id
            pick_len[i] = best_len

        out = []
        i = 0
        while i < n:
            out.append(pick_id[i])
            i += pick_len[i]
        return out

    def encode(self, texts: list[str]) -> list[list[int]]:
        if isinstance(texts, str):        # tolerated, not expected
            return self._encode_one(texts)
        return [self._encode_one(text) for text in texts]

    # -- decoding ----------------------------------------------------------

    def _decode_one(self, ids: list[int]) -> str:
        tokens = self.tokens
        return b"".join(tokens[i] for i in ids).decode("utf-8", errors="replace")

    def decode(self, encoded_texts: list[list[int]]) -> list[str]:
        if encoded_texts and isinstance(encoded_texts[0], int):
            return self._decode_one(encoded_texts)   # tolerated, not expected
        return [self._decode_one(ids) for ids in encoded_texts]

    # -- convenience -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tokens)

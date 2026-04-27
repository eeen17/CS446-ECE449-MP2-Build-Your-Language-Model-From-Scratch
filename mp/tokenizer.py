from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


class BPETokenizer:
    """A minimal byte-level BPE tokenizer.

    Base vocabulary ids 0-255 correspond to raw byte values.
    Learned merges add new token ids starting at 256.
    """

    def __init__(self, merges: list[tuple[int, int]]) -> None:
        self.merges = merges
        self.base_vocab_size = 256
        self.vocab_size = self.base_vocab_size + len(merges)

        # Map pair -> rank (merge order).
        self.merge_ranks: dict[tuple[int, int], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

        # Map token id -> raw bytes represented by that token.
        self.id_to_bytes: dict[int, bytes] = {
            i: bytes([i]) for i in range(self.base_vocab_size)
        }
        for idx, (left, right) in enumerate(merges, start=self.base_vocab_size):
            self.id_to_bytes[idx] = self.id_to_bytes[left] + self.id_to_bytes[right]

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        min_pair_freq: int = 2,
        progress_callback: Callable[[int, int, tuple[int, int], int], None] | None = None,
    ) -> "BPETokenizer":
        """Train a byte-level BPE tokenizer from text.

        TODO:
        Implement standard byte-level BPE training with deterministic behavior.

        Required behavior:
        1) Start from the UTF-8 bytes of the training text, where each byte is one token.
        2) Repeatedly count frequencies of adjacent token pairs in the current token sequence.
        3) Pick exactly one pair to merge each round:
           - prefer higher frequency
           - break ties by lexicographically smaller pair
        4) Stop if:
           - the vocabulary would exceed `vocab_size`, or
           - no pair appears at least `min_pair_freq` times, or
           - the sequence is too short to contain a pair
        5) When merging, replace every non-overlapping occurrence of the chosen pair
           with the next token id (starting at 256, then 257, ...).
        6) Record merges in the order they are learned and return `BPETokenizer(merges)`.

        Edge cases to handle:
        - empty input text should produce a valid tokenizer with no merges
        - `vocab_size < 256` should not silently produce invalid token ids
        - repeated runs on the same text must learn the same merge sequence

        Optional progress reporting:
        - if `progress_callback` is provided, call it after each learned merge with:
          `progress_callback(num_merges_learned, total_possible_merges, best_pair, best_count)`
        """
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids using learned merges.

        TODO:
        Apply the learned merges greedily by merge priority.

        Required behavior:
        1) Convert the input text to its UTF-8 byte sequence.
        2) Repeatedly scan the current token sequence for adjacent pairs that were learned
           during training.
        3) If multiple learned pairs are present, choose the one with the smallest merge rank
           (the pair learned earliest).
        4) Merge one occurrence of that best-ranked pair, update the sequence, and continue.
        5) Stop only when no adjacent pair in the sequence is mergeable.

        Important details:
        - start from raw bytes every time; do not reuse training-time state
        - preserve the exact byte content of the string
        - empty input should return an empty list
        """
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text.

        TODO:
        Reconstruct the original byte stream and decode it as UTF-8.

        Required behavior:
        1) Look up the byte representation for each token id using `self.id_to_bytes`.
        2) Concatenate those byte chunks in order.
        3) Decode the result as UTF-8.

        Important details:
        - decoding should succeed even if byte boundaries are imperfect
        - empty input should return the empty string
        - the output should round-trip correctly for normal UTF-8 text
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save tokenizer merges to JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer merges from JSON and reconstruct a BPETokenizer."""
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        merges_raw = payload.get("merges", [])
        merges = [tuple(pair) for pair in merges_raw]
        return cls(merges)

    @staticmethod
    def _replace_pair(
        token_ids: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:
        """Replace all non-overlapping occurrences of `pair` with `new_token_id`."""
        result: list[int] = []
        i = 0
        while i < len(token_ids):
            if (
                i + 1 < len(token_ids)
                and token_ids[i] == pair[0]
                and token_ids[i + 1] == pair[1]
            ):
                result.append(new_token_id)
                i += 2
            else:
                result.append(token_ids[i])
                i += 1
        return result

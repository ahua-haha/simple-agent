"""Lexicographic fractional order keys."""

from __future__ import annotations

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE = len(ALPHABET)
DEFAULT_KEY = ALPHABET[BASE // 2]


def key_between(previous: str | None, next_: str | None) -> str:
    if previous is not None and next_ is not None and previous >= next_:
        raise ValueError("previous key must be less than next key")

    if previous is None and next_ is None:
        return DEFAULT_KEY

    prefix = ""
    index = 0
    while True:
        previous_digit = _digit_at(previous, index, 0)
        next_digit = _digit_at(next_, index, BASE - 1)
        if next_digit - previous_digit > 1:
            return prefix + ALPHABET[(previous_digit + next_digit) // 2]
        prefix += ALPHABET[previous_digit]
        index += 1


def key_after(previous: str | None) -> str:
    return key_between(previous, None)


def _digit_at(key: str | None, index: int, default: int) -> int:
    if key is None or index >= len(key):
        return default
    return ALPHABET.index(key[index])

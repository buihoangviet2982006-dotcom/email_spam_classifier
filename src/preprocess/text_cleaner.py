"""Shared text preprocessing utilities for training, evaluation, and API inference."""

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
_PUNCT_TRANS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\ufffd": " ",
    }
)


def clean_text(text: str | None) -> str:
    """Normalize email text consistently with the preprocessing notebook.

    - Normalizes Unicode smart quotes/dashes and removes replacement chars.
    - Strips non-ASCII / mojibake characters.
    - Lowercases text and collapses consecutive whitespace.
    """
    if text is None:
        return ""
    normalized = str(text).translate(_PUNCT_TRANS).lower()
    ascii_only = _NON_ASCII_RE.sub(" ", normalized)
    return _WHITESPACE_RE.sub(" ", ascii_only).strip()


def combine_subject_and_body(subject: str | None, text: str) -> str:
    """Combine optional email subject and body before cleaning."""
    parts = [part.strip() for part in (subject or "", text or "") if part and part.strip()]
    return clean_text(" ".join(parts))

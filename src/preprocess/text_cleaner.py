"""Shared text preprocessing utilities for training, evaluation, and API inference."""

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Normalize email text consistently with the preprocessing notebook.

    Lowercases text and collapses consecutive whitespace into a single space.
    """
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(text).lower()).strip()


def combine_subject_and_body(subject: str | None, text: str) -> str:
    """Combine optional email subject and body before cleaning."""
    parts = [part.strip() for part in (subject or "", text or "") if part and part.strip()]
    return clean_text(" ".join(parts))

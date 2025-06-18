"""Simple language detector for English vs Russian.

Logic:
1. If the message is short (< short_threshold characters after trimming), rely on alphabet heuristics:
   - If the text contains only ASCII printable characters, assume English.
   - Otherwise assume Russian (covers Cyrillic and anything non-ASCII).
2. For longer texts, fall back to `langdetect`.
   - Any code starting with "en" counts as English.
   - Everything else defaults to Russian.

Returns:
    "en" or "ru".
"""
from __future__ import annotations

import re
from langdetect import detect as _detect, LangDetectException
import logging

logger = logging.getLogger(__name__)

_ASCII_RE = re.compile(r"^[\x20-\x7E]+$")  # printable ASCII range


def detect_language(text: str, short_threshold: int = 20) -> str:
    """Detect language (en/ru) with heuristics for short texts.

    Args:
        text: Input text.
        short_threshold: Length below which we use alphabet heuristic.

    Returns:
        "en" for English, "ru" for Russian (default fallback).
    """
    if not text:
        logger.debug("[LangDetect] Empty text, defaulting to 'ru'.")
        return "ru"

    text = text.strip()

    # Heuristic for short messages
    if len(text) < short_threshold:
        if _ASCII_RE.match(text):
            logger.debug("[LangDetect] Short ASCII text detected as 'en': %s", text)
            return "en"
        logger.debug("[LangDetect] Short non-ASCII text detected as 'ru': %s", text)
        return "ru"

    # Fallback to langdetect for longer messages
    try:
        code = _detect(text)
        logger.debug("[LangDetect] langdetect returned '%s' for text: %.30s", code, text)
    except LangDetectException:
        logger.warning("[LangDetect] langdetect failed, defaulting to 'ru'.")
        return "ru"

    return "en" if code.startswith("en") else "ru"

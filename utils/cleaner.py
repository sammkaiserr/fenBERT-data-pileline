"""
utils/cleaner.py — Text Cleaning Module

PURPOSE:
    Clean raw text extracted from HTML pages before passing it
    to FinBERT for sentiment analysis. Removes HTML artifacts,
    normalizes whitespace, and handles encoding issues.

SINGLE RESPONSIBILITY:
    This module ONLY cleans text.
    It does NOT scrape, analyze, or write files.

LIBRARIES:
    re (regex)       → Built-in Python module for pattern matching.
                        Used to strip HTML tags and normalize whitespace.

    html             → Built-in Python module for handling HTML entities.
                        Converts &amp; → &, &lt; → <, etc.

WHY clean text before FinBERT?
    → Raw scraped text contains HTML tags (<p>, <img>), special entities
      (&amp;, &nbsp;), and extra whitespace. FinBERT was trained on clean
      English text — feeding it raw HTML degrades accuracy.
"""

import re
import html


def clean_text(raw_text: str) -> str:
    """
    Clean and normalize raw text for sentiment analysis.

    Args:
        raw_text: The raw text string, potentially containing HTML tags,
                  entities, and irregular whitespace.

    Returns:
        Cleaned, normalized text string ready for FinBERT.
        Returns empty string if input is None or empty.

    CLEANING PIPELINE (order matters):
        1. Handle None/empty input
        2. Decode HTML entities (&amp; → &)
        3. Strip HTML tags (<p>, <img>, etc.)
        4. Remove URLs
        5. Normalize whitespace (collapse multiple spaces/newlines)
        6. Strip leading/trailing whitespace
    """
    # ── Step 1: Guard against None or empty input ──────────────
    if not raw_text:
        return ""

    text = raw_text

    # ── Step 2: Decode HTML entities ───────────────────────────
    text = html.unescape(text)

    # ── Step 3: Remove all HTML tags ───────────────────────────
    text = re.sub(r"<[^>]+>", "", text)

    # ── Step 4: Remove URLs ────────────────────────────────────
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # ── Step 5: Normalize whitespace ───────────────────────────
    text = re.sub(r"\s+", " ", text)

    # ── Step 6: Strip leading and trailing whitespace ──────────
    text = text.strip()

    return text


def extract_company_name(title: str) -> str:
    """
    Extract the company name from a financial news headline.

    Args:
        title: The article headline string.

    Returns:
        The extracted company name, or the first meaningful words
        of the title if no known pattern matches.

    STRATEGY (checked in order):
        1. "XXX Share Price Live Updates" → XXX
        2. "Buy/Sell/Reduce/Accumulate XXX; target..." → XXX
        3. "XXX shares ..." or "XXX stock ..." → XXX
        4. "XXX IPO ..." → XXX
        5. Fallback: first 3 words of the title (best effort)

    WHY regex and not NER (Named Entity Recognition)?
        → Financial headlines follow very predictable patterns.
          Regex is faster, has zero dependencies, and works offline.
          NER (spaCy) can be added later for edge cases.
    """
    if not title:
        return "Unknown"

    # Decode any HTML entities first (e.g., M&amp;M → M&M)
    title = html.unescape(title)

    # ── Pattern 1: "HDFC Bank Share Price Live Updates: ..." ───
    # Very common in Economic Times articles.
    match = re.match(r"^(.+?)\s+Share Price Live Updates", title, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # ── Pattern 2: "Buy/Sell/Reduce/Accumulate COMPANY; target..." ─
    # Common in MoneyControl broker recommendation articles.
    match = re.match(
        r"^(?:Buy|Sell|Reduce|Accumulate|Hold)\s+(.+?)(?:;|,|\s+target)",
        title,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # ── Pattern 3: "COMPANY shares ..." or "COMPANY stock ..." ─
    # Common pattern: "TCS shares jump 3% after results"
    match = re.match(r"^(.+?)\s+(?:shares?|stock)\b", title, re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        # Avoid matching "Rs 5 lakh crore" or numbers as company names
        if name and not re.match(r"^(?:Rs|INR|\d)", name, re.IGNORECASE):
            return name

    # ── Pattern 4: "COMPANY IPO ..." ───────────────────────────
    match = re.match(r"^(.+?)\s+IPO\b", title, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # ── Fallback: Use the first 3 meaningful words ─────────────
    # Strip common leading noise words and take the first few words
    # as a best-effort company identifier.
    words = title.split()
    # Take up to 3 words, stop at common separators
    name_words = []
    for word in words[:4]:
        if word in (":", ";", "-", "|", "–", "—"):
            break
        name_words.append(word)

    return " ".join(name_words) if name_words else "Unknown"


"""
scraper/article_scraper.py — Web Scraper for Full Article Extraction

PURPOSE:
    Given an article URL, download the HTML page and extract:
    - Title
    - Full article text (body paragraphs)
    - Published date

SINGLE RESPONSIBILITY:
    This module ONLY extracts article content from HTML.
    It does NOT parse RSS feeds, clean text, or analyze sentiment.

LIBRARIES:
    requests         → Downloads the HTML page. We already use it in the
                        RSS module, so no new dependency is introduced.

    BeautifulSoup    → Parses HTML and lets us navigate the DOM tree using
                        CSS-like selectors. Much safer than regex for HTML.
                        We use the 'lxml' parser for speed (C-based).

WHY BeautifulSoup over regex?
    HTML is NOT a regular language — regex will break on nested tags,
    attributes, and edge cases. BeautifulSoup builds a proper parse tree
    and handles malformed HTML gracefully.
"""

import requests
from bs4 import BeautifulSoup

# Import shared settings from central config
from config import REQUEST_TIMEOUT, USER_AGENT


def scrape_article(url: str) -> dict | None:
    """
    Download an article page and extract its content.

    Args:
        url: Full URL of the article to scrape.

    Returns:
        A dict with keys:
            - title          (str): Article headline
            - text           (str): Full article body text
            - published_date (str): Publication date (if found)
        Returns None if the page could not be fetched or parsed.

    ERROR HANDLING:
        - Network errors   → caught by requests, returns None
        - Missing elements → returns sensible defaults ("" or "Unknown")
        - Never crashes the pipeline — the caller can skip failed articles
    """
    try:
        # ── Step 1: Download the HTML ──────────────────────────────
        # We send a User-Agent header because many news sites block
        # requests that look like automated scripts (no UA = bot).
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

        # Raise an exception for HTTP error codes (4xx, 5xx).
        # This is caught by the except block below.
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        # Covers: ConnectionError, Timeout, HTTPError, TooManyRedirects
        print(f"  ✗ Failed to fetch article: {e}")
        return None

    # ── Step 2: Parse the HTML with BeautifulSoup ──────────────
    # 'lxml' is the fastest HTML parser available for Python.
    # It's written in C and handles malformed HTML well.
    soup = BeautifulSoup(response.text, "lxml")

    # ── Step 3: Extract the title ──────────────────────────────
    # Try the <title> tag first, which every valid HTML page has.
    # .get_text() strips the HTML tags and returns plain text.
    # .strip() removes leading/trailing whitespace.
    title = _extract_title(soup)

    # ── Step 4: Extract the article body text ──────────────────
    # News articles store their content in <p> tags inside an
    # <article> element or a <div> with a class like "article-body".
    # We try multiple strategies to maximize coverage across sites.
    text = _extract_body_text(soup)

    # ── Step 5: Extract the published date ─────────────────────
    # Dates are stored inconsistently across websites.
    # We check common patterns: <time> tag, meta tags, etc.
    published_date = _extract_published_date(soup)

    return {
        "title": title,
        "text": text,
        "published_date": published_date,
    }


def _extract_title(soup: BeautifulSoup) -> str:
    """
    Extract the article title from the parsed HTML.

    Strategy (in priority order):
        1. <h1> tag — most news sites use h1 for the headline
        2. <title> tag — fallback, every HTML page has one
        3. "No Title" — last resort default
    """
    # Strategy 1: <h1> tag (most reliable for article headlines)
    h1_tag = soup.find("h1")
    if h1_tag:
        return h1_tag.get_text(strip=True)

    # Strategy 2: <title> tag (always exists, but may include site name)
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)

    return "No Title"


def _extract_body_text(soup: BeautifulSoup) -> str:
    """
    Extract the main article body text from the parsed HTML.

    Strategy (in priority order):
        1. Find <article> tag → get all <p> tags inside it
        2. Find common article container divs (class-based)
        3. Fallback: get all <p> tags from the entire page

    WHY join <p> tags?
        News articles are structured as a series of <p> paragraphs.
        Joining them with spaces gives us the full readable text.
    """
    # Strategy 1: <article> tag — the semantic HTML5 element for articles.
    # Most modern news sites wrap article content in <article>.
    article_tag = soup.find("article")
    if article_tag:
        paragraphs = article_tag.find_all("p")
        if paragraphs:
            return " ".join(p.get_text(strip=True) for p in paragraphs)

    # Strategy 2: Common CSS class names used by news sites.
    # Different sites use different class names for their article body.
    common_selectors = [
        {"class_": "article-body"},
        {"class_": "story-body"},
        {"class_": "article-content"},
        {"class_": "post-content"},
        {"class_": "entry-content"},
        {"class_": "content-body"},
    ]
    for selector in common_selectors:
        container = soup.find("div", selector)
        if container:
            paragraphs = container.find_all("p")
            if paragraphs:
                return " ".join(p.get_text(strip=True) for p in paragraphs)

    # Strategy 3: Fallback — collect ALL <p> tags on the page.
    # This may include non-article text (nav, footer), but it's
    # better than returning nothing. The cleaner module will tidy it up.
    all_paragraphs = soup.find_all("p")
    if all_paragraphs:
        return " ".join(p.get_text(strip=True) for p in all_paragraphs)

    return ""


def _extract_published_date(soup: BeautifulSoup) -> str:
    """
    Extract the publication date from the parsed HTML.

    Strategy (in priority order):
        1. <time> tag with 'datetime' attribute (semantic HTML5)
        2. <meta> tag with property 'article:published_time' (Open Graph)
        3. <meta> tag with name 'pubdate' or 'publish_date'
        4. "Unknown" fallback
    """
    # Strategy 1: <time datetime="2024-01-15T10:30:00Z">
    # The HTML5 <time> tag is the most reliable source.
    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        return time_tag["datetime"]

    # Strategy 2: Open Graph meta tag — widely used by news sites
    # <meta property="article:published_time" content="2024-01-15">
    og_date = soup.find("meta", property="article:published_time")
    if og_date and og_date.get("content"):
        return og_date["content"]

    # Strategy 3: Other common meta tag patterns
    for attr_name in ["pubdate", "publish_date", "date"]:
        meta_tag = soup.find("meta", attrs={"name": attr_name})
        if meta_tag and meta_tag.get("content"):
            return meta_tag["content"]

    return "Unknown"

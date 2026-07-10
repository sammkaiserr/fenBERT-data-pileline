"""
scraper/rss.py — RSS Feed Scraper Module

PURPOSE:
    Fetch and parse RSS feeds from financial news websites.
    Returns a list of dictionaries with article metadata.

SINGLE RESPONSIBILITY:
    This module ONLY handles RSS feed parsing.
    It does NOT scrape full articles, clean text, or run sentiment analysis.

LIBRARIES:
    feedparser  → Parses RSS/Atom XML feeds into Python dicts.
                   Handles encoding, malformed XML, and date parsing
                   automatically. The gold standard for RSS in Python.

    requests    → Handles the HTTP download. We use requests instead of
                   feedparser's built-in fetcher because macOS Python
                   often has SSL certificate issues. requests ships with
                   certifi (bundled CA certificates) and avoids this.

    datetime    → Fallback timestamp when published date is missing.
"""

import feedparser
import requests
from datetime import datetime

# Import shared settings from the central config file
from config import REQUEST_TIMEOUT, USER_AGENT


def fetch_feed(url: str) -> feedparser.FeedParserDict:
    """
    Download and parse an RSS feed from the given URL.

    Args:
        url: Full URL of the RSS feed endpoint.

    Returns:
        A feedparser.FeedParserDict containing the parsed feed.
        Returns an empty feed object on failure (feed.entries = []).

    WHY requests + feedparser?
        feedparser's built-in HTTP uses urllib, which on macOS often
        fails with SSL_CERTIFICATE_VERIFY_FAILED. requests uses
        certifi's bundled CA certs, which works reliably everywhere.
    """
    print(f"  Fetching: {url}")

    try:
        # requests.get() downloads the RSS XML with proper SSL handling.
        # We pass a User-Agent header because some sites (e.g., Reuters)
        # block requests that look like bots.
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

        # raise_for_status() throws an HTTPError for 4xx/5xx responses
        # so we catch it below instead of silently returning bad data.
        response.raise_for_status()

        # feedparser.parse() accepts raw XML strings — it detects this
        # is content (not a URL) and parses directly. No second HTTP call.
        feed = feedparser.parse(response.text)

    except requests.exceptions.RequestException as e:
        # Catches: ConnectionError, Timeout, HTTPError, DNS failures
        print(f"  ✗ Failed to fetch feed: {e}")
        # Return an empty feed-like object so the caller doesn't crash
        return feedparser.FeedParserDict(entries=[])

    # .bozo is True when feedparser encounters malformed XML.
    # We warn but still return partial data (feedparser is forgiving).
    if feed.bozo:
        print(f"  ⚠ Feed may be malformed: {feed.get('bozo_exception', 'Unknown')}")

    print(f"  ✓ Found {len(feed.entries)} entries")
    return feed


def parse_entries(feed: feedparser.FeedParserDict, source: str) -> list[dict]:
    """
    Extract structured article metadata from parsed feed entries.

    Args:
        feed:   Parsed feed object from fetch_feed().
        source: Human-readable name of the news source (e.g., "Reuters").

    Returns:
        List of dicts, each with exactly these keys:
            - title          (str): Headline of the article
            - source         (str): Name of the news source
            - link           (str): URL to the full article
            - published_date (str): Publication date as string

    WHY plain dicts?
        Keeps downstream modules (article scraper, FinBERT, CSV writer)
        decoupled from feedparser. They receive simple dicts and never
        need to know about feedparser internals.
    """
    articles = []

    for entry in feed.entries:
        article = {
            # .get() with defaults ensures we never crash on missing fields.
            # Different feeds have different field names and some omit fields.
            "title": entry.get("title", "No Title"),
            "source": source,
            "link": entry.get("link", ""),
            "published_date": entry.get(
                "published",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        }
        articles.append(article)

    return articles


def scrape_rss(feeds: list[dict]) -> list[dict]:
    """
    Main entry point — scrape all configured RSS feeds.

    Args:
        feeds: List of feed configs from config.py.
               Each dict has: {"url": "...", "source": "..."}

    Returns:
        Combined list of article dicts from ALL feeds.
        Each dict has keys: title, source, link, published_date.

    WORKFLOW:
        1. Loop through each feed config
        2. Fetch the RSS XML via requests
        3. Parse XML via feedparser
        4. Extract article metadata into clean dicts
        5. Combine all articles and return
    """
    all_articles = []

    print("=" * 60)
    print("  RSS Feed Scraper — Starting")
    print("=" * 60)

    for feed_config in feeds:
        url = feed_config["url"]
        source = feed_config["source"]

        print(f"\n📰 Source: {source}")

        # Step 1: Download and parse the RSS XML
        feed = fetch_feed(url)

        # Step 2: Extract metadata into standardized dicts
        articles = parse_entries(feed, source)

        # Step 3: Add to the combined list
        all_articles.extend(articles)

    print("\n" + "=" * 60)
    print(f"  Scraping complete. Total articles: {len(all_articles)}")
    print("=" * 60)

    return all_articles

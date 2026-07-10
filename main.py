"""
main.py — Pipeline Orchestrator

PURPOSE:
    This is the entry point for the Financial News Sentiment Analyzer.
    It orchestrates the complete pipeline:

        RSS Feeds
            ↓
        Article Extraction (web scraping)
            ↓
        Text Cleaning
            ↓
        FinBERT Sentiment Analysis
            ↓
        CSV Output

SINGLE RESPONSIBILITY:
    This module ONLY coordinates the pipeline.
    It does NOT contain scraping, cleaning, or analysis logic — those
    live in their respective modules. main.py just calls them in order.

USAGE:
    python main.py
"""

from config import RSS_FEEDS
from scraper.rss import scrape_rss
from scraper.article_scraper import scrape_article
from utils.cleaner import clean_text, extract_company_name
from ai.finbert import analyze_sentiment
from utils.csv_writer import save_to_csv


def run_pipeline():
    """
    Execute the full financial news sentiment analysis pipeline.

    WORKFLOW:
        Step 1 — RSS Scraping:
            Fetch article metadata (title, link, source, date) from
            all configured RSS feeds.

        Step 2 — Article Extraction:
            For each article link, download the full HTML page and
            extract the article body text using BeautifulSoup.

        Step 3 — Text Cleaning:
            Clean the raw text (remove HTML tags, URLs, extra whitespace)
            to prepare it for FinBERT.

        Step 4 — Sentiment Analysis:
            Pass the cleaned text through FinBERT to get a sentiment
            label (positive/negative/neutral) and confidence score.

        Step 5 — CSV Output:
            Save all results to output/news.csv using Pandas.

    Returns:
        List of processed article dicts (same data that's written to CSV).
    """
    # ==================================================================
    # STEP 1: Scrape RSS feeds for article metadata
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 1: Scraping RSS Feeds")
    print("=" * 60)

    rss_articles = scrape_rss(RSS_FEEDS)

    if not rss_articles:
        print("\n⚠ No articles found in RSS feeds. Exiting.")
        return []

    # ==================================================================
    # STEP 2 & 3: Extract full article text and clean it
    # ==================================================================
    # We combine extraction + cleaning in one loop to avoid storing
    # large amounts of raw HTML in memory.
    print("\n" + "=" * 60)
    print("  STEP 2: Extracting & Cleaning Article Text")
    print("=" * 60)

    articles_with_text = []

    for i, article in enumerate(rss_articles, start=1):
        print(f"\n  [{i}/{len(rss_articles)}] {article['title'][:60]}...")

        # ── Extract full article text from the URL ─────────────
        scraped = scrape_article(article["link"])

        if scraped and scraped["text"]:
            # Use the scraped full-text for analysis
            raw_text = scraped["text"]
        else:
            # Fallback: use the RSS title as the text for sentiment analysis.
            # Titles are often informative enough for FinBERT.
            print("    ⚠ Could not extract article text. Using title instead.")
            raw_text = article["title"]

        # ── Clean the text ─────────────────────────────────────
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            print("    ⚠ Cleaned text is empty. Skipping.")
            continue

        # Store the article with its cleaned text for the next step
        articles_with_text.append({
            "title": article["title"],
            "source": article["source"],
            "published_date": article["published_date"],
            "cleaned_text": cleaned_text,
        })

    print(f"\n  ✓ Articles with usable text: {len(articles_with_text)}")

    if not articles_with_text:
        print("\n⚠ No articles with extractable text. Exiting.")
        return []

    # ==================================================================
    # STEP 3: Analyze sentiment with FinBERT
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 3: Running FinBERT Sentiment Analysis")
    print("=" * 60)

    # The first call to analyze_sentiment() triggers model loading.
    # Subsequent calls reuse the cached model (lazy singleton).
    print("  Loading FinBERT model (first call only)...\n")

    processed_articles = []

    for i, article in enumerate(articles_with_text, start=1):
        # Run FinBERT on the cleaned text
        result = analyze_sentiment(article["cleaned_text"])

        # Extract the company name from the headline
        company_name = extract_company_name(article["title"])

        # Build the final output dict: Company Name + Sentiment
        processed_article = {
            "Company Name": company_name,
            "Sentiment": result["sentiment"],
        }
        processed_articles.append(processed_article)

        # Progress indicator
        print(
            f"  [{i}/{len(articles_with_text)}] "
            f"{company_name:<30} → {result['sentiment'].upper()}"
        )

    # ==================================================================
    # STEP 4: Save results to CSV
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 4: Saving Results to CSV")
    print("=" * 60)

    filepath = save_to_csv(processed_articles)

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 60)
    print("  ✅ Pipeline Complete!")
    print("=" * 60)
    print(f"  Articles processed: {len(processed_articles)}")
    print(f"  Output file:        {filepath}")

    # Count sentiments for a quick summary
    sentiment_counts = {}
    for article in processed_articles:
        s = article["Sentiment"]
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

    print("  Sentiment breakdown:")
    for sentiment, count in sorted(sentiment_counts.items()):
        print(f"    {sentiment:>8}: {count}")

    # Preview first 5 rows
    print("\n  Preview (first 5):")
    print(f"  {'Company Name':<30} {'Sentiment':<10}")
    print(f"  {'-'*30} {'-'*10}")
    for article in processed_articles[:5]:
        print(f"  {article['Company Name']:<30} {article['Sentiment']:<10}")

    print("=" * 60 + "\n")

    return processed_articles


# ---------------------------------------------------------------------------
# Entry point — run the pipeline when this script is executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()

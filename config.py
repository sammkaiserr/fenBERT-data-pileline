"""
config.py — Central Configuration for Financial News Analyzer

All project-wide settings live here:
- RSS feed URLs and source names
- HTTP request settings
- Output paths
- Model configuration

WHY a separate config file?
  → Single source of truth. When you add a new RSS feed or change
    an output path, you edit ONE file instead of hunting through modules.
"""

# ---------------------------------------------------------------------------
# RSS Feed Sources
# ---------------------------------------------------------------------------
# Each entry is a dict with:
#   "url"    → the RSS feed endpoint
#   "source" → human-readable name (stored alongside every article)
#
# You can add as many feeds as you like — the scraper loops over this list.
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    {
        "url": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
        "source": "Reuters",
    },
    {
        "url": "https://www.moneycontrol.com/rss/latestnews.xml",
        "source": "MoneyControl",
    },
    {
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "source": "Economic Times Markets",
    },
]

# ---------------------------------------------------------------------------
# HTTP Request Settings
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 15  # seconds — prevents hanging on unresponsive servers

# User-Agent header — some sites block requests without a realistic UA string.
# This mimics a standard Chrome browser on macOS.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Output Settings
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"             # folder for CSV results
OUTPUT_FILENAME = "news.csv"      # default output file name

# ---------------------------------------------------------------------------
# FinBERT Model Settings
# ---------------------------------------------------------------------------

FINBERT_MODEL_NAME = "ProsusAI/finbert"   # Hugging Face model identifier
MAX_TOKEN_LENGTH = 512                    # FinBERT's max input token length

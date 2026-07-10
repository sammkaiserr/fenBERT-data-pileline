"""
utils/csv_writer.py — CSV Output Module

PURPOSE:
    Save processed news articles (with sentiment analysis results)
    to a CSV file using Pandas.

SINGLE RESPONSIBILITY:
    This module ONLY writes data to CSV.
    It does NOT scrape, clean, or analyze text.

LIBRARIES:
    pandas  → The standard Python data analysis library. We use it here
              for DataFrame creation and CSV export. Pandas handles:
              - Column ordering
              - Proper CSV escaping (commas in text, quotes, etc.)
              - UTF-8 encoding
              - index=False to avoid writing row numbers

    os      → Built-in module for file path operations and directory creation.

WHY Pandas instead of csv.writer?
    → Pandas DataFrames give us column validation, easy reordering,
      and a one-liner to_csv(). The csv module requires manual handling
      of headers, row writing, and encoding. Since we'll use Pandas
      elsewhere (data analysis, future features), it's worth using here.
"""

import os
import pandas as pd

from config import OUTPUT_DIR, OUTPUT_FILENAME


def save_to_csv(articles: list[dict], filename: str = None) -> str:
    """
    Save a list of processed article dicts to a CSV file.

    Args:
        articles: List of dicts, each containing:
                  - Company Name   (str)
                  - Sentiment      (str): positive / negative / neutral
        filename: Optional custom filename. Defaults to config.OUTPUT_FILENAME.

    Returns:
        The absolute path to the created CSV file.

    WORKFLOW:
        1. Ensure the output directory exists
        2. Build the full file path
        3. Create a Pandas DataFrame from the article list
        4. Write the DataFrame to CSV
        5. Return the file path
    """
    # ── Step 1: Create the output directory if it doesn't exist ─
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 2: Build the full file path ───────────────────────
    output_file = filename if filename else OUTPUT_FILENAME
    filepath = os.path.join(OUTPUT_DIR, output_file)

    # ── Step 3: Create a Pandas DataFrame ──────────────────────
    # The 'columns' parameter ensures consistent column order.
    df = pd.DataFrame(articles, columns=[
        "Company Name",
        "Sentiment",
    ])

    # ── Step 4: Write to CSV ──────────────────────────────────
    # index=False prevents row numbers in output.
    df.to_csv(filepath, index=False, encoding="utf-8")

    print(f"  ✓ Results saved to: {filepath}")
    print(f"  ✓ Total records: {len(df)}")

    return filepath


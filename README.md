# 📈 AI-Powered Financial News Sentiment Analysis

An end-to-end pipeline that scrapes financial news from RSS feeds, extracts article text, and analyzes sentiment using **FinBERT** (a BERT model fine-tuned on financial text).

## 🏗️ Architecture

```
RSS Feeds (MoneyControl, Economic Times, Reuters)
        ↓
   RSS Scraper (feedparser + requests)
        ↓
   Article Extractor (requests + BeautifulSoup)
        ↓
   Text Cleaner (regex + html)
        ↓
   FinBERT Sentiment Analysis (Hugging Face Transformers)
        ↓
   CSV Output (Pandas)
```

## 📁 Project Structure

```
financial-news-analyzer/
├── main.py                    # Pipeline orchestrator
├── config.py                  # Central configuration
├── requirements.txt           # Dependencies
├── scraper/
│   ├── rss.py                 # RSS feed scraper
│   └── article_scraper.py     # Web scraper (BeautifulSoup)
├── ai/
│   └── finbert.py             # FinBERT sentiment analysis
├── utils/
│   ├── cleaner.py             # Text cleaning + company name extraction
│   └── csv_writer.py          # CSV output (Pandas)
└── output/
    └── news.csv               # Generated results
```

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/sammkaiserr/fenBERT-data-pileline.git
cd fenBERT-data-pileline
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline
```bash
python main.py
```

### 5. View results
```bash
cat output/news.csv
```

## 📊 Output Format

```csv
Company Name,Sentiment
HDFC Bank,negative
TCS,neutral
City Union Bank,positive
Wipro,negative
Infosys,neutral
```

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `feedparser` | RSS/Atom feed parsing |
| `requests` | HTTP requests |
| `beautifulsoup4` | HTML parsing |
| `transformers` | FinBERT model (Hugging Face) |
| `torch` | PyTorch inference |
| `pandas` | CSV output |

## 📌 Key Features

- **Modular architecture** — each module has a single responsibility
- **Graceful error handling** — failed articles don't crash the pipeline
- **Smart company extraction** — regex patterns extract company names from headlines
- **Lazy model loading** — FinBERT loads once and caches for all predictions
- **Configurable** — add/remove RSS sources by editing `config.py`
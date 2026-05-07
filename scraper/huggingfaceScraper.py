"""
preprocess.py
-------------
Scrapes headlines from a provided CSV of URLs to generate the dataset for Hugging Face.
This was the script that generated SGavin/CIS5190_NewsSource
"""

import logging
import time
import random
from urllib.parse import urlparse
import argparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CATEGORY_MAP = {
    "politics": "politics", "business": "business", "tech": "technology",
    "entertainment": "entertainment", "sports": "sports", "health": "health",
    "world": "world", "us": "us_news", "lifestyle": "lifestyle",
    "opinion": "opinion", "weather": "weather", "crime": "crime_justice",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

def _identify_source(url: str):
    """Extracts and identifies the news source from the URL."""
    domain = urlparse(url).netloc.lower()
    if "foxnews" in domain or "fox" in domain: return "fox"
    if "nbc" in domain: return "nbc"
    return None

def _fetch_headline(url: str, session: requests.Session, max_retries: int = 3):
    """Fetches the headline from the target URL using OpenGraph tags or the title element."""
    for attempt in range(1, max_retries + 1):
        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            resp = session.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            og = soup.find("meta", property="og:title")
            if og and og.get("content", "").strip():
                return og["content"].strip()
            return soup.title.string.strip() if soup.title else None
        except Exception:
            time.sleep((2 ** attempt) + random.uniform(0.5, 1.0))
    return None

def scrape_urls_to_csv(
    input_csv: str,
    output_csv: str = "headlines.csv",
    url_column: int = 0,
    delay_range: tuple = (0.5, 2.0),
) -> pd.DataFrame:
    """Reads a CSV of URLs, scrapes their headlines, and saves the results to a new CSV."""
    df_raw = pd.read_csv(input_csv, header=None)
    urls = df_raw.iloc[:, url_column].dropna().astype(str).tolist()
    logger.info(f"Loaded {len(urls)} URLs from {input_csv}")

    session = requests.Session()
    records = []

    for i, url in enumerate(urls):
        url = url.strip()
        source = _identify_source(url)
        if not source: continue

        headline = _fetch_headline(url, session)
        if not headline: continue

        records.append({
            "news_source": source,
            "headline": headline,
            "url": url,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(urls)}")
        time.sleep(random.uniform(*delay_range))

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df)} rows to {output_csv}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape news headlines from a list of URLs.")
    parser.add_argument("input_csv", help="Path to input CSV containing URLs")
    parser.add_argument("--output", default="headlines.csv", help="Path to output CSV")

    args = parser.parse_args()
    
    scrape_urls_to_csv(args.input_csv, output_csv=args.output)
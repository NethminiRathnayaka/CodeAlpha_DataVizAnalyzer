"""
Task 1: Web Scraping Module
===========================
Extracts unstructured/structured product data and reviews from web sources using
BeautifulSoup, requests, and structured parsers, with fallback synthetic generators.
"""

import os
import re
import time
import random
import logging
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

class ProductScraper:
    """
    Robust Web Scraper for extracting product details and customer reviews.
    Supports BooksToScrape (standard scraping sandbox) and custom URLs.
    """

    def __init__(self, base_url: str = "http://books.toscrape.com/", delay: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def scrape_books_catalog(self, max_pages: int = 3) -> pd.DataFrame:
        """
        Scrapes books from books.toscrape.com across multiple catalog pages.
        Returns a structured DataFrame.
        """
        records: List[Dict[str, Any]] = []
        logger.info(f"Starting catalog scrape from {self.base_url} (Max pages: {max_pages})")

        for page in range(1, max_pages + 1):
            url = f"{self.base_url}/catalogue/page-{page}.html" if page > 1 else f"{self.base_url}/index.html"
            try:
                resp = self.session.get(url, timeout=10)
                if resp.status_code != 200:
                    logger.warning(f"Page {page} returned status {resp.status_code}, terminating pagination.")
                    break
                
                soup = BeautifulSoup(resp.content, "lxml")
                articles = soup.find_all("article", class_="product_pod")

                if not articles:
                    logger.info(f"No products found on page {page}.")
                    break

                for art in articles:
                    title_elem = art.h3.find("a")
                    title = title_elem.get("title", title_elem.text).strip() if title_elem else "Unknown Title"
                    
                    price_elem = art.find("p", class_="price_color")
                    raw_price = price_elem.text.strip() if price_elem else "£0.00"
                    
                    rating_classes = art.find("p", class_="star-rating").get("class", [])
                    rating_word = [c for c in rating_classes if c != "star-rating"]
                    rating_val = RATING_MAP.get(rating_word[0], 3) if rating_word else 3

                    avail_elem = art.find("p", class_="instock availability")
                    availability = avail_elem.text.strip() if avail_elem else "In stock"
                    
                    rel_link = title_elem.get("href", "") if title_elem else ""
                    if rel_link.startswith("catalogue/"):
                        full_link = f"{self.base_url}/{rel_link}"
                    else:
                        full_link = f"{self.base_url}/catalogue/{rel_link}"

                    records.append({
                        "product_id": f"BOOK-{len(records)+1:04d}",
                        "product_name": title,
                        "category": "Books & Literature",
                        "raw_price": raw_price,
                        "rating": rating_val,
                        "stock_status": availability,
                        "product_url": full_link,
                        "source": "books.toscrape.com",
                    })

                logger.info(f"Page {page} scraped successfully ({len(articles)} items).")
                time.sleep(self.delay)

            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                break

        df = pd.DataFrame(records)
        return df


def generate_benchmark_reviews(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, multifaceted benchmark dataset of 300+ customer reviews
    spanning 5 core tech categories with varied sentiment, prices, and ratings.
    Used for offline execution, testing, and benchmark comparisons.
    """
    random.seed(seed)

    categories = {
        "Smartphones & Accessories": [
            "ProMax Ultra 5G Phone", "Galaxy Edge Fold", "SlimGrip Magnetic Case", 
            "UltraCharge 65W GaN Charger", "Nova Zen Smartphone 128GB"
        ],
        "Laptops & Computers": [
            "AeroBook 14 OLED Laptop", "Predator X Gaming Rig", "ErgoMechanical Wireless Keyboard",
            "UltraVision 27 4K Monitor", "Precision Laser Gaming Mouse"
        ],
        "Audio & Headphones": [
            "AcousticPro Noise-Cancelling Headphones", "BassPulse True Wireless Earbuds",
            "StudioReference Desktop Monitors", "SoundWave Bluetooth Portable Speaker", "ClearVoice USB Podcast Mic"
        ],
        "Smart Wearables": [
            "FitPulse HR Smartwatch", "AeroBand Fitness Tracker", "Titanium Rugged GPS Watch",
            "SleepSense Smart Ring", "VisionAir AR Smart Glasses"
        ],
        "E-Readers & Tablets": [
            "PageGlow 7-inch Paperwhite E-Reader", "SlatePro 11 Drawing Tablet",
            "PocketReader Color E-Ink", "TabAir Lite 10.1 Display", "StylusPen Ultra-Sensitive"
        ]
    }

    positive_templates = [
        ("Absolute game changer! {feature} exceeded my expectations.", "Incredible quality and value for money. The {aspect} is buttery smooth and responsive. Highly recommend to everyone!"),
        ("Best purchase of the year!", "I was skeptical at first, but the {aspect} blew me away. Super fast shipping, beautiful packaging, and flawless performance."),
        ("Exceptional build and performance.", "The craftsmanship is top-tier. Battery lasts all day, and the {feature} works like a charm. 10/10 purchase."),
        ("Very impressed with customer service and design.", "Great ergonomic feel. I use this every single day for work and leisure. The {aspect} makes daily workflow effortless."),
        ("Phenomenal product! Worth every penny.", "Sturdy construction, crisp aesthetics, and rock-solid reliability. You truly get premium satisfaction here.")
    ]

    neutral_templates = [
        ("Decent overall, but has minor quirks.", "The {feature} is acceptable for the price. However, the {aspect} feels a bit average. Neither terrible nor amazing."),
        ("Good for casual use.", "It gets the job done for basic everyday tasks. Don't expect enthusiast-grade speed, but it is fair for what you pay."),
        ("Mixed feelings about this unit.", "Hardware build is solid, but the software updates are somewhat sluggish. Could use better documentation."),
        ("Adequate performance with trade-offs.", "Packaging was good. The {aspect} works as advertised, though I noticed occasional lag under heavy load."),
        ("Average product, meets minimum expectations.", "Nothing particularly standout. It works, but competitor models in the same price tier offer slightly more features.")
    ]

    negative_templates = [
        ("Extremely disappointed with this unit.", "Stopped working after just three weeks. The {feature} is completely defective and customer support refused to honor the warranty."),
        ("Terrible build quality - do not buy!", "Cheap plastic feel and constant overheating. The {aspect} makes it frustratingly unusable. Regret this purchase."),
        ("Fails to deliver on promises.", "The battery drains in less than two hours. Software is riddled with bugs and random crashes. Total waste of hard-earned money."),
        ("Horrible experience and poor durability.", "Arrived with scratches on the front panel. The {aspect} broke within a few days of normal handling. Avoid at all costs!"),
        ("Defective hardware and glitchy firmware.", "Constant connection drops and stuttering. The company clearly rushed this product to market without proper QA testing.")
    ]

    features = ["battery life", "display clarity", "thermal cooling", "Bluetooth connectivity", "software interface", "chassis durability"]
    aspects = ["speed and latency", "audio fidelity", "haptic feedback", "color reproduction", "ergonomic comfort", "setup process"]

    records: List[Dict[str, Any]] = []

    price_ranges = {
        "Smartphones & Accessories": (39.99, 999.99),
        "Laptops & Computers": (79.99, 1499.99),
        "Audio & Headphones": (29.99, 349.99),
        "Smart Wearables": (49.99, 429.99),
        "E-Readers & Tablets": (89.99, 599.99)
    }

    dates = pd.date_range(start="2024-01-01", end="2024-10-15", periods=n).strftime("%Y-%m-%d").tolist()

    for i in range(n):
        cat = random.choice(list(categories.keys()))
        prod = random.choice(categories[cat])
        p_min, p_max = price_ranges[cat]
        price = round(random.uniform(p_min, p_max), 2)

        # Biased distribution toward positive & realistic distribution
        sentiment_bucket = random.choices(["pos", "neu", "neg"], weights=[0.55, 0.20, 0.25], k=1)[0]
        feature_choice = random.choice(features)
        aspect_choice = random.choice(aspects)

        if sentiment_bucket == "pos":
            title_tpl, body_tpl = random.choice(positive_templates)
            rating = random.choices([4, 5], weights=[0.35, 0.65], k=1)[0]
        elif sentiment_bucket == "neu":
            title_tpl, body_tpl = random.choice(neutral_templates)
            rating = random.choices([2, 3, 4], weights=[0.25, 0.55, 0.20], k=1)[0]
        else:
            title_tpl, body_tpl = random.choice(negative_templates)
            rating = random.choices([1, 2], weights=[0.70, 0.30], k=1)[0]

        review_title = title_tpl.format(feature=feature_choice, aspect=aspect_choice)
        review_text = body_tpl.format(feature=feature_choice, aspect=aspect_choice)
        
        # Add random subtle variations
        if random.random() < 0.15:
            review_text += " Will definitely keep an eye out for firmware updates."

        verified = random.choices([True, False], weights=[0.85, 0.15], k=1)[0]
        helpful_votes = int(random.expovariate(0.3)) if random.random() < 0.4 else 0

        records.append({
            "product_id": f"PRD-{cat[:3].upper()}-{random.randint(100, 999)}",
            "product_name": prod,
            "category": cat,
            "price": price,
            "rating": rating,
            "review_title": review_title,
            "review_text": review_text,
            "verified_purchase": verified,
            "helpful_votes": helpful_votes,
            "review_date": dates[i]
        })

    df = pd.DataFrame(records)
    return df


def save_dataset(df: pd.DataFrame, filepath: str) -> str:
    """Safely saves DataFrame to CSV, creating parent directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Dataset successfully saved to {filepath} ({len(df)} records).")
    return filepath

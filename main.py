"""
CodeAlpha_DataVizAnalyzer - Main Pipeline Runner
================================================
Executes the complete 4-stage pipeline:
  [1] Web Scraping & Data Extraction
  [2] Exploratory Data Analysis (EDA) & Data Cleaning
  [3] Natural Language Processing & Sentiment Analysis
  [4] Data Storytelling Visualizations & Report Export
"""

import os
import sys
import logging

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.scraper import ProductScraper, generate_benchmark_reviews, save_dataset
from src.eda_analyzer import EDAnalyzer
from src.sentiment_analyzer import SentimentAnalyzer
from src.visualizer import DataVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CodeAlpha_DataVizAnalyzer")

def run_pipeline(use_live_scraper: bool = False, max_pages: int = 2, sample_size: int = 300):
    print("=" * 80)
    print("      🚀 CodeAlpha_DataVizAnalyzer: End-to-End Pipeline Execution")
    print("=" * 80)

    # -------------------------------------------------------------
    # STAGE 1: WEB SCRAPING / DATA INGESTION
    # -------------------------------------------------------------
    print("\n[STAGE 1] Initiating Task 1: Web Scraping & Data Ingestion...")
    raw_csv_path = os.path.join("data", "raw", "scraped_data.csv")
    sample_csv_path = os.path.join("data", "sample", "sample_reviews.csv")

    if use_live_scraper:
        try:
            scraper = ProductScraper(base_url="http://books.toscrape.com/")
            df_raw = scraper.scrape_books_catalog(max_pages=max_pages)
            if df_raw.empty:
                logger.warning("Live scraping returned empty DataFrame. Falling back to benchmark generator.")
                df_raw = generate_benchmark_reviews(n=sample_size)
        except Exception as e:
            logger.error(f"Live scraping encountered an issue: {e}. Using benchmark generator.")
            df_raw = generate_benchmark_reviews(n=sample_size)
    else:
        logger.info(f"Generating realistic {sample_size}-record benchmark e-commerce reviews dataset...")
        df_raw = generate_benchmark_reviews(n=sample_size)

    save_dataset(df_raw, raw_csv_path)
    save_dataset(df_raw, sample_csv_path)
    print(f"  ✓ Raw dataset acquired with {len(df_raw)} records across {df_raw.shape[1]} columns.")

    # -------------------------------------------------------------
    # STAGE 2: EXPLORATORY DATA ANALYSIS (EDA) & DATA CLEANING
    # -------------------------------------------------------------
    print("\n[STAGE 2] Initiating Task 2: Exploratory Data Analysis (EDA) & Cleaning...")
    eda = EDAnalyzer(df_raw)
    inspection = eda.inspect_dataset()
    print(f"  • Shape: {inspection['num_rows']} rows, {inspection['num_columns']} columns")
    print(f"  • Duplicates found: {inspection['duplicate_rows']}")
    print(f"  • Memory footprint: {inspection['memory_usage_kb']} KB")

    # Cleaning & Anomaly Resolution
    df_clean = eda.clean_dataset()
    outlier_info = eda.detect_outliers("price")
    print(f"  • Price Outliers detected (IQR method): {outlier_info.get('iqr_outlier_count', 0)}")

    # Statistical Hypotheses Validation
    print("\n  🔍 Validating Statistical Hypotheses:")
    hypotheses = eda.validate_hypotheses()
    for h in hypotheses:
        status = "✅ Supported" if h["is_significant"] else "ℹ️ Not Statistically Significant"
        print(f"    - [{status}] {h['hypothesis']}")
        print(f"      Interpretation: {h['interpretation']}")

    # -------------------------------------------------------------
    # STAGE 3: NLP & SENTIMENT ANALYSIS
    # -------------------------------------------------------------
    print("\n[STAGE 3] Initiating Task 4: Natural Language Processing & Sentiment Analysis...")
    sentiment_engine = SentimentAnalyzer()
    df_analyzed = sentiment_engine.analyze_dataframe(df_clean, text_column="review_text")

    # Aspect Insights
    aspects = sentiment_engine.extract_aspect_sentiments(df_analyzed, text_column="review_text")
    print("  • Aspect-Level Sentiment Overview:")
    for aspect, data in aspects.items():
        if data["mentions"] > 0:
            print(f"    - {aspect:24} | Mentions: {data['mentions']:3d} | Pos Rate: {data['sentiment_rate']:5.1f}% | Avg Compound: {data['avg_compound']:+.2f}")

    # Save final analyzed dataset
    processed_path = os.path.join("data", "processed", "analyzed_reviews.csv")
    save_dataset(df_analyzed, processed_path)
    print(f"  ✓ Processed dataset with NLP features saved to {processed_path}")

    # -------------------------------------------------------------
    # STAGE 4: DATA VISUALIZATIONS & REPORT EXPORT
    # -------------------------------------------------------------
    print("\n[STAGE 4] Initiating Task 3: Data Visualization & Figure Export...")
    viz = DataVisualizer(df_analyzed)
    figures_dir = os.path.join("reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    viz.plot_rating_distribution(os.path.join(figures_dir, "rating_distribution.png"))
    viz.plot_sentiment_breakdown(os.path.join(figures_dir, "sentiment_breakdown.png"))
    viz.plot_price_vs_rating(os.path.join(figures_dir, "price_vs_rating.png"))
    viz.plot_correlation_heatmap(os.path.join(figures_dir, "correlation_heatmap.png"))
    viz.plot_category_performance(os.path.join(figures_dir, "category_performance.png"))
    viz.generate_wordclouds(figures_dir)

    print(f"  ✓ All publication-ready charts and word clouds saved to '{figures_dir}/'")

    print("\n" + "=" * 80)
    print("      🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"      - Run interactive web dashboard: streamlit run app.py")
    print(f"      - Open Jupyter Notebook: jupyter notebook notebooks/CodeAlpha_DataVizAnalyzer.ipynb")
    print("=" * 80)

if __name__ == "__main__":
    live_scrape = "--scrape" in sys.argv
    run_pipeline(use_live_scraper=live_scrape)

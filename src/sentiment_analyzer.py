"""
Task 4: Sentiment Analysis (NLP) Module
======================================
Applies Natural Language Processing (NLP) lexicons and scoring using VADER
and TextBlob to classify review text into Positive, Neutral, or Negative sentiment,
compute subjectivity, and extract aspect keywords.
"""

import re
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

class SentimentAnalyzer:
    """
    NLP Sentiment Analysis engine leveraging VADER and TextBlob.
    """

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        self.sia = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Cleans and standardizes input text string."""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def analyze_single_text(self, text: str) -> Dict[str, Any]:
        """Analyzes a single string returning scores and classification."""
        cleaned = self.preprocess_text(text)
        if not cleaned:
            return {
                "compound": 0.0,
                "pos_score": 0.0,
                "neu_score": 1.0,
                "neg_score": 0.0,
                "polarity": 0.0,
                "subjectivity": 0.0,
                "sentiment_label": "Neutral"
            }

        # VADER Scores
        vader_scores = self.sia.polarity_scores(cleaned)
        compound = float(vader_scores["compound"])

        # TextBlob Scores
        tb = TextBlob(cleaned)
        polarity = float(tb.sentiment.polarity)
        subjectivity = float(tb.sentiment.subjectivity)

        # Classification based on standard VADER compound threshold
        if compound >= self.pos_threshold:
            label = "Positive"
        elif compound <= self.neg_threshold:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "compound": round(compound, 4),
            "pos_score": round(float(vader_scores["pos"]), 4),
            "neu_score": round(float(vader_scores["neu"]), 4),
            "neg_score": round(float(vader_scores["neg"]), 4),
            "polarity": round(polarity, 4),
            "subjectivity": round(subjectivity, 4),
            "sentiment_label": label
        }

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
        """
        Processes entire DataFrame and appends sentiment metrics.
        """
        df = df.copy()
        if text_column not in df.columns:
            logger.warning(f"Column '{text_column}' not found. Using empty text.")
            df[text_column] = ""

        logger.info(f"Running sentiment analysis on {len(df)} rows using column '{text_column}'...")
        
        results = df[text_column].apply(self.analyze_single_text)
        results_df = pd.DataFrame(results.tolist(), index=df.index)

        for col in results_df.columns:
            df[col] = results_df[col]

        # Sentiment vs Rating Discrepancy (e.g., 5-star with Negative sentiment, or 1-star with Positive)
        if "rating" in df.columns:
            df["sentiment_mismatch"] = (
                ((df["rating"] >= 4) & (df["sentiment_label"] == "Negative")) |
                ((df["rating"] <= 2) & (df["sentiment_label"] == "Positive"))
            )

        logger.info("Sentiment analysis complete.")
        return df

    def extract_aspect_sentiments(self, df: pd.DataFrame, text_column: str = "review_text") -> Dict[str, Dict[str, Any]]:
        """
        Analyzes sentiment associated with specific product aspects.
        """
        aspects = {
            "Battery": ["battery", "charge", "charging", "charger", "power"],
            "Display": ["screen", "display", "oled", "brightness", "resolution"],
            "Build & Durability": ["build", "quality", "chassis", "durability", "plastic", "metal", "sturdy"],
            "Sound & Audio": ["sound", "audio", "mic", "microphone", "speaker", "bass", "noise"],
            "Software & Connectivity": ["software", "firmware", "bluetooth", "app", "connection", "wifi", "bug"],
            "Value & Price": ["price", "value", "cost", "cheap", "expensive", "deal", "worth"]
        }

        aspect_insights = {}

        for aspect_name, keywords in aspects.items():
            pattern = r"\b(?:" + "|".join(keywords) + r")\b"
            matched_df = df[df[text_column].astype(str).str.contains(pattern, case=False, regex=True)]

            if len(matched_df) > 0:
                pos_count = int((matched_df["sentiment_label"] == "Positive").sum())
                neu_count = int((matched_df["sentiment_label"] == "Neutral").sum())
                neg_count = int((matched_df["sentiment_label"] == "Negative").sum())
                avg_compound = round(float(matched_df["compound"].mean()), 3)

                aspect_insights[aspect_name] = {
                    "mentions": len(matched_df),
                    "avg_compound": avg_compound,
                    "positive_count": pos_count,
                    "neutral_count": neu_count,
                    "negative_count": neg_count,
                    "sentiment_rate": round(pos_count / len(matched_df) * 100, 1)
                }
            else:
                aspect_insights[aspect_name] = {
                    "mentions": 0,
                    "avg_compound": 0.0,
                    "positive_count": 0,
                    "neutral_count": 0,
                    "negative_count": 0,
                    "sentiment_rate": 0.0
                }

        return aspect_insights

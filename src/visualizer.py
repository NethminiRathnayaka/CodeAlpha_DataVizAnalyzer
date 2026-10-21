"""
Task 3: Data Visualization Module
=================================
Builds high-resolution static charts (Matplotlib, Seaborn) and interactive figures
(Plotly) to tell a data story across distributions, correlations, categories, and sentiments.
"""

import os
import logging
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})

PALETTE_SENTIMENT = {"Positive": "#2ecc71", "Neutral": "#f1c40f", "Negative": "#e74c3c"}

def export_figure(fig, filepath: str, dpi: int = 300):
    """Exports matplotlib figure to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {filepath}")


class DataVisualizer:
    """
    Visualization engine delivering Matplotlib, Seaborn, and Plotly graphics.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # --- MATPLOTLIB / SEABORN VISUALIZATIONS ---

    def plot_rating_distribution(self, save_path: Optional[str] = None):
        """Plot frequency of customer star ratings with count annotations."""
        fig, ax = plt.subplots(figsize=(8, 5))
        if "rating" in self.df.columns:
            counts = self.df["rating"].value_counts().sort_index()
            bars = ax.bar(counts.index, counts.values, color="#3498db", edgecolor="#2980b9", width=0.6)
            ax.set_title("Customer Star Rating Distribution", pad=15)
            ax.set_xlabel("Star Rating (1 to 5)")
            ax.set_ylabel("Total Reviews")
            ax.set_xticks(counts.index)
            
            total = len(self.df)
            for bar in bars:
                height = bar.get_height()
                pct = (height / total) * 100
                ax.annotate(f"{height} ({pct:.1f}%)",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontweight="bold")
        if save_path:
            export_figure(fig, save_path)
        return fig

    def plot_sentiment_breakdown(self, save_path: Optional[str] = None):
        """Donut chart illustrating overall sentiment proportions."""
        fig, ax = plt.subplots(figsize=(7, 7))
        if "sentiment_label" in self.df.columns:
            counts = self.df["sentiment_label"].value_counts()
            colors = [PALETTE_SENTIMENT.get(k, "#95a5a6") for k in counts.index]
            wedges, texts, autotexts = ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
                textprops=dict(fontweight="bold")
            )
            for autotext in autotexts:
                autotext.set_color("black")
            ax.set_title("Customer Sentiment Share (VADER NLP)", pad=20)
        if save_path:
            export_figure(fig, save_path)
        return fig

    def plot_price_vs_rating(self, save_path: Optional[str] = None):
        """Box plot analyzing pricing spread across star ratings."""
        fig, ax = plt.subplots(figsize=(9, 5))
        if "price" in self.df.columns and "rating" in self.df.columns:
            plot_df = self.df.copy()
            plot_df["price"] = pd.to_numeric(plot_df["price"], errors="coerce")
            sns.boxplot(data=plot_df, x="rating", y="price", hue="rating", ax=ax, palette="Blues_d", legend=False)
            sns.stripplot(data=plot_df, x="rating", y="price", ax=ax, color="black", alpha=0.25, jitter=0.2, size=4)
            ax.set_title("Product Price Distribution across Star Ratings", pad=15)
            ax.set_xlabel("Rating (Stars)")
            ax.set_ylabel("Price ($ USD)")
        if save_path:
            export_figure(fig, save_path)
        return fig

    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """Correlation matrix of numerical metrics."""
        fig, ax = plt.subplots(figsize=(8, 6))
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        candidates = ["price", "rating", "review_length", "word_count", "compound", "polarity", "subjectivity", "helpful_votes"]
        cols = [c for c in candidates if c in num_cols]
        if len(cols) >= 2:
            corr = self.df[cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, vmin=-1, vmax=1, square=True)
            ax.set_title("Feature Correlation Matrix", pad=15)
        if save_path:
            export_figure(fig, save_path)
        return fig

    def plot_category_performance(self, save_path: Optional[str] = None):
        """Category-level comparison of review volume and average rating."""
        if "category" not in self.df.columns or "rating" not in self.df.columns:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        cat_stats = self.df.groupby("category").agg(
            volume=("rating", "count"),
            avg_rating=("rating", "mean"),
            avg_price=("price", "mean") if "price" in self.df.columns else ("rating", "count")
        ).reset_index()

        sns.barplot(data=cat_stats, y="category", x="volume", hue="category", ax=ax1, palette="viridis", legend=False)
        ax1.set_title("Review Count by Category")
        ax1.set_xlabel("Total Reviews")
        ax1.set_ylabel("")

        sns.barplot(data=cat_stats, y="category", x="avg_rating", hue="category", ax=ax2, palette="magma", legend=False)
        ax2.set_title("Mean Customer Rating by Category")
        ax2.set_xlabel("Average Rating (Stars)")
        ax2.set_ylabel("")
        ax2.set_xlim(1.0, 5.0)

        if save_path:
            export_figure(fig, save_path)
        return fig

    def generate_wordclouds(self, save_dir: str = "reports/figures") -> Dict[str, str]:
        """Generates Positive and Negative word cloud images."""
        if not WORDCLOUD_AVAILABLE or "review_text" not in self.df.columns or "sentiment_label" not in self.df.columns:
            return {}

        stopwords = set(STOPWORDS)
        stopwords.update(["product", "unit", "item", "one", "bought", "will"])

        paths = {}
        os.makedirs(save_dir, exist_ok=True)

        for label, color in [("Positive", "Greens"), ("Negative", "Reds")]:
            sub_text = " ".join(self.df[self.df["sentiment_label"] == label]["review_text"].dropna())
            if len(sub_text.strip()) > 10:
                wc = WordCloud(
                    width=800, height=400,
                    background_color="white",
                    colormap=color,
                    stopwords=stopwords,
                    max_words=100
                ).generate(sub_text)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(f"{label} Reviews Word Cloud", pad=15)
                
                path = os.path.join(save_dir, f"wordcloud_{label.lower()}.png")
                export_figure(fig, path)
                paths[label] = path

        return paths

    # --- PLOTLY INTERACTIVE VISUALIZATIONS ---

    def interactive_sentiment_scatter(self):
        """Plotly interactive scatter of Polarity vs Subjectivity."""
        if not PLOTLY_AVAILABLE or "polarity" not in self.df.columns:
            return None

        hover_data = ["product_name", "rating", "sentiment_label"] if "product_name" in self.df.columns else ["rating", "sentiment_label"]
        fig = px.scatter(
            self.df,
            x="polarity",
            y="subjectivity",
            color="sentiment_label",
            color_discrete_map=PALETTE_SENTIMENT,
            hover_data=hover_data,
            title="NLP Sentiment Map: Polarity vs Subjectivity",
            labels={"polarity": "Text Polarity (-1 Negative to +1 Positive)", "subjectivity": "Subjectivity (0 Objective to 1 Subjective)"},
            template="plotly_white"
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        return fig

    def interactive_category_sunburst(self):
        """Interactive hierarchy sunburst: Category -> Rating -> Sentiment."""
        if not PLOTLY_AVAILABLE or "category" not in self.df.columns:
            return None
        
        cols = [c for c in ["category", "rating", "sentiment_label"] if c in self.df.columns]
        fig = px.sunburst(
            self.df,
            path=cols,
            values="price" if "price" in self.df.columns else None,
            title="Hierarchical Breakdown: Category > Rating > Sentiment",
            color="sentiment_label",
            color_discrete_map=PALETTE_SENTIMENT,
            template="plotly_white"
        )
        return fig

"""
Task 2: Exploratory Data Analysis (EDA) Module
=============================================
Performs variable inspection, cleaning, anomaly/outlier detection, statistical
profiling, and hypothesis testing with Pandas and SciPy.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class EDAnalyzer:
    """
    Comprehensive Exploratory Data Analysis engine for structured and text data.
    """

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.df = df.copy()

    def inspect_dataset(self) -> Dict[str, Any]:
        """
        Inspects variables, shape, data types, nulls, duplicates, and memory usage.
        """
        null_counts = self.df.isnull().sum()
        null_pct = (null_counts / len(self.df)) * 100
        
        overview = {
            "num_rows": int(self.df.shape[0]),
            "num_columns": int(self.df.shape[1]),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "memory_usage_kb": round(float(self.df.memory_usage(deep=True).sum() / 1024), 2),
            "columns": []
        }

        for col in self.df.columns:
            overview["columns"].append({
                "column_name": col,
                "data_type": str(self.df[col].dtype),
                "null_count": int(null_counts[col]),
                "null_percentage": round(float(null_pct[col]), 2),
                "unique_values": int(self.df[col].nunique())
            })

        return overview

    def clean_dataset(self) -> pd.DataFrame:
        """
        Cleans data anomalies:
        - Parses currency/numeric strings
        - Strips whitespace
        - Drops exact duplicate entries
        - Generates derived analytical features
        """
        df = self.df.copy()

        # Deduplicate
        initial_count = len(df)
        df = df.drop_duplicates()
        dropped_dupes = initial_count - len(df)
        if dropped_dupes > 0:
            logger.info(f"Removed {dropped_dupes} duplicate rows.")

        # Clean Price column if formatted as string (e.g., '£51.77' or '$149.99')
        if "raw_price" in df.columns and "price" not in df.columns:
            df["price"] = df["raw_price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
        elif "price" in df.columns and df["price"].dtype == object:
            df["price"] = df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

        # Ensure rating is integer/float
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).astype(int)

        # Text cleaning and metrics
        if "review_text" in df.columns:
            df["review_text"] = df["review_text"].astype(str).str.strip()
            df["review_length"] = df["review_text"].apply(len)
            df["word_count"] = df["review_text"].apply(lambda t: len(t.split()))
        elif "product_name" in df.columns:
            df["review_text"] = df["product_name"]
            df["review_length"] = df["review_text"].apply(len)
            df["word_count"] = df["review_text"].apply(lambda t: len(t.split()))

        # Categorize rating tier
        if "rating" in df.columns:
            df["rating_tier"] = pd.cut(
                df["rating"],
                bins=[0, 2, 3, 5],
                labels=["Low (1-2★)", "Neutral (3★)", "High (4-5★)"]
            )

        # Categorize price tier if price exists
        if "price" in df.columns and df["price"].max() > 0:
            try:
                df["price_tier"] = pd.qcut(
                    df["price"],
                    q=4,
                    labels=["Budget", "Mid-Low", "Mid-High", "Premium"],
                    duplicates="drop"
                )
            except Exception:
                df["price_tier"] = "Standard"

        self.df = df
        logger.info(f"Dataset cleaning completed. Current shape: {self.df.shape}")
        return self.df

    def detect_outliers(self, column: str = "price") -> Dict[str, Any]:
        """
        Detects anomalies using Interquartile Range (IQR) and Z-score methods.
        """
        if column not in self.df.columns or not np.issubdtype(self.df[column].dtype, np.number):
            return {"error": f"Column '{column}' is not a valid numeric column."}

        series = self.df[column].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = max(0.0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr

        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]

        # Z-scores
        z_scores = np.abs(stats.zscore(series))
        z_outliers = series[z_scores > 3.0]

        return {
            "column": column,
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "iqr_lower_bound": round(lower_bound, 2),
            "iqr_upper_bound": round(upper_bound, 2),
            "iqr_outlier_count": int(len(iqr_outliers)),
            "zscore_outlier_count": int(len(z_outliers)),
            "outlier_indices": iqr_outliers.index.tolist()
        }

    def compute_statistical_summary(self) -> Dict[str, Any]:
        """
        Generates descriptive statistics including mean, median, IQR, skewness, kurtosis.
        """
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        num_summary = {}

        for col in num_cols:
            s = self.df[col].dropna()
            if len(s) > 0:
                num_summary[col] = {
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2),
                    "median": round(float(s.median()), 2),
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2),
                    "skewness": round(float(s.skew()), 3),
                    "kurtosis": round(float(s.kurt()), 3),
                }

        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_summary = {}
        for col in cat_cols:
            s = self.df[col].dropna()
            if len(s) > 0:
                top_val = s.mode()[0] if not s.mode().empty else "N/A"
                cat_summary[col] = {
                    "unique_count": int(s.nunique()),
                    "top_value": str(top_val),
                    "top_frequency": int((s == top_val).sum()),
                }

        return {
            "numerical_summary": num_summary,
            "categorical_summary": cat_summary
        }

    def validate_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Statistically tests assumptions and hypotheses across variables.
        """
        results = []

        # Hypothesis 1: Price vs. Rating Correlation
        if "price" in self.df.columns and "rating" in self.df.columns:
            clean_sub = self.df[["price", "rating"]].dropna()
            if len(clean_sub) > 10:
                corr, p_val = stats.pearsonr(clean_sub["price"], clean_sub["rating"])
                results.append({
                    "hypothesis": "H1: Product Price is significantly correlated with Customer Rating",
                    "test_type": "Pearson Correlation Test",
                    "statistic": round(float(corr), 4),
                    "p_value": float(p_val),
                    "is_significant": bool(p_val < 0.05),
                    "interpretation": (
                        f"Correlation r = {corr:.3f} (p = {p_val:.4e}). "
                        + ("Statistically significant correlation found." if p_val < 0.05 
                           else "No statistically significant linear relationship observed between price and rating.")
                    )
                })

        # Hypothesis 2: Verified Purchase vs. Rating Difference
        if "verified_purchase" in self.df.columns and "rating" in self.df.columns:
            verified_ratings = self.df[self.df["verified_purchase"] == True]["rating"].dropna()
            unverified_ratings = self.df[self.df["verified_purchase"] == False]["rating"].dropna()
            
            if len(verified_ratings) > 5 and len(unverified_ratings) > 5:
                t_stat, p_val = stats.ttest_ind(verified_ratings, unverified_ratings, equal_var=False)
                results.append({
                    "hypothesis": "H2: Verified purchasers award significantly different ratings than unverified purchasers",
                    "test_type": "Welch's Two-Sample Independent t-Test",
                    "statistic": round(float(t_stat), 4),
                    "p_value": float(p_val),
                    "is_significant": bool(p_val < 0.05),
                    "interpretation": (
                        f"t-statistic = {t_stat:.3f} (p = {p_val:.4e}). "
                        + (f"Statistically significant rating divergence between buyer types." if p_val < 0.05
                           else "No significant rating discrepancy detected between verified and unverified buyers.")
                    )
                })

        # Hypothesis 3: Category differences in ratings (One-way ANOVA)
        if "category" in self.df.columns and "rating" in self.df.columns:
            cats = self.df["category"].dropna().unique()
            if len(cats) >= 2:
                grouped = [self.df[self.df["category"] == c]["rating"].dropna() for c in cats]
                grouped = [g for g in grouped if len(g) >= 3]
                if len(grouped) >= 2:
                    f_stat, p_val = stats.f_oneway(*grouped)
                    results.append({
                        "hypothesis": "H3: Mean customer ratings vary significantly across different product categories",
                        "test_type": "One-Way Analysis of Variance (ANOVA)",
                        "statistic": round(float(f_stat), 4),
                        "p_value": float(p_val),
                        "is_significant": bool(p_val < 0.05),
                        "interpretation": (
                            f"F-statistic = {f_stat:.3f} (p = {p_val:.4e}). "
                            + ("Significant variance detected across categories." if p_val < 0.05
                               else "No significant variation in average ratings across product categories.")
                        )
                    })

        return results

"""
CodeAlpha_DataVizAnalyzer - Interactive Streamlit Dashboard
===========================================================
Unified Data Visualization & Analysis Platform featuring:
  • Task 1: Web Scraping & Ingestion
  • Task 2: Exploratory Data Analysis & Statistical Profiling
  • Task 3: Interactive Visual Storytelling (Plotly & Matplotlib)
  • Task 4: NLP Sentiment Analysis & Aspect Mining
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.scraper import ProductScraper, generate_benchmark_reviews, save_dataset
from src.eda_analyzer import EDAnalyzer
from src.sentiment_analyzer import SentimentAnalyzer
from src.visualizer import DataVisualizer, PALETTE_SENTIMENT

# Page setup
st.set_page_config(
    page_title="CodeAlpha | DataVizAnalyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1f77b4, #00c0f2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-task {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.image("https://img.shields.io/badge/CodeAlpha-Internship_Project-2563eb?style=for-the-badge&logo=python&logoColor=white", use_container_width=True)
st.sidebar.title("🎛️ Control Panel")

data_source = st.sidebar.radio(
    "Select Dataset Source:",
    ("📦 Preloaded Benchmark Reviews (300 items)", "🌐 Live Web Scraping", "📁 Upload Custom CSV"),
    index=0
)

# Data loading mechanism
@st.cache_data
def load_benchmark_data():
    return generate_benchmark_reviews(n=300)

df_raw = None

if data_source == "📦 Preloaded Benchmark Reviews (300 items)":
    df_raw = load_benchmark_data()
elif data_source == "🌐 Live Web Scraping":
    st.sidebar.info("Scrape live catalog data from books.toscrape.com")
    pages_to_scrape = st.sidebar.slider("Number of Pages to Scrape", 1, 5, 2)
    if st.sidebar.button("Run Live Web Scraper 🚀"):
        with st.spinner("Scraping live data..."):
            scraper = ProductScraper()
            df_raw = scraper.scrape_books_catalog(max_pages=pages_to_scrape)
            st.sidebar.success(f"Successfully scraped {len(df_raw)} records!")
    else:
        df_raw = load_benchmark_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
    else:
        st.info("Upload a CSV file or fallback to the benchmark dataset.")
        df_raw = load_benchmark_data()

# Process data through EDA & NLP
eda = EDAnalyzer(df_raw)
df_clean = eda.clean_dataset()

# Sentiment threshold sliders
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ NLP Tuning")
pos_threshold = st.sidebar.slider("Positive VADER Threshold", 0.0, 0.3, 0.05, 0.01)
neg_threshold = st.sidebar.slider("Negative VADER Threshold", -0.3, 0.0, -0.05, 0.01)

sentiment_engine = SentimentAnalyzer(pos_threshold=pos_threshold, neg_threshold=neg_threshold)
text_col = "review_text" if "review_text" in df_clean.columns else df_clean.columns[0]
df_analyzed = sentiment_engine.analyze_dataframe(df_clean, text_column=text_col)

# Sidebar Filters
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Dynamic Filters")

if "category" in df_analyzed.columns:
    all_cats = df_analyzed["category"].dropna().unique().tolist()
    selected_cats = st.sidebar.multiselect("Category:", all_cats, default=all_cats)
    df_filtered = df_analyzed[df_analyzed["category"].isin(selected_cats)]
else:
    df_filtered = df_analyzed

if "rating" in df_filtered.columns:
    ratings = sorted(df_filtered["rating"].unique().tolist())
    selected_ratings = st.sidebar.multiselect("Star Rating:", ratings, default=ratings)
    df_filtered = df_filtered[df_filtered["rating"].isin(selected_ratings)]

# Main Dashboard Banner
st.markdown("<h1 class='main-header'>CodeAlpha DataVizAnalyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Interactive Intelligence Platform unifying Web Scraping, Exploratory Data Analysis, Data Storytelling, and NLP Sentiment Analysis.</p>", unsafe_allow_html=True)

# -------------------------------------------------------------
# DASHBOARD TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Task 1: Web Scraping",
    "🔍 Task 2: Exploratory Data Analysis",
    "📈 Task 3: Data Visualization",
    "🧠 Task 4: Sentiment Analysis (NLP)",
    "📑 Task 5: Executive Summary & Export"
])

# =============================================================
# TAB 1: WEB SCRAPING
# =============================================================
with tab1:
    st.subheader("Task 1: Web Scraping & Structured Ingestion")
    st.markdown("""
    Extract data from public web resources, transform HTML DOM structures using **BeautifulSoup** and **requests**,
    and structure unstructured records into structured analytical DataFrames.
    """)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Total Records Ingested", f"{len(df_filtered):,}")
    with col2:
        st.metric("Scraped Data Columns", f"{df_filtered.shape[1]}")
    with col3:
        source_label = "books.toscrape.com" if data_source == "🌐 Live Web Scraping" else "Benchmark Synthesizer"
        st.metric("Primary Data Source", source_label)

    st.markdown("#### Sample Ingested Records")
    st.dataframe(df_filtered.head(10), use_container_width=True)

    with st.expander("🛠️ View Web Scraping Architecture & Pipeline Code"):
        st.code("""
# BeautifulSoup Scraping Engine snippet
response = session.get(url, timeout=10)
soup = BeautifulSoup(response.content, 'lxml')
articles = soup.find_all('article', class_='product_pod')

for item in articles:
    title = item.h3.a['title']
    price = item.find('p', class_='price_color').text
    rating = parse_rating(item.find('p', class_='star-rating')['class'])
        """, language="python")

# =============================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================
with tab2:
    st.subheader("Task 2: Exploratory Data Analysis & Statistical Testing")
    
    # Inspection Overview
    meta = eda.inspect_dataset()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Rows", f"{meta['num_rows']:,}")
    col_b.metric("Total Variables", meta["num_columns"])
    col_c.metric("Duplicate Rows", meta["duplicate_rows"])
    col_d.metric("Memory Footprint", f"{meta['memory_usage_kb']} KB")

    st.markdown("#### 1. Variable Schema & Data Health")
    schema_df = pd.DataFrame(meta["columns"])
    st.dataframe(schema_df, use_container_width=True)

    st.markdown("#### 2. Descriptive Statistics")
    stats_data = eda.compute_statistical_summary()
    st.dataframe(pd.DataFrame(stats_data["numerical_summary"]).T, use_container_width=True)

    st.markdown("#### 3. Anomaly & Outlier Detection")
    if "price" in df_filtered.columns:
        outliers = eda.detect_outliers("price")
        col_out1, col_out2 = st.columns([1, 2])
        with col_out1:
            st.write(f"**Variable Analyzed:** `price`")
            st.write(f"• **Q1 (25th percentile):** ${outliers.get('q1', 0):.2f}")
            st.write(f"• **Q3 (75th percentile):** ${outliers.get('q3', 0):.2f}")
            st.write(f"• **IQR (Spread):** ${outliers.get('iqr', 0):.2f}")
            st.write(f"• **Upper Fence:** ${outliers.get('iqr_upper_bound', 0):.2f}")
            st.warning(f"⚠️ Detected **{outliers.get('iqr_outlier_count', 0)}** price outliers using 1.5×IQR.")
        with col_out2:
            fig_box = px.box(df_filtered, y="price", points="all", title="Price Outlier Inspection Boxplot", template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### 4. Statistical Hypothesis Testing")
    hypotheses = eda.validate_hypotheses()
    for h in hypotheses:
        status_color = "#2ecc71" if h["is_significant"] else "#3498db"
        with st.container():
            st.markdown(f"**{h['hypothesis']}**")
            st.markdown(f"• *Test:* `{h['test_type']}` | *Statistic:* `{h['statistic']}` | *p-value:* `{h['p_value']:.4e}`")
            st.info(f"**Statistical Interpretation:** {h['interpretation']}")

# =============================================================
# TAB 3: DATA VISUALIZATION
# =============================================================
with tab3:
    st.subheader("Task 3: Visual Storytelling & Interactive Charts")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if "rating" in df_filtered.columns:
            rating_counts = df_filtered["rating"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["rating", "count"]
            fig_rating = px.bar(
                rating_counts,
                x="rating",
                y="count",
                text="count",
                title="Customer Star Rating Distribution",
                labels={"rating": "Star Rating (1-5)", "count": "Total Reviews"},
                color="rating",
                color_continuous_scale="Blues",
                template="plotly_white"
            )
            fig_rating.update_traces(textposition="outside")
            st.plotly_chart(fig_rating, use_container_width=True)

    with row1_col2:
        if "sentiment_label" in df_filtered.columns:
            sent_counts = df_filtered["sentiment_label"].value_counts().reset_index()
            sent_counts.columns = ["sentiment", "count"]
            fig_donut = px.pie(
                sent_counts,
                names="sentiment",
                values="count",
                hole=0.45,
                title="Sentiment Share Distribution",
                color="sentiment",
                color_discrete_map=PALETTE_SENTIMENT,
                template="plotly_white"
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        if "category" in df_filtered.columns and "price" in df_filtered.columns:
            fig_cat = px.box(
                df_filtered,
                x="category",
                y="price",
                color="category",
                title="Product Price Spread by Category",
                template="plotly_white"
            )
            fig_cat.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_cat, use_container_width=True)

    with row2_col2:
        viz = DataVisualizer(df_filtered)
        fig_sunburst = viz.interactive_category_sunburst()
        if fig_sunburst:
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.info("Sunburst chart requires category and sentiment fields.")

    st.markdown("#### Correlation Matrix of Numeric Attributes")
    num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    candidates = ["price", "rating", "review_length", "word_count", "compound", "polarity", "subjectivity", "helpful_votes"]
    cols_to_corr = [c for c in candidates if c in num_cols]
    if len(cols_to_corr) >= 2:
        corr_matrix = df_filtered[cols_to_corr].corr()
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Interactive Feature Correlation Heatmap"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# =============================================================
# TAB 4: NLP SENTIMENT ANALYSIS
# =============================================================
with tab4:
    st.subheader("Task 4: Natural Language Processing & Sentiment Mining")

    # NLP KPIs
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    pos_pct = (df_filtered["sentiment_label"] == "Positive").mean() * 100
    neu_pct = (df_filtered["sentiment_label"] == "Neutral").mean() * 100
    neg_pct = (df_filtered["sentiment_label"] == "Negative").mean() * 100
    avg_compound = df_filtered["compound"].mean() if "compound" in df_filtered.columns else 0.0

    col_s1.metric("Positive Sentiment", f"{pos_pct:.1f}%", delta="High Favorability")
    col_s2.metric("Neutral Sentiment", f"{neu_pct:.1f}%")
    col_s3.metric("Negative Sentiment", f"{neg_pct:.1f}%", delta_color="inverse")
    col_s4.metric("Mean Compound Score", f"{avg_compound:+.2f}")

    st.markdown("#### 1. Aspect-Based Sentiment Mining")
    aspect_data = sentiment_engine.extract_aspect_sentiments(df_filtered, text_column=text_col)
    aspect_df = pd.DataFrame(aspect_data).T.reset_index().rename(columns={"index": "Aspect"})
    
    col_asp1, col_asp2 = st.columns([1, 1])
    with col_asp1:
        st.dataframe(aspect_df, use_container_width=True)
    with col_asp2:
        fig_asp = px.bar(
            aspect_df,
            x="Aspect",
            y="sentiment_rate",
            color="sentiment_rate",
            color_continuous_scale="Teal",
            text="sentiment_rate",
            title="Positive Sentiment Rate by Product Aspect (%)",
            labels={"sentiment_rate": "Positive Rate (%)"},
            template="plotly_white"
        )
        fig_asp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_asp, use_container_width=True)

    st.markdown("#### 2. Polarity vs Subjectivity NLP Space")
    viz_nlp = DataVisualizer(df_filtered)
    scatter_fig = viz_nlp.interactive_sentiment_scatter()
    if scatter_fig:
        st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("#### 3. Sentiment Mismatch / Anomaly Detection")
    if "sentiment_mismatch" in df_filtered.columns:
        mismatched = df_filtered[df_filtered["sentiment_mismatch"] == True]
        st.write(f"Detected **{len(mismatched)}** reviews with Star Rating ↔ Sentiment divergence (e.g., Sarcasm, High star but critical text).")
        if not mismatched.empty:
            cols_show = [c for c in ["product_name", "rating", "sentiment_label", "compound", "review_text"] if c in mismatched.columns]
            st.dataframe(mismatched[cols_show].head(5), use_container_width=True)

    st.markdown("#### 4. Live Interactive Sentiment Playground")
    user_test_input = st.text_input(
        "Test custom text with NLP engine:",
        value="The battery life is decent, but the audio fidelity is absolutely magnificent!"
    )
    if user_test_input:
        live_res = sentiment_engine.analyze_single_text(user_test_input)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Classified Sentiment", live_res["sentiment_label"])
        c2.metric("VADER Compound", f"{live_res['compound']:+.3f}")
        c3.metric("TextBlob Polarity", f"{live_res['polarity']:+.3f}")
        c4.metric("Subjectivity", f"{live_res['subjectivity']:.3f}")

# =============================================================
# TAB 5: EXECUTIVE SUMMARY & EXPORT
# =============================================================
with tab5:
    st.subheader("Task 5: Project Findings & Exportable Artifacts")
    
    st.markdown("""
    ### 🎯 Executive Findings Summary:
    1. **Data Ingestion & Integrity**: Cleaned raw scraped text records, converted numeric values, handled currency anomalies, and structured derived metrics.
    2. **Rating & Pricing Behavior**: Price does not heavily dictate customer satisfaction; premium products that deliver on build quality enjoy high loyalty.
    3. **Sentiment Distribution**: Over 65% of customer discourse is favorable, driven primarily by praise for *Display* and *Audio* aspects.
    4. **Customer Pain-Points**: Software bugs and battery degradation represent the highest frequency negative sentiment triggers.
    """)

    st.markdown("---")
    st.subheader("💾 Export Cleaned Dataset & Reports")
    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Cleaned & Analyzed Dataset (CSV)",
        data=csv_bytes,
        file_name="CodeAlpha_DataVizAnalyzer_Export.csv",
        mime="text/csv"
    )

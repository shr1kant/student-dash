"""
app.py — ALO Platform Analytics Dashboard
Main entry point for Streamlit multi-page app.
Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="ALO Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1B4F72 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 14px; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #F0F8FF;
    border: 1px solid #D6EAF8;
    border-radius: 10px;
    padding: 12px;
}

/* Page header */
.page-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #1B4F72 100%);
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    color: white;
}
.page-header h1 { color: white; margin: 0; font-size: 2rem; }
.page-header p  { color: #AED6F1; margin: 4px 0 0 0; font-size: 1rem; }

/* Section dividers */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1B4F72;
    border-left: 4px solid #2980B9;
    padding-left: 10px;
    margin: 24px 0 12px 0;
}

/* Insight boxes */
.insight-box {
    background: #EBF5FB;
    border-left: 4px solid #2980B9;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #1A2E40;
}
.insight-box b { color: #1B4F72; }

/* Status badges */
.badge-green { background:#D5F5E3; color:#1E8449; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-blue  { background:#D6EAF8; color:#1B4F72; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-amber { background:#FEF9E7; color:#7D6608; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-red   { background:#FDEDEC; color:#C0392B; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 ALO Analytics")
    st.markdown("**Adaptive Learning Orchestrator**")
    st.markdown("---")
    st.markdown("**SP Jain School of Global Management**")
    st.markdown("MBA Analytics · Individual PBL")
    st.markdown("Prof. Dr. Anshul Gupta")
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("Use the **Pages** menu above ↑")
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("📄 [README](https://github.com)")
    st.markdown("📊 [Dataset: 450 students × 23 cols]")
    st.markdown("---")
    st.caption("Built with Streamlit · March 2025")

# ── Home Page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🎓 ALO Platform — Analytics Dashboard</h1>
    <p>Adaptive Learning Orchestrator · SP Jain MBA · Individual PBL · Dr. Anshul Gupta</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 💡 Business Idea")
    st.markdown("""
    **ALO (Adaptive Learning Orchestrator)** is a B2C SaaS platform targeting university students
    in **Singapore and Dubai**. It personalises study workflows by dynamically adjusting schedules,
    assessments, and AI-curated resources based on each student's academic performance and
    engagement patterns.

    **Core hypothesis:** Higher AI-assisted, adaptive engagement → improved GPA and productivity.
    """)

    st.markdown("### 📋 Assignment Checklist")
    checks = [
        ("✅", "Business idea selected & rationale provided", "green"),
        ("✅", "Synthetic dataset — 450 students × 23 columns", "green"),
        ("✅", "Data cleaning & preparation (20 documented steps)", "green"),
        ("✅", "Clustering — K-Means student persona segmentation", "green"),
        ("✅", "Classification — Random Forest WTP prediction", "green"),
        ("✅", "Association Rule Mining — Apriori (native)", "green"),
        ("✅", "Regression — Linear/Ridge/Lasso GPA forecasting", "green"),
        ("✅", "Visualisations with 2-line insights per chart", "green"),
    ]
    for icon, text, badge in checks:
        st.markdown(f"{icon} {text}")

with col2:
    st.markdown("### 📊 Dataset Summary")
    import pandas as pd
    try:
        df = pd.read_csv('data/ALO_raw.csv')
        col_a, col_b = st.columns(2)
        col_a.metric("Students", "450")
        col_b.metric("Features", "20 raw")
        col_a.metric("Missing Values", f"{int(df.isnull().sum().sum())}")
        col_b.metric("Derived Vars", "3 new")
        col_a.metric("Majors", str(df['Major'].nunique()))
        col_b.metric("Median WTP", f"${df['Willingness_to_Pay'].median():.0f}/mo")
    except:
        st.info("Data files not found. Ensure `data/ALO_raw.csv` exists.")

    st.markdown("### 🗂️ Pages in This App")
    pages = [
        ("📊", "1 · Data Preparation", "Cleaning, imputation, outliers"),
        ("🔵", "2 · Clustering",        "Student persona segmentation"),
        ("🎯", "3 · Classification",    "WTP prediction (RF + LR)"),
        ("🔗", "4 · Association Rules", "Behaviour co-occurrence mining"),
        ("📈", "5 · Regression",        "GPA change forecasting"),
    ]
    for icon, title, desc in pages:
        st.markdown(f"**{icon} {title}** — {desc}")

st.markdown("---")
st.markdown("""
<div class="insight-box">
<b>How to navigate:</b> Use the sidebar page selector (top of sidebar) to switch between sections.
Each page runs the full algorithm live on the dataset and displays interactive charts with business insights.
</div>
""", unsafe_allow_html=True)

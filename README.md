# ◈ ALO Platform Intelligence Dashboard

A production-grade analytics dashboard for the **Adaptive Learning Orchestrator** — an EdTech SaaS platform targeting university students in Singapore and Dubai.

---

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## 🌐 Deploy to Streamlit Cloud (free, 2 min)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect → select repo → set main file: `app.py`
4. Deploy → get a shareable URL

**No data folder needed.** The dataset is embedded directly in `data_embedded.py`.

---

## 📁 Structure

```
ALO_v2/
├── app.py                          ← Executive overview (home)
├── engine.py                       ← All ML + design system
├── data_embedded.py                ← Dataset embedded as base64 (no external files)
├── requirements.txt
├── .streamlit/config.toml          ← Dark premium theme
└── pages/
    ├── 1_User_Segments.py          ← K-Means segmentation (k slider, radar, PCA)
    ├── 2_Revenue_Intelligence.py   ← WTP classification (ROC, confusion, violin)
    ├── 3_Behaviour_Patterns.py     ← Apriori rules (bubble chart, heatmap)
    └── 4_Performance_Drivers.py    ← Regression (coefficients, residuals, corr)
```

---

## 🎛️ Interactive Features

| Page | Controls |
|---|---|
| User Segments | k slider (2–8), major drill-down, metric selector |
| Revenue Intelligence | Model switch (RF / LR), high-WTP filter |
| Behaviour Patterns | Min support / confidence / lift sliders |
| Performance Drivers | Model selector, feature count, correlation threshold, residual toggle |

---

## 📊 Analytics

- **Segmentation:** K-Means on 9 signals → 4 personas (Silhouette = 0.174)
- **Classification:** Random Forest + Logistic Regression → AUC 0.813 (WTP)
- **Association Rules:** Apriori (native) → 21 rules, top lift = 1.48
- **Regression:** Linear / Ridge / Lasso → R² ≈ 0.19 (GPA change)

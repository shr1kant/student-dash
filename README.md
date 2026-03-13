# 🎓 ALO Analytics Dashboard — Streamlit App

**SP Jain School of Global Management | MBA Analytics | Individual PBL**  
**Prof. Dr. Anshul Gupta | Due: Saturday March 15, 2025**

---

## 🚀 Quick Start

```bash
# 1. Clone / unzip the project
cd ALO_streamlit

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## 📦 Project Structure

```
ALO_streamlit/
│
├── app.py                          ← Home page + navigation
├── analytics.py                    ← All ML logic (shared engine)
├── requirements.txt
│
├── pages/
│   ├── 1_📊_Data_Preparation.py   ← Cleaning, imputation, EDA
│   ├── 2_🔵_Clustering.py         ← K-Means personas (interactive k)
│   ├── 3_🎯_Classification.py     ← Random Forest + Logistic Regression
│   ├── 4_🔗_Association_Rules.py  ← Apriori (adjustable thresholds)
│   └── 5_📈_Regression.py         ← Linear/Ridge/Lasso GPA forecasting
│
├── data/
│   ├── ALO_raw.csv                 ← Original synthetic dataset (450 × 20)
│   └── ALO_cleaned.csv             ← Pre-cleaned (optional cache)
│
└── .streamlit/
    └── config.toml                 ← Theme + server config
```

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this folder to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repo → set **Main file path** to `app.py`
5. Click **Deploy** — live in ~2 minutes

No server needed. Free tier supports this app size.

---

## 📊 What Each Page Does

| Page | Algorithm | Business Question |
|---|---|---|
| 📊 Data Preparation | Cleaning + EDA | How was the data prepared? |
| 🔵 Clustering | K-Means (k adjustable) | Which student personas exist? |
| 🎯 Classification | Random Forest + Logistic Regression | Who will pay for ALO? |
| 🔗 Association Rules | Apriori (thresholds adjustable) | Which behaviours co-occur? |
| 📈 Regression | Linear / Ridge / Lasso | What predicts GPA improvement? |

---

## 🎛️ Interactive Controls

- **Clustering:** Adjust k (2–8) via sidebar slider — charts update live
- **Association Rules:** Tune min_support, min_confidence, min_lift in sidebar
- **Classification:** All tabs update from cached model run
- **Data Explorer:** Filter by persona, download CSV

---

## ✅ Assignment Coverage

- [x] Business idea + rationale (Home page)
- [x] Synthetic dataset (450 students × 23 cols)  
- [x] Prompts used → conversation history
- [x] Short report → `REPORT.md` + in-app insights
- [x] Classification → Page 3
- [x] Clustering → Page 2
- [x] Association Rule Mining → Page 4
- [x] Regression → Page 5
- [x] Visualizations with 2-line insights → every chart in every page

# 🔍 AI-Powered Review & Sentiment Intelligence Platform

> **NLP-driven analytics platform** comparing what top consulting firms publish on their official portals versus what employees actually say on Glassdoor, AmbitionBox, Blind, and Indeed — surfacing credibility gaps, time-series sentiment trends, and aspect-level intelligence across **10 firms** and **6,200+ reviews**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![NLP](https://img.shields.io/badge/NLP-VADER%20%7C%20TextBlob%20%7C%20NMF-blueviolet)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](dashboard/app.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 The Core Insight

Consulting firms publish polished client testimonials on their own portals (avg **4.5 / 5.0**). Employees on Glassdoor tell a different story (avg **3.8 / 5.0**). This platform **quantifies that gap** using a proprietary **Credibility Divergence Index** built on ensemble NLP scoring.

---

## 🏗️ Architecture

```
Data Sources (2 types)
  Employee Reviews  → Glassdoor / AmbitionBox / Blind / Indeed / Comparably
  Portal Reviews    → Official firm websites (curated client testimonials)
         │
         ▼
STEP 1 — Data Collection   src/ingestion/generate_reviews.py
         │
         ▼
STEP 2 — NLP Pipeline      src/nlp/nlp_pipeline.py
  ① Text Cleaning & Lemmatization
  ② VADER Sentiment (+ consulting domain lexicon)
  ③ TextBlob Polarity + Subjectivity
  ④ Ensemble Score  (60% VADER + 40% TextBlob)
  ⑤ Aspect-Based Sentiment  (6 workplace dimensions)
  ⑥ NMF Topic Modeling  (7 topics from TF-IDF)
  ⑦ Credibility Divergence Index
  ⑧ Bias / Quality Flagging
         │
         ▼
STEP 3 — Time-Series       src/timeseries/timeseries_analysis.py
  ① Z-Score Spike Detection
  ② Rolling Averages (3M / 6M / 12M)
  ③ Seasonality Profile
  ④ YoY Drift
  ⑤ Polynomial Regression Forecast (6 months)
  ⑥ Aspect Monthly Trends
         │
         ▼
STEP 4 — Dashboard         dashboard/app.py
  6-page Streamlit app with Plotly
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-intelligence-platform.git
cd sentiment-intelligence-platform

pip install -r requirements.txt
python -c "import nltk; [nltk.download(p, quiet=True) for p in ['vader_lexicon','stopwords','punkt','punkt_tab','wordnet']]"

# Run full pipeline (all 3 steps)
python src/pipeline/run_pipeline.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 📊 Dashboard Pages

| Page | Content |
|---|---|
| Executive Brief | KPIs, firm ranking, platform breakdown, rating vs sentiment scatter |
| Firm Deep-Dive | Spider chart, role/dept/status sentiment, pros & cons frequency |
| Portal vs Reality | Divergence Index leaderboard, dual rating comparison, credibility scatter |
| Topic & Aspect Intel | NMF topic sunburst, sentiment heatmap, 6-aspect radar + comparison |
| Time-Series & Spikes | Rolling trends, anomaly markers, YoY drift, forecast, seasonality |
| Review Explorer | Full-text search, sentiment cards, sort & filter |

---

## 📁 File Structure

```
sentiment-intelligence-platform/
├── src/
│   ├── ingestion/generate_reviews.py     # Step 1: data collection
│   ├── nlp/nlp_pipeline.py               # Step 2: NLP (8 sub-steps)
│   ├── timeseries/timeseries_analysis.py # Step 3: time-series (6 sub-steps)
│   └── pipeline/run_pipeline.py          # Master runner
├── dashboard/app.py                      # 6-page Streamlit dashboard
├── data/
│   ├── raw/                              # Generated on first run
│   └── processed/                        # All NLP + analytics outputs
├── requirements.txt
└── README.md
```

---

## 🧠 NLP Methods

**Ensemble Formula:** `score = 0.6 × VADER_compound + 0.4 × TextBlob_polarity`

**Consulting Domain Lexicon additions to VADER:**
- burnout: -3.0, toxic: -3.5, meritocracy: +2.5, mentorship: +2.0

**Divergence Index:** `MinMaxScaler(portal_sentiment − employee_sentiment) × 100`
- 0–39: 🟢 Low Divergence | 40–69: 🟡 Moderate | 70–100: 🔴 High

---

## 📄 License

MIT

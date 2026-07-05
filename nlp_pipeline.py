"""
═══════════════════════════════════════════════════════════════════════════════
STEP 2 — NLP PROCESSING PIPELINE
═══════════════════════════════════════════════════════════════════════════════

What this file does (8 sequential steps):
------------------------------------------
① TEXT CLEANING       — lowercase, remove URLs/special chars, lemmatize
② VADER SENTIMENT     — rule-based, tuned with consulting-domain lexicon
                         (adds words like: burnout=-3.0, meritocracy=+2.5)
③ TEXTBLOB SENTIMENT  — pattern-based polarity + subjectivity score
④ ENSEMBLE SCORE      — 60% VADER + 40% TextBlob (more robust than either alone)
⑤ ASPECT SENTIMENT    — extracts sentiment specifically for 6 workplace dimensions:
                         work_life_balance, compensation, culture,
                         leadership, career_growth, work_quality
⑥ TOPIC MODELING      — NMF on TF-IDF matrix → 7 coherent themes
⑦ DIVERGENCE SCORING  — portal score minus employee score per firm → Credibility Index
⑧ BIAS FLAGGING       — flags suspicious reviews (too short, too subjective, etc.)

Why ensemble instead of just one model:
  VADER is excellent on short, punchy text (e.g. "brutal hours, toxic culture")
  TextBlob handles longer, nuanced text better
  Combined they cover both styles that appear in consulting reviews

Output files:
  data/processed/employee_reviews_processed.csv
  data/processed/portal_reviews_processed.csv
  data/processed/firm_divergence_summary.csv
  data/processed/monthly_timeseries.csv
  data/processed/topic_distribution.csv
  data/processed/topic_keywords.json
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import re, json, os, warnings
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
for pkg in ["stopwords","punkt","punkt_tab","wordnet","vader_lexicon"]:
    try: nltk.download(pkg, quiet=True)
    except: pass

STOP  = set(stopwords.words("english"))
LEM   = WordNetLemmatizer()
VADER = SentimentIntensityAnalyzer()

RAW  = Path("data/raw")
PROC = Path("data/processed")

# ─── Consulting domain lexicon additions ──────────────────────────────────────
# These words appear in consulting reviews but VADER doesn't know them
DOMAIN_LEXICON = {
    "meritocracy":2.5, "intellectually":1.5, "rigor":1.8, "prestigious":1.5,
    "accelerated":1.5, "ownership":1.2, "mentorship":2.0, "impactful":2.0,
    "exposure":1.2, "launchpad":1.5, "stimulating":2.0, "consequential":1.5,
    "burnout":-3.0, "toxic":-3.5, "cutthroat":-2.5, "unsustainable":-2.5,
    "political":-2.0, "revolving":-2.0, "opaque":-1.8, "siloed":-1.5,
    "micromanaged":-2.5, "underpaid":-2.0, "attrition":-1.5, "demoralizing":-2.5,
    "grueling":-2.0, "unaware":-1.2, "over-promises":-2.0,
}
for word, score in DOMAIN_LEXICON.items():
    VADER.lexicon[word] = score

TOPIC_NAMES = {
    0:"Career Growth & Exits", 1:"Work-Life Balance", 2:"Culture & Values",
    3:"Compensation & Benefits", 4:"Leadership & Management",
    5:"Project Quality & Impact", 6:"Diversity & Inclusion",
}

ASPECTS = {
    "work_life_balance":["balance","hours","weekend","travel","overtime","flexible","remote","burnout","life","nights"],
    "compensation":     ["salary","pay","bonus","compensation","underpaid","raise","package","stipend","benefits"],
    "culture":          ["culture","toxic","inclusive","diverse","environment","morale","team","collaborative","political"],
    "leadership":       ["leadership","management","partner","director","mentor","micromanage","transparency","trust"],
    "career_growth":    ["promotion","growth","learning","career","opportunity","upward","stuck","develop","advance"],
    "work_quality":     ["interesting","project","client","challenging","meaningful","impact","repetitive","boring","quality"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ① — TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text):
    if not isinstance(text, str): return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^a-z\s\'\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text):
    try:
        tokens = nltk.word_tokenize(text)
    except:
        tokens = text.split()
    return [LEM.lemmatize(t) for t in tokens if t not in STOP and len(t) > 2]

def preprocess(df, text_col="review_text"):
    print("    ① Cleaning and tokenizing text...")
    df = df.copy()
    df["text_clean"] = df[text_col].apply(clean_text)
    df["tokens"]     = df["text_clean"].apply(tokenize)
    df["token_str"]  = df["tokens"].apply(lambda t: " ".join(t))
    df["word_count"] = df["text_clean"].apply(lambda t: len(t.split()))
    df["char_count"] = df["text_clean"].apply(len)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ② — VADER SENTIMENT (domain-tuned)
# ═══════════════════════════════════════════════════════════════════════════════

def run_vader(df):
    print("    ② Running VADER sentiment (consulting-tuned lexicon)...")
    def score(text):
        s = VADER.polarity_scores(text)
        label = "positive" if s["compound"]>=0.05 else "negative" if s["compound"]<=-0.05 else "neutral"
        return {"vader_compound":round(s["compound"],4),
                "vader_pos":round(s["pos"],4),
                "vader_neg":round(s["neg"],4),
                "vader_neu":round(s["neu"],4),
                "vader_label":label}
    return pd.concat([df, df["text_clean"].apply(score).apply(pd.Series)], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ③ — TEXTBLOB SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_textblob(df):
    print("    ③ Running TextBlob polarity + subjectivity...")
    def score(text):
        b = TextBlob(text)
        p = round(b.sentiment.polarity, 4)
        s = round(b.sentiment.subjectivity, 4)
        label = "positive" if p>0.05 else "negative" if p<-0.05 else "neutral"
        return {"tb_polarity":p, "tb_subjectivity":s, "tb_label":label}
    return pd.concat([df, df["text_clean"].apply(score).apply(pd.Series)], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ④ — ENSEMBLE SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def run_ensemble(df):
    print("    ④ Computing ensemble score (60% VADER + 40% TextBlob)...")
    df = df.copy()
    df["ensemble_score"] = (0.6*df["vader_compound"] + 0.4*df["tb_polarity"]).round(4)
    df["ensemble_label"] = df["ensemble_score"].apply(
        lambda s: "positive" if s>=0.05 else "negative" if s<=-0.05 else "neutral")
    df["confidence"] = df["ensemble_score"].abs().clip(0,1).round(3)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ⑤ — ASPECT-BASED SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_aspects(df):
    print("    ⑤ Extracting aspect-level sentiment (6 workplace dimensions)...")
    def extract(row):
        result = {}
        text = row["text_clean"]
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = text.split(".")
        for asp, kws in ASPECTS.items():
            relevant = [s for s in sentences if any(k in s for k in kws)]
            if relevant:
                sc = VADER.polarity_scores(" ".join(relevant))["compound"]
            else:
                hits = [k for k in kws if k in text]
                sc   = VADER.polarity_scores(" ".join(hits))["compound"] if hits else 0.0
            result[f"aspect_{asp}"] = round(sc, 3)
        return result
    asp_df = df.apply(extract, axis=1).apply(pd.Series)
    return pd.concat([df, asp_df], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ⑥ — TOPIC MODELING (NMF on TF-IDF)
# ═══════════════════════════════════════════════════════════════════════════════

def run_topics(df, n_topics=7):
    print(f"    ⑥ NMF topic modeling → {n_topics} topics...")
    texts = df["token_str"].fillna("")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=3)
    X     = tfidf.fit_transform(texts)
    nmf   = NMF(n_components=n_topics, random_state=42, max_iter=500)
    W     = nmf.fit_transform(X)
    feat  = tfidf.get_feature_names_out()
    kws   = {TOPIC_NAMES[i]: [feat[j] for j in row.argsort()[-10:][::-1]]
             for i, row in enumerate(nmf.components_)}
    PROC.mkdir(parents=True, exist_ok=True)
    with open(PROC/"topic_keywords.json","w") as f:
        json.dump(kws, f, indent=2)
    df = df.copy()
    df["dominant_topic"]   = W.argmax(axis=1)
    df["topic_name"]       = df["dominant_topic"].map(TOPIC_NAMES)
    df["topic_confidence"] = W.max(axis=1).round(3)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ⑦ — DIVERGENCE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_divergence(emp_df, portal_df):
    print("    ⑦ Computing Portal ↔ Employee Divergence Index...")
    emp_agg = emp_df.groupby("firm").agg(
        emp_avg_sentiment   =("ensemble_score","mean"),
        emp_avg_rating      =("overall_rating","mean"),
        emp_pct_negative    =("ensemble_label", lambda x:(x=="negative").mean()),
        emp_pct_positive    =("ensemble_label", lambda x:(x=="positive").mean()),
        emp_review_count    =("review_id","count"),
        emp_avg_wlb         =("aspect_work_life_balance","mean"),
        emp_avg_comp        =("aspect_compensation","mean"),
        emp_avg_culture     =("aspect_culture","mean"),
        emp_avg_leadership  =("aspect_leadership","mean"),
        emp_avg_growth      =("aspect_career_growth","mean"),
    ).reset_index()

    port_agg = portal_df.groupby("firm").agg(
        portal_avg_rating   =("portal_rating","mean"),
        portal_count        =("review_id","count"),
    ).reset_index()
    port_agg["portal_sentiment"] = (port_agg["portal_avg_rating"]-3)/2

    merged = emp_agg.merge(port_agg, on="firm", how="left")
    merged["divergence_raw"]   = (merged["portal_sentiment"] - merged["emp_avg_sentiment"]).clip(0,1)
    scaler = MinMaxScaler(feature_range=(0,100))
    merged["divergence_index"] = scaler.fit_transform(merged[["divergence_raw"]]).round(1)
    merged["credibility_gap"]  = merged["divergence_index"].apply(
        lambda x: "🔴 High" if x>=70 else "🟡 Moderate" if x>=40 else "🟢 Low")
    return merged.round(4)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ⑧ — BIAS FLAGGING
# ═══════════════════════════════════════════════════════════════════════════════

def flag_bias(df):
    print("    ⑧ Flagging suspicious / low-quality reviews...")
    def flags(row):
        f = []
        if row.get("tb_subjectivity",0) > 0.85: f.append("high_subjectivity")
        if row.get("word_count",50)     < 15:    f.append("too_short")
        if row.get("confidence",1)      < 0.10:  f.append("ambiguous")
        return "|".join(f) if f else "clean"
    df = df.copy()
    df["bias_flags"] = df.apply(flags, axis=1)
    df["is_clean"]   = df["bias_flags"] == "clean"
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    PROC.mkdir(parents=True, exist_ok=True)

    print("  Loading raw data...")
    emp    = pd.read_csv(RAW/"employee_reviews_raw.csv", parse_dates=["date"])
    portal = pd.read_csv(RAW/"portal_reviews_raw.csv",  parse_dates=["date"])

    print(f"\n  Processing {len(emp):,} employee reviews:")
    emp = preprocess(emp)
    emp = run_vader(emp)
    emp = run_textblob(emp)
    emp = run_ensemble(emp)
    emp = run_aspects(emp)
    emp = run_topics(emp)
    emp = flag_bias(emp)
    emp.to_csv(PROC/"employee_reviews_processed.csv", index=False)

    print(f"\n  Processing {len(portal):,} portal reviews:")
    portal = preprocess(portal, "review_text")
    portal = run_vader(portal)
    portal = run_textblob(portal)
    portal = run_ensemble(portal)
    portal.to_csv(PROC/"portal_reviews_processed.csv", index=False)

    print("\n  Building firm-level analytics tables...")
    div = compute_divergence(emp, portal)
    div.to_csv(PROC/"firm_divergence_summary.csv", index=False)

    # Monthly time-series aggregate
    emp["date"] = pd.to_datetime(emp["date"])
    monthly = emp.groupby(["firm","year","month"]).agg(
        avg_sentiment   =("ensemble_score","mean"),
        avg_rating      =("overall_rating","mean"),
        review_count    =("review_id","count"),
        pct_negative    =("ensemble_label",lambda x:(x=="negative").mean()),
        pct_positive    =("ensemble_label",lambda x:(x=="positive").mean()),
        avg_wlb         =("aspect_work_life_balance","mean"),
        avg_culture     =("aspect_culture","mean"),
        avg_comp        =("aspect_compensation","mean"),
        avg_growth      =("aspect_career_growth","mean"),
        avg_leadership  =("aspect_leadership","mean"),
    ).reset_index().round(4)
    monthly.to_csv(PROC/"monthly_timeseries.csv", index=False)

    # Topic distribution
    topics = emp.groupby(["firm","topic_name"]).agg(
        count        =("review_id","count"),
        avg_sentiment=("ensemble_score","mean"),
    ).reset_index()
    topics.to_csv(PROC/"topic_distribution.csv", index=False)

    print("\n  ✓ Outputs:")
    for f in sorted(PROC.glob("*.csv")):
        rows = len(pd.read_csv(f))
        print(f"    {f.name:<45} {rows:>6,} rows")
    return emp, portal, div, monthly

if __name__ == "__main__":
    run_pipeline()

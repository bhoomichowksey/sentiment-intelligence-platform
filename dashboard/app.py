"""
═══════════════════════════════════════════════════════════════════════════════
STEP 5 — ADVANCED PROFESSIONAL DASHBOARD
═══════════════════════════════════════════════════════════════════════════════

6-page Streamlit dashboard with professional design:
  Page 1 — Executive Intelligence Brief
  Page 2 — Firm Deep-Dive
  Page 3 — Portal vs Reality (Divergence Analysis)
  Page 4 — Topic & Aspect Intelligence
  Page 5 — Time-Series, Spikes & Forecast
  Page 6 — Review Explorer (full-text search)

Launch: streamlit run dashboard/app.py
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Intelligence | Consulting Firms",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Hide Streamlit chrome */
  #MainMenu {visibility:hidden;} footer {visibility:hidden;}
  .block-container {padding-top:1.5rem; padding-bottom:1rem;}

  /* Sidebar */
  [data-testid="stSidebar"] {background: linear-gradient(180deg,#0D1B2A 0%,#1B2E45 100%);}
  [data-testid="stSidebar"] * {color:#E8EDF2 !important;}
  [data-testid="stSidebar"] .stRadio label {
    background:rgba(255,255,255,0.05); border-radius:8px;
    padding:8px 12px; margin-bottom:4px; cursor:pointer;
    transition:background 0.2s;
  }
  [data-testid="stSidebar"] .stRadio label:hover {background:rgba(255,255,255,0.12);}

  /* KPI cards */
  .kpi-card {
    background:#FFFFFF; border-radius:12px; padding:18px 22px;
    border-left:4px solid #1A6EBD; box-shadow:0 2px 8px rgba(0,0,0,.08);
    transition: transform 0.15s;
  }
  .kpi-card:hover {transform:translateY(-2px); box-shadow:0 4px 14px rgba(0,0,0,.12);}
  .kpi-label {color:#64748B;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
  .kpi-value {color:#0F172A;font-size:28px;font-weight:800;line-height:1.2;margin:4px 0;}
  .kpi-delta-pos {color:#16A34A;font-size:12px;font-weight:600;}
  .kpi-delta-neg {color:#DC2626;font-size:12px;font-weight:600;}

  /* Section headers */
  .section-header {
    color:#0F172A; font-size:22px; font-weight:700;
    padding-bottom:6px; border-bottom:2px solid #E2E8F0; margin-bottom:16px;
  }

  /* Review cards */
  .review-card {
    background:#FFFFFF; border-radius:10px; padding:16px 20px;
    margin-bottom:12px; box-shadow:0 1px 4px rgba(0,0,0,.06);
    border-left:4px solid #CBD5E1;
  }
  .review-card.positive {border-left-color:#16A34A;}
  .review-card.negative {border-left-color:#DC2626;}
  .review-card.neutral  {border-left-color:#D97706;}

  /* Divider */
  hr {border:none; border-top:1px solid #E2E8F0; margin:20px 0;}

  /* Plotly charts */
  .js-plotly-plot .plotly .main-svg {border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ─── Color Palette ─────────────────────────────────────────────────────────────
C = {
    "navy":   "#0D1B2A", "blue":  "#1A6EBD", "sky": "#38BDF8",
    "green":  "#16A34A", "red":   "#DC2626",  "amber":"#D97706",
    "slate":  "#64748B", "light": "#F8FAFC",  "white":"#FFFFFF",
    "border": "#E2E8F0", "text":  "#0F172A",
}
FIRM_PAL   = px.colors.qualitative.Bold
MBB        = {"McKinsey & Company","Boston Consulting Group","Bain & Company"}
BIG4       = {"Deloitte Consulting","PwC Advisory","EY Consulting","KPMG Advisory"}
PROC       = Path("data/processed")

# ─── Data Loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_emp():      return pd.read_csv(PROC/"employee_reviews_processed.csv", parse_dates=["date"])
@st.cache_data
def load_portal():   return pd.read_csv(PROC/"portal_reviews_processed.csv",  parse_dates=["date"])
@st.cache_data
def load_div():      return pd.read_csv(PROC/"firm_divergence_summary.csv")
@st.cache_data
def load_monthly():  return pd.read_csv(PROC/"monthly_timeseries.csv")
@st.cache_data
def load_ts():       return pd.read_csv(PROC/"timeseries_enriched.csv")
@st.cache_data
def load_fc():       return pd.read_csv(PROC/"sentiment_forecast.csv")
@st.cache_data
def load_drift():    return pd.read_csv(PROC/"yoy_drift.csv")
@st.cache_data
def load_topics():   return pd.read_csv(PROC/"topic_distribution.csv")
@st.cache_data
def load_aspects():  return pd.read_csv(PROC/"aspect_monthly_trend.csv")
@st.cache_data
def load_season():
    try:
        with open(PROC/"seasonality_profile.json") as f: return json.load(f)
    except: return {}

def chart_theme():
    return dict(plot_bgcolor=C["white"], paper_bgcolor=C["white"],
                font=dict(family="Inter", size=12, color=C["text"]),
                margin=dict(t=40,b=30,l=10,r=10))

# ─── KPI Card ──────────────────────────────────────────────────────────────────
def kpi(col, label, value, delta=None, delta_label="", good="up"):
    if delta is not None:
        good_pos = (delta >= 0 and good == "up") or (delta < 0 and good == "down")
        dcls     = "kpi-delta-pos" if good_pos else "kpi-delta-neg"
        arr      = "▲" if delta >= 0 else "▼"
        dtag     = f'<div class="{dcls}">{arr} {abs(delta):.2f} {delta_label}</div>'
    else: dtag = ""
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {dtag}
    </div>""", unsafe_allow_html=True)


# ─── Load All Data ─────────────────────────────────────────────────────────────
emp     = load_emp()
portal  = load_portal()
div_df  = load_div()
monthly = load_monthly()
ts      = load_ts()
fc      = load_fc()
drift   = load_drift()
topics  = load_topics()
aspects = load_aspects()
season  = load_season()

firms_all = sorted(emp["firm"].unique().tolist())
years_all = sorted(emp["year"].unique().tolist())
plats_all = sorted(emp["platform"].unique().tolist())

# ─── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px">
      <div style="font-size:28px">🔍</div>
      <div style="font-size:16px;font-weight:800;color:#F1F5F9">Sentiment Intelligence</div>
      <div style="font-size:11px;color:#94A3B8;margin-top:2px">Top 10 Consulting Firms</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "📋  Executive Brief",
        "🏢  Firm Deep-Dive",
        "⚖️   Portal vs Reality",
        "🧩  Topic & Aspect Intel",
        "📈  Time-Series & Spikes",
        "🔎  Review Explorer",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown('<div style="font-size:11px;color:#64748B;font-weight:600;text-transform:uppercase;letter-spacing:.05em">FILTERS</div>', unsafe_allow_html=True)

    sel_firms = st.multiselect("Firms", firms_all, default=firms_all)
    sel_years = st.multiselect("Years", years_all, default=years_all)
    sel_plats = st.multiselect("Platforms", plats_all, default=plats_all)

    if not sel_firms: sel_firms = firms_all
    if not sel_years: sel_years = years_all
    if not sel_plats: sel_plats = plats_all

    st.divider()
    st.markdown(f'<div style="font-size:10px;color:#475569">VADER · TextBlob · NMF Topics<br>Ensemble NLP · Z-Score Anomaly<br>Poly Regression Forecast</div>', unsafe_allow_html=True)

# Apply filters
ef  = emp[emp["firm"].isin(sel_firms) & emp["year"].isin(sel_years) & emp["platform"].isin(sel_plats)]
tsf = ts[ts["firm"].isin(sel_firms)  & ts["year"].isin(sel_years)]
df  = div_df[div_df["firm"].isin(sel_firms)]
tf  = topics[topics["firm"].isin(sel_firms)]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE BRIEF
# ═══════════════════════════════════════════════════════════════════════════════
if "Executive Brief" in page:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
      <div style="font-size:32px">📋</div>
      <div>
        <div style="font-size:26px;font-weight:800;color:{C['text']}">Executive Intelligence Brief</div>
        <div style="color:{C['slate']};font-size:13px">{len(ef):,} reviews · {ef['firm'].nunique()} firms · {ef['platform'].nunique()} platforms · {int(ef['year'].min())}–{int(ef['year'].max())}</div>
      </div>
    </div>
    <hr>""", unsafe_allow_html=True)

    # KPI Row
    avg_s   = ef["ensemble_score"].mean()
    pct_pos = (ef["ensemble_label"]=="positive").mean()*100
    pct_neg = (ef["ensemble_label"]=="negative").mean()*100
    avg_r   = ef["overall_rating"].mean()
    max_div = df["divergence_index"].max()
    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1,"Network Sentiment",f"{avg_s:+.3f}", delta=avg_s, delta_label="score")
    kpi(c2,"Positive Reviews",f"{pct_pos:.1f}%")
    kpi(c3,"Negative Reviews",f"{pct_neg:.1f}%", delta=-pct_neg, good="down")
    kpi(c4,"Avg Employee Rating",f"{avg_r:.2f} / 5.0")
    kpi(c5,"Max Credibility Gap",f"{max_div:.0f} / 100")

    st.markdown("<hr>",unsafe_allow_html=True)
    col1, col2 = st.columns([1.3,1], gap="large")

    with col1:
        firm_s = ef.groupby("firm")["ensemble_score"].mean().reset_index().sort_values("ensemble_score")
        firm_s["tier"] = firm_s["firm"].apply(lambda f:"MBB" if f in MBB else "Big 4" if f in BIG4 else "Tier 2")
        bar_c = [C["blue"] if t=="MBB" else C["sky"] if t=="Big 4" else C["amber"] for t in firm_s["tier"]]
        fig = go.Figure()
        fig.add_bar(x=firm_s["ensemble_score"], y=firm_s["firm"], orientation="h",
                    marker_color=bar_c, marker_line_width=0,
                    text=[f"{v:+.3f}" for v in firm_s["ensemble_score"]], textposition="outside",
                    textfont=dict(size=11))
        fig.add_vline(x=0, line_color=C["slate"], line_width=1)
        fig.update_layout(**chart_theme(), title="Employee Sentiment Score by Firm",
                          title_font_size=14, height=380,
                          xaxis=dict(range=[-0.6,0.8], title="Ensemble Sentiment (-1 to +1)", gridcolor=C["border"]),
                          yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        lc = ef["ensemble_label"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=lc.index.tolist(), values=lc.values.tolist(),
            hole=0.60,
            marker=dict(colors=[C["green"],C["amber"],C["red"]], line=dict(color=C["white"], width=2)),
            textinfo="label+percent", textfont_size=12,
        ))
        fig2.update_layout(**chart_theme(), title="Sentiment Distribution",
                           title_font_size=14, height=380,
                           annotations=[dict(text=f"{avg_s:+.3f}<br>score",
                                            x=0.5,y=0.5,font_size=16,showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        # Platform sentiment
        ps = ef.groupby("platform")["ensemble_score"].mean().reset_index().sort_values("ensemble_score", ascending=False)
        fig3 = px.bar(ps, x="platform", y="ensemble_score", color="ensemble_score",
                      color_continuous_scale=[[0,C["red"]],[0.5,C["amber"]],[1,C["green"]]],
                      title="Sentiment by Platform", text=ps["ensemble_score"].apply(lambda x:f"{x:+.3f}"),
                      labels={"ensemble_score":"Avg Sentiment","platform":""})
        fig3.update_traces(textposition="outside", textfont_size=11)
        fig3.update_layout(**chart_theme(), title_font_size=14, height=320,
                           coloraxis_showscale=False, yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fa = ef.groupby("firm").agg(avg_r=("overall_rating","mean"),
            avg_s=("ensemble_score","mean"), n=("review_id","count")).reset_index()
        fa["tier"] = fa["firm"].apply(lambda f:"MBB" if f in MBB else "Big 4" if f in BIG4 else "Tier 2")
        fig4 = px.scatter(fa, x="avg_r", y="avg_s", size="n", color="tier",
                          text="firm", size_max=40,
                          color_discrete_map={"MBB":C["blue"],"Big 4":C["sky"],"Tier 2":C["amber"]},
                          title="Star Rating vs NLP Sentiment Score",
                          labels={"avg_r":"Avg Star Rating","avg_s":"Sentiment Score","tier":"Tier"})
        fig4.update_traces(textposition="top center", textfont_size=8)
        fig4.update_layout(**chart_theme(), title_font_size=14, height=320,
                           xaxis=dict(gridcolor=C["border"]),yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FIRM DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Firm Deep-Dive" in page:
    st.markdown(f'<div style="font-size:26px;font-weight:800;color:{C["text"]}">🏢 Firm Deep-Dive — Employee Lens</div><hr>', unsafe_allow_html=True)

    sel = st.selectbox("Select Firm", firms_all)
    fd  = emp[emp["firm"]==sel]
    tier= "MBB" if sel in MBB else "Big 4" if sel in BIG4 else "Tier 2"
    st.markdown(f'<span style="background:{C["blue"]}22;color:{C["blue"]};border-radius:6px;padding:3px 10px;font-size:12px;font-weight:700">{tier}</span>', unsafe_allow_html=True)
    st.markdown("")

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1,"Sentiment Score",   f"{fd['ensemble_score'].mean():+.3f}")
    kpi(c2,"Avg Star Rating",   f"{fd['overall_rating'].mean():.2f}")
    kpi(c3,"% Negative",        f"{(fd['ensemble_label']=='negative').mean()*100:.1f}%")
    kpi(c4,"Model Confidence",  f"{fd['confidence'].mean():.2f}")
    kpi(c5,"Avg Subjectivity",  f"{fd['tb_subjectivity'].mean():.2f}")
    st.markdown("<hr>",unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        # Spider chart of sub-ratings
        rcols  = [c for c in fd.columns if c.startswith("rating_")]
        labels = [c.replace("rating_","").replace("_"," ").replace("and","&").title() for c in rcols]
        vals   = fd[rcols].mean().tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=labels+[labels[0]],
                      fill="toself", fillcolor=f"rgba(26,110,189,0.15)",
                      line=dict(color=C["blue"],width=2.5), name=sel))
        # Benchmark line (industry avg = 3.9)
        bench = [3.9]*len(labels)
        fig.add_trace(go.Scatterpolar(r=bench+[bench[0]], theta=labels+[labels[0]],
                      line=dict(color=C["slate"],width=1,dash="dot"),
                      name="Industry Avg", fill="none"))
        fig.update_layout(**chart_theme(), polar=dict(radialaxis=dict(range=[1,5], gridcolor=C["border"])),
                          title=f"Sub-Rating Profile vs Industry Average",
                          title_font_size=14, height=400,
                          legend=dict(x=0.8, y=0.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Role sentiment
        rs = fd.groupby("reviewer_role")["ensemble_score"].mean().reset_index().sort_values("ensemble_score")
        bar_c = [C["green"] if v>0.05 else C["red"] if v<-0.05 else C["amber"] for v in rs["ensemble_score"]]
        fig2 = go.Figure()
        fig2.add_bar(x=rs["ensemble_score"], y=rs["reviewer_role"], orientation="h",
                     marker_color=bar_c, marker_line_width=0,
                     text=[f"{v:+.3f}" for v in rs["ensemble_score"]], textposition="outside")
        fig2.add_vline(x=0, line_color=C["slate"], line_width=1)
        fig2.update_layout(**chart_theme(), title="Sentiment by Role",
                           title_font_size=14, height=400,
                           xaxis=dict(range=[-0.7,0.8],gridcolor=C["border"]),
                           yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        ds = fd.groupby("department")["ensemble_score"].mean().reset_index().sort_values("ensemble_score")
        fig3 = px.bar(ds, x="ensemble_score", y="department", orientation="h",
                      color="ensemble_score", color_continuous_scale=[[0,C["red"]],[0.5,C["amber"]],[1,C["green"]]],
                      title="Sentiment by Department",
                      labels={"ensemble_score":"Sentiment","department":""})
        fig3.add_vline(x=0, line_color=C["slate"], line_dash="dash", line_width=1)
        fig3.update_layout(**chart_theme(), coloraxis_showscale=False,
                           title_font_size=14, height=340, yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        ss = fd.groupby("employment_status")["ensemble_score"].mean().reset_index()
        fig4 = px.bar(ss, x="employment_status", y="ensemble_score",
                      color="ensemble_score", color_continuous_scale=[[0,C["red"]],[0.5,C["amber"]],[1,C["green"]]],
                      title="Current vs Former Employee Sentiment",
                      text=ss["ensemble_score"].apply(lambda x:f"{x:+.3f}"),
                      labels={"ensemble_score":"Sentiment","employment_status":""})
        fig4.update_traces(textposition="outside", textfont_size=13)
        fig4.update_layout(**chart_theme(), coloraxis_showscale=False,
                           title_font_size=14, height=340, yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig4, use_container_width=True)

    # Pros & Cons
    st.markdown(f'<div class="section-header">Most Cited Pros & Cons</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2, gap="large")
    with col5:
        p = pd.Series(" | ".join(fd["pros"].dropna()).split(" | ")).value_counts().head(10).reset_index()
        p.columns=["Item","Count"]
        fig5 = px.bar(p,x="Count",y="Item",orientation="h",
                      color_discrete_sequence=[C["green"]], title="✅ Top Pros")
        fig5.update_layout(**chart_theme(),title_font_size=14,height=320,yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig5, use_container_width=True)
    with col6:
        c = pd.Series(" | ".join(fd["cons"].dropna()).split(" | ")).value_counts().head(10).reset_index()
        c.columns=["Item","Count"]
        fig6 = px.bar(c,x="Count",y="Item",orientation="h",
                      color_discrete_sequence=[C["red"]], title="⚠️ Top Cons")
        fig6.update_layout(**chart_theme(),title_font_size=14,height=320,yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PORTAL vs REALITY
# ═══════════════════════════════════════════════════════════════════════════════
elif "Portal vs Reality" in page:
    st.markdown(f'<div style="font-size:26px;font-weight:800;color:{C["text"]}">⚖️ Portal vs Reality — Credibility Divergence</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{C["slate"]};font-size:13px;margin-bottom:12px">Quantifying the gap between what firms <b>publish about themselves</b> and what employees <b>actually say</b> on review platforms.</div><hr>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.3,1], gap="large")
    with col1:
        dv = df.sort_values("divergence_index")
        bar_c = [C["green"] if x<40 else C["amber"] if x<70 else C["red"] for x in dv["divergence_index"]]
        fig = go.Figure()
        fig.add_bar(x=dv["divergence_index"], y=dv["firm"], orientation="h",
                    marker_color=bar_c, marker_line_width=0,
                    text=[f"{v:.0f}" for v in dv["divergence_index"]], textposition="outside",
                    textfont=dict(size=12,color=C["text"]))
        fig.add_vline(x=40,line_dash="dash",line_color=C["amber"],
                      annotation_text="Moderate",annotation_font_color=C["amber"])
        fig.add_vline(x=70,line_dash="dash",line_color=C["red"],
                      annotation_text="High Gap",annotation_font_color=C["red"])
        fig.update_layout(**chart_theme(), title="Credibility Divergence Index (0=Authentic → 100=Curated)",
                          title_font_size=14, height=400, xaxis=dict(range=[0,118],gridcolor=C["border"]),
                          yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        comp = df.copy()
        fig2 = go.Figure()
        fig2.add_bar(name="Portal Rating (Curated)", x=comp["firm"], y=comp["portal_avg_rating"],
                     marker_color=C["sky"], marker_line_width=0)
        fig2.add_bar(name="Employee Rating (Real)", x=comp["firm"], y=comp["emp_avg_rating"],
                     marker_color=C["blue"], marker_line_width=0)
        fig2.update_layout(**chart_theme(), barmode="group",
                           title="Portal vs Employee Average Rating",
                           title_font_size=14, height=400,
                           yaxis=dict(range=[0,5.8],gridcolor=C["border"]),
                           xaxis_tickangle=-35,
                           legend=dict(x=0,y=1.1,orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: divergence vs negative sentiment
    fig3 = px.scatter(df, x="divergence_index", y="emp_pct_negative",
                      size="emp_review_count", color="credibility_gap",
                      text="firm", size_max=40,
                      color_discrete_map={"🔴 High":C["red"],"🟡 Moderate":C["amber"],"🟢 Low":C["green"]},
                      title="Credibility Gap Index vs % Negative Reviews",
                      labels={"divergence_index":"Divergence Index (0–100)",
                              "emp_pct_negative":"Fraction of Negative Reviews",
                              "credibility_gap":"Gap Level"})
    fig3.update_traces(textposition="top center", textfont_size=9)
    fig3.update_layout(**chart_theme(), height=380,
                       xaxis=dict(gridcolor=C["border"]),yaxis=dict(gridcolor=C["border"]))
    st.plotly_chart(fig3, use_container_width=True)

    # Summary table
    st.markdown(f'<div class="section-header">Divergence Summary Table</div>', unsafe_allow_html=True)
    show_cols = {
        "firm":"Firm","portal_avg_rating":"Portal Rating","emp_avg_rating":"Employee Rating",
        "emp_avg_sentiment":"Emp Sentiment","emp_pct_negative":"% Negative",
        "divergence_index":"Divergence Index","credibility_gap":"Gap Label"
    }
    disp = df[[c for c in show_cols if c in df.columns]].rename(columns=show_cols)
    disp = disp.sort_values("Divergence Index", ascending=False).round(3)
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TOPIC & ASPECT INTEL
# ═══════════════════════════════════════════════════════════════════════════════
elif "Topic & Aspect" in page:
    st.markdown(f'<div style="font-size:26px;font-weight:800;color:{C["text"]}">🧩 Topic & Aspect Intelligence</div><hr>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        tv = tf.groupby(["firm","topic_name"])["count"].sum().reset_index()
        fig = px.sunburst(tv, path=["firm","topic_name"], values="count",
                          color="count", color_continuous_scale="Blues",
                          title="Review Volume: Firm → Topic")
        fig.update_layout(**chart_theme(), height=440, title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ts2 = tf.groupby(["firm","topic_name"])["avg_sentiment"].mean().reset_index()
        piv = ts2.pivot(index="firm",columns="topic_name",values="avg_sentiment")
        fig2 = px.imshow(piv, color_continuous_scale=[[0,C["red"]],[0.5,C["amber"]],[1,C["green"]]],
                         zmin=-0.3, zmax=0.5, title="Sentiment Heatmap: Firm × Topic",
                         text_auto=".2f", aspect="auto")
        fig2.update_layout(**chart_theme(), height=440, title_font_size=14)
        st.plotly_chart(fig2, use_container_width=True)

    # Aspect radar
    st.markdown(f'<div class="section-header">Aspect-Level Sentiment Radar</div>', unsafe_allow_html=True)
    ASPECT_MAP = {
        "aspect_work_life_balance":"Work-Life",
        "aspect_compensation":"Compensation",
        "aspect_culture":"Culture",
        "aspect_leadership":"Leadership",
        "aspect_career_growth":"Career Growth",
        "aspect_work_quality":"Work Quality",
    }
    aa = ef.groupby("firm")[[*ASPECT_MAP]].mean().reset_index()
    lbls = list(ASPECT_MAP.values())
    fig3 = go.Figure()
    for _, row in aa.iterrows():
        vs = [row[k] for k in ASPECT_MAP]
        fig3.add_trace(go.Scatterpolar(r=vs+[vs[0]], theta=lbls+[lbls[0]],
                       fill="toself", opacity=0.4, name=row["firm"]))
    fig3.update_layout(**chart_theme(),
                       polar=dict(radialaxis=dict(range=[-0.5,0.6],gridcolor=C["border"])),
                       title="6-Dimension Aspect Sentiment — All Firms",
                       title_font_size=14, height=500,
                       legend=dict(x=1.05,y=0.5))
    st.plotly_chart(fig3, use_container_width=True)

    # Aspect comparison bars
    sel_asp = st.selectbox("Compare single aspect across firms",
        ["Work-Life","Compensation","Culture","Leadership","Career Growth","Work Quality"])
    rev_map = {v:k for k,v in ASPECT_MAP.items()}
    key = rev_map[sel_asp]
    ad = ef.groupby("firm")[key].mean().reset_index().sort_values(key)
    fig4 = px.bar(ad, x=key, y="firm", orientation="h",
                  color=key, color_continuous_scale=[[0,C["red"]],[0.5,C["amber"]],[1,C["green"]]],
                  title=f"'{sel_asp}' Aspect Sentiment — All Firms",
                  text=ad[key].apply(lambda x:f"{x:+.3f}"),
                  labels={key:"Sentiment Score"})
    fig4.add_vline(x=0, line_color=C["slate"], line_dash="dash")
    fig4.update_traces(textposition="outside")
    fig4.update_layout(**chart_theme(), coloraxis_showscale=False,
                       title_font_size=14, height=380, yaxis=dict(gridcolor=C["border"]))
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — TIME-SERIES & SPIKES
# ═══════════════════════════════════════════════════════════════════════════════
elif "Time-Series" in page:
    st.markdown(f'<div style="font-size:26px;font-weight:800;color:{C["text"]}">📈 Time-Series, Demand Spikes & Forecast</div><hr>', unsafe_allow_html=True)

    firm_sel = st.selectbox("Focus Firm", ["All Firms"]+firms_all)

    if firm_sel == "All Firms":
        ts_data = tsf.copy()
    else:
        ts_data = ts[ts["firm"]==firm_sel].copy()

    ts_data["period"] = pd.to_datetime(
        ts_data["year"].astype(str)+"-"+ts_data["month"].astype(str).str.zfill(2)+"-01")

    # Main trend chart
    fig = go.Figure()
    if firm_sel == "All Firms":
        for firm, grp in ts_data.groupby("firm"):
            grp = grp.sort_values("period")
            if "sentiment_ma6" in grp.columns:
                fig.add_scatter(x=grp["period"], y=grp["sentiment_ma6"],
                                mode="lines", name=firm, opacity=0.75, line=dict(width=1.8))
    else:
        grp = ts_data.sort_values("period")
        if "sentiment_ma6" in grp.columns:
            if "sentiment_stddev" in grp.columns:
                fig.add_scatter(
                    x=pd.concat([grp["period"],grp["period"][::-1]]),
                    y=pd.concat([grp["sentiment_ma6"]+grp["sentiment_stddev"].fillna(0),
                                 (grp["sentiment_ma6"]-grp["sentiment_stddev"].fillna(0))[::-1]]),
                    fill="toself", fillcolor="rgba(26,110,189,0.10)",
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="Confidence Band")
            fig.add_scatter(x=grp["period"], y=grp["avg_sentiment"],
                            mode="markers", marker=dict(size=4,color=C["slate"],opacity=0.5),
                            name="Monthly Avg")
            fig.add_scatter(x=grp["period"], y=grp["sentiment_ma3"],
                            mode="lines", line=dict(color=C["sky"],width=1.8,dash="dot"),
                            name="3M Rolling Avg")
            fig.add_scatter(x=grp["period"], y=grp["sentiment_ma6"],
                            mode="lines", line=dict(color=C["blue"],width=2.5),
                            name="6M Rolling Avg")
        if "event_flag" in grp.columns:
            spk = grp[grp["event_flag"]==True]
            if not spk.empty:
                fig.add_scatter(x=spk["period"], y=spk["avg_sentiment"],
                                mode="markers", marker=dict(size=14,color=C["red"],
                                symbol="triangle-up",line=dict(width=1,color=C["white"])),
                                name="⚠️ Anomaly")
        # Forecast
        fc2 = fc[fc["firm"]==firm_sel].copy()
        if not fc2.empty:
            fc2["period"] = pd.to_datetime(fc2["year"].astype(str)+"-"+fc2["month"].astype(str).str.zfill(2)+"-01")
            fig.add_scatter(x=fc2["period"], y=fc2["forecast_sentiment"],
                            mode="lines+markers", line=dict(color=C["amber"],dash="dash",width=2),
                            marker=dict(size=6, symbol="circle-open"),
                            name="📊 Forecast (6M)")

    fig.update_layout(**chart_theme(), height=420,
                      title=f"Sentiment Time-Series — {firm_sel}",
                      title_font_size=14, xaxis_title="",
                      yaxis_title="Sentiment Score",
                      yaxis=dict(gridcolor=C["border"]),
                      xaxis=dict(gridcolor=C["border"]),
                      legend=dict(x=0,y=-0.2,orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        # YoY drift
        dr2 = drift[drift["firm"].isin(sel_firms)]
        fig2 = px.line(dr2, x="year", y="avg_sentiment", color="firm",
                       color_discrete_sequence=FIRM_PAL,
                       title="Year-over-Year Sentiment Drift",
                       labels={"avg_sentiment":"Avg Sentiment","year":"Year"},
                       markers=True)
        fig2.update_layout(**chart_theme(), height=340, title_font_size=14,
                           yaxis=dict(gridcolor=C["border"]),xaxis=dict(gridcolor=C["border"]),
                           legend=dict(font_size=9))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Month-of-year seasonality
        mn = ef.groupby("month")["ensemble_score"].mean().reset_index()
        mnames={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        mn["month_name"] = mn["month"].map(mnames)
        fig3 = px.bar(mn, x="month_name", y="ensemble_score",
                      color="ensemble_score",
                      color_continuous_scale=[[0,C["red"]],[0.5,C["amber"]],[1,C["green"]]],
                      title="Seasonal Pattern: Avg Sentiment by Month",
                      labels={"ensemble_score":"Avg Sentiment","month_name":"Month"})
        fig3.update_layout(**chart_theme(), coloraxis_showscale=False,
                           height=340, title_font_size=14, yaxis=dict(gridcolor=C["border"]))
        st.plotly_chart(fig3, use_container_width=True)

    # Anomaly table
    if "event_flag" in ts.columns:
        ev = ts[ts["event_flag"]==True][
            ["firm","year","month","avg_sentiment","review_count","event_type","volume_zscore","sentiment_zscore"]
        ].sort_values("sentiment_zscore").head(20)
        st.markdown(f'<div class="section-header">⚠️ Detected Anomalies — {len(ts[ts["event_flag"]==True])} total events</div>', unsafe_allow_html=True)
        st.dataframe(ev.round(3), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — REVIEW EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif "Review Explorer" in page:
    st.markdown(f'<div style="font-size:26px;font-weight:800;color:{C["text"]}">🔎 Review Explorer</div><hr>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1: srch = st.text_input("🔍 Search text", placeholder="burnout, culture, promotion...")
    with col2: sf   = st.multiselect("Sentiment", ["positive","neutral","negative"], default=["positive","neutral","negative"])
    with col3: rf   = st.multiselect("Role", sorted(emp["reviewer_role"].unique()), default=sorted(emp["reviewer_role"].unique()))

    view = ef[ef["ensemble_label"].isin(sf) & ef["reviewer_role"].isin(rf)].copy()
    if srch:
        view = view[view["review_text"].str.lower().str.contains(srch.lower(), na=False)]

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"Matching Reviews",   f"{len(view):,}")
    kpi(c2,"Avg Sentiment",      f"{view['ensemble_score'].mean():+.3f}")
    kpi(c3,"Avg Rating",         f"{view['overall_rating'].mean():.2f}")
    kpi(c4,"% Negative",         f"{(view['ensemble_label']=='negative').mean()*100:.1f}%")
    st.markdown("<hr>",unsafe_allow_html=True)

    # Sort
    sort_by = st.selectbox("Sort by", ["Date (newest)","Sentiment (most positive)","Sentiment (most negative)","Rating (highest)","Helpful count"])
    sort_map = {
        "Date (newest)":          ("date",False),
        "Sentiment (most positive)":("ensemble_score",False),
        "Sentiment (most negative)":("ensemble_score",True),
        "Rating (highest)":       ("overall_rating",False),
        "Helpful count":          ("helpful_count",False),
    }
    sc, sa = sort_map[sort_by]
    view = view.sort_values(sc, ascending=sa)

    st.markdown(f"Showing **{min(50,len(view))}** of **{len(view):,}** reviews")

    for _, r in view.head(50).iterrows():
        sc2 = r.get("ensemble_label","neutral")
        scol= C["green"] if sc2=="positive" else C["red"] if sc2=="negative" else C["amber"]
        stars = "★"*int(r.get("overall_rating",0)) + "☆"*(5-int(r.get("overall_rating",0)))
        st.markdown(f"""
        <div class="review-card {sc2}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <div>
              <span style="font-weight:700;color:{C['text']};font-size:14px">{r.get('firm','')}</span>
              <span style="background:{scol}22;color:{scol};border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;margin-left:8px">{sc2.upper()}</span>
              <span style="color:{C['slate']};font-size:11px;margin-left:8px">{r.get('reviewer_role','')} · {r.get('department','')}</span>
            </div>
            <div style="color:{C['slate']};font-size:11px">{str(r.get('date',''))[:10]} · {r.get('platform','')}</div>
          </div>
          <div style="font-weight:600;font-size:14px;margin-bottom:6px;color:{C['text']}">{r.get('review_title','')}</div>
          <div style="color:#374151;font-size:13px;line-height:1.6;margin-bottom:10px">{r.get('review_text','')}</div>
          <div style="display:flex;gap:20px;font-size:12px;color:{C['slate']}">
            <span>⭐ <b style="color:{C['text']}">{stars}</b> {r.get('overall_rating','?')}/5</span>
            <span>📊 Score: <b style="color:{scol}">{r.get('ensemble_score',0):+.3f}</b></span>
            <span>🎯 Confidence: <b>{r.get('confidence',0):.2f}</b></span>
            <span>✍️ Subjectivity: <b>{r.get('tb_subjectivity',0):.2f}</b></span>
            <span>👍 {r.get('helpful_count',0)} helpful</span>
            <span>🏷 {r.get('bias_flags','')}</span>
          </div>
          {"<div style='margin-top:8px;font-size:12px;color:#16A34A'>✅ <b>Pros:</b> "+str(r.get('pros',''))+"</div>" if pd.notna(r.get('pros')) else ""}
          {"<div style='font-size:12px;color:#DC2626'>⚠️ <b>Cons:</b> "+str(r.get('cons',''))+"</div>" if pd.notna(r.get('cons')) else ""}
        </div>""", unsafe_allow_html=True)

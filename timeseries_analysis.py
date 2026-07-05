"""
═══════════════════════════════════════════════════════════════════════════════
STEP 3 — TIME-SERIES ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

What this file does:
--------------------
Takes the monthly aggregates from Step 2 and applies time-series techniques:

① SPIKE DETECTION    — Z-score method: flags months where review volume or
                        sentiment dropped abnormally (|Z| > 1.5).
                        These correlate with real events: layoffs, scandal,
                        bad press, or cultural shifts.

② ROLLING AVERAGES   — 3M / 6M / 12M rolling windows to smooth noise and
                        reveal underlying sentiment trends per firm.

③ SEASONALITY        — Month-of-year effects (e.g. Jan reviews are always
                        more negative — bonus season disappointment).
                        Day-of-week effects (weekend reviews are harsher).

④ YoY DRIFT          — Year-over-year change in average sentiment per firm.
                        Positive drift = improving culture.
                        Negative drift = deteriorating culture.

⑤ FORECASTING        — Polynomial regression (degree 2) to predict next 6
                        months of sentiment per firm.
                        Why poly not linear? Sentiment has curves — firms
                        often improve, plateau, then slip again.

⑥ ASPECT TRENDS      — Monthly trajectory for each of the 6 workplace
                        dimensions (work-life, comp, culture, etc.)

Output files:
  data/processed/timeseries_enriched.csv
  data/processed/yoy_drift.csv
  data/processed/sentiment_forecast.csv
  data/processed/aspect_monthly_trend.csv
  data/processed/seasonality_profile.json
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import json, warnings
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")
PROC = Path("data/processed")


def load():
    emp     = pd.read_csv(PROC/"employee_reviews_processed.csv", parse_dates=["date"])
    monthly = pd.read_csv(PROC/"monthly_timeseries.csv")
    return emp, monthly


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ① — SPIKE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_spikes(monthly, z_thresh=1.5):
    print("  ① Spike detection (Z-score method)...")
    result = []
    for firm, grp in monthly.groupby("firm"):
        grp = grp.sort_values(["year","month"]).copy()
        mu_v, sd_v = grp["review_count"].mean(), grp["review_count"].std()
        mu_s, sd_s = grp["avg_sentiment"].mean(), grp["avg_sentiment"].std()
        sd_v = sd_v if sd_v > 0 else 1
        sd_s = sd_s if sd_s > 0 else 1
        grp["volume_zscore"]    = ((grp["review_count"] - mu_v) / sd_v).round(3)
        grp["sentiment_zscore"] = ((grp["avg_sentiment"] - mu_s) / sd_s).round(3)
        grp["volume_spike"]     = grp["volume_zscore"].abs() > z_thresh
        grp["sentiment_crash"]  = grp["sentiment_zscore"] < -z_thresh
        grp["event_flag"]       = grp["volume_spike"] | grp["sentiment_crash"]
        def classify(row):
            if row["volume_spike"] and row["sentiment_crash"]: return "Crisis Event"
            if row["volume_spike"]:                            return "Volume Spike"
            if row["sentiment_crash"]:                         return "Sentiment Crash"
            return "Normal"
        grp["event_type"] = grp.apply(classify, axis=1)
        result.append(grp)
    out = pd.concat(result, ignore_index=True)
    print(f"    → {out['event_flag'].sum()} anomalous months detected across all firms")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ② — ROLLING AVERAGES
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_averages(monthly):
    print("  ② Computing rolling averages (3M / 6M / 12M)...")
    result = []
    for firm, grp in monthly.groupby("firm"):
        grp = grp.sort_values(["year","month"]).copy()
        for w in [3, 6, 12]:
            grp[f"sentiment_ma{w}"] = grp["avg_sentiment"].rolling(w, min_periods=1).mean().round(4)
            grp[f"volume_ma{w}"]    = grp["review_count"].rolling(w, min_periods=1).mean().round(1)
        grp["sentiment_mom"]    = grp["avg_sentiment"].diff(3).round(4)   # 3-month momentum
        grp["sentiment_stddev"] = grp["avg_sentiment"].rolling(6, min_periods=1).std().round(4)
        result.append(grp)
    return pd.concat(result, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ③ — SEASONALITY
# ═══════════════════════════════════════════════════════════════════════════════

def seasonality_profile(emp):
    print("  ③ Building seasonality profile (month-of-year effects)...")
    by_month = emp.groupby("month")["ensemble_score"].mean().round(4).to_dict()
    by_dow   = emp.groupby(emp["date"].dt.dayofweek)["ensemble_score"].mean().round(4)
    by_dow   = {int(k): float(v) for k, v in by_dow.items()}
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    profile = {
        "by_month_number": by_month,
        "by_month_name":   {month_names[k]: v for k, v in by_month.items()},
        "by_day_of_week":  by_dow,
        "insight": "Jan/Feb typically more negative (bonus season). Jun-Aug spike in volume (intern class reviewing)."
    }
    with open(PROC/"seasonality_profile.json","w") as f:
        json.dump(profile, f, indent=2)
    return profile


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ④ — YoY DRIFT
# ═══════════════════════════════════════════════════════════════════════════════

def yoy_drift(monthly):
    print("  ④ Computing year-over-year sentiment drift...")
    yearly = monthly.groupby(["firm","year"]).agg(
        avg_sentiment   =("avg_sentiment","mean"),
        avg_rating      =("avg_rating","mean"),
        review_count    =("review_count","sum"),
        pct_negative    =("pct_negative","mean"),
    ).reset_index()
    yearly["sentiment_yoy"] = yearly.groupby("firm")["avg_sentiment"].diff().round(4)
    yearly["rating_yoy"]    = yearly.groupby("firm")["avg_rating"].diff().round(3)
    yearly["drift_label"]   = yearly["sentiment_yoy"].apply(
        lambda x: "↑ Improving" if pd.notna(x) and x>0.03
             else "↓ Declining" if pd.notna(x) and x<-0.03
             else "→ Stable"   if pd.notna(x) else "—")
    return yearly


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ⑤ — FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

def forecast_sentiment(monthly, horizon=6):
    print(f"  ⑤ Forecasting next {horizon} months (polynomial regression)...")
    forecasts = []
    for firm, grp in monthly.groupby("firm"):
        grp = grp.sort_values(["year","month"]).copy()
        grp["t"] = range(len(grp))
        X = grp[["t"]].values
        y = grp["avg_sentiment"].values
        poly  = PolynomialFeatures(degree=2)
        Xp    = poly.fit_transform(X)
        model = LinearRegression().fit(Xp, y)
        last_yr, last_mo = int(grp["year"].iloc[-1]), int(grp["month"].iloc[-1])
        future = []
        ym = (last_yr, last_mo)
        for _ in range(horizon):
            mo = ym[1]%12+1
            yr = ym[0]+(1 if ym[1]==12 else 0)
            ym = (yr,mo)
            future.append(ym)
        t_fut  = np.arange(len(grp), len(grp)+horizon).reshape(-1,1)
        preds  = model.predict(poly.transform(t_fut)).clip(-1,1)
        for (yr,mo), pred in zip(future, preds):
            forecasts.append({"firm":firm,"year":yr,"month":mo,
                              "forecast_sentiment":round(float(pred),4),"is_forecast":True})
    return pd.DataFrame(forecasts)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP ⑥ — ASPECT MONTHLY TRENDS
# ═══════════════════════════════════════════════════════════════════════════════

def aspect_trends(emp):
    print("  ⑥ Building aspect-level monthly trends...")
    asp_cols = [c for c in emp.columns if c.startswith("aspect_")]
    return emp.groupby(["firm","year","month"])[asp_cols].mean().reset_index().round(4)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_timeseries():
    emp, monthly = load()
    monthly_spikes   = detect_spikes(monthly)
    monthly_rolling  = rolling_averages(monthly_spikes)
    _                = seasonality_profile(emp)
    drift            = yoy_drift(monthly)
    fc               = forecast_sentiment(monthly)
    asp              = aspect_trends(emp)

    monthly_rolling.to_csv(PROC/"timeseries_enriched.csv", index=False)
    drift.to_csv(PROC/"yoy_drift.csv", index=False)
    fc.to_csv(PROC/"sentiment_forecast.csv", index=False)
    asp.to_csv(PROC/"aspect_monthly_trend.csv", index=False)

    print("\n  ✓ Time-series outputs saved:")
    for name in ["timeseries_enriched.csv","yoy_drift.csv","sentiment_forecast.csv","aspect_monthly_trend.csv"]:
        rows = len(pd.read_csv(PROC/name))
        print(f"    {name:<40} {rows:>5,} rows")
    return monthly_rolling, drift, fc, asp

if __name__ == "__main__":
    run_timeseries()

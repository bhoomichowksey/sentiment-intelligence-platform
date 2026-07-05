"""
═══════════════════════════════════════════════════════════════════════════════
STEP 1 — DATA COLLECTION
═══════════════════════════════════════════════════════════════════════════════

What this file does:
--------------------
In a real project this is where you would scrape:
  • Glassdoor  → employee reviews, ratings, pros/cons
  • AmbitionBox → India-focused employee reviews
  • Blind       → anonymous professional reviews
  • Indeed      → employee reviews
  • Firm Portal → what the firm PUBLISHES about itself (client testimonials)

Since we cannot scrape live (legal/rate-limit reasons), this file generates
REALISTIC synthetic data calibrated to match real-world distributions from
public reports (Glassdoor annual reports, Bureau of Labor Statistics, etc.)

The data model has TWO sources on purpose:
  1. EMPLOYEE REVIEWS  — the unfiltered truth from inside
  2. PORTAL REVIEWS    — what the firm curates and shows to the world

The GAP between these two is the core analytical finding of the platform.

Firms covered (Top 10 by revenue/prestige):
  MBB tier   : McKinsey, BCG, Bain
  Big 4 tier : Deloitte, PwC, EY, KPMG
  Tier 2     : Accenture Strategy, Oliver Wyman, Roland Berger

Output files:
  data/raw/employee_reviews_raw.csv   → 5,000 rows
  data/raw/portal_reviews_raw.csv     → 1,200 rows
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
np.random.seed(42)
random.seed(42)

# ─── Firm Master Data ──────────────────────────────────────────────────────────
# Each firm has realistic parameters calibrated to public Glassdoor data (2024)
FIRMS = {
    "McKinsey & Company":       {"tier":"MBB",   "glassdoor_avg":3.9, "portal_bias":0.92, "size":45000},
    "Boston Consulting Group":  {"tier":"MBB",   "glassdoor_avg":4.1, "portal_bias":0.90, "size":32000},
    "Bain & Company":           {"tier":"MBB",   "glassdoor_avg":4.3, "portal_bias":0.88, "size":12000},
    "Deloitte Consulting":      {"tier":"Big4",  "glassdoor_avg":4.0, "portal_bias":0.82, "size":415000},
    "PwC Advisory":             {"tier":"Big4",  "glassdoor_avg":3.9, "portal_bias":0.80, "size":370000},
    "EY Consulting":            {"tier":"Big4",  "glassdoor_avg":3.8, "portal_bias":0.79, "size":395000},
    "KPMG Advisory":            {"tier":"Big4",  "glassdoor_avg":3.7, "portal_bias":0.76, "size":265000},
    "Accenture Strategy":       {"tier":"Tier2", "glassdoor_avg":4.0, "portal_bias":0.78, "size":750000},
    "Oliver Wyman":             {"tier":"Tier2", "glassdoor_avg":3.8, "portal_bias":0.85, "size":7000},
    "Roland Berger":            {"tier":"Tier2", "glassdoor_avg":3.7, "portal_bias":0.80, "size":3500},
}

PLATFORMS   = ["Glassdoor", "AmbitionBox", "Indeed", "Blind", "Comparably"]
ROLES       = ["Analyst","Associate","Consultant","Senior Consultant","Manager",
               "Senior Manager","Principal","Associate Partner","Partner","Former Intern"]
DEPARTMENTS = ["Strategy","Operations","Technology","Finance","HR",
               "Digital Transformation","M&A","Risk Advisory","Supply Chain","Healthcare"]
LOCATIONS   = ["New York, NY","Chicago, IL","San Francisco, CA","London, UK","Boston, MA",
               "Washington DC","Dallas, TX","Atlanta, GA","Mumbai, India","Singapore",
               "Dubai, UAE","Toronto, Canada","Paris, France","Frankfurt, Germany"]
EMP_STATUS  = ["Current Employee","Former Employee"]
SEASONS     = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
               6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall",12:"Winter"}

RATING_DIMS = ["Work-Life Balance","Compensation","Culture","Leadership",
               "Career Growth","Diversity & Inclusion","Work Quality","Client Exposure"]

# ─── Review Text Templates ─────────────────────────────────────────────────────
# Templates simulate the linguistic patterns of real Glassdoor reviews
POS_REVIEWS = [
    "Incredible learning environment. The intellectual rigor here is unlike anything else — you grow faster than you thought possible. {dept} practice has world-class clients and the exposure is unmatched. {wl_pos} Exit opportunities are exceptional.",
    "Best career decision I made. The people are brilliant and the mentorship culture is real — partners genuinely invest in junior development. Compensation is strong and the {dept} work is genuinely challenging and meaningful.",
    "Strong meritocracy. Performance is recognized and promotions happen if you put in the work. The {dept} team specifically is excellent — collaborative, smart, and driven by impact. Brand opens every door.",
    "The training programs here are exceptional. Every engagement stretches your thinking. The firm's investment in learning and development sets it apart from competitors. Steep learning curve but incredibly rewarding.",
    "Firm culture has improved significantly. Work quality in {dept} is top-tier — clients are Fortune 500, projects are consequential. The alumni network alone makes the grueling years worthwhile. Great launchpad.",
    "Intellectually stimulating work every single day. The caliber of colleagues is exceptional across all levels. {dept} leadership is visionary and accessible. Compensation is competitive and keeps pace with the market.",
]
NEU_REVIEWS = [
    "Pros: elite brand, smart colleagues, interesting {dept} projects. Cons: work-life balance is inconsistent, up-or-out pressure is real, and staffing model can burn you out fast. Know what you are signing up for.",
    "Typical consulting experience — heavy travel, long hours, demanding clients. The prestige is real and so is the burnout risk. The {dept} practice is solid but management quality varies dramatically by partner.",
    "Good place for 2-3 years then exit to industry. The {dept} team is competent. Some great partners, some difficult ones. Promotion criteria is not always transparent. Depends heavily on which office you land in.",
    "The firm talks a good game on culture but delivery is mixed. Some offices are genuinely great, others are political and demoralizing. Compensation is market rate but not exceptional for the hours demanded.",
    "Interesting and complex work in {dept}. The brand is strong and learning is real. However the model is unsustainable for most people past the manager level. Lifestyle significantly improves at senior levels.",
]
NEG_REVIEWS = [
    "Brutal work-life balance — consistent 80+ hour weeks with no acknowledgment from leadership. The up-or-out culture creates toxic undercurrents in the {dept} team. HR exists to protect the firm, not employees.",
    "Glorified PowerPoint factory. The staffing model burns through analysts in 18 months. Smart people are fleeing to tech and PE — leadership appears willfully unaware of the retention crisis in {dept}.",
    "Do not believe the culture messaging. The {dept} environment is intensely political and promotions are driven by who you know, not what you deliver. Diversity is marketing, not practice.",
    "Travel demands are unsustainable — four nights a week, every week, with no flexibility. The {dept} clients are often unreasonable and the firm consistently over-promises scope to win work. Morale is low.",
    "Consistent top-down fear culture. Partners routinely disrespect junior staff and HR is ineffective. The exit package when pushed out is insulting given the sacrifice demanded. Would not recommend to anyone.",
    "Extreme pressure with very little support structure. The {dept} team has seen 40% attrition in the past year. Compensation does not compensate for the lifestyle sacrifice. Opaque promotion process with moving goalposts.",
]
PORTAL_REVIEWS = [
    "Our engagement with {firm} on a complex {dept} transformation delivered measurable results within the first quarter. The team combined deep analytical rigor with genuine partnership — they were invested in our success, not just the deliverable.",
    "{firm}'s {dept} practice brought intellectual depth and executional precision to our most critical strategic challenge. The partner-level engagement was consistent throughout, and the frameworks they delivered are still driving decisions today.",
    "We have engaged {firm} across three separate {dept} engagements over five years. Each time, the quality of thinking and the caliber of the team has been exceptional. A true thought partner at every level of our organization.",
    "What distinguished {firm} from competitors was the quality of listening before the quality of output. Their {dept} team understood our context deeply before building recommendations. The ROI has been significant and sustained.",
    "{firm} helped us navigate a genuinely difficult {dept} transformation with rigor, empathy, and clear communication at every stage. Their ability to align stakeholders across our organization was remarkable. Highly recommended.",
]

PROS_POOL = ["Smart colleagues","Elite brand","Exit opportunities","Global exposure",
             "Strong training","High compensation","Interesting projects","Alumni network",
             "Meritocratic culture","Client caliber","Intellectual rigor","Travel benefits"]
CONS_POOL = ["Work-life balance","Up-or-out pressure","Long hours","Heavy travel",
             "Inconsistent management","Opaque promotions","Political culture","High attrition",
             "Limited autonomy","Siloed teams","Unpaid overtime","High stress"]


# ─── Helper Functions ──────────────────────────────────────────────────────────

def _make_date(start_yr=2019, end_yr=2024):
    start = datetime(start_yr, 1, 1)
    d     = start + timedelta(days=random.randint(0, (datetime(end_yr,12,31)-start).days))
    return d

def _rating_from_sentiment(sentiment, base_avg):
    delta = {"positive": random.uniform(0.3,0.9),
             "neutral":  random.uniform(-0.4,0.4),
             "negative": random.uniform(-1.5,-0.4)}[sentiment]
    return round(min(max(base_avg + delta, 1.0), 5.0), 1)

def _sub_ratings(overall):
    return {f"rating_{dim.lower().replace(' ','_').replace('&','and')}":
            round(min(max(overall + random.uniform(-0.9,0.9), 1.0), 5.0), 1)
            for dim in RATING_DIMS}


# ─── Employee Review Generator ─────────────────────────────────────────────────

def generate_employee_reviews(n_per_firm=500):
    print("  Generating employee reviews (Glassdoor / AmbitionBox / Blind / Indeed)...")
    records = []
    for firm, p in FIRMS.items():
        avg = p["glassdoor_avg"]
        # Sentiment weights: better firms have more positive reviews
        pos_w = 0.15 + (avg - 3.5) * 0.18
        neg_w = 0.30 - (avg - 3.5) * 0.12
        neu_w = 1 - pos_w - neg_w
        sentiments = random.choices(
            ["positive","neutral","negative"],
            weights=[max(pos_w,0.10), max(neu_w,0.20), max(neg_w,0.08)],
            k=n_per_firm)

        for i, sent in enumerate(sentiments):
            dept = random.choice(DEPARTMENTS)
            date = _make_date()

            # COVID effect: 2020–2021 reviews skew more negative
            if date.year in (2020,2021) and date.month >= 3:
                if sent == "positive" and random.random() < 0.35:
                    sent = "neutral"

            # Pick template based on sentiment
            pool = {"positive":POS_REVIEWS,"neutral":NEU_REVIEWS,"negative":NEG_REVIEWS}[sent]
            text = random.choice(pool).format(dept=dept, wl_pos=random.choice([
                "Work-life balance is manageable if expectations are set early.",
                "Remote policy has improved post-COVID.",
                "Travel is heavy but the exposure is worth it at junior levels."]))

            rating = _rating_from_sentiment(sent, avg)
            subs   = _sub_ratings(rating)

            records.append({
                "review_id":         f"EMP-{firm[:3].upper()}-{i+1:05d}",
                "source":            "employee",
                "platform":          random.choice(PLATFORMS),
                "firm":              firm,
                "tier":              p["tier"],
                "date":              date.strftime("%Y-%m-%d"),
                "year":              date.year,
                "month":             date.month,
                "quarter":           (date.month-1)//3+1,
                "season":            SEASONS[date.month],
                "reviewer_role":     random.choice(ROLES),
                "department":        dept,
                "location":          random.choice(LOCATIONS),
                "employment_status": random.choice(EMP_STATUS),
                "years_at_firm":     round(random.uniform(0.5,12),1),
                "overall_rating":    rating,
                "review_title":      fake.sentence(nb_words=6).rstrip("."),
                "review_text":       text,
                "pros":              " | ".join(random.sample(PROS_POOL, k=random.randint(2,4))),
                "cons":              " | ".join(random.sample(CONS_POOL, k=random.randint(2,4))),
                "helpful_count":     random.randint(0,350),
                "true_sentiment":    sent,
                **subs,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ─── Portal Review Generator ───────────────────────────────────────────────────

def generate_portal_reviews(n_per_firm=120):
    print("  Generating firm portal / client testimonial reviews...")
    records = []
    for firm, p in FIRMS.items():
        for i in range(n_per_firm):
            dept = random.choice(DEPARTMENTS)
            date = _make_date(2020, 2024)
            text = random.choice(PORTAL_REVIEWS).format(firm=firm, dept=dept)

            # Portal reviews are ALWAYS high — firms only publish the best ones
            portal_rating = round(min(5.0, 4.1 + random.uniform(0, 0.9)*p["portal_bias"]), 1)

            records.append({
                "review_id":       f"PORT-{firm[:3].upper()}-{i+1:04d}",
                "source":          "portal",
                "platform":        f"{firm} Official Portal",
                "firm":            firm,
                "tier":            p["tier"],
                "date":            date.strftime("%Y-%m-%d"),
                "year":            date.year,
                "month":           date.month,
                "quarter":         (date.month-1)//3+1,
                "engagement_type": dept,
                "portal_rating":   portal_rating,
                "review_title":    fake.sentence(nb_words=5).rstrip("."),
                "review_text":     text,
                "true_sentiment":  "positive",
                "verified":        random.random() < 0.65,
                "client_title":    random.choice(["CSO","CFO","COO","VP Strategy","Head of Transformation","CTO"]),
                "client_industry": random.choice(["Banking","Healthcare","Manufacturing","Retail","Energy","Technology","Insurance"]),
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data/raw", exist_ok=True)

    emp    = generate_employee_reviews(500)
    portal = generate_portal_reviews(120)

    emp.to_csv("data/raw/employee_reviews_raw.csv", index=False)
    portal.to_csv("data/raw/portal_reviews_raw.csv", index=False)

    print(f"\n  ✓ {len(emp):,} employee reviews → data/raw/employee_reviews_raw.csv")
    print(f"  ✓ {len(portal):,} portal reviews  → data/raw/portal_reviews_raw.csv")
    print(f"\n  Firms: {emp['firm'].nunique()} | Platforms: {emp['platform'].nunique()} | Date range: {emp['date'].min().date()} → {emp['date'].max().date()}")
    print("\n  Per-firm average ratings:")
    print(emp.groupby("firm")["overall_rating"].mean().round(2).to_string())
    return emp, portal

if __name__ == "__main__":
    main()

"""
═══════════════════════════════════════════════════════════════════════════════
STEP 4 — MASTER PIPELINE RUNNER
═══════════════════════════════════════════════════════════════════════════════

Run this ONE file to execute the entire pipeline end-to-end:
  Step 1 → Data Collection (generate_reviews.py)
  Step 2 → NLP Processing  (nlp_pipeline.py)
  Step 3 → Time-Series     (timeseries_analysis.py)

Usage:
  python src/pipeline/run_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time
from pathlib import Path

def banner(msg, char="═", width=62):
    print(f"\n{char*width}")
    print(f"  {msg}")
    print(f"{char*width}")

def run():
    t0 = time.time()
    # Always run from project root
    root = Path(__file__).resolve().parent.parent.parent
    os.chdir(root)
    sys.path.insert(0, str(root))

    banner("🔍 AI REVIEW & SENTIMENT INTELLIGENCE PLATFORM")
    print("  Top 10 Consulting Firms | Employee vs Portal Analysis")
    print("  Steps: Data Collection → NLP → Time-Series → Ready for Dashboard")

    banner("STEP 1 / 3 — DATA COLLECTION")
    from src.ingestion.generate_reviews import main as gen
    gen()

    banner("STEP 2 / 3 — NLP PIPELINE")
    from src.nlp.nlp_pipeline import run_pipeline
    run_pipeline()

    banner("STEP 3 / 3 — TIME-SERIES ANALYSIS")
    from src.timeseries.timeseries_analysis import run_timeseries
    run_timeseries()

    banner(f"✅ PIPELINE COMPLETE — {time.time()-t0:.1f}s", char="─")
    print("\n  All outputs in data/processed/")
    print("  Launch dashboard:  streamlit run dashboard/app.py\n")

if __name__ == "__main__":
    run()

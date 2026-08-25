# NBA Contract Value Analysis

## Overview

This project scrapes real NBA player statistics and salary data, trains ML models to predict what players *should* earn based on performance, then identifies underpaid stars ("High ROI") and overpaid contracts ("Cap Clogs") to generate data-driven trade recommendations.

**Pipeline:** Scrape Stats → Clean & Merge Salaries → Train Model → Categorize Players → Generate Trade Proposals

---

## Files

| File | Purpose |
|------|---------|
| `nba_scraper_2020_2025.ipynb` | Jupyter notebook that scrapes per-game stats from Basketball Reference (2020–2025 seasons). Outputs raw stats with fallback salary estimation. |
| `clean_and_combine_salaries.py` | Cleans real salary data from two sources (Kaggle + Basketball Reference), scrapes 2025-26 stats, merges everything into the final analysis-ready CSV. |
| `predict_salary.py` | CLI tool to train and evaluate salary prediction models (Random Forest, XGBoost, Ridge, Lasso, Gradient Boost). Reports MAE, R², feature importances. |
| `trade_analysis.py` | Trains on historical salary patterns, predicts fair market value for current players, categorizes them as High ROI / Cap Clog, and generates a mock trade recommendation for a specified team. |
| `three_pt_premium.py` | Analyzes the salary premium paid to 3-point shooters vs efficient interior scorers. Adjustable thresholds for 3PM, 2PA, 2P%, and min games. Shows year-over-year trends, top earners, and best value players in each group. |
| `ui/app.py` | Unified Streamlit dashboard with a home page and navigation to all analyses: Salary Predictions, Trade Recommendations, and 3-Point Premium. |
| `ui/predictions.py` | Streamlit web UI for the salary prediction model. Provides interactive model/year selection and displays top 10 tables for highest predicted, most overpaid, and most underpaid players. |
| `ui/trade_recommendation.py` | Streamlit page for team-level ROI and cap-clog analysis with trade proposals, financial impact, and business-case output. |
| `ui/three_pt_premium.py` | Streamlit page for 3-point premium analysis with threshold controls, trend tables, and top-earner lists by scorer archetype. |
| `requirements.txt` | Python dependencies (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, plotly, torch, streamlit). |
| `nba_stats_and_salaries_2020_2025.csv` | Final merged dataset: 1,612 player-seasons with real stats + real salaries (Years 2023–2026). |
| `real_salaries_combined.csv` | Intermediate file: cleaned salary data from both sources before merging with stats. |
| `kaggle_old_salaries.csv` | Source: Kaggle NBA salary dataset covering 2022-23 through 2024-25 seasons. |
| `br_salaries_2025.csv` | Source: Basketball Reference contract page for the 2025-26 season. |

---

## Key Changes & Improvements

### Data Cleaning
- **Salary parsing**: Stripped `$`, commas, quotes, and trailing whitespace from salary strings; converted `$0` entries to NaN (player not under contract that year).
- **Name normalization**: Unicode decomposition (e.g., `Jokić` → `Jokic`, `Dončić` → `Doncic`) for cross-dataset matching — achieved 98-99% match rates.
- **Duplicate column fix**: The Kaggle CSV had a mislabeled duplicate `2024/2025` column that was actually `2025/2026` — identified via value matching against the BR file.
- **TOT row handling**: Players traded mid-season appear multiple times on Basketball Reference; kept only the `TOT` (total) row and removed partial-team duplicates.
- **Header cleanup**: BR salary CSV had a multi-level header that caused column name mangling — fixed with `skiprows` and manual column assignment.

### Model Improvements
- **Temporal train/test split**: Originally the model trained on ALL data and predicted the same data (R² ≈ 1.0, no meaningful categorization). Fixed by training on *past* seasons and predicting for the *current* year — reveals true market inefficiencies.
- **Real salary data**: Replaced formula-generated salary estimates with actual NBA contract data from Kaggle (2023–2025) and Basketball Reference (2026), eliminating the circular dependency between features and target.

### Analysis
- **Added 2025-26 season**: Scraped current season stats (734 players) to pair with the most complete salary source (490 players from BR), giving better coverage for the target prediction year.
- **Final dataset**: 1,612 player-seasons with verified salary data across 4 seasons, enabling robust train/predict splits.
- **3PT premium study**: Identified a consistent ~22% salary premium for 3-point shooters over interior scorers despite interior players having higher eFG% — suggests market overvaluation of perimeter shooting.

---

## Streamlit UI

The project includes both a unified dashboard and standalone pages.

### Recommended: Unified Dashboard (Home + 3 Analysis Pages)

```bash
python -m streamlit run ui/app.py
```

The unified app opens on a **Home** page and lets you choose between 3 analysis options:

1. **Salary Predictions**
  - Model selector (Random Forest, Ridge, Lasso, Gradient Boost, XGBoost)
  - Top 10 highest predicted salaries
  - Top 10 most overpaid
  - Top 10 most underpaid

2. **Trade Recommendations**
  - Parser-equivalent controls in UI (data file, year, team)
  - Season category summary (High ROI / Fair Value / Cap Clog)
  - Top High ROI and Top Cap Clog tables
  - Team salary overview, cap-clog list, proposed trade, financial impact, business case

3. **3-Point Premium**
  - Threshold controls (min 3PM, min 2PA, min 2P%, min games)
  - Group salary comparison metrics
  - Year-over-year premium trend table
  - Top 10 paid players by scorer type

### Standalone Pages (Still Available)

```bash
python -m streamlit run ui/predictions.py
python -m streamlit run ui/trade_recommendation.py
python -m streamlit run ui/three_pt_premium.py
```

# NBA Contract Value Analysis

A data-driven NBA analytics platform that scrapes real player statistics and salary data, trains ML models to predict fair market value, and surfaces actionable insights through an interactive Streamlit dashboard.

**Pipeline:** Scrape Stats → Clean & Merge Salaries → Train Model → Categorize Players → Generate Trade Proposals

---

## Dashboard Pages

### 🏠 Home

The landing page provides an overview of the project and quick navigation to every analysis module. Includes a summary of data sources, technologies used, and key insights.

![Home Page](images/home.png)

---

### 💰 Salary Predictions

Trains ML models (Random Forest, Ridge, Lasso, Gradient Boost, XGBoost) on past seasons (2023–2025) and predicts what every player *should* earn in 2026 based on performance. Surfaces the top overpaid and underpaid players in the league with a 7% annual salary-cap adjustment.

![Salary Predictions](images/salary_predictions.png)

---

### 🎯 Trade Recommendations

Team-level analysis that categorizes every player as High ROI, Fair Value, or Cap Clog. Select a team (or let the app pick the worst offender) to see its salary breakdown, biggest liabilities, a mock trade proposal, financial impact, and a ready-to-present business case.

![Trade Recommendations](images/trade_recommendations.png)

---

### ⭐ 3-Point Premium

Investigates whether NBA teams overpay 3-point specialists relative to efficient interior scorers. Adjustable thresholds for 3PM, 2PA, 2P%, and minimum games let you explore the ~22% salary premium across seasons, top earners by archetype, and best-value players in each group.

![3-Point Premium](images/three_pt_premium.png)

---

### 🤖 GM Advisor

A chatbot interface where you can discuss trades, signings, cap strategy, and long-term roster plans. Attach outputs from the other pages as context and export a downloadable business plan. Currently runs on a mock LLM client — drop in an OpenAI key to go live.

![GM Advisor](images/gm_advisor.png)

---

### 🎬 Play Recognition 🚧

*In the works.* A vision-model pipeline to identify NBA plays from 1–5 still frames. Today the page lets you view and edit the canonical play catalog (`data/plays.md`); the training and prediction backends are stubbed and ready for a future classifier.

![Play Recognition](images/play_recognition.png)

---

## Quick Start

```bash
pip install -r requirements.txt
python -m streamlit run ui/app.py
```

## Data Sources

| File | Description |
|------|-------------|
| `nba_stats_and_salaries_2020_2026.csv` | 1,612 player-seasons with real stats + real salaries (2023–2026) |
| `kaggle_old_salaries.csv` | Kaggle NBA salary dataset (2022-23 through 2024-25) |
| `br_salaries_2025.csv` | Basketball Reference contracts for the 2025-26 season |

## Technologies

- **Modeling**: Scikit-Learn, XGBoost
- **Data**: Pandas, NumPy
- **Visualization**: Streamlit, Plotly, Matplotlib, Seaborn
- **Deep Learning** *(planned)*: PyTorch (play recognition)

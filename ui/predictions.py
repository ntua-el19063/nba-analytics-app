"""
NBA Salary Predictions — Streamlit UI
Run with: streamlit run ui/predictions.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from src.predict_salary import get_model, load_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="NBA Salary Predictions", layout="wide")
st.title("NBA Salary Predictions")
st.caption("Temporal model: trains on 2023–2025, predicts all 2026 players (7% annual cap adjustment)")

# --- Input: Model selection only ---
model_name = st.selectbox(
    "Model",
    options=["random_forest", "ridge", "lasso", "gradient_boost", "xgboost"],
    index=0,
)

CAP_INCREASE = 0.07
PREDICT_YEAR = 2026

if st.button("Run Prediction"):
    with st.spinner("Training model..."):
        # Load CSV and prepare features/target.
        # load_data() converts Awards strings to recency-weighted numeric scores,
        # selects the 15 feature columns, fills NaN with 0, and drops invalid rows.
        X, y, full_df = load_data("data/nba_stats_and_salaries_2020_2026.csv")

        # Temporal split: train on past seasons, predict current
        train_mask = full_df['Year'] < PREDICT_YEAR
        test_mask = full_df['Year'] == PREDICT_YEAR

        X_train = X[train_mask].copy()
        y_train = y[train_mask].copy()
        X_test = X[test_mask].copy()
        y_test = y[test_mask].copy()

        # Adjust training salaries to 2026 cap dollars
        # A player earning $30M in 2023 → ~$30M * 1.07^3 = $36.8M in 2026 terms
        train_years = full_df.loc[train_mask, 'Year']
        cap_adjustment = (1 + CAP_INCREASE) ** (PREDICT_YEAR - train_years)
        y_train = y_train * cap_adjustment.values

        # Scale and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = get_model(model_name)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

    # Build results dataframe
    results = pd.DataFrame({
        'Player': full_df.loc[X_test.index, 'Player'].values,
        'Actual Salary': y_test.values,
        'Predicted Salary': y_pred.astype(int),
        # Positive Difference = underpaid, Negative = overpaid
        'Difference': (y_pred - y_test.values).astype(int),
    })

    # Show metrics
    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"${mae:,.0f}")
    m2.metric("R² Score", f"{r2:.4f}")
    m3.metric("Players predicted", len(results))

    # Display 3 tables
    st.subheader("Top 10 Highest Predicted Salaries")
    top_predicted = results.nlargest(10, 'Predicted Salary')[['Player', 'Predicted Salary', 'Actual Salary']].reset_index(drop=True)
    top_predicted.index += 1
    st.dataframe(top_predicted.style.format({'Predicted Salary': '${:,.0f}', 'Actual Salary': '${:,.0f}'}), use_container_width=True)

    st.subheader("Top 10 Most Overpaid (Actual > Predicted)")
    most_overpaid = results.nsmallest(10, 'Difference')[['Player', 'Actual Salary', 'Predicted Salary', 'Difference']].reset_index(drop=True)
    most_overpaid.index += 1
    st.dataframe(most_overpaid.style.format({'Actual Salary': '${:,.0f}', 'Predicted Salary': '${:,.0f}', 'Difference': '${:,.0f}'}), use_container_width=True)

    st.subheader("Top 10 Most Underpaid (Predicted > Actual)")
    most_underpaid = results.nlargest(10, 'Difference')[['Player', 'Actual Salary', 'Predicted Salary', 'Difference']].reset_index(drop=True)
    most_underpaid.index += 1
    st.dataframe(most_underpaid.style.format({'Actual Salary': '${:,.0f}', 'Predicted Salary': '${:,.0f}', 'Difference': '${:,.0f}'}), use_container_width=True)

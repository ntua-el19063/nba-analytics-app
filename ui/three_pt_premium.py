"""
3-Point Premium Analysis — Streamlit UI
Run with: streamlit run ui/three_pt_premium.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="3PT Premium Analysis", layout="wide")
st.title("3-Point Shooter Salary Premium")
st.caption("Do NBA teams overpay 3-point shooters relative to efficient interior scorers?")

# --- Inputs ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    min_3pm = st.number_input("Min 3PM per game", value=2.0, step=0.5, min_value=0.5)
with col2:
    min_2pa = st.number_input("Min 2PA per game (interior)", value=5.0, step=0.5, min_value=1.0)
with col3:
    min_2pct = st.number_input("Min 2P% (interior)", value=0.52, step=0.01, min_value=0.40, max_value=0.75)
with col4:
    min_games = st.number_input("Min games played", value=20, step=5, min_value=1)

DATA_PATH = "data/nba_stats_and_salaries_2020_2026.csv"

if st.button("Analyze"):
    df = pd.read_csv(DATA_PATH)
    df = df[df['G'] >= min_games].copy()

    # Categorize players
    three_pt = df[df['3P'] >= min_3pm].copy()
    three_pt['Scorer Type'] = '3PT Specialist'

    interior = df[
        (df['3P'] < min_3pm) &
        (df['2PA'] >= min_2pa) &
        (df['2P%'] >= min_2pct)
    ].copy()
    interior['Scorer Type'] = 'Interior Scorer'

    if len(three_pt) == 0 or len(interior) == 0:
        st.warning("One or both groups are empty. Adjust thresholds.")
    else:
        # Summary metrics
        avg_sal_3 = three_pt['Salary'].mean()
        avg_sal_i = interior['Salary'].mean()
        premium_pct = ((avg_sal_3 - avg_sal_i) / avg_sal_i) * 100

        st.subheader("Group Comparison")
        m1, m2, m3 = st.columns(3)
        m1.metric("3PT Specialist Avg Salary", f"${avg_sal_3:,.0f}", f"{len(three_pt)} players")
        m2.metric("Interior Scorer Avg Salary", f"${avg_sal_i:,.0f}", f"{len(interior)} players")
        m3.metric("3PT Salary Premium", f"{premium_pct:+.1f}%")

        # Detailed comparison table
        comparison = pd.DataFrame({
            'Metric': ['Avg Salary', 'Median Salary', 'Avg PPG', 'Avg MPG', 'Avg eFG%', 'Cost per PPG'],
            '3PT Specialists': [
                f"${avg_sal_3:,.0f}",
                f"${three_pt['Salary'].median():,.0f}",
                f"{three_pt['PTS'].mean():.1f}",
                f"{three_pt['MP'].mean():.1f}",
                f"{three_pt['eFG%'].mean():.3f}",
                f"${avg_sal_3 / three_pt['PTS'].mean():,.0f}",
            ],
            'Interior Scorers': [
                f"${avg_sal_i:,.0f}",
                f"${interior['Salary'].median():,.0f}",
                f"{interior['PTS'].mean():.1f}",
                f"{interior['MP'].mean():.1f}",
                f"{interior['eFG%'].mean():.3f}",
                f"${avg_sal_i / interior['PTS'].mean():,.0f}",
            ],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        # Year-over-year trend
        st.subheader("Year-over-Year Premium Trend")
        years = sorted(df['Year'].unique())
        trend_data = []
        for year in years:
            y3 = three_pt[three_pt['Year'] == year]
            yi = interior[interior['Year'] == year]
            if len(y3) > 0 and len(yi) > 0:
                avg3 = y3['Salary'].mean()
                avgi = yi['Salary'].mean()
                prem = ((avg3 - avgi) / avgi) * 100
                trend_data.append({'Year': int(year), '3PT Avg': avg3, 'Interior Avg': avgi, 'Premium %': prem, 'N (3PT)': len(y3), 'N (Int)': len(yi)})

        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            st.dataframe(trend_df.style.format({
                '3PT Avg': '${:,.0f}',
                'Interior Avg': '${:,.0f}',
                'Premium %': '{:+.1f}%',
            }), use_container_width=True, hide_index=True)

        # Top earners in each group
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Top 10 Highest Paid 3PT Specialists")
            top3 = three_pt.nlargest(10, 'Salary')[['Player', 'Year', 'Team', '3P', 'PTS', 'Salary']].reset_index(drop=True)
            top3.index += 1
            st.dataframe(top3.style.format({'Salary': '${:,.0f}'}), use_container_width=True)

        with col_b:
            st.subheader("Top 10 Highest Paid Interior Scorers")
            topi = interior.nlargest(10, 'Salary')[['Player', 'Year', 'Team', '2P%', 'PTS', 'Salary']].reset_index(drop=True)
            topi.index += 1
            st.dataframe(topi.style.format({'Salary': '${:,.0f}', '2P%': '{:.3f}'}), use_container_width=True)

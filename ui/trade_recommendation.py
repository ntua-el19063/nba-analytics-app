"""
NBA Trade Recommendation Analysis — Streamlit UI
Run with: streamlit run ui/trade_recommendation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from src.trade_analysis import load_and_train, categorize_players, generate_trade_recommendation

st.set_page_config(page_title="NBA Trade Recommendation", layout="wide")
st.title("🏀 NBA Trade Recommendation Analysis")
st.caption("Analyze player value using predictive salary modeling and identify optimal trade scenarios")

# =============================================================================
# SIDEBAR: INPUT CONTROLS
# =============================================================================
st.sidebar.header("⚙️ Analysis Parameters")

# Data file selection
data_file = st.sidebar.selectbox(
    "Data File",
    options=[
        "data/nba_stats_and_salaries_2020_2025.csv",
        "data/nba_stats_and_salaries_2020_2026.csv",
    ],
    index=0,
)

# Year selection
year = st.sidebar.number_input(
    "Season Year",
    min_value=2020,
    max_value=2030,
    value=2026,
    step=1,
)

# Team selection
team_options = [
    "AUTO (Worst Cap Clog)",
    "ATL", "BOS", "BRK", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
    "GSW", "HOU", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP",
    "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR",
    "UTA", "WAS",
]

team = st.sidebar.selectbox(
    "Team to Analyze",
    options=team_options,
    index=0,
)

# Run button
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# =============================================================================
# MAIN ANALYSIS
# =============================================================================
if run_analysis:
    with st.spinner("Loading data and training model..."):
        df_all, model = load_and_train(data_file, target_year=year)
    
    with st.spinner("Categorizing players..."):
        season_df = categorize_players(df_all, year=year)
    
    # Summary statistics
    st.header("📊 Season Summary")
    counts = season_df['Category'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(season_df))
    with col2:
        st.metric("High ROI", counts.get('High ROI', 0))
    with col3:
        st.metric("Fair Value", counts.get('Fair Value', 0))
    with col4:
        st.metric("Cap Clogs", counts.get('Cap Clog', 0))
    
    # =============================================================================
    # TOP HIGH ROI PLAYERS
    # =============================================================================
    st.header("⭐ Top High ROI Players (Best Value)")
    top_roi = season_df[season_df['Category'] == 'High ROI'].nlargest(10, 'Salary_Diff').copy()
    
    if not top_roi.empty:
        top_roi_display = top_roi[['Player', 'Team', 'PTS', 'Salary', 'Predicted_Salary', 'Salary_Diff']].copy()
        top_roi_display.columns = ['Player', 'Team', 'PPG', 'Actual Salary', 'Fair Value', 'Surplus Value']
        top_roi_display = top_roi_display.reset_index(drop=True)
        top_roi_display.index += 1
        
        st.dataframe(
            top_roi_display.style.format({
                'PPG': '{:.1f}',
                'Actual Salary': '${:,.0f}',
                'Fair Value': '${:,.0f}',
                'Surplus Value': '${:,.0f}',
            }),
            use_container_width=True
        )
    else:
        st.info("No High ROI players found for this season.")
    
    # =============================================================================
    # TOP CAP CLOGS
    # =============================================================================
    st.header("⚠️ Top Cap Clogs (Most Overpaid)")
    top_clogs = season_df[season_df['Category'] == 'Cap Clog'].nsmallest(10, 'Salary_Diff').copy()
    
    if not top_clogs.empty:
        top_clogs_display = top_clogs[['Player', 'Team', 'PTS', 'Salary', 'Predicted_Salary', 'Salary_Diff']].copy()
        top_clogs_display.columns = ['Player', 'Team', 'PPG', 'Actual Salary', 'Fair Value', 'Overpay']
        top_clogs_display = top_clogs_display.reset_index(drop=True)
        top_clogs_display.index += 1
        
        st.dataframe(
            top_clogs_display.style.format({
                'PPG': '{:.1f}',
                'Actual Salary': '${:,.0f}',
                'Fair Value': '${:,.0f}',
                'Overpay': '${:,.0f}',
            }),
            use_container_width=True
        )
    else:
        st.info("No Cap Clogs found for this season.")
    
    # =============================================================================
    # TEAM-SPECIFIC ANALYSIS
    # =============================================================================
    st.header("🎯 Trade Recommendation")
    
    # Determine which team to analyze
    if team == "AUTO (Worst Cap Clog)":
        worst_clog = season_df[season_df['Category'] == 'Cap Clog'].nsmallest(1, 'Salary_Diff')
        if not worst_clog.empty:
            selected_team = worst_clog.iloc[0]['Team']
            st.info(f"📍 Auto-selected team with biggest cap clog: **{selected_team}**")
        else:
            st.warning("No teams with cap clogs found.")
            selected_team = None
    else:
        selected_team = team
    
    if selected_team:
        team_players = season_df[season_df['Team'] == selected_team].copy()
        
        if team_players.empty:
            st.error(f"No players found for team '{selected_team}'")
            st.info(f"Available teams: {', '.join(sorted(season_df['Team'].unique()))}")
        else:
            # --- Team Overview ---
            st.subheader(f"Team Salary Overview — {selected_team}")
            team_summary = team_players[['Player', 'PTS', 'TRB', 'AST', 'MP', 'Salary', 'Predicted_Salary', 'Category']].copy()
            team_summary.columns = ['Player', 'PPG', 'RPG', 'APG', 'MPG', 'Actual Salary', 'Fair Value', 'Category']
            team_summary = team_summary.sort_values('PPG', ascending=False)
            team_summary = team_summary.reset_index(drop=True)
            team_summary.index += 1
            
            st.dataframe(
                team_summary.style.format({
                    'PPG': '{:.1f}',
                    'RPG': '{:.1f}',
                    'APG': '{:.1f}',
                    'MPG': '{:.1f}',
                    'Actual Salary': '${:,.0f}',
                    'Fair Value': '${:,.0f}',
                }),
                use_container_width=True
            )
            
            # --- Cap Clogs on this team ---
            cap_clogs = team_players[team_players['Category'] == 'Cap Clog'].sort_values('Salary_Diff')
            
            if not cap_clogs.empty:
                st.subheader(f"Cap Clogs on {selected_team} (Overpaid Players)")
                for idx, (_, player) in enumerate(cap_clogs.iterrows(), 1):
                    overpay = player['Salary'] - player['Predicted_Salary']
                    with st.expander(f"{idx}. {player['Player']} — Overpaid by ${overpay:,.0f}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Actual Salary", f"${player['Salary']:,.0f}")
                        with col2:
                            st.metric("Fair Value", f"${player['Predicted_Salary']:,.0f}")
                        with col3:
                            st.metric("Overpay", f"${overpay:,.0f}")
                        
                        st.caption(f"{player['PTS']:.1f} PTS / {player['TRB']:.1f} REB / {player['AST']:.1f} AST in {player['MP']:.1f} MPG")
                
                # --- Proposed Trade ---
                st.subheader(f"Proposed Trade for {selected_team}")
                
                trade_away = cap_clogs.iloc[0]
                other_teams = season_df[season_df['Team'] != selected_team]
                high_roi = other_teams[other_teams['Category'] == 'High ROI'].sort_values('Salary_Diff', ascending=False)
                
                # Find best pair
                target_salary = trade_away['Salary']
                best_pair = None
                best_production_gain = -float('inf')
                
                for i in range(min(20, len(high_roi))):
                    for j in range(i + 1, min(20, len(high_roi))):
                        p1 = high_roi.iloc[i]
                        p2 = high_roi.iloc[j]
                        combined_salary = p1['Salary'] + p2['Salary']
                        if 0.8 * target_salary <= combined_salary <= 1.1 * target_salary:
                            combined_production = p1['Predicted_Salary'] + p2['Predicted_Salary']
                            production_gain = combined_production - trade_away['Predicted_Salary']
                            if production_gain > best_production_gain:
                                best_production_gain = production_gain
                                best_pair = (p1, p2)
                
                if best_pair is None and len(high_roi) >= 2:
                    best_pair = (high_roi.iloc[0], high_roi.iloc[1])
                
                if best_pair:
                    p1, p2 = best_pair
                    
                    # Trade details
                    trade_col1, trade_col2 = st.columns(2)
                    
                    with trade_col1:
                        st.markdown(f"**{selected_team} SENDS:**")
                        st.markdown(f"❌ **{trade_away['Player']}**")
                        st.caption(f"Salary: ${trade_away['Salary']:,.0f}")
                        st.caption(f"Production Value: ${trade_away['Predicted_Salary']:,.0f}")
                        st.caption(f"{trade_away['PTS']:.1f} PTS / {trade_away['TRB']:.1f} REB / {trade_away['AST']:.1f} AST")
                    
                    with trade_col2:
                        st.markdown(f"**{selected_team} RECEIVES:**")
                        st.markdown(f"✓ **{p1['Player']}** ({p1['Team']})")
                        st.caption(f"Salary: ${p1['Salary']:,.0f} | Value: ${p1['Predicted_Salary']:,.0f}")
                        st.caption(f"{p1['PTS']:.1f} PTS / {p1['TRB']:.1f} REB / {p1['AST']:.1f} AST")
                        
                        st.markdown(f"✓ **{p2['Player']}** ({p2['Team']})")
                        st.caption(f"Salary: ${p2['Salary']:,.0f} | Value: ${p2['Predicted_Salary']:,.0f}")
                        st.caption(f"{p2['PTS']:.1f} PTS / {p2['TRB']:.1f} REB / {p2['AST']:.1f} AST")
                    
                    # Financial Impact
                    st.subheader("💰 Financial Impact")
                    incoming = p1['Salary'] + p2['Salary']
                    outgoing = trade_away['Salary']
                    cap_saved = outgoing - incoming
                    production_before = trade_away['Predicted_Salary']
                    production_after = p1['Predicted_Salary'] + p2['Predicted_Salary']
                    
                    impact_col1, impact_col2 = st.columns(2)
                    with impact_col1:
                        st.metric("Salary Going Out", f"${outgoing:,.0f}")
                        st.metric("Salary Coming In", f"${incoming:,.0f}")
                        st.metric("Cap Space Saved", f"${cap_saved:,.0f}", delta=f"${cap_saved:,.0f}")
                    
                    with impact_col2:
                        st.metric("Production Traded", f"${production_before:,.0f}")
                        st.metric("Production Acquired", f"${production_after:,.0f}")
                        st.metric("Net Production Gain", f"${production_after - production_before:,.0f}", 
                                 delta=f"${production_after - production_before:,.0f}")
                    
                    # Business Case
                    st.subheader("📋 Business Case")
                    st.markdown(f"""
**Why this trade works under the NBA Salary Cap:**

1. **CAP RELIEF**: By trading {trade_away['Player']} (${outgoing:,.0f}), {selected_team} frees 
   ${cap_saved:,.0f} in cap space — money that can be used to retain key players 
   or sign free agents next offseason.

2. **PRODUCTION UPGRADE**: The two incoming players ({p1['Player']} + {p2['Player']}) 
   combine for {p1['PTS'] + p2['PTS']:.1f} PPG vs {trade_away['PTS']:.1f} PPG from the outgoing player.
   Their combined production value (${production_after:,.0f}) exceeds the single 
   player's value (${production_before:,.0f}) by ${production_after - production_before:,.0f}.

3. **DEPTH > STAR POWER**: Two reliable role players provide more consistent 
   nightly output, injury insurance, and lineup flexibility compared to 
   one overpaid underperformer.

4. **ASSET FLEXIBILITY**: If {selected_team} later needs to make another move, two 
   moderate contracts are much easier to trade than one bloated deal. 
   Teams around the league actively seek affordable, productive players.

**VERDICT**: This trade transforms a negative-value contract into two positive-
value assets while improving both on-court production AND financial flexibility.
                    """)
                else:
                    st.warning(f"Could not find suitable trade partner players with complementary salaries.")
            else:
                st.success(f"✓ {selected_team} has no significant Cap Clogs — their contracts are efficient!")

else:
    st.info("👈 **Adjust the parameters on the left and click 'Run Analysis' to begin**")

"""
NBA ROI & Trade Recommendation Analysis
========================================
This script:
1. Uses the prediction model to estimate what each player SHOULD earn
2. Compares predicted salary vs actual salary to find:
   - "High ROI" players (underpaid — produce more than they cost)
   - "Cap Clogs" (overpaid — cost more than their production merits)
3. Generates a Mock Trade Recommendation for a specific team

Usage:
    python trade_analysis.py                  # Analyze all teams, pick best trade
    python trade_analysis.py --team LAL       # Focus on Lakers
    python trade_analysis.py --team BKN       # Focus on Nets
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# =============================================================================
# 1. LOAD DATA & TRAIN MODEL
# =============================================================================

def load_and_train(csv_path="data/nba_stats_and_salaries_2020_2025.csv", target_year=2026):
    """
    Load data, train model on PAST seasons, predict fair salary for target year.
    
    Key insight: Training on historical data and predicting for the current year
    reveals which players are over/underpaid relative to what the market normally
    pays for their stat profile. Young stars on rookie deals show as High ROI,
    aging veterans on legacy contracts show as Cap Clogs.
    """
    df = pd.read_csv(csv_path)

    feature_cols = ['Age', 'G', 'GS', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK',
                    'TOV', 'FG%', '3P%', 'FT%', 'eFG%']
    available = [c for c in feature_cols if c in df.columns]

    # Split: train on past seasons, predict for target year
    train_df = df[df['Year'] < target_year].copy()
    target_df = df[df['Year'] == target_year].copy()

    X_train = train_df[available].fillna(0)
    y_train = train_df['Salary']
    valid_train = y_train > 0
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]

    X_target = target_df[available].fillna(0)

    # Train model on historical salary patterns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_target_scaled = scaler.transform(X_target)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Predict "fair salary" for target year players based on historical patterns
    target_df = target_df.copy()
    target_df['Predicted_Salary'] = model.predict(X_target_scaled).astype(int)
    target_df['Salary_Diff'] = target_df['Predicted_Salary'] - target_df['Salary']
    # Positive diff = underpaid (they produce more than they're paid)
    # Negative diff = overpaid (they cost more than their production warrants)

    return target_df, model


# =============================================================================
# 2. CATEGORIZE PLAYERS: HIGH ROI vs CAP CLOGS
# =============================================================================

def categorize_players(df, year=2026):
    """
    Categorize players from a specific season.
    
    High ROI:  Players whose predicted salary is significantly ABOVE their actual salary.
               They produce like a $20M player but only cost $8M → great value.
    
    Cap Clog:  Players whose predicted salary is significantly BELOW their actual salary.
               They produce like a $10M player but cost $30M → overpaid.
    """
    season = df[df['Year'] == year].copy()

    # Calculate value ratio: predicted / actual
    # > 1.0 means underpaid (good value), < 1.0 means overpaid
    season['Value_Ratio'] = season['Predicted_Salary'] / season['Salary']

    # Categorize
    # High ROI: predicted salary > 1.5x their actual pay (getting 50%+ more production than paid for)
    # Cap Clog: predicted salary < 0.7x their actual pay (paying 30%+ more than their production merits)
    season['Category'] = 'Fair Value'
    season.loc[season['Value_Ratio'] >= 1.5, 'Category'] = 'High ROI'
    season.loc[season['Value_Ratio'] <= 0.7, 'Category'] = 'Cap Clog'

    return season


# =============================================================================
# 3. GENERATE MOCK TRADE RECOMMENDATION
# =============================================================================

def generate_trade_recommendation(season_df, team_code):
    """
    Generate a trade recommendation for a specific team.
    Strategy: Trade away the biggest Cap Clog, acquire two High ROI role players
    that improve both production and cap flexibility.
    """
    team_players = season_df[season_df['Team'] == team_code].copy()

    if team_players.empty:
        print(f"No players found for team '{team_code}'")
        print(f"Available teams: {sorted(season_df['Team'].unique())}")
        return

    # Find Cap Clogs on this team (sorted by most overpaid)
    cap_clogs = team_players[team_players['Category'] == 'Cap Clog'].sort_values('Salary_Diff')
    # Find High ROI players on OTHER teams (best values available)
    other_teams = season_df[season_df['Team'] != team_code]
    high_roi = other_teams[other_teams['Category'] == 'High ROI'].sort_values('Salary_Diff', ascending=False)

    if cap_clogs.empty:
        print(f"\n{team_code} has no significant Cap Clogs — this team's contracts are efficient!")
        print("\nTeam Salary Overview:")
        print(team_players[['Player', 'Salary', 'Predicted_Salary', 'Value_Ratio', 'Category']]
              .sort_values('Salary', ascending=False).head(10).to_string(index=False))
        return

    # Pick the worst Cap Clog to trade away
    trade_away = cap_clogs.iloc[0]

    # Find two High ROI players whose combined salary ≈ the Cap Clog's salary
    target_salary = trade_away['Salary']
    best_pair = None
    best_production_gain = -float('inf')

    for i in range(min(20, len(high_roi))):
        for j in range(i + 1, min(20, len(high_roi))):
            p1 = high_roi.iloc[i]
            p2 = high_roi.iloc[j]
            combined_salary = p1['Salary'] + p2['Salary']
            # Combined salary should be within 80-110% of the traded player
            if 0.8 * target_salary <= combined_salary <= 1.1 * target_salary:
                combined_production = p1['Predicted_Salary'] + p2['Predicted_Salary']
                production_gain = combined_production - trade_away['Predicted_Salary']
                if production_gain > best_production_gain:
                    best_production_gain = production_gain
                    best_pair = (p1, p2)

    # If no perfect salary match, just pick the two best available
    if best_pair is None and len(high_roi) >= 2:
        best_pair = (high_roi.iloc[0], high_roi.iloc[1])

    # =========================================================================
    # PRINT THE REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"  MOCK TRADE RECOMMENDATION — {team_code}")
    print("=" * 70)

    # --- Team Overview ---
    print(f"\n{'─' * 70}")
    print(f"  TEAM SALARY OVERVIEW ({team_code})")
    print(f"{'─' * 70}")
    team_summary = team_players[['Player', 'PTS', 'Salary', 'Predicted_Salary', 'Category']].copy()
    team_summary['Salary'] = team_summary['Salary'].apply(lambda x: f"${x:,.0f}")
    team_summary['Predicted_Salary'] = team_summary['Predicted_Salary'].apply(lambda x: f"${x:,.0f}")
    team_summary = team_summary.sort_values('PTS', ascending=False)
    print(team_summary.head(10).to_string(index=False))

    # --- Cap Clogs ---
    print(f"\n{'─' * 70}")
    print(f"  CAP CLOGS ON {team_code} (Overpaid Players)")
    print(f"{'─' * 70}")
    for _, player in cap_clogs.iterrows():
        overpay = player['Salary'] - player['Predicted_Salary']
        print(f"  • {player['Player']}")
        print(f"    Actual Salary:    ${player['Salary']:>12,.0f}")
        print(f"    Fair Value:       ${player['Predicted_Salary']:>12,.0f}")
        print(f"    OVERPAID BY:      ${overpay:>12,.0f}")
        print(f"    Stats: {player['PTS']:.1f} PTS / {player['TRB']:.1f} REB / {player['AST']:.1f} AST in {player['MP']:.1f} MPG")
        print()

    # --- Proposed Trade ---
    if best_pair:
        p1, p2 = best_pair
        print(f"{'─' * 70}")
        print(f"  PROPOSED TRADE")
        print(f"{'─' * 70}")
        print(f"\n  {team_code} SENDS:")
        print(f"    ❌ {trade_away['Player']}")
        print(f"       Salary: ${trade_away['Salary']:,.0f}  |  Production Value: ${trade_away['Predicted_Salary']:,.0f}")
        print(f"       Stats: {trade_away['PTS']:.1f} PTS / {trade_away['TRB']:.1f} REB / {trade_away['AST']:.1f} AST")

        print(f"\n  {team_code} RECEIVES:")
        print(f"    ✓ {p1['Player']} ({p1['Team']})")
        print(f"       Salary: ${p1['Salary']:,.0f}  |  Production Value: ${p1['Predicted_Salary']:,.0f}")
        print(f"       Stats: {p1['PTS']:.1f} PTS / {p1['TRB']:.1f} REB / {p1['AST']:.1f} AST")
        print(f"    ✓ {p2['Player']} ({p2['Team']})")
        print(f"       Salary: ${p2['Salary']:,.0f}  |  Production Value: ${p2['Predicted_Salary']:,.0f}")
        print(f"       Stats: {p2['PTS']:.1f} PTS / {p2['TRB']:.1f} REB / {p2['AST']:.1f} AST")

        # --- Financial Impact ---
        incoming = p1['Salary'] + p2['Salary']
        outgoing = trade_away['Salary']
        cap_saved = outgoing - incoming
        production_before = trade_away['Predicted_Salary']
        production_after = p1['Predicted_Salary'] + p2['Predicted_Salary']

        print(f"\n{'─' * 70}")
        print(f"  FINANCIAL IMPACT")
        print(f"{'─' * 70}")
        print(f"  Salary going out:     ${outgoing:>12,.0f}")
        print(f"  Salary coming in:     ${incoming:>12,.0f}")
        print(f"  Net cap space saved:  ${cap_saved:>12,.0f}")
        print(f"\n  Production traded:    ${production_before:>12,.0f} (model-estimated value)")
        print(f"  Production acquired:  ${production_after:>12,.0f} (combined)")
        print(f"  Net production gain:  ${production_after - production_before:>12,.0f}")

        # --- Business Case ---
        print(f"\n{'─' * 70}")
        print(f"  BUSINESS CASE")
        print(f"{'─' * 70}")
        print(f"""
  Why this trade works under the NBA Salary Cap:

  1. CAP RELIEF: By trading {trade_away['Player']} (${outgoing:,.0f}), {team_code} frees
     ${cap_saved:,.0f} in cap space — money that can be used to retain key players
     or sign free agents next offseason.

  2. PRODUCTION UPGRADE: The two incoming players ({p1['Player']} + {p2['Player']})
     combine for {p1['PTS'] + p2['PTS']:.1f} PPG vs {trade_away['PTS']:.1f} PPG from the outgoing player.
     Their combined production value (${production_after:,.0f}) exceeds the single
     player's value (${production_before:,.0f}) by ${production_after - production_before:,.0f}.

  3. DEPTH > STAR POWER: Two reliable role players provide more consistent
     nightly output, injury insurance, and lineup flexibility compared to
     one overpaid underperformer.

  4. ASSET FLEXIBILITY: If {team_code} later needs to make another move, two 
     moderate contracts are much easier to trade than one bloated deal.
     Teams around the league actively seek affordable, productive players.

  VERDICT: This trade transforms a negative-value contract into two positive-
  value assets while improving both on-court production AND financial flexibility.
""")

    print("=" * 70)


# =============================================================================
# 4. MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA ROI & Trade Analysis")
    parser.add_argument('--team', '-t', type=str, default=None,
                        help="Team code to analyze (e.g., LAL, BKN, GSW, MIA)")
    parser.add_argument('--year', '-y', type=int, default=2026,
                        help="Season year to analyze (default: 2026)")
    parser.add_argument('--data', '-d', type=str,
                        default='data/nba_stats_and_salaries_2020_2025.csv')
    args = parser.parse_args()

    print("Loading data and training model...")
    df_all, model = load_and_train(args.data, target_year=args.year)

    print("Categorizing players...")
    season_df = categorize_players(df_all, year=args.year)

    # Summary stats
    counts = season_df['Category'].value_counts()
    print(f"\n{args.year} Season Player Categories:")
    print(f"  High ROI (underpaid stars):  {counts.get('High ROI', 0)}")
    print(f"  Fair Value:                  {counts.get('Fair Value', 0)}")
    print(f"  Cap Clogs (overpaid):        {counts.get('Cap Clog', 0)}")

    # Top High ROI
    print(f"\n{'─' * 50}")
    print("TOP 10 HIGH ROI PLAYERS (Best Value in the League)")
    print(f"{'─' * 50}")
    top_roi = season_df[season_df['Category'] == 'High ROI'].nlargest(10, 'Salary_Diff')
    for _, p in top_roi.iterrows():
        surplus = p['Predicted_Salary'] - p['Salary']
        print(f"  {p['Player']:20s} ({p['Team']}) — {p['PTS']:.1f} PPG, "
              f"Paid ${p['Salary']:>10,.0f}, Worth ${p['Predicted_Salary']:>10,.0f} "
              f"(+${surplus:,.0f} surplus)")

    # Top Cap Clogs
    print(f"\n{'─' * 50}")
    print("TOP 10 CAP CLOGS (Most Overpaid Players)")
    print(f"{'─' * 50}")
    top_clogs = season_df[season_df['Category'] == 'Cap Clog'].nsmallest(10, 'Salary_Diff')
    for _, p in top_clogs.iterrows():
        overpay = p['Salary'] - p['Predicted_Salary']
        print(f"  {p['Player']:20s} ({p['Team']}) — {p['PTS']:.1f} PPG, "
              f"Paid ${p['Salary']:>10,.0f}, Worth ${p['Predicted_Salary']:>10,.0f} "
              f"(-${overpay:,.0f} overpay)")

    # Trade recommendation
    if args.team:
        generate_trade_recommendation(season_df, args.team)
    else:
        # Auto-pick the team with the worst Cap Clog
        worst_clog = season_df[season_df['Category'] == 'Cap Clog'].nsmallest(1, 'Salary_Diff')
        if not worst_clog.empty:
            auto_team = worst_clog.iloc[0]['Team']
            print(f"\n\nAuto-selecting team with biggest cap clog: {auto_team}")
            generate_trade_recommendation(season_df, auto_team)

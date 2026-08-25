"""
NBA Salary Prediction Model
============================
Predicts a player's annual salary based on their per-game stats.

Usage:
    python predict_salary.py                          # defaults to RandomForestRegressor
    python predict_salary.py --model ridge            # use Ridge regression
    python predict_salary.py --model xgboost          # use XGBoost
    python predict_salary.py --model lasso            # use Lasso regression

The model parameter lets you swap algorithms without changing anything else.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso


# =============================================================================
# 1. CHOOSE YOUR MODEL
# =============================================================================

def get_model(model_name: str):
    """
    Returns an sklearn-compatible regression model based on the name you pass.
    
    Supported models:
        'random_forest' - Random Forest (good default, handles non-linear relationships)
        'xgboost'       - XGBoost (usually best accuracy, slower)
        'ridge'         - Ridge Regression (fast, linear, good baseline)
        'lasso'         - Lasso Regression (linear, also does feature selection)
        'gradient_boost'- Gradient Boosting (similar to XGBoost but sklearn-native)
    """
    models = {
        'random_forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1000),  # Larger alpha for salary scale
        'gradient_boost': GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    # XGBoost is optional (needs extra install)
    if model_name == 'xgboost':
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        except ImportError:
            print("XGBoost not installed. Falling back to Random Forest.")
            return models['random_forest']

    if model_name not in models:
        print(f"Unknown model '{model_name}'. Available: {list(models.keys()) + ['xgboost']}")
        print("Falling back to Random Forest.")
        return models['random_forest']

    return models[model_name]


# =============================================================================
# 2. LOAD AND PREPARE DATA
# =============================================================================

def score_awards(awards_str):
    """
    Convert Awards string (e.g., 'MVP-1,AS,NBA1') to a numeric score.
    Higher score = more prestigious awards that season.
    
    Scoring:
        MVP-1: 10 (winner), MVP-2 to MVP-5: 5-2 pts (top 5 voting)
        AS (All-Star): 3
        NBA1/NBA2/NBA3 (All-NBA teams): 5/3/2
        DPOY-1: 4, DEF1/DEF2: 2/1
        ROY-1, MIP-1, 6MOY-1: 3 each
    """
    if pd.isna(awards_str) or awards_str in ('0', '0.0', ''):
        return 0
    
    score = 0
    awards = str(awards_str).split(',')
    
    for award in awards:
        award = award.strip()
        # MVP voting (top 5 matters most)
        if award.startswith('MVP-'):
            rank = int(award.split('-')[1])
            if rank == 1: score += 10
            elif rank <= 5: score += max(6 - rank, 1)  # 2nd=5, 3rd=4, 4th=3, 5th=2
        # All-Star
        elif award == 'AS':
            score += 3
        # All-NBA teams
        elif award == 'NBA1': score += 5
        elif award == 'NBA2': score += 3
        elif award == 'NBA3': score += 2
        # Defensive awards
        elif award.startswith('DPOY-'):
            rank = int(award.split('-')[1])
            if rank == 1: score += 4
        elif award == 'DEF1': score += 2
        elif award == 'DEF2': score += 1
        # Other major awards
        elif award in ('ROY-1', 'MIP-1', '6MOY-1'):
            score += 3
    
    return score


def load_data(csv_path: str = "data/nba_stats_and_salaries_2020_2026.csv"):
    """
    Loads the merged stats+salary CSV file.
    Returns features (X) and target (y = Salary).
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from '{csv_path}'")
    print(f"Seasons: {sorted(df['Year'].unique())}")
    print(f"Columns: {df.columns.tolist()}\n")

    # ----- Define which columns we use as features -----
    # These are the per-game stats that best relate to a player's value.
    # You can add/remove features here to experiment!
    feature_cols = [
        'Age',    # Player age
        'G',      # Games played
        'GS',     # Games started
        'MP',     # Minutes per game
        'PTS',    # Points per game
        'TRB',    # Total rebounds per game
        'AST',    # Assists per game
        'STL',    # Steals per game
        'BLK',    # Blocks per game
        'TOV',    # Turnovers per game
        'FG%',    # Field goal percentage
        '3P%',    # Three-point percentage
        'FT%',    # Free throw percentage
        'eFG%',   # Effective FG% (accounts for 3-pointers being worth more)
        'Awards', # Season awards (MVP, All-NBA, All-Star, etc.) - captures "star power"
    ]

    # Only keep columns that actually exist in the data
    available_features = [col for col in feature_cols if col in df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        print(f"Note: These features weren't found in the data: {missing}")

    print(f"Using {len(available_features)} features: {available_features}\n")

    # Convert Awards string to numeric score before extracting features
    if 'Awards' in df.columns:
        df['Awards'] = df['Awards'].apply(score_awards)

        # Apply recency weighting: recent awards matter more for salary negotiations.
        # For each player-season, we sum their awards from prior seasons with decay:
        #   same year: weight 1.0, 1 year ago: 0.7, 2 years ago: 0.5, 3+ years ago: 0.3
        decay_weights = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.3}

        df = df.sort_values(by=['Player', 'Year']).reset_index(drop=True)
        weighted_awards = []

        for idx, row in df.iterrows():
            player_history = df[(df['Player'] == row['Player']) & (df['Year'] <= row['Year'])]
            total = 0.0
            for _, hist_row in player_history.iterrows():
                years_ago = int(row['Year'] - hist_row['Year'])
                weight = decay_weights.get(years_ago, 0.3)
                total += hist_row['Awards'] * weight
            weighted_awards.append(round(total, 1))

        df['Awards'] = weighted_awards
        print(f"Awards converted to recency-weighted scores (max: {df['Awards'].max()})")
        print(f"  Weights: current year=1.0, 1yr ago=0.7, 2yr ago=0.5, 3+yr ago=0.3")

    # X = features, y = what we're predicting (Salary)
    X = df[available_features].copy()
    y = df['Salary'].copy()

    # Fill any missing values with 0 (some players have NaN for 3P% if they never shot 3s)
    X.fillna(0, inplace=True)

    # Remove rows where salary is 0 or missing (bad data)
    valid = y > 0
    X = X[valid]
    y = y[valid]

    return X, y, df[valid]


# =============================================================================
# 3. TRAIN AND EVALUATE
# =============================================================================

def train_and_evaluate(model_name: str = 'random_forest', csv_path: str = "data/nba_stats_and_salaries_2020_2026.csv", filter_year: int = None):
    """
    Temporal prediction pipeline: train on 2023-2025, predict all of 2026.
    
    Uses a 7% annual salary cap increase to adjust historical salaries to 2026 dollars
    before training, so the model learns production→salary in today's cap environment.
    
    Args:
        filter_year: Ignored (kept for CLI compatibility). Always predicts 2026.
    """
    CAP_INCREASE = 0.07  # 7% annual salary cap increase
    PREDICT_YEAR = 2026

    # Load data
    X, y, full_df = load_data(csv_path)

    # ----- Temporal split: train on past, predict current -----
    train_mask = full_df['Year'] < PREDICT_YEAR
    test_mask = full_df['Year'] == PREDICT_YEAR

    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_test = X[test_mask].copy()
    y_test = y[test_mask].copy()

    # ----- Adjust training salaries to 2026 cap dollars -----
    # A player earning $30M in 2023 would earn ~$30M * 1.07^3 = $36.8M in 2026 cap terms.
    train_years = full_df.loc[train_mask, 'Year']
    cap_adjustment = (1 + CAP_INCREASE) ** (PREDICT_YEAR - train_years)
    y_train = y_train * cap_adjustment.values

    print(f"Training set: {len(X_train)} rows (Years < {PREDICT_YEAR}, adjusted +7%/yr to {PREDICT_YEAR} dollars)")
    print(f"Test set:     {len(X_test)} rows (Year = {PREDICT_YEAR})\n")

    # ----- Scale features -----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----- Get the model -----
    model = get_model(model_name)
    print(f"Training model: {model.__class__.__name__}...")

    # ----- Train -----
    model.fit(X_train_scaled, y_train)

    # ----- Predict on 2026 players -----
    y_pred = model.predict(X_test_scaled)

    # ----- Evaluate -----
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"RESULTS ({model.__class__.__name__}) — Temporal: Train <{PREDICT_YEAR}, Test ={PREDICT_YEAR}")
    print(f"{'='*50}")
    print(f"  Mean Absolute Error:  ${mae:,.0f}")
    print(f"    → On average, predictions are off by this much")
    print(f"  R² Score:             {r2:.4f}")
    print(f"    → 1.0 = perfect, 0.0 = no better than guessing the average")
    print(f"  Cap adjustment:       +{CAP_INCREASE*100:.0f}% per year applied to training salaries")
    print(f"{'='*50}\n")

    # ----- Show feature importance (for tree-based models) -----
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        print("Top features by importance:")
        for feat, imp in importances.head(10).items():
            print(f"  {feat:>6}: {'█' * int(imp * 50)} ({imp:.3f})")
        print()

    # ----- Build results for all 2026 players -----
    results = pd.DataFrame({
        'Actual Salary': y_test.values,
        'Predicted Salary': y_pred.astype(int),
        'Difference': (y_pred - y_test.values).astype(int),
    }, index=X_test.index)

    # Merge back player names for display
    results['Player'] = full_df.loc[results.index, 'Player'].values
    results['Year'] = full_df.loc[results.index, 'Year'].values
    results = results[['Player', 'Year', 'Actual Salary', 'Predicted Salary', 'Difference']]

    display_results = results.copy()
    print(f"Predictions for all {len(display_results)} players in {PREDICT_YEAR}:")
    print(display_results.head(15).to_string(index=False))

    # ----- Top 10 Lists -----
    print(f"\n{'='*70}")
    print("TOP 10 HIGHEST PREDICTED SALARIES")
    print(f"{'='*70}")
    top_predicted = display_results.nlargest(10, 'Predicted Salary')
    for _, row in top_predicted.iterrows():
        print(f"  {row['Player']:25s} {int(row['Year'])}  Pred: ${row['Predicted Salary']:>12,}  Actual: ${int(row['Actual Salary']):>12,}")

    print(f"\n{'='*70}")
    print("TOP 10 MOST OVERPAID (Actual > Predicted)")
    print(f"{'='*70}")
    most_overpaid = display_results.nsmallest(10, 'Difference')  # Negative diff = overpaid
    for _, row in most_overpaid.iterrows():
        diff = int(row['Difference'])
        print(f"  {row['Player']:25s} {int(row['Year'])}  Overpaid by: ${abs(diff):>12,}  (Actual: ${int(row['Actual Salary']):,})")

    print(f"\n{'='*70}")
    print("TOP 10 MOST UNDERPAID (Predicted > Actual)")
    print(f"{'='*70}")
    most_underpaid = display_results.nlargest(10, 'Difference')  # Positive diff = underpaid
    for _, row in most_underpaid.iterrows():
        diff = int(row['Difference'])
        print(f"  {row['Player']:25s} {int(row['Year'])}  Underpaid by: ${diff:>12,}  (Actual: ${int(row['Actual Salary']):,})")

    return model, scaler, results


# =============================================================================
# 4. RUN FROM COMMAND LINE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict NBA player salary from per-game stats")
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='random_forest',
        help="Model to use: random_forest, xgboost, ridge, lasso, gradient_boost"
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/nba_stats_and_salaries_2020_2026.csv',
        help="Path to the merged stats+salary CSV file"
    )
    args = parser.parse_args()

    train_and_evaluate(model_name=args.model, csv_path=args.data)

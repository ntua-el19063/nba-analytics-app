"""
Clean and combine real NBA salary data from multiple sources.

Sources:
- kaggle_old_salaries.csv: Covers 2022-23, 2023-24, 2024-25 seasons (our Years 2023, 2024, 2025)
- br_salaries_2025.csv:    Covers 2025-26 season (our Year 2026)

This script:
1. Scrapes 2025-26 season stats from Basketball Reference (Year 2026)
2. Cleans both salary CSVs (removes $, commas, whitespace; handles $0 as missing)
3. Combines salary data: Kaggle (2023-2025) + BR (2026)
4. Merges stats (2023-2026) with real salaries
5. Saves the final analysis-ready CSV
"""

import pandas as pd
import numpy as np
import re
import unicodedata


def clean_salary_value(val):
    """Convert salary strings like '$48,070,014 ' to integer. Returns 0 for invalid."""
    if pd.isna(val):
        return 0
    val = str(val).strip().replace('$', '').replace(',', '').strip()
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def normalize_name(name):
    """Normalize player names for cross-dataset matching (handles accents, case)."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    # Decompose unicode and remove combining marks (e.g., Jokić -> Jokic)
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_name.lower().strip()


def scrape_2026_stats():
    """Scrape 2025-26 season stats from Basketball Reference."""
    print("Scraping 2025-26 season stats from Basketball Reference...")
    url = "https://www.basketball-reference.com/leagues/NBA_2026_per_game.html"
    tables = pd.read_html(url)
    stats = tables[0]

    # Remove header-repeat rows
    stats = stats[stats['Player'] != 'Player'].reset_index(drop=True)
    stats['Year'] = 2026

    # Handle traded players — same logic as nba_scraper notebook
    # TOT/2TM/3TM/4TM are "combined totals" rows for traded players.
    # We keep only the total row (so we don't double-count stats) but replace
    # the team code with the player's last team from the individual-team rows.
    trade_codes = ['TOT', '2TM', '3TM', '4TM']

    # Create Trades column: 2TM=1, 3TM=2, 4TM=3, TOT=1, else 0
    def get_trades(team):
        if pd.isna(team):
            return 0
        match = re.match(r'(\d+)TM', str(team))
        if match:
            return int(match.group(1)) - 1
        elif team == 'TOT':
            return 1
        return 0

    stats['Trades'] = stats['Team'].apply(get_trades)

    trade_rows = stats[stats['Team'].isin(trade_codes)].copy()
    team_stats_rows = stats[~stats['Team'].isin(trade_codes)].copy()

    # For each traded player, find their last team from the partial rows
    def get_last_team(player, year):
        matches = team_stats_rows[(team_stats_rows['Player'] == player) & (team_stats_rows['Year'] == year)]
        if not matches.empty:
            return matches.iloc[-1]['Team']
        return None

    trade_rows['Team'] = trade_rows.apply(
        lambda row: get_last_team(row['Player'], row['Year']) or row['Team'],
        axis=1
    )

    # Remove individual team rows for players who have a total row
    players_with_total = set(zip(trade_rows['Player'], trade_rows['Year']))
    team_stats_rows_clean = team_stats_rows[
        ~team_stats_rows.apply(lambda row: (row['Player'], row['Year']) in players_with_total, axis=1)
    ]

    # Recombine: total rows + non-traded player rows
    stats = pd.concat([trade_rows, team_stats_rows_clean], ignore_index=True)
    stats.sort_values(by=['Year', 'Player'], inplace=True)
    stats.reset_index(drop=True, inplace=True)

    # Convert numeric columns
    numeric_cols = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
                    '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    for col in numeric_cols:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors='coerce')
    stats[numeric_cols] = stats[numeric_cols].fillna(0)

    # Handle Awards column
    if 'Awards' not in stats.columns:
        stats['Awards'] = 0.0
    else:
        stats['Awards'] = stats['Awards'].fillna(0)

    print(f"  Got {len(stats)} players for 2025-26 season")
    return stats


def clean_kaggle_salaries(filepath="data/kaggle_old_salaries.csv"):
    """
    Clean the Kaggle NBA salary file.
    Columns: Player Id, Player Name, 2022/2023, 2023/2024, 2024/2025, 2025/2026
    """
    df = pd.read_csv(filepath)
    df.columns = ['Player_Id', 'Player', 'Sal_2023', 'Sal_2024', 'Sal_2025', 'Sal_2026']

    for col in ['Sal_2023', 'Sal_2024', 'Sal_2025', 'Sal_2026']:
        df[col] = df[col].apply(clean_salary_value)

    df['Player'] = df['Player'].str.strip()

    # Melt wide → long
    salary_long = df.melt(
        id_vars=['Player'],
        value_vars=['Sal_2023', 'Sal_2024', 'Sal_2025', 'Sal_2026'],
        var_name='Year_Col',
        value_name='Salary'
    )
    year_map = {'Sal_2023': 2023, 'Sal_2024': 2024, 'Sal_2025': 2025, 'Sal_2026': 2026}
    salary_long['Year'] = salary_long['Year_Col'].map(year_map)
    salary_long = salary_long[salary_long['Salary'] > 0][['Player', 'Year', 'Salary']].reset_index(drop=True)

    print(f"Kaggle cleaned: {len(salary_long)} records")
    print(f"  Year breakdown: {salary_long.groupby('Year').size().to_dict()}")
    return salary_long


def clean_br_salaries(filepath="data/br_salaries_2025.csv"):
    """Clean the Basketball Reference 2025-26 salary file."""
    df = pd.read_csv(filepath, skiprows=1)
    df.columns = ['Rk', 'Player', 'Team', 'Sal_2026', 'Sal_2027', 'Sal_2028',
                  'Sal_2029', 'Sal_2030', 'Sal_2031', 'Guaranteed', 'Player_Id']

    df['Player'] = df['Player'].str.strip()
    df = df[df['Player'] != 'Player'].reset_index(drop=True)
    df['Salary'] = df['Sal_2026'].apply(clean_salary_value)
    df['Year'] = 2026
    df = df[df['Salary'] > 0][['Player', 'Year', 'Salary']].drop_duplicates(subset=['Player'], keep='first')

    print(f"BR cleaned: {len(df)} players for 2025-26 (Year 2026)")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  CLEANING AND COMBINING NBA SALARY DATA")
    print("=" * 60)

    # 1. Scrape 2026 stats
    print("\n--- Step 1: Scrape 2025-26 Stats ---")
    stats_2026 = scrape_2026_stats()

    # 2. Load existing stats (2020-2025) and drop old Salary column
    print("\n--- Step 2: Load Existing Stats ---")
    existing = pd.read_csv("data/nba_stats_and_salaries_2020_2025.csv")
    if 'Salary' in existing.columns:
        existing = existing.drop(columns=['Salary'])
    existing_years = sorted(existing['Year'].unique())
    print(f"  {len(existing)} rows, years: {existing_years}")

    # 3. Combine stats: existing (2020-2025) + new 2026
    print("\n--- Step 3: Combine Stats ---")
    common_cols = [c for c in existing.columns if c in stats_2026.columns]
    all_stats = pd.concat([existing[common_cols], stats_2026[common_cols]], ignore_index=True)
    print(f"  Combined stats: {len(all_stats)} rows, years: {sorted(all_stats['Year'].unique())}")

    # 4. Clean salary data
    print("\n--- Step 4: Clean Salary Data ---")
    kaggle_sal = clean_kaggle_salaries()
    br_sal = clean_br_salaries()

    # Use Kaggle for 2023-2025, BR for 2026 (490 players vs Kaggle's 135)
    combined_sal = pd.concat([
        kaggle_sal[kaggle_sal['Year'].isin([2023, 2024, 2025])],
        br_sal
    ], ignore_index=True)
    print(f"\n  Combined salaries: {len(combined_sal)} records")
    print(f"  Year breakdown: {combined_sal.groupby('Year').size().to_dict()}")

    # 5. Merge stats + salaries using normalized names
    print("\n--- Step 5: Merge Stats + Salaries ---")
    # MERGE EXPLANATION:
    # We use a left join to preserve all stats rows, attaching salary where available.
    # The '_match' column is a normalized (lowercase, accent-stripped) version of Player
    # to handle name variations between datasets (e.g., "Nikola Jokić" vs "Nikola Jokic").
    # We join on both '_match' AND 'Year' to ensure we match the correct season's salary
    # (a player's salary changes year-to-year). After merging, we drop the helper column.
    all_stats['_match'] = all_stats['Player'].apply(normalize_name)
    combined_sal['_match'] = combined_sal['Player'].apply(normalize_name)

    merged = all_stats.merge(
        combined_sal[['_match', 'Year', 'Salary']],
        on=['_match', 'Year'],
        how='left'
    ).drop(columns=['_match'])

    # Report match rates
    print("  Match rates:")
    for year in sorted(merged['Year'].unique()):
        ym = merged['Year'] == year
        matched = (merged.loc[ym, 'Salary'] > 0).sum()
        total = ym.sum()
        pct = matched / total * 100 if total else 0
        print(f"    Year {year}: {matched}/{total} ({pct:.0f}%)")

    # Keep only rows with real salary data
    final = merged[merged['Salary'].notna() & (merged['Salary'] > 0)].copy()
    final['Salary'] = final['Salary'].astype(int)

    # Remove exact duplicate rows (same player, year, and stats appearing multiple times)
    before = len(final)
    final = final.drop_duplicates(subset=['Player', 'Year'], keep='last').reset_index(drop=True)
    dropped = before - len(final)
    if dropped > 0:
        print(f"\n  Removed {dropped} duplicate rows (same player+year)")

    print(f"\n  FINAL DATASET: {len(final)} rows with real salaries")
    print(f"  Years: {sorted(final['Year'].unique())}")

    # Show head of final dataframe
    print("\n  Final DataFrame preview:")
    print(final.head(10).to_string())

    # Save
    output = "data/nba_stats_and_salaries_2020_2026.csv"
    final.to_csv(output, index=False)
    print(f"\n  Saved: {output}")

    # Also save the combined salary reference file
    combined_sal.to_csv("data/real_salaries_combined.csv", index=False)

    # Show top earners
    print("\n  Top 10 earners (2025-26 season):")
    top = final[final['Year'] == 2026].nlargest(10, 'Salary')
    for _, row in top.iterrows():
        player = row['Player']
        team = row['Team']
        pts = row['PTS']
        sal = row['Salary']
        print(f"    {player:25s} {team:4s} {pts:5.1f} PPG  ${sal:>12,}")
        # : — starts the format spec | > — right-align |  12 — minimum width of 12 characters | , — thousands separator

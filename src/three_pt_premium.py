"""
3-Point Shooter Salary Premium Analysis
========================================
Examines whether NBA teams overpay 3-point shooters relative to high-volume
2-point scorers with similar or better efficiency.

Compares two groups:
  GROUP A: "3PT Specialists" — Players averaging > X made 3-pointers per game
  GROUP B: "Interior Scorers" — Players below the 3PT threshold but with high
           2-point volume and good 2P%

All thresholds are adjustable via command-line arguments.

Usage:
    python three_pt_premium.py
    python three_pt_premium.py --min-3pm 2.5 --min-games 30
    python three_pt_premium.py --min-2pa 6 --min-2pct 0.55 --min-games 40
"""

import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="3PT Shooter Salary Premium Analysis")

    # Adjustable inputs
    parser.add_argument('--min-3pm', type=float, default=2.0,
                        help="Minimum 3-pointers made per game to qualify as '3PT Specialist' (default: 2.0)")
    parser.add_argument('--min-games', type=int, default=20,
                        help="Minimum games played per season to be included (default: 20)")
    parser.add_argument('--min-2pa', type=float, default=5.0,
                        help="Minimum 2-point attempts per game for 'Interior Scorer' group (default: 5.0)")
    parser.add_argument('--min-2pct', type=float, default=0.52,
                        help="Minimum 2P%% for 'Interior Scorer' group (default: 0.52)")
    parser.add_argument('--data', type=str, default='data/nba_stats_and_salaries_2020_2025.csv',
                        help="Path to the stats+salary CSV")
    parser.add_argument('--year', type=int, default=None,
                        help="Analyze a specific year only (default: all years)")

    return parser.parse_args()


def load_and_filter(args):
    """Load data and apply minimum games filter."""
    df = pd.read_csv(args.data)
    df = df[df['G'] >= args.min_games].copy()
    if args.year:
        df = df[df['Year'] == args.year]
    return df


def categorize_players(df, args):
    """Split players into 3PT Specialists vs Interior Scorers."""
    # Group A: 3PT Specialists (high-volume 3PT makers)
    three_pt = df[df['3P'] >= args.min_3pm].copy()
    three_pt['Scorer Type'] = '3PT Specialist'

    # Group B: Interior Scorers (below 3PT threshold, high 2PA with good 2P%)
    interior = df[
        (df['3P'] < args.min_3pm) &
        (df['2PA'] >= args.min_2pa) &
        (df['2P%'] >= args.min_2pct)
    ].copy()
    interior['Scorer Type'] = 'Interior Scorer'

    return three_pt, interior


def format_salary(val):
    """Format salary as $XX.XM or $X.XM."""
    if val >= 1_000_000:
        return f"${val / 1_000_000:.1f}M"
    return f"${val:,.0f}"


def print_report(three_pt, interior, args):
    """Print formatted comparison report."""
    width = 75

    print("\n" + "=" * width)
    print("  3-POINT SHOOTER SALARY PREMIUM ANALYSIS")
    print("=" * width)

    # Parameters
    print(f"\n{'─' * width}")
    print("  PARAMETERS")
    print(f"{'─' * width}")
    print(f"  Min games played:       {args.min_games}")
    print(f"  3PT Specialist:         ≥ {args.min_3pm} made 3s per game")
    print(f"  Interior Scorer:        < {args.min_3pm} 3PM, ≥ {args.min_2pa} 2PA, ≥ {args.min_2pct:.0%} 2P%")
    if args.year:
        print(f"  Season:                 {args.year} only")
    else:
        print(f"  Seasons:                All available")

    # Summary stats
    print(f"\n{'─' * width}")
    print("  GROUP COMPARISON")
    print(f"{'─' * width}")

    header = f"  {'Metric':<30} {'3PT Specialists':>18} {'Interior Scorers':>18}"
    print(header)
    print(f"  {'─' * 66}")

    n3 = len(three_pt)
    ni = len(interior)
    print(f"  {'Player-seasons':<30} {n3:>18,} {ni:>18,}")

    if n3 == 0 or ni == 0:
        print("\n  ⚠ One or both groups are empty. Adjust thresholds and try again.")
        return

    avg_sal_3 = three_pt['Salary'].mean()
    avg_sal_i = interior['Salary'].mean()
    med_sal_3 = three_pt['Salary'].median()
    med_sal_i = interior['Salary'].median()
    avg_pts_3 = three_pt['PTS'].mean()
    avg_pts_i = interior['PTS'].mean()
    avg_mpg_3 = three_pt['MP'].mean()
    avg_mpg_i = interior['MP'].mean()
    avg_efg_3 = three_pt['eFG%'].mean()
    avg_efg_i = interior['eFG%'].mean()

    print(f"  {'Avg Salary':<30} {format_salary(avg_sal_3):>18} {format_salary(avg_sal_i):>18}")
    print(f"  {'Median Salary':<30} {format_salary(med_sal_3):>18} {format_salary(med_sal_i):>18}")
    print(f"  {'Avg PPG':<30} {avg_pts_3:>18.1f} {avg_pts_i:>18.1f}")
    print(f"  {'Avg MPG':<30} {avg_mpg_3:>18.1f} {avg_mpg_i:>18.1f}")
    print(f"  {'Avg eFG%':<30} {avg_efg_3:>18.3f} {avg_efg_i:>18.3f}")

    # Premium calculation
    premium_pct = ((avg_sal_3 - avg_sal_i) / avg_sal_i) * 100
    cost_per_pt_3 = avg_sal_3 / avg_pts_3 if avg_pts_3 > 0 else 0
    cost_per_pt_i = avg_sal_i / avg_pts_i if avg_pts_i > 0 else 0

    print(f"\n  {'─' * 66}")
    print(f"  {'3PT SALARY PREMIUM':<30} {premium_pct:>+18.1f}%")
    print(f"  {'Cost per PPG (3PT)':<30} {format_salary(cost_per_pt_3):>18}")
    print(f"  {'Cost per PPG (Interior)':<30} {format_salary(cost_per_pt_i):>18}")
    efficiency_gap = ((cost_per_pt_3 - cost_per_pt_i) / cost_per_pt_i) * 100
    print(f"  {'Cost-per-point premium':<30} {efficiency_gap:>+18.1f}%")

    # Year-over-year trend
    if not args.year and three_pt['Year'].nunique() > 1:
        print(f"\n{'─' * width}")
        print("  YEAR-OVER-YEAR TREND")
        print(f"{'─' * width}")
        print(f"  {'Year':<8} {'3PT Avg Salary':>16} {'Interior Avg':>16} {'Premium':>12} {'N(3PT)':>8} {'N(Int)':>8}")
        print(f"  {'─' * 66}")

        years = sorted(set(three_pt['Year'].unique()) | set(interior['Year'].unique()))
        for yr in years:
            yr3 = three_pt[three_pt['Year'] == yr]
            yri = interior[interior['Year'] == yr]
            if len(yr3) > 0 and len(yri) > 0:
                a3 = yr3['Salary'].mean()
                ai = yri['Salary'].mean()
                prem = ((a3 - ai) / ai) * 100
                print(f"  {yr:<8} {format_salary(a3):>16} {format_salary(ai):>16} {prem:>+11.1f}% {len(yr3):>8} {len(yri):>8}")

    # Top paid 3PT specialists
    print(f"\n{'─' * width}")
    print("  TOP 10 HIGHEST PAID 3PT SPECIALISTS")
    print(f"{'─' * width}")
    top3 = three_pt.nlargest(10, 'Salary')
    print(f"  {'Player':<24} {'Team':>4} {'Year':>5} {'3PM':>5} {'3P%':>6} {'PTS':>5} {'Salary':>14}")
    print(f"  {'─' * 66}")
    for _, r in top3.iterrows():
        sal = format_salary(r['Salary'])
        print(f"  {r['Player']:<24} {r['Team']:>4} {int(r['Year']):>5} {r['3P']:>5.1f} {r['3P%']:>6.3f} {r['PTS']:>5.1f} {sal:>14}")

    # Top paid interior scorers
    print(f"\n{'─' * width}")
    print("  TOP 10 HIGHEST PAID INTERIOR SCORERS")
    print(f"{'─' * width}")
    topi = interior.nlargest(10, 'Salary')
    print(f"  {'Player':<24} {'Team':>4} {'Year':>5} {'2PA':>5} {'2P%':>6} {'PTS':>5} {'Salary':>14}")
    print(f"  {'─' * 66}")
    for _, r in topi.iterrows():
        sal = format_salary(r['Salary'])
        print(f"  {r['Player']:<24} {r['Team']:>4} {int(r['Year']):>5} {r['2PA']:>5.1f} {r['2P%']:>6.3f} {r['PTS']:>5.1f} {sal:>14}")

    # Best value in each group (highest PPG per $1M)
    print(f"\n{'─' * width}")
    print("  BEST VALUE: MOST PPG PER $1M SALARY")
    print(f"{'─' * width}")

    three_pt_copy = three_pt.copy()
    three_pt_copy['PPG_per_1M'] = three_pt_copy['PTS'] / (three_pt_copy['Salary'] / 1_000_000)
    best_val_3 = three_pt_copy.nlargest(5, 'PPG_per_1M')

    interior_copy = interior.copy()
    interior_copy['PPG_per_1M'] = interior_copy['PTS'] / (interior_copy['Salary'] / 1_000_000)
    best_val_i = interior_copy.nlargest(5, 'PPG_per_1M')

    print(f"\n  3PT Specialists — Best Value:")
    print(f"  {'Player':<24} {'PTS':>5} {'Salary':>12} {'PPG/$1M':>9}")
    for _, r in best_val_3.iterrows():
        print(f"  {r['Player']:<24} {r['PTS']:>5.1f} {format_salary(r['Salary']):>12} {r['PPG_per_1M']:>9.2f}")

    print(f"\n  Interior Scorers — Best Value:")
    print(f"  {'Player':<24} {'PTS':>5} {'Salary':>12} {'PPG/$1M':>9}")
    for _, r in best_val_i.iterrows():
        print(f"  {r['Player']:<24} {r['PTS']:>5.1f} {format_salary(r['Salary']):>12} {r['PPG_per_1M']:>9.2f}")

    # Verdict
    print(f"\n{'─' * width}")
    print("  VERDICT")
    print(f"{'─' * width}")
    if premium_pct > 0:
        print(f"""
  3-point shooters command a {premium_pct:.1f}% salary premium over interior scorers
  with comparable or better efficiency (eFG%).

  Cost per point scored:
    • 3PT Specialists:  {format_salary(cost_per_pt_3)} per PPG
    • Interior Scorers: {format_salary(cost_per_pt_i)} per PPG

  Teams pay {efficiency_gap:.1f}% MORE per point of production for perimeter shooting.
  This suggests the market overvalues 3-point volume relative to efficient
  interior scoring — a potential arbitrage opportunity for cap-conscious teams.
""")
    else:
        print(f"""
  Interior scorers actually earn MORE ({abs(premium_pct):.1f}% premium) than 3PT
  specialists with the current thresholds. The 3PT overpay narrative does not
  hold under these parameters. Try adjusting --min-3pm or --min-2pct.
""")

    print("=" * width)


if __name__ == "__main__":
    args = parse_args()

    print("Loading data...")
    df = load_and_filter(args)
    print(f"  {len(df)} player-seasons after filtering (min {args.min_games} games)")

    three_pt, interior = categorize_players(df, args)
    print_report(three_pt, interior, args)

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

RISK_FREE_RATE = 0.04

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_eden_csv(path, ticker):
    df = pd.read_csv(path)
    df = df[['Date', 'Close']].copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df = df.rename(columns={'Close': ticker})
    return df


def load_candidate_csv(path, ticker):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df = df.rename(columns={'Close': ticker})
    return df


def build_price_matrix(eden_assets, candidate=None):
    merged = None
    for asset in eden_assets:
        df = load_eden_csv(asset['path'], asset['ticker'])
        merged = df if merged is None else merged.merge(df, on='Date', how='inner')
    if candidate is not None:
        cand_df = load_candidate_csv(candidate['path'], 'Candidate')
        merged = merged.merge(cand_df, on='Date', how='inner')
    merged = merged.set_index('Date').sort_index()
    return merged.dropna()


def compute_returns(price_df):
    return price_df.pct_change().dropna()


def risk_decomposition(returns_df, weights):
    w = np.array(weights)
    cov = returns_df.cov()
    port_vol_daily = math.sqrt(w @ cov.values @ w)
    marginal = (cov.values @ w) / port_vol_daily
    component = w * marginal
    pct_contrib = pd.Series(
        (component / port_vol_daily) * 100,
        index=returns_df.columns
    )
    port_vol_ann = port_vol_daily * np.sqrt(252)
    port_ret_ann = returns_df.mean() @ w * 252
    port_sharpe = (port_ret_ann - RISK_FREE_RATE) / port_vol_ann
    return pct_contrib, port_vol_ann, port_sharpe


def run_analysis(eden_assets, candidates):
    results = {}
    sharpe_dict = {}
    vol_dict = {}

    eden_total = sum(a['market value of shares'] for a in eden_assets)

    prices = build_price_matrix(eden_assets, candidate=None)
    returns = compute_returns(prices)
    weights = np.array([a['market value of shares'] / eden_total for a in eden_assets])

    contrib, vol, sharpe = risk_decomposition(returns, weights)
    results['Current'] = contrib
    sharpe_dict['Current'] = sharpe
    vol_dict['Current'] = vol

    print(f"\nBaseline portfolio")
    print(f"  Vol (ann.): {vol * 100:.2f}%   Sharpe: {sharpe:.3f}")
    print(contrib.round(2).to_string())

    for cand in candidates:
        label = cand['ticker']
        prices = build_price_matrix(eden_assets, candidate=cand)
        returns = compute_returns(prices)
        new_total = eden_total + cand['market value of shares']
        weights = np.array(
            [a['market value of shares'] / new_total for a in eden_assets] +
            [cand['market value of shares'] / new_total]
        )
        contrib, vol, sharpe = risk_decomposition(returns, weights)
        contrib = contrib.rename({'Candidate': label})
        results[label] = contrib
        sharpe_dict[label] = sharpe
        vol_dict[label] = vol

        print(f"\n+ {label}")
        print(f"  Vol (ann.): {vol * 100:.2f}%   Sharpe: {sharpe:.3f}")
        print(contrib.round(2).to_string())

    return results, sharpe_dict, vol_dict


COLORS = {
    'Current': '#546E7A',
    'Orion Chips': '#E65100',
    'Cleanergy': '#2E7D32',
    'NovaTerra AI': '#1565C0',
}

DARK_BG = '#111111'
PANEL_BG = '#1C1C1C'
GRID_COL = '#2E2E2E'
TEXT_COL = '#DDDDDD'
SPINE_COL = '#3A3A3A'


def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    ax.yaxis.grid(True, color=GRID_COL, linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)


def build_table(results):
    eden_assets = ['ASML', 'GLD', 'NOVO-B', 'PVH', 'XLU']
    cand_labels = ['Orion Chips', 'Cleanergy', 'NovaTerra AI']
    row_order = eden_assets + cand_labels
    table = pd.DataFrame(index=row_order)
    for scenario, contrib in results.items():
        col = {}
        for asset in eden_assets:
            col[asset] = contrib.get(asset, np.nan)
        if scenario != 'Current':
            col[scenario] = contrib.get(scenario, np.nan)
        table[scenario] = pd.Series(col)
    return table, row_order, eden_assets, cand_labels


def plot_absolute_contributions(results):
    table, row_order, eden_assets, cand_labels = build_table(results)
    scenarios = list(results.keys())
    equal_risk = 100 / 6

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    x = np.arange(len(row_order))
    bar_width = 0.18

    for i, scenario in enumerate(scenarios):
        vals = [table.loc[row, scenario] if row in table.index else np.nan
                for row in row_order]
        plot_vals = [v if not (isinstance(v, float) and np.isnan(v)) else 0 for v in vals]
        offset = (i - len(scenarios) / 2 + 0.5) * bar_width
        ax.bar(x + offset, plot_vals, bar_width,
               label=scenario, color=COLORS[scenario], alpha=0.9)

    ax.axhline(y=equal_risk, linestyle='--', color='white', linewidth=1.0,
               label=f'Equal risk parity ({equal_risk:.1f}%)')

    ax.axvline(x=len(eden_assets) - 0.5, color=SPINE_COL,
               linewidth=1.0, linestyle=':')

    ax.text(len(eden_assets) - 0.5 - 2.3, ax.get_ylim()[1] * 0.92,
            'Existing holdings', color=TEXT_COL, fontsize=8, alpha=0.7)
    ax.text(len(eden_assets) - 0.5 + 0.1, ax.get_ylim()[1] * 0.92,
            'Candidates', color=TEXT_COL, fontsize=8, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(row_order, rotation=20, ha='right')
    ax.set_ylabel('Risk Contribution (%)')
    ax.set_title('Component Risk Contribution by Asset and Scenario', pad=12)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL,
              edgecolor=SPINE_COL, loc='upper left')

    plt.tight_layout()
    plt.savefig('chart1_risk_contributions.png', dpi=150,
                bbox_inches='tight', facecolor=DARK_BG)
    plt.show()


def plot_delta(results):
    table, row_order, eden_assets, cand_labels = build_table(results)
    scenarios = list(results.keys())
    cand_scenarios = [s for s in scenarios if s != 'Current']
    baseline = table['Current'].fillna(0)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    x = np.arange(len(row_order))
    w = 0.26

    for i, scenario in enumerate(cand_scenarios):
        col = table[scenario].fillna(0)
        delta = col - baseline
        offset = (i - len(cand_scenarios) / 2 + 0.5) * w
        ax.bar(x + offset, delta, w,
               label=scenario, color=COLORS[scenario], alpha=0.9)

    ax.axhline(y=0, color=TEXT_COL, linewidth=0.8)
    ax.axvline(x=len(eden_assets) - 0.5, color=SPINE_COL,
               linewidth=1.0, linestyle=':')

    ax.text(len(eden_assets) - 0.5 - 2.3,
            ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else ax.get_ylim()[0] * 0.9,
            'Existing holdings', color=TEXT_COL, fontsize=8, alpha=0.7)
    ax.text(len(eden_assets) - 0.5 + 0.1,
            ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else ax.get_ylim()[0] * 0.9,
            'Candidates', color=TEXT_COL, fontsize=8, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(row_order, rotation=20, ha='right')
    ax.set_ylabel('Change in risk contribution vs baseline (pp)')
    ax.set_title('Risk Contribution Change vs Current Portfolio', pad=12)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL,
              edgecolor=SPINE_COL)

    plt.tight_layout()
    plt.savefig('chart2_risk_delta.png', dpi=150,
                bbox_inches='tight', facecolor=DARK_BG)
    plt.show()


def plot_sharpe(sharpe_dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style_ax(ax)

    scenarios = list(sharpe_dict.keys())
    sharpes = list(sharpe_dict.values())
    baseline = sharpes[0]

    bar_colors = []
    for i, (s, v) in enumerate(zip(scenarios, sharpes)):
        if i == 0:
            bar_colors.append(COLORS['Current'])
        elif v > baseline:
            bar_colors.append('#388E3C')
        else:
            bar_colors.append('#C62828')

    bars = ax.bar(scenarios, sharpes, color=bar_colors, alpha=0.9, width=0.5)

    ax.axhline(y=baseline, linestyle='--', color='white', linewidth=1.0,
               label=f'Baseline ({baseline:.3f})')

    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom',
                color=TEXT_COL, fontsize=10)

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.set_ylabel('Annualised Sharpe Ratio')
    ax.set_title('Portfolio Sharpe Ratio by Scenario', pad=12)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL,
              edgecolor=SPINE_COL)

    plt.tight_layout()
    plt.savefig('chart3_sharpe.png', dpi=150,
                bbox_inches='tight', facecolor=DARK_BG)
    plt.show()


def main():
    eden_assets = [
        {'ticker': 'ASML', 'market value of shares': 5000, 'path': os.path.join(DATA_DIR, 'ASML_1d_data.csv')},
        {'ticker': 'NOVO-B', 'market value of shares': 5000, 'path': os.path.join(DATA_DIR, 'NOVO-B.CO_1d_data.csv')},
        {'ticker': 'PVH', 'market value of shares': 7500, 'path': os.path.join(DATA_DIR, 'PVH_1d_data.csv')},
        {'ticker': 'XLU', 'market value of shares': 7500, 'path': os.path.join(DATA_DIR, 'XLU_1d_data.csv')},
        {'ticker': 'GLD', 'market value of shares': 10000, 'path': os.path.join(DATA_DIR, 'GLD_1d_data.csv')},
    ]

    candidates = [
        {'ticker': 'Orion Chips', 'market value of shares': 5000, 'path': os.path.join(DATA_DIR, 'orion_eod.csv')},
        {'ticker': 'Cleanergy', 'market value of shares': 5000, 'path': os.path.join(DATA_DIR, 'cleanergy_eod.csv')},
        {'ticker': 'NovaTerra AI', 'market value of shares': 5000, 'path': os.path.join(DATA_DIR, 'novaterra_eod.csv')},
    ]

    results, sharpe_dict, vol_dict = run_analysis(eden_assets, candidates)

    plot_absolute_contributions(results)
    plot_delta(results)
    plot_sharpe(sharpe_dict)


if __name__ == '__main__':
    main()

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# fix total investment for current portfolio
# be sure dates from oldest to newest 
  
def csv_cleaning(df):
    ''' formatting the csv files to have the same structure '''
    df = df.drop(['Open','High','Low', 'Adj Close', 'Volume'], axis = 1)
    df = df.iloc[::-1].reset_index(drop = True)
    df['Date'] = pd.to_datetime(df['Date']).astype(str)
    return df

def calculate_return(df):
    ''' calculate the return of the csv files '''

    returns = df.copy()
    returns = returns.set_index('Date')
    # pandas function to calculate return ( new - old ) / old
    returns = returns.pct_change()
    # dropping missing data because the first value is missing
    returns = returns.dropna()
    return returns

def compute_risk_contribution_analysis(test_portfolio, current_portfolio, portfolio_weights, price_matrix, candidate):

        for asset in test_portfolio:
            # computing the weight of the asset
            total_value = sum(asset['market value of shares']for asset in test_portfolio)
            portfolio_weights.append(asset['market value of shares']/total_value)
            asset_df = pd.read_csv(asset['path'])
            if asset in current_portfolio: 
                asset_df = csv_cleaning(asset_df)
            asset_df = asset_df.rename(columns = {'Close':asset['ticker']})
            asset_df = asset_df.rename (columns = {candidate: 'Candidate' })
            asset_df = asset_df.sort_values("Date")
            # building a price matrix with the close price for each day for all the assets.
            price_matrix = price_matrix.merge(asset_df, how='outer')
        # dropping missing values because not all the csv files start from the same date
        # we include only the dates where we have information of all the assets. 
        price_matrix = price_matrix.dropna()
        print(portfolio_weights)

        return_df = calculate_return(price_matrix)
        # buildind the covariance matrix to see if their return follow a similar trend
        cov_matrix = return_df.cov()
        portfolio_weights = np.array(portfolio_weights)
        portfolio_volatility = math.sqrt(portfolio_weights @ cov_matrix @ portfolio_weights)
        marginal_risk_contribution = (cov_matrix @ portfolio_weights)/portfolio_volatility

        risk_contribution = portfolio_weights * marginal_risk_contribution

        print("risk contribution\n", risk_contribution)
        # we remove 1 because we do not want to include the "Date" column. 
        target_risk_contribution = portfolio_volatility / (len(price_matrix.columns)-1)
        print("target risk contribution\n", target_risk_contribution)

        print("actual risk - target risk\n", risk_contribution-target_risk_contribution)
        risk_share = risk_contribution / portfolio_volatility
        print("risk_contribution / portfolio_volatility\n", risk_share)
        print("risk_contribution/target_risk_contribution\n", risk_contribution/target_risk_contribution)

        return risk_share

def apply_plot_style():
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "font.family": "DejaVu Sans"
    })

def plot_stacked_risk_contributions(risk_share_table: pd.DataFrame):
    apply_plot_style()

    colors = [
    "#D4E6F1",  # ASML
    "#A9CCE3",  # NOVO-B
    "#7FB3D5",  # PVH
    "#5499C7",  # XLU
    "#2471A3",  # GLD
    "#154360"   # Candidate
]
    plot_df = risk_share_table.fillna(0).T

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_df.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    width=0.72,
    color=colors
)

    ax.set_title("Portfolio Risk Composition")
    ax.set_xlabel("")
    ax.set_ylabel("Risk contribution (%)")
    ax.grid(axis="y")
    ax.legend(title="Asset", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")

    plt.tight_layout()
    plt.show()

def table_for_presentation(risk_share_table):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    table = ax.table(
        cellText=risk_share_table.round(3).values,
        rowLabels=risk_share_table.index,
        colLabels=risk_share_table.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    plt.title(
        "Risk Contribution Comparison",
        fontsize=16,
        weight='bold'
    )

    plt.tight_layout()
    plt.show()

    
def main():
    current_portfolio = [
        {'ticker': 'ASML', 'market value of shares': 5000, 'path': 'ASML_1d_data.csv'},
        {'ticker': 'NOVO-B', 'market value of shares': 5000, 'path': 'NOVO-B.CO_1d_data.csv' },
        {'ticker': 'PVH', 'market value of shares': 7500, 'path': 'PVH_1d_data.csv'},
        {'ticker': 'XLU', 'market value of shares': 7500, 'path': 'XLU_1d_data.csv' },
        {'ticker': 'GLD', 'market value of shares': 10000, 'path': 'GLD_1d_data.csv'}  
    ]

    possible_stocks = [
    { 'ticker': 'Orion Chips', 'market value of shares': 5000, 'path': 'orion_eod.csv'},
    {'ticker': 'Cleanergy', 'market value of shares': 5000, 'path': 'cleanergy_eod.csv'},
    {'ticker': 'NovaTerra AI', 'market value of shares': 5000, 'path': 'novaterra_eod.csv'}
    ]
    price_matrix = pd.DataFrame(columns=('Date',))
    portfolio_weights = []
    results = {}
    results['current portfolio'] = compute_risk_contribution_analysis(current_portfolio, current_portfolio, portfolio_weights, price_matrix, '')
    # building 3 different test_portfolio, one for each scenario. 
    for candidate in possible_stocks: 
        test_portfolio = current_portfolio + [candidate]
        price_matrix = pd.DataFrame(columns=('Date',))
        portfolio_weights = []
        results[candidate['ticker']] = compute_risk_contribution_analysis(test_portfolio, current_portfolio, portfolio_weights, price_matrix, candidate['ticker'])
    # reordering rows for visual purposes
    table = pd.DataFrame(results)
    candidate_row = table.loc[['Candidate']]
    table = table.drop('Candidate')
    table = pd.concat([table, candidate_row])
    print(table)
    plot_stacked_risk_contributions(table)
    table_for_presentation(table)

main()

    



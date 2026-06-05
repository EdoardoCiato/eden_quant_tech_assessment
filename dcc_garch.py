# Libraries to use in the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import math



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

def fit_univariate_garch(return_series):
    # Clean the series
    y = pd.to_numeric(return_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    # Optional but common in finance: scale returns to percentages
    y = y * 100

    # Fit GARCH(1,1)
    am = arch_model(
        y,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )
    res = am.fit(disp="off")

    # Conditional volatility and standardized residuals
    cond_vol = res.conditional_volatility
    std_resid = res.std_resid

    return res, cond_vol, std_resid

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
    portfolio_weights =[]
    total_value = sum(asset['market value of shares']for asset in current_portfolio)
    for asset in current_portfolio:
        # computing the weight of the asset
        portfolio_weights.append(asset['market value of shares']/total_value)
        asset_df = pd.read_csv(asset['path'])
        asset_df = csv_cleaning(asset_df)
        asset_df = asset_df.rename(columns = {'Close':asset['ticker']})
        # building a price matrix with the close price for each day for all the assets.
        price_matrix = price_matrix.merge(asset_df, on="Date", how="outer")
    # dropping missing values because not all the csv files start from the same date
    # we include only the dates where we have information of all the assets. 
    price_matrix = price_matrix.sort_values("Date").reset_index(drop=True)
    price_matrix = price_matrix.dropna()
    return_df = calculate_return(price_matrix)
    portfolio_weights = np.array(portfolio_weights)
    portfolio_return =  return_df @ portfolio_weights

    portfolio_res, portfolio_cond_vol, portfolio_std_resid = fit_univariate_garch(portfolio_return)
    
    for candidate in possible_stocks:
         
        candidate_series = pd.read_csv(candidate['path'])
        candidate_series['Date'] = pd.to_datetime(candidate_series['Date']).astype(str)
        candidate_return = calculate_return(candidate_series)
        candidate_return = candidate_return.squeeze()
        candidate_res, candidate_cond_vol, candidate_std_resid = fit_univariate_garch(candidate_return)
        portfolio_std_resid = pd.DataFrame(portfolio_std_resid).rename(columns = {'std_resid': 'portfolio'})
        std_res_matrix = portfolio_std_resid.merge(candidate_std_resid, how='outer', on = 'Date')
        std_res_matrix = std_res_matrix.rename(columns = {'std_resid': candidate['ticker']})
        std_res_matrix = std_res_matrix.dropna()
                        
        long_run_correlation = std_res_matrix.T @ std_res_matrix 
        long_run_correlation = long_run_correlation / len(std_res_matrix['portfolio'])
        print(long_run_correlation.shape)

        dynamic_matrix = long_run_correlation
        for i in len(std_res_matrix['portfolio']):
             
             





        
main()

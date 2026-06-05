import pandas as pd
def hhi_index(market_shares):
     hhi = sum(share ** 2 for share in market_shares)
     return round(hhi)

def main():
    market_shares = [
        {'ticker': 'Orion', 'market share': [31, 22, 19, 14, 9, 5]},
        {'ticker': 'Cleanergy', 'market share': [32, 24, 18, 14, 8, 4]},
        {'ticker': 'NovaTerra AI', 'market share': [28, 25, 18, 17, 8, 4]}
    ]
    results = {}
    for candidate in market_shares:
        results[candidate['ticker']] = hhi_index(candidate['market share'])
    print(results)
    results = pd.Series(results)
    print(results)

main()
import pandas as pd
from mlfinlab.online_portfolio_selection import BAH


def main():
    asset_price = pd.read_csv("../tests/test_data/stock_prices.csv")
    asset_name = list(asset_price.columns)
    bah = BAH()

if __name__ == "__main__":
    main()
'''
This is a sub-module of the portfolio-optimisation module for estimation of expected returns
'''

class ReturnsEstimation:
    '''
    This class contains methods for estimating expected returns. A good estimation of the expected returns is very important
    for portfolio optimisation problems and so it is necessary to use good estimates of returns and not just rely on
    simple techniques.
    '''

    def __init__(self):
        return

    @staticmethod
    def calculate_mean_historical_returns(asset_prices, resample_by=None, frequency=252):
        '''
        Calculate the annualised mean historical returns from asset price data

        :param asset_prices: (pd.DataFrame) asset price data
        :return: (np.array) returns per asset
        '''

        # Resample the asset prices
        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.mean() * frequency
        return returns

    @staticmethod
    def calculate_exponential_historical_returns(asset_prices, resample_by=None, frequency=252, span=500):
        '''
        Calculate the exponentially-weighted mean of (daily) historical returns, giving
        higher weight to more recent data.

        :param asset_prices: (pd.DataFrame) asset price data
        :return: (np.array) returns per asset
        '''

        # Resample the asset prices
        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.ewm(span=span).mean().iloc[-1] * frequency
        return returns

    @staticmethod
    def calculate_returns(asset_prices, resample_by=None):
        '''
        Calculate the annualised mean historical returns from asset price data

        :param asset_prices: (pd.Dataframe) a dataframe of historical asset prices (daily close)
        :param resample_by: (str) specifies how to resample the prices - weekly, daily, monthly etc.. Defaults to
                                  None for no resampling
        :return: (pd.Dataframe) stock returns
        '''

        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        asset_returns = asset_prices.pct_change()
        asset_returns = asset_returns.dropna(how='all')
        return asset_returns

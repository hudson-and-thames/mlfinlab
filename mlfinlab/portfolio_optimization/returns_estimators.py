

# pylint: disable=missing-module-docstring
class ReturnsEstimation:
    """
    This class contains methods for estimating expected returns. A good estimation of the asset expected returns is very important
    for portfolio optimisation problems and so it is necessary to use good estimates of returns and not just rely on
    simple techniques.
    """

    def __init__(self):
        return

    @staticmethod
    def calculate_mean_historical_returns(asset_prices, resample_by=None, frequency=252):
        """
        Calculates the annualised mean historical returns from asset price data.

        :param asset_prices: (pd.DataFrame) Asset price data
        :param resample_by: (str) Period to resample data ['D','W','M' etc.] None for no resampling
        :param frequency: (int) Average number of observations per year
        :return: (pd.Series) Annualized mean historical returns per asset
        """

        # Resample the asset prices
        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.mean() * frequency
        return returns

    @staticmethod
    def calculate_exponential_historical_returns(asset_prices, resample_by=None, frequency=252, span=500):
        """
        Calculates the exponentially-weighted annualized mean of historical returns, giving
        higher weight to more recent data.

        :param asset_prices: (pd.DataFrame) Asset price data
        :param resample_by: (str) Period to resample data ['D','W','M' etc.] None for no resampling
        :param frequency: (int) Average number of observations per year
        :param span: (int) Window length to use in pandas ewm function
        :return: (pd.Series) Exponentially-weighted mean of historical returns
        """

        # Resample the asset prices
        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.ewm(span=span).mean().iloc[-1] * frequency
        return returns

    @staticmethod
    def calculate_returns(asset_prices, resample_by=None):
        """
        Calculates a dataframe of returns from a dataframe of prices.

        :param asset_prices: (pd.Dataframe) Historical asset prices
        :param resample_by: (str) Period to resample data ['D','W','M' etc.] None for no resampling
        :return: (pd.Dataframe) Returns per asset
        """

        if resample_by:
            asset_prices = asset_prices.resample(resample_by).last()
        asset_returns = asset_prices.pct_change()
        asset_returns = asset_returns.dropna(how='all')
        return asset_returns

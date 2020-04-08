import pandas as pd
import numpy as np


# initial check
# def initial_check(asset_prices, expected_asset_returns, covariance_matrix):
#     if asset_prices is None and (expected_asset_returns is None or covariance_matrix is None):
#         raise ValueError("Either supply your own asset returns matrix or pass the asset prices as input")
#
#     if asset_prices is not None:
#         if not isinstance(asset_prices, pd.DataFrame):
#             raise ValueError("Asset prices matrix must be a dataframe")
#         if not isinstance(asset_prices.index, pd.DatetimeIndex):
#             raise ValueError("Asset prices dataframe must be indexed by date.")


# utility methods
def calculate_covariance(asset_names, asset_prices, covariance_matrix, resample_by, returns_estimator):
    # Calculate covariance of returns or use the user specified covariance matrix
    if covariance_matrix is None:
        returns = returns_estimator.calculate_returns(asset_prices=asset_prices, resample_by=resample_by)
        covariance_matrix = returns.cov()
    covariance = pd.DataFrame(covariance_matrix, index=asset_names, columns=asset_names)

    return covariance_matrix


# Calculate the expected returns if the user does not supply any returns
def calculate_expected_asset_returns(self, asset_prices, expected_asset_returns, resample_by):
    if expected_asset_returns is None:
        if expected_returns == "mean":
            expected_asset_returns = self.returns_estimator.calculate_mean_historical_returns(
                    asset_prices=asset_prices,
                    resample_by=resample_by)
        elif self.calculate_expected_returns == "exponential":
            expected_asset_returns = self.returns_estimator.calculate_exponential_historical_returns(
                    asset_prices=asset_prices,
                    resample_by=resample_by)
        else:
            raise ValueError("Unknown returns specified. Supported returns - mean, exponential")

    expected_asset_returns = np.array(expected_asset_returns).reshape((len(expected_asset_returns), 1))
    return expected_asset_returns


# Calculate the portfolio risk and return if it has not been calculated
def calculate_portfolio_risk(portfolio_risk, covariance_matrix, weights):
    if portfolio_risk is None:
        portfolio_risk = np.dot(weights, np.dot(covariance_matrix.values, weights.T))
    return portfolio_risk


# Calculate the portfolio return
def calculate_portfolio_return(portfolio_return, weights, expected_asset_returns):
    if portfolio_return is None:
        portfolio_return = np.dot(weights, expected_asset_returns)
    return portfolio_return

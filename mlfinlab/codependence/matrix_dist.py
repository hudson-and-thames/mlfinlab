import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import mlfinlab as ml

def kl_dist(p, q):
    """
    Calculates the Kullback-Leibler distance between two coor matricies.
    
    """
    #kl_div formula
    #np.sum(np.where(p != 0, p * np.log(p / q), 0))
    #import scipy.special as sc
    n = np.trace(np.linalg.inv(q).dot(q))
    tmp = 0.5 * (np.log( np.linalg.det(q) / np.linalg.det(p) ) + np.trace((np.linalg.inv(q).dot(p)) - n ) )
    return tmp
    #return sc.kl_div(p.to_numpy(), q.to_numpy())
    
def norm_dist(p, q ,d=2):
    """
    Calculates the norm distance between two matricies
    :param: d (default=2) for Frobenius distance
    """
    return np.linalg.norm(q - p, d)

stock_prices = pd.read_csv('../tests/test_data/stock_prices.csv', parse_dates=True, index_col='Date')
stock_prices = stock_prices.dropna(axis=1)
stock_prices = stock_prices.iloc[:, :5]
tn_relation = stock_prices.shape[0] / stock_prices.shape[1]
stock_prices.head()



# A class that has the Minimum Covariance Determinant estimator
risk_estimators = ml.portfolio_optimization.RiskEstimators()

# Finding the Minimum Covariance Determinant estimator on price data and with set random seed to 0
min_cov_det = risk_estimators.minimum_covariance_determinant(stock_prices, price_data=True, random_state=0)

# For the simple covariance, we need to transform the stock prices to returns

# A class with function to calculate returns from prices
returns_estimation = ml.portfolio_optimization.ReturnsEstimators()

# Calcualting the data set of returns
stock_returns = returns_estimation.calculate_returns(stock_prices)

# Finding the simple covariance matrix from a series of returns
cov_matrix = stock_returns.cov()

# Transforming Minimum Covariance Determinant estimator from np.array to pd.DataFrame
min_cov_det = pd.DataFrame(min_cov_det, index=cov_matrix.index, columns=cov_matrix.columns)

print('The Minimum Covariance Determinant estimator is:')
min_cov_det

corr = ml.portfolio_optimization.risk_estimators.RiskEstimators.cov_to_corr(cov_matrix)
kl_dist(corr, corr)

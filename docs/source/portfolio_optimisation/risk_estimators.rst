.. _portfolio_optimisation-risk_estimators:


===============
Risk Estimators
===============
This class includes functions for different ways of calculating and adjusting covariance matrices, de-noising the
covariance and the correlation matrices, and other helpful methods.

.. tip::
   **Underlying Literature**

   The following sources elaborate extensively on the topic:

   - **Scikit-learn User Guide on Covariance estimation** `available here <https://scikit-learn.org/stable/modules/covariance.html>`__. *Describes covariance estimators in this class*
   - **Machine Learning for Asset Managers** *by* Marcos Lopez de Prado `available here <https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545>`__. *Chapter 2 describes the motivation and the algorithm of de-noising and de-toning the covariance matrix*
   - **Financial applications of random matrix theory: Old laces and new pieces** *by* Potter M.,  J.P. Bouchaud *and* L. Laloux `available here <https://arxiv.org/abs/physics/0507111>`__. *Describes the process of de-noising the covariance matrix*

Minimum Covariance Determinant
##############################

Minimum Covariance Determinant (MCD) is a robust estimator of covariance.

Description of the MCD according to the **Scikit-learn User Guide on Covariance estimation**:

"The idea is to find a given proportion (h) of “good” observations that are not outliers and compute their empirical
covariance matrix. This empirical covariance matrix is then rescaled to compensate for the performed selection of observations".

Our method is a wrapper for the sklearn MinCovDet class. For more details about the function and its parameters, please
visit `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html>`__.

Maximum Likelihood Covariance Estimator (Empirical Covariance)
##############################################################

Maximum Likelihood Estimator of a sample is an unbiased estimator of the corresponding population’s covariance matrix.

Description of the Empirical Covariance according to the **Scikit-learn User Guide on Covariance estimation**:

"The covariance matrix of a data set is known to be well approximated by the classical maximum likelihood estimator,
provided the number of observations is large enough compared to the number of features (the variables describing the
observations). More precisely, the Maximum Likelihood Estimator of a sample is an unbiased estimator of the corresponding
population’s covariance matrix".

Our method is a wrapper for the sklearn EmpiricalCovariance class. For more details about the function and its parameters,
please visit `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html>`__.

Covariance Estimator with Shrinkage
###################################

Shrinkage allows one to avoid the inability to invert the covariance matrix due to numerical reasons. Shrinkage consists
of reducing the ratio between the smallest and the largest eigenvalues of the empirical covariance matrix.

The following Shrinkage methods are supported:

- Basic shrinkage
- Ledoit-Wolf shrinkage
- Oracle Approximating shrinkage

Our methods here are wrappers for the sklearn ShrunkCovariance, LedoitWolf, and OAS classes.
For more details about the function and its parameters, please visit `sklearn documentation <https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance>`__.

Semi-Covariance Matrix
#######################

The Semi-Covariance matrix is used to measure the downside volatility of a portfolio and can be used as a measure to minimize it.
This metric also allows to measure the volatility of returns below a specific threshold. An example of Semi-Covaraicne usage
can be found `here <https://www.solactive.com/wp-content/uploads/2018/04/Solactive_Minimum-Downside-Volatility-Indices.pdf>`__.

According to the above-mentioned paper:
"Each element in the Semi-Covariance matrix is calculated as:

.. math::
    SemiCov_{ij} = \frac{1}{T}\sum_{t=1}^{T}[Min(R_{i,t}-B,0)*Min(R_{j,t}-B,0)]

where :math:`T` is the number of observations, :math:`R_{i,t}` is the return of an asset :math:`i` at time :math:`t`, and :math:`B` is the threshold return.

If the :math:`B` is set to zero, the volatility of negative returns is measured".

Exponentially-Weighted Covariance Matrix
########################################

Each element in the Exponentially-weighted Covariance matrix is the last element from an exponentially weighted moving average
a series based on series of covariances between returns of the corresponding assets. It's used to give greater weight to most
relevant observations in computing the covariance.

First, we calculate the series of covariances for every observation time :math:`t` between each two elements :math:`i` and :math:`j`:

.. math::
    CovarSeries_{i,j}^{t} = (R_{i}^{t} - Mean(R_{i})) * (R_{j}^{t} - Mean(R_{j}))

Then we apply the exponential weighted moving average based on the obtained series with decay in terms of span, as :math:`\alpha=\frac{2}{span+1}`, for :math:`span \ge 1`

.. math::
    ExponentialCovariance_{i,j} = ExponentialWeightedMovingAverage(CovarSeries_{i,j})[T]

.. tip::
    This and above methods are described in more detail in the Risk Estimators Notebook.

De-noising the Covariance and the Correlation Matrices
######################################################

The main idea behind de-noising is to separate the noise-related eigenvalues from the signal-related ones. This is achieved
by fitting the Marcenko-Pastur distribution of the empirical distribution of eigenvalues using a Kernel Density Estimate (KDE).

The de-noising algorithm works as follows:

1. A correlation is calculated from the covariance matrix (if the input is the covariance matrix).

2. Eigenvalues and eigenvectors of the correlation matrix are calculated.

3. A maximum theoretical eigenvalue is found by fitting Marcenko-Pastur distribution to the empirical distribution of the correlation matrix eigenvalues. The empirical distribution is obtained through kernel density estimation. The fit of the M-P distribution is done by minimizing the Sum of Squared estimate of Errors between the theoretical pdf and the kernel. The minimization is done by adjusting the variation of the M-P distribution.

4. The eigenvalues of the correlation matrix are sorted and the eigenvalues higher than the maximum theoretical eigenvalue are set to their average value. This is how the eigenvalues associated with noise are shrinked. The de-noised covariance matrix is then calculated back from new eigenvalues and eigenvectors.

.. tip::

    This algorithm is described in more detail in the work **Financial applications of random matrix theory: Old laces and new pieces** *by* Potter M.,  J.P. Bouchaud *and* L. Laloux `available here <https://arxiv.org/abs/physics/0507111>`__.

    Examples of how to use this method are also available in the NCO Notebook.

Transforming Covariance to Correlation and Back
###############################################

These are simple methods for transforming the covariance matrix to the correlation matrix and back to the covariance matrix.

The formula for calculation of the correlation matrix from the covariance matrix:

.. math::

    D = \sqrt{diag(CovMatrix)}

    CorrMatrix = D^{-1} * CovMatrix * D^{-1}

The formula for calculation of the covariance matrix from the correlation matrix:

.. math::

    CovMatrix = D * CorrMatrix * D

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.risk_estimators

    .. autoclass:: RiskEstimators
        :members:

        .. automethod:: __init__

Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.portfolio_optimization.portfolio_optimization import RiskEstimators

    # Reading data
    stock_returns = pd.read_csv('DATA_FILE_PATH', parse_dates=True, index_col='Date')

    # The class that contains the TIC algorithm
    risk_est = RiskEstimators()

    # Calculating the Minimum Covariance Determinant estimator
    # We use price_data=False, as the input is dataframe with returns and not prices.
    mcd_cov = risk_est.minimum_covariance_determinant(stock_returns, price_data=False)

    # Calculating the Empirical Covariance estimator
    empirical_cov = risk_est.empirical_covariance(stock_returns, price_data=False)

    # Calculating the Shrinked Covariances on price data with every method
    # The alpha for Basic Shrinkage used is 0.1
    shrinked_cov = risk_est.shrinked_covariance(stock_returns, price_data=False,
                                                shrinkage_type='all', basic_shrinkage=0.1)

    # Separating the Shrinked covariances for every method
    shrinked_cov_basic, shrinked_cov_lw, shrinked_cov_oas = shrinked_cov

    # Calculating the Semi-Covariance with a threshold return of 0
    semi_cov = risk_est.semi_covariance(stock_returns, price_data=False, threshold_return=0)

    # Calculating the Exponential Covariance for a span of 60
    exponential_cov = risk_est.exponential_covariance(stock_returns, price_data=False, window_span=60)

    # Relation of number of observations T to the number of variables N (T/N)
    tn_relation = stock_returns.shape[0] / stock_returns.shape[1]

    # Finding the simple covariance matrix from a series of returns
    cov_matrix = stock_returns.cov()

    # Finding the De-noised Сovariance matrix
    cov_matrix_denoised = risk_est.denoise_covariance(cov_matrix, tn_relation)

    # Finding the simple correlation matrix from a series of returns
    corr_matrix = stock_returns.corr()

    # Finding the De-noised Correlation matrix
    corr_matrix_denoised = risk_est.denoise_covariance(corr_matrix, tn_relation)

    # Transforming our covariance matrix to a correlation matrix
    corr_matrix = risk_est.cov_to_corr(cov_matrix)

    # The standard deviation to use when calculating the covaraince matrix back
    std = np.diag(cov_matrix) ** (1/2)

    # And back to the covariance matrix
    cov_matrix_again = risk_est.corr_to_cov(corr_matrix, std)

Research Notebooks
##################

The following research notebook can be used to better understand how the algorithms within this module can be used on real data.

* `Risk Estimators Notebook`_

.. _Risk Estimators Notebook: https://github.com/hudson-and-thames/research/blob/master/RiskEstimators/RiskEstimators.ipynb


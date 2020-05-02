.. _portfolio_optimisation-risk_estimators:


===============
Risk Estimators
===============
This class includes functions for calculating different types of covariance matrices, de-noising, and other helpful methods.

.. tip::
   **Underlying Literature**

   The following sources elaborate extensively on the topic:

   - **Scikit-learn User Guide on Covariance estimation** `available here <https://scikit-learn.org/stable/modules/covariance.html#robust-covariance>`__. *Describes the algorithms of covariance matrix estimators in more detail.*
   - **Minimum covariance determinant** *by* Mia Hubert *and* Michiel Debruyne `available here <https://wis.kuleuven.be/stat/robust/papers/2010/wire-mcd.pdf>`__. *Detailed description of the minimum covariance determinant (MCD) estimator.*
   - **A well-conditioned estimator for large-dimensional covariance matrices** *by* Olivier Ledoit *and* Michael `available here <http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf>`__. *Introduces the Ledoit-Wolf shrinkage method.*
   - **Shrinkage Algorithms for MMSE Covariance Estimation** *by* Y. Chen, A. Wiesel, Y.C. Eldar and A.O. Hero `available here <https://webee.technion.ac.il/people/YoninaEldar/104.pdf>`__. *Introduces the Oracle Approximating shrinkage method.*
   - **Minimum Downside Volatility Indices** *by* Solactive AG - German Index Engineering `available here <https://www.solactive.com/wp-content/uploads/2018/04/Solactive_Minimum-Downside-Volatility-Indices.pdf>`__. *Describes examples of use of the Semi-Covariance matrix.*
   - **Financial applications of random matrix theory: Old laces and new pieces** *by* Potter M., J.P. Bouchaud, L. Laloux `available here <https://arxiv.org/abs/physics/0507111>`__. *Describes the process of de-noising of the covariance matrix.*
   - **A Robust Estimator of the Efficient Frontier** *by* Marcos Lopez de Prado `available here <https://papers.ssrn.com/sol3/abstract_id=3469961>`__. *Describes the De-noising Covariance/Correlation Matrix algorithm.*

Minimum Covariance Determinant
##############################

Minimum Covariance Determinant (MCD) is a robust estimator of covariance that was introduced by P.J. Rousseeuw.

Following the **Scikit-learn User Guide on Covariance estimation**:

"The outliers are appearing in real data sets and seriously affect the Empirical covariance estimator and the Covariance estimators with shrinkage.
For this reason, a robust covariance estimator is needed in order to discard/downweight the outliers in the data".

"The basic idea of the algorithm is to find a set of observations that are not outliers and compute their empirical covariance matrix,
which is then rescaled to compensate for the performed selection of observations".

Our method is a wrapper for the sklearn MinCovDet class. For more details about the function and its parameters, please
visit `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html>`__.

Maximum Likelihood Covariance Estimator (Empirical Covariance)
##############################################################

Maximum Likelihood Estimator of a sample is an unbiased estimator of the corresponding population’s covariance matrix.

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

**Basic shrinkage**

Following the **Scikit-learn User Guide on Covariance estimation**:

"This shrinkage is done by shifting every eigenvalue according to a given offset, which is equivalent to finding the l2-penalized Maximum Likelihood Estimator of the covariance matrix".

"Shrinkage boils down to a simple a convex transformation":

.. math::

      \sum_{shrunk} = (1 - \alpha)\sum_{unshrunk} + \alpha\frac{Tr \sum_{unshrunk}}{p}Id

"The amount of shrinkage :math:`\alpha` is setting a trade-off between bias and variance".

**Ledoit-Wolf shrinkage**

"The Ledoit-Wolf shrinkage is based on computing the optimal shrinkage coefficient :math:`\alpha` that minimizes the Mean Squared Error between the estimated and the real covariance matrix".

**Oracle Approximating shrinkage**

"Assuming that the data are Gaussian distributed, Chen et al. derived a formula aimed at choosing a shrinkage coefficient :math:`\alpha`
that yields a smaller Mean Squared Error than the one given by Ledoit and Wolf’s formula".

"The resulting estimator is known as the Oracle Shrinkage Approximating estimator of the covariance".

Our methods here are wrappers for the sklearn ShrunkCovariance, LedoitWolf, and OAS classes.
For more details about the function and its parameters, please visit `sklearn documentation <https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance>`__.

.. tip::

      Shrinkage methods are described in greater detail in the works listed in the introduction.


Semi-Covariance Matrix
#######################

Semi-Covariance matrix is used to measure the downside volatility of a portfolio and can be used as a measure to minimize it.
This metric also allows measuring the volatility of returns below a specific threshold.

According to the **Minimum Downside Volatility Indices** paper:

"Each element in the Semi-Covariance matrix is calculated as:

.. math::

      SemiCov_{ij} = \frac{1}{T}\sum_{t=1}^{T}[Min(R_{i,t}-B,0)*Min(R_{j,t}-B,0)]

where :math:`T` is the number of observations, :math:`R_{i,t}` is the return of an asset :math:`i` at time :math:`t`, and :math:`B` is the threshold return.
If the :math:`B` is set to zero, the volatility of negative returns is measured."


.. tip::
      An example of Semi-Covariance usage can be found `here <https://www.solactive.com/wp-content/uploads/2018/04/Solactive_Minimum-Downside-Volatility-Indices.pdf>`__.

Exponentially-Weighted Covariance Matrix
########################################

Each element in the Exponentially-weighted Covariance matrix is the last element from an exponentially weighted moving average
series based on series of covariances between returns of the corresponding assets. It's used to give greater weight to most
relevant observations in computing the covariance.

Each element is calculated as follows:

.. math::
      :nowrap:

      \begin{align*}
      Covar_{i,j}^{t} = (R_{i}^{t} - Mean(R_{i})) * (R_{j}^{t} - Mean(R_{j}))
      \end{align*}

      \begin{align*}
      Decay = \frac{2}{span+1}
      \end{align*}

      \begin{align*}
      EWMA(Covar_{i,j})_{t} = ((Covar_{i,j}^{t} - Covar_{i,j}^{t-1}) * Decay) + Covar_{i,j}^{t-1}
      \end{align*}

      \begin{align*}
      ExponentialCovariance_{i,j (Decay)} = EWMA(Covar)_{T}
      \end{align*}

Where :math:`R_{i}^{t}` is the return of :math:`i` -th asset for :math:`t` -th observation,
:math:`T` is the total number of observations, :math:`Covar_{i,j}` is the series of correlations between :math:`i` -th
and :math:`j` -th asset, :math:`EWMA(Covar)_{t}` is the :math:`t` -th observation of exponentially-weighted
moving average of :math:`Covar` .


De-noising Covariance/Correlation Matrix
########################################

The main idea behind de-noising is to separate the noise-related eigenvalues from the signal-related ones. This is achieved
through fitting the Marcenko-Pastur distribution of the empirical distribution of eigenvalues using a Kernel Density Estimate (KDE).

The de-noising function works as follows:

- The given covariance matrix is transformed to the correlation matrix.

- The eigenvalues and eigenvectors of the correlation matrix are calculated.

- Using the Kernel Density Estimate algorithm a kernel of the eigenvalues is estimated.

- The Marcenko-Pastur pdf is fitted to the KDE estimate using the variance as the parameter for the optimization.

- From the obtained Marcenko-Pastur distribution, the maximum theoretical eigenvalue is calculated using the formula from the "Instability caused by noise" part.

- The eigenvalues in the set that are above the theoretical value are all set to their average value. For example, we have a set of 5 sorted eigenvalues ( :math:`\lambda_1` ... :math:`\lambda_5` ), 2 of which are above the maximum theoretical value, then we set :math:`\lambda_4^{NEW} = \lambda_5^{NEW} = \frac{\lambda_4^{OLD} + \lambda_5^{OLD}}{2}`

- The new set of eigenvalues with the set of eigenvectors is used to obtain the new de-noised correlation matrix.

- The new correlation matrix is then transformed back to the new de-noised covariance matrix.

(If the correlation matrix is given as an input, the first and the last steps of the algorithm are omitted)

.. tip::

    The de-noising algorithm is described in more detail in the work **A Robust Estimator of the Efficient Frontier** *by* Marcos Lopez de Prado `available here <https://papers.ssrn.com/abstract_id=3469961>`_.

Transforming Covariance to Correlation and Back
#############################################################

Helper methods for transforming the covariance matrix to the correlation matrix and back to the covariance matrix.

.. tip::
    This and above the methods are described in more detail in the Risk Estimators Notebook.

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.risk_estimators

    .. autoclass:: RiskEstimators
        :members:

        .. automethod:: __init__


Example
########
Below is an example of using the functions from the Risk Estimators module.

.. code-block::

    import pandas as pd
    import numpy as np
    from mlfinlab.portfolio_optimization import RiskEstimators

    # Import price data
    stock_prices = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

    # A class that has needed functions
    risk_estimators = RiskEstimators()

    # Finding the MCD estimator on price data
    min_cov_det = risk_estimators.minimum_covariance_determinant(stock_prices, price_data=True)

    # Finding the Empirical Covariance on price data
    empirical_cov = risk_estimators.empirical_covariance(stock_prices, price_data=True)

    # Finding the Shrinked Covariances on price data with every method
    shrinked_cov = risk_estimators.shrinked_covariance(stock_prices, price_data=True,
                                                       shrinkage_type='all', basic_shrinkage=0.1)

    # Finding the Semi-Covariance on price data
    semi_cov = risk_estimators.semi_covariance(stock_prices, price_data=True, threshold_return=0)

    # Finding the Exponential Covariance on price data and span of 60
    exponential_cov = risk_estimators.exponential_covariance(stock_prices, price_data=True,
                                                             window_span=60)

    # Relation of number of observations T to the number of variables N (T/N)
    tn_relation = stock_prices.shape[0] / stock_prices.shape[1]

    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01

    # Finding the De-noised Сovariance matrix
    cov_matrix_denoised = risk_estimators.denoise_covariance(cov_matrix, tn_relation,
                                                             kde_bwidth)

Research Notebooks
##################

The following research notebook can be used to better understand how the algorithms within this module can be used on real data.

* `Risk Estimators Notebook`_

.. _Risk Estimators Notebook: https://github.com/hudson-and-thames/research/blob/master/RiskEstimators/RiskEstimators.ipynb


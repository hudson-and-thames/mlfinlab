.. _portfolio_optimisation-risk_estimators:

.. |br| raw:: html

    <br>

.. |h3| raw:: html

    <h3>

.. |h3_| raw:: html

    </h3>

.. |h4| raw:: html

    <h4>

.. |h4_| raw:: html

    </h4>

.. |h5| raw:: html

    <h5>

.. |h5_| raw:: html

    </h5>


===============
Risk Estimators
===============
Risk is a very important part of finance and the performance of large number of investment strategies are dependent on the
efficient estimation of underlying portfolio risk. There are different ways of representing risk but the most widely used is a
covariance matrix. This means that an accurate calculation of the covariances is essential for an accurate representation of risk.
This class provides functions for calculating different types of covariance matrices, de-noising, de-toning and other helpful methods.

.. tip::
   |h4| Underlying Literature |h4_|

   The following sources elaborate extensively on the topic:

   - **Scikit-learn User Guide on Covariance estimation** `available here <https://scikit-learn.org/stable/modules/covariance.html#robust-covariance>`__. *Describes the algorithms of covariance matrix estimators in more detail.*
   - **Minimum covariance determinant** *by* Mia Hubert *and* Michiel Debruyne `available here <https://wis.kuleuven.be/stat/robust/papers/2010/wire-mcd.pdf>`__. *Detailed description of the minimum covariance determinant (MCD) estimator.*
   - **A well-conditioned estimator for large-dimensional covariance matrices** *by* Olivier Ledoit *and* Michael `available here <http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf>`__. *Introduces the Ledoit-Wolf shrinkage method.*
   - **Shrinkage Algorithms for MMSE Covariance Estimation** *by* Y. Chen, A. Wiesel, Y.C. Eldar and A.O. Hero `available here <https://webee.technion.ac.il/people/YoninaEldar/104.pdf>`__. *Introduces the Oracle Approximating shrinkage method.*
   - **Minimum Downside Volatility Indices** *by* Solactive AG - German Index Engineering `available here <https://www.solactive.com/wp-content/uploads/2018/04/Solactive_Minimum-Downside-Volatility-Indices.pdf>`__. *Describes examples of use of the Semi-Covariance matrix.*
   - **Financial applications of random matrix theory: Old laces and new pieces** *by* Potter M., J.P. Bouchaud, L. Laloux `available here <https://arxiv.org/abs/physics/0507111>`__. *Describes the process of de-noising of the covariance matrix.*
   - **A Robust Estimator of the Efficient Frontier** *by* Marcos Lopez de Prado `available here <https://papers.ssrn.com/sol3/abstract_id=3469961>`__. *Describes the Constant Residual Eigenvalue Method for De-noising Covariance/Correlation Matrix.*
   - **Machine Learning for Asset Managers** *by* Marcos Lopez de Prado `available here <https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545>`__. *Describes the Targeted Shrinkage De-noising and the De-toning methods for Covariance/Correlation Matrices.*


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

Implementation
**************

.. py:currentmodule:: mlfinlab.portfolio_optimization.risk_estimators

.. autoclass:: RiskEstimators
   :members: __init__, minimum_covariance_determinant


----

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

Implementation
**************

.. autoclass:: RiskEstimators
   :noindex:
   :members: empirical_covariance


----

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

Implementation
**************

.. autoclass:: RiskEstimators
   :noindex:
   :members: shrinked_covariance


----

Semi-Covariance Matrix
######################

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

Implementation
**************

.. autoclass:: RiskEstimators
   :noindex:
   :members: semi_covariance


----

Exponentially-Weighted Covariance Matrix
########################################

Each element in the Exponentially-weighted Covariance matrix is the last element from an exponentially weighted moving average
series based on series of covariances between returns of the corresponding assets. It's used to give greater weight to most
relevant observations in computing the covariance.

Each element is calculated as follows:

.. math::
      :nowrap:

      \begin{align*}
      \sum_{i,j}^{t} = (R_{i}^{t} - Mean(R_{i})) * (R_{j}^{t} - Mean(R_{j}))
      \end{align*}

      \begin{align*}
      Decay = \frac{2}{span+1}
      \end{align*}

      \begin{align*}
      EWMA(\sum_{i,j})_{t} = ((\sum_{i,j}^{t} - \sum_{i,j}^{t-1}) * Decay) + \sum_{i,j}^{t-1}
      \end{align*}

      \begin{align*}
      ExponentialCovariance_{i,j (Decay)} = EWMA(\sum)_{T}
      \end{align*}

Where :math:`R_{i}^{t}` is the return of :math:`i^{th}` asset for :math:`t^{th}` observation,
:math:`T` is the total number of observations, :math:`\sum_{i,j}` is the series of covariances between :math:`i^{th}`
and :math:`j^{th}` asset, :math:`EWMA(\sum)_{t}` is the :math:`t^{th}` observation of exponentially-weighted
moving average of :math:`\sum`.

Implementation
**************

.. autoclass:: RiskEstimators
   :noindex:
   :members: exponential_covariance


----

De-noising and De-toning Covariance/Correlation Matrix
######################################################

Two methods for de-noising are implemented in this module:

- Constant Residual Eigenvalue Method
- Targeted Shrinkage

Constant Residual Eigenvalue De-noising Method
**********************************************

The main idea behind the Constant Residual Eigenvalue de-noising method is to separate the noise-related eigenvalues from
the signal-related ones. This is achieved by fitting the Marcenko-Pastur distribution of the empirical distribution of
eigenvalues using a Kernel Density Estimate (KDE).

The de-noising function works as follows:

- The given covariance matrix is transformed to the correlation matrix.

- The eigenvalues and eigenvectors of the correlation matrix are calculated.

- Using the Kernel Density Estimate algorithm a kernel of the eigenvalues is estimated.

- The Marcenko-Pastur pdf is fitted to the KDE estimate using the variance as the parameter for the optimization.

- From the obtained Marcenko-Pastur distribution, the maximum theoretical eigenvalue is calculated using the formula
  from the **Instability caused by noise** part of `A Robust Estimator of the Efficient Frontier paper <https://papers.ssrn.com/sol3/abstract_id=3469961>`__.

- The eigenvalues in the set that are below the theoretical value are all set to their average value.
  For example, we have a set of 5 eigenvalues sorted in the descending order ( :math:`\lambda_1` ... :math:`\lambda_5` ),
  3 of which are below the maximum theoretical value, then we set

.. math::

    \lambda_3^{NEW} = \lambda_4^{NEW} = \lambda_5^{NEW} = \frac{\lambda_3^{OLD} + \lambda_4^{OLD} + \lambda_5^{OLD}}{3}

- Eigenvalues above the maximum theoretical value are left intact.

.. math::

    \lambda_1^{NEW} = \lambda_1^{OLD}

    \lambda_2^{NEW} = \lambda_2^{OLD}

- The new set of eigenvalues with the set of eigenvectors is used to obtain the new de-noised correlation matrix.
  :math:`\tilde{C}` is the de-noised correlation matrix, :math:`W` is the eigenvectors matrix,
  and :math:`\Lambda` is the diagonal matrix with new eigenvalues.

.. math::

    \tilde{C} = W \Lambda W'

- To rescale :math:`\tilde{C}` so that the main diagonal consists of 1s the following transformation is made.
  This is how the final :math:`C_{denoised}` is obtained.

.. math::

    C_{denoised} = \tilde{C} [(diag[\tilde{C}])^\frac{1}{2}(diag[\tilde{C}])^{\frac{1}{2}'}]^{-1}

- The new correlation matrix is then transformed back to the new de-noised covariance matrix.

(If the correlation matrix is given as an input, the first and the last steps of the algorithm are omitted)

.. tip::
    The Constant Residual Eigenvalue de-noising method is described in more detail in the work
    **A Robust Estimator of the Efficient Frontier** *by* Marcos Lopez de Prado `available here <https://papers.ssrn.com/abstract_id=3469961>`_.

    Lopez de Prado suggests that this de-noising algorithm is preferable as it removes the noise while preserving the signal.

Targeted Shrinkage De-noising
*****************************

The main idea behind the Targeted Shrinkage de-noising method is to shrink the eigenvectors/eigenvalues that are
noise-related. This is done by shrinking the correlation matrix calculated from noise-related eigenvectors/eigenvalues
and then adding the correlation matrix composed from signal-related eigenvectors/eigenvalues.

The de-noising function works as follows:

- The given covariance matrix is transformed to the correlation matrix.

- The eigenvalues and eigenvectors of the correlation matrix are calculated and sorted in the descending order.

- Using the Kernel Density Estimate algorithm a kernel of the eigenvalues is estimated.

- The Marcenko-Pastur pdf is fitted to the KDE estimate using the variance as the parameter for the optimization.

- From the obtained Marcenko-Pastur distribution, the maximum theoretical eigenvalue is calculated using the formula
  from the **Instability caused by noise** part of `A Robust Estimator of the Efficient Frontier <https://papers.ssrn.com/sol3/abstract_id=3469961>`__.

- The correlation matrix composed from eigenvectors and eigenvalues related to noise (eigenvalues below the maximum
  theoretical eigenvalue) is shrunk using the :math:`\alpha` variable.

.. math::

    C_n = \alpha W_n \Lambda_n W_n' + (1 - \alpha) diag[W_n \Lambda_n W_n']

- The shrinked noise correlation matrix is summed to the information correlation matrix.

.. math::

    C_i = W_i \Lambda_i W_i'

    C_{denoised} = C_n + C_i

- The new correlation matrix is then transformed back to the new de-noised covariance matrix.

(If the correlation matrix is given as an input, the first and the last steps of the algorithm are omitted)

De-toning
*********

De-noised correlation matrix from the previous methods can also be de-toned by excluding a number of first
eigenvectors representing the market component.

According to Lopez de Prado:

"Financial correlation matrices usually incorporate a market component. The market component is characterized by the
first eigenvector, with loadings :math:`W_{n,1} \approx N^{-\frac{1}{2}}, n = 1, ..., N.`
Accordingly, a market component affects every item of the covariance matrix. In the context of clustering
applications, it is useful to remove the market component, if it exists (a hypothesis that can be
tested statistically)."

"By removing the market component, we allow a greater portion of the correlation to be explained
by components that affect specific subsets of the securities. It is similar to removing a loud tone
that prevents us from hearing other sounds"

"The detoned correlation matrix is singular, as a result of eliminating (at least) one eigenvector.
This is not a problem for clustering applications, as most approaches do not require the invertibility
of the correlation matrix. Still, **a detoned correlation matrix** :math:`C_{detoned}` **cannot be used directly for**
**mean-variance portfolio optimization**."

The de-toning function works as follows:

- De-toning is applied on the de-noised correlation matrix.

- The correlation matrix representing the market component is calculated from market component eigenvectors and eigenvalues
  and then subtracted from the de-noised correlation matrix. This way the de-toned correlation matrix is obtained.

.. math::

    \hat{C} = C_{denoised} - W_m \Lambda_m W_m'

- De-toned correlation matrix :math:`\hat{C}` is then rescaled so that the main diagonal consists of 1s

.. math::

    C_{detoned} = \hat{C} [(diag[\hat{C}])^\frac{1}{2}(diag[\hat{C}])^{\frac{1}{2}'}]^{-1}

.. tip::
    For a more detailed description of de-noising and de-toning, please read Chapter 2 of the book
    **Machine Learning for Asset Managers** *by* Marcos Lopez de Prado.

.. tip::
    This and above the methods are described in more detail in the Risk Estimators Notebook.


Implementation
**************

.. autoclass:: RiskEstimators
   :noindex:
   :members: denoise_covariance


----

Example Code
############

.. code-block::

    import pandas as pd
    import numpy as np
    from mlfinlab.portfolio_optimization import RiskEstimators, ReturnsEstimators

    # Import price data
    stock_returns = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

    # Class that have needed functions
    risk_estimators = RiskEstimators()
    returns_estimators = ReturnsEstimators()

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

    # Series of returns from series of prices
    stock_returns = ret_est.calculate_returns(stock_prices)
	
    # Finding the simple covariance matrix from a series of returns
    cov_matrix = stock_returns.cov()

    # Finding the Constant Residual Eigenvalue De-noised Сovariance matrix
    const_resid_denoised = risk_estimators.denoise_covariance(cov_matrix, tn_relation,
                                                              denoise_method='const_resid_eigen',
                                                              detone=False, kde_bwidth=kde_bwidth)

    # Finding the Targeted Shrinkage De-noised Сovariance matrix
    targ_shrink_denoised = risk_estimators.denoise_covariance(cov_matrix, tn_relation,
                                                              denoise_method='target_shrink',
                                                              detone=False, kde_bwidth=kde_bwidth)

    # Finding the Constant Residual Eigenvalue De-noised and De-toned Сovariance matrix
    const_resid_detoned = risk_estimators.denoise_covariance(cov_matrix, tn_relation,
                                                             denoise_method='const_resid_eigen',
                                                             detone=True, market_component=1,
                                                             kde_bwidth=kde_bwidth)

Research Notebooks
##################

The following research notebook can be used to better understand how the algorithms within this module can be used on real data.

* `Risk Estimators Notebook`_

.. _Risk Estimators Notebook: https://github.com/hudson-and-thames/research/blob/master/Portfolio%20Optimisation%20Tutorials/Risk%20Estimators/RiskEstimators.ipynb

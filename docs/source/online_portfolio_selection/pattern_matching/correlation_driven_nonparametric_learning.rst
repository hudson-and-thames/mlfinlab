.. _online_portfolio_selection-pattern_matching-correlation_driven_nonparametric_learning:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

=========================================
Correlation Driven Nonparametric Learning
=========================================

Correlation Driven Nonparametric Learning strategies look at historical market sequences to identify similarly correlated periods.
Existing pattern matching strategies attempt to exploit and identify the correlation between different market windows by using
the Euclidean distance to measure the similarity between two market windows. However, the traditional Euclidean distance between
windows does not effectively capture the linear relation. CORN utilizes the Pearson correlation coefficient instead of Euclidean
distances to capture the whole market direction.

Three different variations of the CORN strategies are implemented in the Online Portfolio Selection module.

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Pattern%20Matching.ipynb>`_
    provides a more detailed exploration of the strategies.

1. Correlation Driven Nonparametric Learning (CORN)
###################################################

CORN formally defines a similar set to be one that satisfies the following equation:

.. math::
    C_t(w,\rho) = \left\lbrace w < i < t+1 \bigg\vert \frac{cov(x^{i-1}_{i-w}, x^t_{t-w+1})}{std(x^{i-1}_{i-w})std(x^t_{t-w+1})} \geq \rho\right\rbrace

:math:`W` represents the number of windows to lookback, and :math:`\rho` is the correlation coefficient threshold.

For the specific correlation calculation, each market window of w periods is concatenated to obtain a univariate correlation coefficient between the two windows.

Once all the similar historical periods are identified, the strategy returns weights that will maximize returns for the period.

.. math::
    b_{t+1}(w,\rho) = \underset{b \in \Delta_m}{\arg \max} \underset{i \in C_t(w,\rho)}{\prod}(b \cdot x_i)

- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p(t)` is the price at time :math:`t`.
- :math:`w` is the number of windows to lookback.
- :math:`cov` is the covariance term.
- :math:`std` is the standard deviation term.
- :math:`rho` is the correlation threshold.
- :math:`C_t` is the set of similar periods.

CORN Parameters
---------------

The optimal parameters for CORN are dependant on each dataset. For NYSE, :math:`\rho` of 0.3 and window of 1 had the highest returns.
SP500 images indicate an optimal rho of 0 and window of 6. Most of the times the window values should be less than 7 with a strong
inclination to 0 with rho values being on the lower end from 0 to 0.4.

.. image:: pattern_matching_images/nyse_corn.png
   :width: 49 %

.. image:: pattern_matching_images/sp500_corn.png
   :width: 49 %

CORN Implementation
-------------------

.. automodule:: mlfinlab.online_portfolio_selection.pattern_matching.correlation_driven_nonparametric_learning

    .. autoclass:: CORN
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__


2. Correlation Driven Nonparametric Learning - Uniform (CORN-U)
###############################################################

Because the CORN strategies are dependent on the parameters, we propose a more generic one that takes an
ensemble approach to reduce variability. One possible CORN ensemble is the CORN-U method.

CORN-U generates a set of experts with different window sizes and the same :math:`\rho` value. After all the expert's
weights are calculated, weights are evenly distributed among all experts to represent the strategy as a universal portfolio.

After gathering results for all the experts, the total portfolio weight will be determined by:

.. math::
    b_{t+1}=\frac{\sum_{w, \rho}q(w,\rho)S_t(w,\rho)b_{t+1}(w,\rho)}{\sum_{w, \rho}q(w,\rho)S_t(w,\rho)}

- :math:`S(t)` is the total portfolio returns at time :math:`t`.
- :math:`q(w, \rho)` is the weight allocation constant. For CORN-U, this is a uniform distribution.

CORN-U Parameters
-----------------

The optimal parameters for CORN-U follow the best parameters for parent class CORN. The most important parameter that affects
returns tend to be the :math:`\rho` value and a range between 0 and 0.4 works for most datasets. Window ranges
are trickier as they tend to be either just 1 or a much larger value.

.. image:: pattern_matching_images/nyse_cornu.png
   :width: 49 %

.. image:: pattern_matching_images/sp500_cornu.png
   :width: 49 %

CORN-U Implementation
---------------------

.. automodule:: mlfinlab.online_portfolio_selection.pattern_matching.correlation_driven_nonparametric_learning_uniform

    .. autoclass:: CORNU
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__

3. Correlation Driven Nonparametric Learning - K (CORN-K)
#########################################################

CORN-K further improves the CORN-U by generating more parameters of experts. There is more variability as
different ranges of window and :math:`\rho` value are considered to create more options.

The most important part of the CORN-K, however, is the capital allocation method. Unlike CORN-U, which uniformly
distributes capital among all the experts, CORN-K selects the top-k best performing experts until the last period
and equally allocate capital among them. This prunes the experts that have less optimal returns and puts more weight on the performing ones.

CORN-K takes in 3 parameters: window, rho, and k.

CORN-K Parameters
-----------------

The most important parameter for CORN-K is k, and most of the times this should always be set at 1 or 2 for the highest returns. A low
value of k effectively prunes the less performing experts. Rho of 1 is good for datasets that have optimal CORN rho value of 0,
but if the optimal CORN rho is slightly above that rho should be changed to a value higher then 3 to capture the range. Typically,
range of [3, 5] worked for preliminary datasets. Window values also depend on each dataset, but the best
guess would be on the lower range of 1 or higher value of 7.

.. image:: pattern_matching_images/nyse_cornk.png
   :width: 49 %

.. image:: pattern_matching_images/sp500_cornk.png
   :width: 49 %

CORN-K Implementation
---------------------

.. automodule:: mlfinlab.online_portfolio_selection.pattern_matching.correlation_driven_nonparametric_learning_k

    .. autoclass:: CORNK
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__

Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection import *

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # CORN
    # Compute Correlation Driven Nonparametric Learning with no given weights, window of 1, and rho of 0.3.
    corn = CORN(window=1, rho=0.3)
    corn.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Correlation Driven Nonparametric Learning with user given weights, window of 3, and rho of 0.5.
    corn1 = CORN(window=3, rho=0.5)
    corn1.allocate(asset_prices=stock_prices, weights=some_weight)

    # CORN-U
    # Compute Correlation Driven Nonparametric Learning - Uniform with no given weights, window range of 10, and rho of 0.3.
    cornu = CORNU(window=10, rho=0.3)
    cornu.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Correlation Driven Nonparametric Learning - Uniform with user given weights, window range of 5, and rho of 0.1.
    cornu1 = CORNU(window=5, rho=0.1)
    cornu1.allocate(asset_prices=stock_prices, weights=some_weight)

    # CORN-K
    # Compute Correlation Driven Nonparametric Learning - K with no given weights, window range of 10, rho of 7, and k of 2.
    cornk = CORNK(window=10, rho=7, k=2)
    cornk.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Correlation Driven Nonparametric Learning - K with user given weights, window range of 5, rho of 3, and k of 1.
    cornk1 = CORNK(window=5, rho=3, k=1)
    cornk1.allocate(asset_prices=stock_prices, weights=some_weight)

    # Recalculate k for cornk1 to save computational time of generating all experts.
    cornk1.recalculate_k(k=3)

    # Get the latest predicted weights.
    corn.weights

    # Get all weights for the strategy.
    cornk.all_weights

    # Get portfolio returns.
    cornu.portfolio_return

    # Get each object of the generated experts.
    cornk1.experts

    # Get each experts parameters.
    cornu.expert_params

    # Get all expert's portfolio returns over time.
    cornu1.expert_portfolio_returns

    # Get capital allocation weights.
    cornk1.weights_on_experts

.. tip::

    Strategies were implemented with modifications from `Li, B., Hoi, S.C., & Gopalkrishnan, V. (2011). CORN: Correlation-driven nonparametric
    learning approach for portfolio selection. ACM TIST, 2,
    21:1-21:29. <https://dl.acm.org/doi/abs/10.1145/1961189.1961193>`_

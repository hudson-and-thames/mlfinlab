.. _online_portfolio_selection-mean_reversion-robust_median_reversion:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

=======================
Robust Median Reversion
=======================

Robust Median Reversion extends the previous Online Moving Average Reversion by introducing L1 median of the specified windows.
Instead of reverting to a moving average, RMR reverts to the L1 median estimator, which proves to be a more effective method of
predicting the next period's price because financial data is inherently noisy and contains many outliers.

L1-median is calculated with the following equation:

.. math::
    \mu = \underset{\mu}{\arg \min}\overset{k-1}{\underset{i=0}{\sum}}||p_{t-i} - \mu ||

where :math:`k` is the number of historical price windows, and :math:`\mu` represents the predicted price.

The calculation of L1-median is computationally inefficient, so the algorithm will be using the Modified Weiszfeld Algorithm.

.. math::
    \hat{x}_{t+1} = T(\mu) = (1 - \frac{\eta(\mu)}{\gamma(\mu)})^+ \: \tilde{T}(\mu + \min(1,\frac{\eta(\mu)}{\gamma(\mu)})\mu

.. math::
    \eta(\mu) = 1 \text{ if } \mu =\text{ any price and }0 \text{ otherwise.}

.. math::
    \gamma(\mu)=\left|\left|\underset{p_{t-i} \neq \mu}{\sum}\frac{p_{t-i}-\mu}{||p_{t-i}-\mu||}\right|\right|

.. math::
    \tilde{T}(\mu)=\left\lbrace \underset{p_{t-i}\neq \mu}{\sum}\frac{1}{||p_{t-i}-\mu||}\right\rbrace^{-1}\underset{p_{t-i}\neq \mu}{\sum}\frac{p_{t-i}}{||p_{t-i}-\mu||}

Then next portfolio weights will use the predicted price to produce the optimal portfolio weights.

.. math::
    b_{t+1} = b_{t} - \min \left \lbrace 0, \frac{\hat{x}_{t+1} b_t-\epsilon}{||\hat{x}_{t+1}-\bar{x}_{t+1}\cdot\textbf{1}||^2}\right \rbrace \cdot (\hat{x}_{t+1}-\bar{x}_{t+1}\cdot\textbf{1})

- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p(t)` is the price at time :math:`t`.
- :math:`\mu_t` is the projected price.
- :math:`\hat{x}` is the projected price relative.
- :math:`\bar{x}` is the mean of the projected price relative.


.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Mean%20Reversion.ipynb>`_
    provides a more detailed exploration of the strategies.

Parameters
----------

Similarly to OLMAR, the parameters primarily depend on the window value. N_iteration of 200 typically had
the highest results with a :math:`\epsilon` range of 15 to 25. Ultimately, the window range that decides
the period of mean reversion was the most influential parameter to affect the portfolio's results.

.. image:: mean_reversion_images/nyse_rmr.png
   :width: 32 %

.. image:: mean_reversion_images/djia_rmr.png
   :width: 32 %

.. image:: mean_reversion_images/msci_rmr.png
   :width: 32 %

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.mean_reversion.robust_median_reversion

    .. autoclass:: RMR
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

    # Compute Robust Median Reversion with no given weights, epsilon of 15, n_iteration of 100, and window of 7.
    rmr = RMR(epsilon=15, n_iteration=100, window=7)
    rmr.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Robust Median Reversion with given user weights, epsilon of 25, n_iteration of 500, and window of 21.
    rmr1 = RMR(epsilon=25, n_iteration=500, window=21)
    rmr1.allocate(asset_prices=stock_prices, weights=some_weight)

    # Get the latest predicted weights.
    rmr.weights

    # Get all weights for the strategy.
    rmr.all_weights

    # Get portfolio returns.
    rmr1.portfolio_return

.. tip::

    Strategies were implemented with modifications from `D. Huang, J. Zhou, B. Li, S. C. H. Hoi and S. Zhou, "Robust Median Reversion Strategy for
    Online Portfolio Selection," in IEEE Transactions on Knowledge and Data Engineering, vol. 28,
    no. 9, pp. 2480-2493, 1 Sept. 2016. <https://www.ijcai.org/Proceedings/13/Papers/296.pdf>`_


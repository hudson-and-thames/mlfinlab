.. _online_portfolio_selection-momentum-follow_the_regularized_leader:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

=============================
Follow the Regularized Leader
=============================

Follow the Regularized Leader adds an additional regularization term to the objective function for Follow the Leader to prevent a drastic deviation in each period.

.. math::
    b_{t+1} = \underset{b \in \Delta_m}{\arg\max} \overset{t}{\underset{n=1}{\sum}} \: \log(b \cdot x_n) - \frac{\beta}{2}R(b)

- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p_t` is the price at time :math:`t`.
- :math:`\beta` is a penalty variable for the regularization.
- :math:`R(b)` is the regularization term for follow the regularized leader.
- :math:`\Delta_m` is the simplex domain. The sum of all elements is 1, and each element is in the range of [0, 1].

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Momentum.ipynb>`_
    provides a more detailed exploration of the strategies.

Parameters
----------

The optimal parameters depend on each dataset. For NYSE, a high regularization was an effective method to generate high returns
as :math:`\beta` of 20 was optimal.

.. image:: momentum_images/nyse_ftrl_beta_0_1.png
   :width: 49 %

.. image:: momentum_images/nyse_ftrl_beta_1_100.png
   :width: 49 %

However, for the MSCI dataset, we see that regularization is an ineffective means to follow the momentum strategy.
The highest returns are results with :math:`\beta` of 0.2. Lower values of beta tend to follow the Uniform Constant Rebalanced Portfolio results.

.. image:: momentum_images/msci_ftrl_beta_0_1.png
   :width: 49 %

.. image:: momentum_images/msci_ftrl_beta_1_100.png
   :width: 49 %

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.momentum.follow_the_regularized_leader

    .. autoclass:: FTRL
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

    # Compute Follow the Regularized Leader with no given weights and beta of 10.
    ftrl = FTRL(beta=10)
    ftrl.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Follow the Leader with given user weights and beta of 0.4.
    ftrl = FTRL(beta=0.4)
    ftrl.allocate(asset_prices=stock_prices, weights=some_weight)

    # Get the latest predicted weights.
    ftrl.weights

    # Get all weights for the strategy.
    ftrl.all_weights

    # Get portfolio returns.
    ftrl.portfolio_return

.. tip::

    Strategies were implemented with modifications from `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

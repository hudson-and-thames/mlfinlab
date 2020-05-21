.. _online_portfolio_selection-benchmarks-best_constant_rebalanced_portfolio:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

==============================
Best Constant Rebalanced Portfolio
==============================

Best Constant Rebalanced Portfolio is a strategy that is implemented in hindsight, which is similar to Best Stock. It uses the same weight
for all time periods. However, it determines those weights by having the complete market sequence of the past. The objective function for
BCRP looks to maximize portfolio returns with the equation below.

.. math::
    b^{\bf{\star}} = \underset{b^n \in \Delta_m}{\arg\max} \: S_n(CRP(b)) = \underset{b \in \Delta_m}{\arg\max} \overset{n}{\underset{t=1}{\prod}} \:  b^{\top}x_t

Once the optimal weight has been determined, the final returns can be calculated by using the CRP returns equation.

.. math::
    S_n(BCRP) = \underset{b \in \Delta_m}{\max} \: S_n(CRP(b)) = S_n(CRP(b^{\bf \star}))

Implementation
##############

.. automodule:: mlfinlab.online_portfolio_selection.benchmarks.best_constant_rebalanced_portfolio

    .. autoclass:: BCRP
        :members:
        :show-inheritance:
        :inherited-members:


Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection.benchmarks.best_constant_rebalanced_portfolio import BCRP

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute Best Constant Rebalanced Portfolio weights with no weights given.
    bcrp = CRP()
    bcrp.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Get the latest predicted weights.
    bcrp.weights

    # Get all weights for the strategy.
    bcrp.all_weights

    # Get portfolio returns.
    bcrp.portfolio_return

Research Notebooks
##################

The following research notebooks provides a more detailed exploration of the strategies.

* `Benchmarks Notebook`_

.. _Benchmarks Notebook: https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Introduction%20to%20Online%20Portfolio%20Selection.ipynb

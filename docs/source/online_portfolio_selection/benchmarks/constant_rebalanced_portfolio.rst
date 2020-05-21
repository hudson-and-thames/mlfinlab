.. _online_portfolio_selection-benchmarks-constant_rebalanced_portfolio:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

==============================
Constant Rebalanced Portfolio
==============================

Constant Rebalanced Portfolio rebalances to a certain portfolio weight every time period. This particular weight can be set by the user,
and if there are no inputs, it will automatically allocate equal weights to all assets. The total returns for a CRP can be calculated by
taking the cumulative product of the weight and relative returns matrix.

.. math::
    S_n(CRP(b)) = \overset{n}{\underset{t=1}{\prod}} \:  b^{\top}x_t

Once the initial portfolio has been determined, the final weights can be represented as buying and holding the initial weight.

.. math::
    S_n(BEST) = \underset{b \in \Delta_m}{\max} b \cdot \left(\overset{n}{\underset{t=1}{\bigodot}}  x_t \right) = S_n(BAH(b_0))

Implementation
##############

.. automodule:: mlfinlab.online_portfolio_selection.benchmarks.constant_rebalanced_portfolio

    .. autoclass:: CRP
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__


Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection.benchmarks.constant_rebalanced_portfolio import CRP

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute Constant Rebalanced Portfolio weights with no weights given.
    crp = CRP()
    crp.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Constant Rebalanced Portfolio weights with given weights.
    crp = CRP()
    crp.allocate(asset_prices=stock_prices, weights=some_weight)

    # Get the latest predicted weights.
    crp.weights

    # Get all weights for the strategy.
    crp.all_weights

    # Get portfolio returns.
    crp.portfolio_return

Research Notebooks
##################

The following research notebooks provides a more detailed exploration of the strategies.

* `Benchmarks Notebook`_

.. _Benchmarks Notebook: https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Introduction%20to%20Online%20Portfolio%20Selection.ipynb

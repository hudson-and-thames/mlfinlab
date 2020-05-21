.. _online_portfolio_selection-benchmarks-buy_and_hold:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

==============================
Buy and Hold
==============================

Buy and Hold is a strategy where an investor invests in an initial portfolio and never rebalances it. The portfolio weights, however, change
as time goes by because the underlying assets change in prices.

Returns for Buy and Hold can be calculated by multiplying the initial weight and the cumulative product of relative returns.

.. math::
    S_n(BAH(b_1)) = b_1 \cdot \left(\overset{n}{\underset{t=1}{\bigodot}} x_t\right)

Implementation
##############

.. automodule:: mlfinlab.online_portfolio_selection.benchmarks.buy_and_hold

    .. autoclass:: BAH
        :members:
        :show-inheritance:
        :inherited-members:


Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection.benchmarks.buy_and_hold import BAH

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute Buy and Hold weights with no weights given.
    bah = BAH()
    bah.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Buy and Hold weights with given weights.
    bah = BAH()
    bah.allocate(asset_prices=stock_prices, weights=some_weight)

    # Get the latest predicted weights.
    bah.weights

    # Get all weights for the strategy.
    bah.all_weights

    # Get portfolio returns.
    bah.portfolio_return

Research Notebooks
##################

The following research notebooks provides a more detailed exploration of the strategies.

* `Benchmarks Notebook`_

.. _Benchmarks Notebook: https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Introduction%20to%20Online%20Portfolio%20Selection.ipynb

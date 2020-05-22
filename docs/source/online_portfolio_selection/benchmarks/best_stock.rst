.. _online_portfolio_selection-benchmarks-best_stock:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

==========
Best Stock
==========

Best Stock strategy chooses the best performing asset in hindsight.

The best performing asset is determined with an argmax equation stated below. The portfolio selection strategy searches for the asset
that increases the most in price for the given time period.

.. math::
    b_0 = \underset{b \in \Delta_m}{\arg\max} \: b \cdot \left(\overset{n}{\underset{t=1}{\bigodot}}  x_t \right)

Once the initial portfolio has been determined, the final weights can be represented as buying and holding the initial weight.

.. math::
    S_n(BEST) = \underset{b \in \Delta_m}{\max} b \cdot \left(\overset{n}{\underset{t=1}{\bigodot}}  x_t \right) = S_n(BAH(b_0))

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Introduction%20to%20Online%20Portfolio%20Selection.ipynb>`_
    provides a more detailed exploration of the strategies.

Implementation
##############

.. automodule:: mlfinlab.online_portfolio_selection.benchmarks.best_stock

    .. autoclass:: BestStock
        :members:
        :show-inheritance:
        :inherited-members:


Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection.benchmarks.best_stock import BestStock

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute Best Stock weights with no weights given.
    beststock = BestStock()
    beststock.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Get the latest predicted weights.
    beststock.weights

    # Get all weights for the strategy.
    beststock.all_weights

    # Get portfolio returns.
    beststock.portfolio_return

.. _online_portfolio_selection-momentum-follow_the_leader:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

=================
Follow the Leader
=================

The biggest drawback of using Exponential Gradient is the failure to look at the changes before the latest period.
Follow the Leader mediates this shortfall by directly tracking the Best Constant Rebalanced Portfolio; therefore, FTL
looks at the whole history of the data and calculates the portfolio weights that would have had the maximum returns.

.. math::
    b_{t+1} = b^{\bf{\star}}_t = \underset{b \in \Delta_m}{\arg\max} \overset{t}{\underset{\tau=1}{\sum}} \: \log(b \cdot x_{\tau})

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Momentum.ipynb>`_
    provides a more detailed exploration of the strategies.


Implementation
##############

.. automodule:: mlfinlab.online_portfolio_selection.momentum.follow_the_leader

    .. autoclass:: FTL
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__

Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection.momentum.follow_the_leader import FTL

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute Follow the Leader with no given weights.
    ftl = FTL()
    ftl.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Follow the Leader with given weights.
    ftl = FTL()
    ftl.allocate(asset_prices=stock_prices, weights=some_weight)

    # Get the latest predicted weights.
    ftl.weights

    # Get all weights for the strategy.
    ftl.all_weights

    # Get portfolio returns.
    ftl.portfolio_return

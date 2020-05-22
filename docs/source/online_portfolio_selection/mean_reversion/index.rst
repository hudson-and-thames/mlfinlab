.. _online_portfolio_selection-mean_reversion-index:

.. note::
    The portfolio optimisation module contains different algorithms that are used for asset allocation and optimising strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

==============
Mean Reversion
==============

.. toctree::
    :maxdepth: 3
    :caption: Passive Aggressive Mean Reversion
    :hidden:

    passive_aggressive_mean_reversion

.. toctree::
    :maxdepth: 3
    :caption: Confidence Weighted Mean Reversion
    :hidden:

    confidence_weighted_mean_reversion

.. toctree::
    :maxdepth: 3
    :caption: Online Moving Average Reversion
    :hidden:

    online_moving_average_reversion

.. toctree::
    :maxdepth: 3
    :caption: Robust Median Reversion
    :hidden:

    robust_median_reversion

Mean Reversion is an effective quantitative strategy based on the theory that prices will revert back to its historical mean.
A basic example of mean reversion follows the benchmark of Constant Rebalanced Portfolio. By setting a predetermined allocation of
weight to each asset, the portfolio shifts its weights from increasing to decreasing ones.

Through this documentation, the importance of hyperparameters is highlighted as the choices greatly affect the outcome of returns.
A lot of the hyperparameters for traditional research has been chosen by looking at the data in hindsight, and fundamental analysis
of each dataset and market structure is required to profitably implement this strategy in a real-time market scenario.

There are four different mean reversion strategies implemented in the Online Portfolio Selection module.

1. Passive Aggressive Mean Reversion

2. Confidence Weighted Mean Reversion

3. Online Moving Average Reversion

4. Robust Median Reversion

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Mean%20Reversion.ipynb>`_
    provides a more detailed exploration of the strategies.

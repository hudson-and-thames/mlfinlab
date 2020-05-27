.. _online_portfolio_selection-mean_reversion:

==============
Mean Reversion
==============

Mean Reversion is an effective quantitative strategy based on the theory that prices will revert back to its historical mean.
A basic example of mean reversion follows the benchmark of Constant Rebalanced Portfolio. By setting a predetermined allocation of
weight to each asset, the portfolio shifts its weights from increasing to decreasing ones.

Through this documentation, the importance of hyperparameters is highlighted as the choices greatly affect the outcome of returns.
A lot of the hyperparameters for traditional research has been chosen by looking at the data in hindsight, and fundamental analysis
of each dataset and market structure is required to profitably implement this strategy in a real-time market scenario.

There are four mean reversion strategies implemented in the Online Portfolio Selection module.

.. toctree::
    :maxdepth: 1

    mean_reversion/passive_aggressive_mean_reversion
    mean_reversion/confidence_weighted_mean_reversion
    mean_reversion/online_moving_average_reversion
    mean_reversion/robust_median_reversion

.. tip::

    The following `mean reversion <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Mean%20Reversion.ipynb>`_
    notebook provides a more detailed exploration of the strategies.

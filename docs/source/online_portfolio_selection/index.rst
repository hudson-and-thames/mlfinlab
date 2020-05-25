.. _online_portfolio_selection-index:

==========================
Online Portfolio Selection
==========================

Online Portfolio Selection
##########################

In general, most of these strategies will follow the structure of the parent class: OLPS.

The parent class exists to quickly build a new strategy. Each strategy is modularized to ensure maximum
efficiency to switch around the update algorithms.

.. automodule:: mlfinlab.online_portfolio_selection.online_portfolio_selection

    .. autoclass:: OLPS
        :members:

There are four different strategies currently implemented in the Online Portfolio Selection module.

Benchmarks
##########

Before we dive into the more interesting and complex models of portfolio selection, we will begin our analysis with benchmarks.
As unappealing as benchmarks are, traditional strategies such as tracking the S&P 500 have been hugely successful.

Typically these are implemented in hindsight, so future data is often incorporated within the selection algorithm. For real-life
applications, we do not have access to future data from the present, so strategies here should be taken with a grain of salt.

There are four benchmarks strategies implemented in the Online Portfolio Selection module.

.. toctree::
    :maxdepth: 1

    benchmarks/buy_and_hold
    benchmarks/best_stock
    benchmarks/constant_rebalanced_portfolio
    benchmarks/best_constant_rebalanced_portfolio

.. tip::

    The following `benchmarks <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Introduction%20to%20Online%20Portfolio%20Selection.ipynb>`_
    notebook provides a more detailed exploration of the strategies.


Momentum
########

Momentum strategies have been a popular quantitative strategy in recent decades as the simple but powerful trend-following
allows investors to exponentially increase their returns. This module will implement two types of momentum strategy with one
following the best-performing assets in the last period and the other following the Best Constant Rebalanced Portfolio until the last period.

There are three momentum strategies implemented in the Online Portfolio Selection module.

.. toctree::
    :maxdepth: 1

    momentum/exponential_gradient
    momentum/follow_the_leader
    momentum/follow_the_regularized_leader

.. tip::

    The following research `momentum <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Momentum.ipynb>`_
    notebook provides a more detailed exploration of the strategies.

Mean Reversion
##############

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

Pattern Matching
################

Pattern matching locates similarly acting historical market windows and make future predictions based on the similarity.
Traditional quantitative strategies such as momentum and mean reversion focus on the directionality of the market trends.
The underlying assumption that the immediate past trends will continue is simple but does not always perform the best in real markets.
Pattern matching strategies combine the strengths of both by exploiting the statistical correlations of the current market window to the past.

There are three pattern matching strategies implemented in the Online Portfolio Selection module.

.. toctree::
    :maxdepth: 1

    pattern_matching/correlation_driven_nonparametric_learning
    pattern_matching/symmetric_correlation_driven_nonparametric_learning
    pattern_matching/functional_correlation_driven_nonparametric_learning

.. tip::

    The following `pattern matching <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Pattern%20Matching.ipynb>`_
    notebook provides a more detailed exploration of the strategies.

Universal Portfolio
###################

For the ensemble methods of Universal Portfolio, there is a sub-parent class of Universal Portfolio.

Universal Portfolio effectively acts as a fund of funds. It is possible to generate differents experts
with different parameters and gather the performance through different methods.

.. automodule:: mlfinlab.online_portfolio_selection.universal_portfolio

    .. autoclass:: UP
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__

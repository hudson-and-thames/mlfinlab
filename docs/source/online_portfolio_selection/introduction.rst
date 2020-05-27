.. _online_portfolio_selection-introduction:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.

============
Introduction
============

Online Portfolio Selection
==========================

Online Portfolio Selection can broadly be divided into two parts: Portfolio Selection and Online Learning.

Portfolio Selection is a sequential allocation among a set of assets to maximize the final return of investment.
Every day a portfolio manager is given a task to decide the allocation of capital, and we formulate the problem
so that the weights and the daily returns can be represented in a vector format. The product of the two will represent
the daily returns of the strategy.

Online Learning utilizes computationally efficient algorithms to handle large scale applications. You could have
the best strategy in the world that predicts the stock market movements 100% of the time. However, if your strategy
takes one day to run, you would not be able to capture the opportunity presented by the situation. It is imperative
that these update algorithms can be done in a set amount of time and preferably a quick one to that. This would
actually expand the application of this selection algorithm to not only a daily or weekly time frame to something
that can be applied to even intraday or mid-frequency settings as well.

It is also noteworthy to understand that traditional theories for portfolio selection, such as Markowitz’s
Portfolio Theory, optimize the balance between the portfolio's risks and returns. However, online portfolio
selection is founded on the capital growth theory, which solely focuses on maximizing the returns of the current
portfolio. Because the capital growth theory primarily relies on the Kelly criterion, traditional metrics such as
Sharpe ratios and maximum drawdowns are more so less useful. The primary metric in this situation becomes the log
of final wealth, which in turn indicates the maximum of final wealth.

.. image:: images/diagram.png
   :width: 99 %

Four different strategies are currently implemented in the Online Portfolio Selection module with the
above diagram available for a general overview of some of the strategies.

1. Benchmarks

2. Momentum

3. Mean Reversion

4. Pattern Matching

All of the online portfolio selection strategies will be built on top of the base constructor class ``OLPS``.

Import
======

Strategies can be imported by using variations of the following lines.

.. code-block::

    # Import all strategies.
    from mlfinlab.online_portfolio_selection import *

    # Import all benchmark strategies.
    from mlfinlab.online_portfolio_selection.benchmarks import *

    # Import all momentum strategies.
    from mlfinlab.online_portfolio_selection.benchmarks import *

    # Import all mean reversion strategies.
    from mlfinlab.online_portfolio_selection.mean_reversion import *

    # Import all pattern matching strategies.
    from mlfinlab.online_portfolio_selection.pattern_matching import *

    # Import a specific buy and hold strategy.
    from mlfinlab.online_portfolio_selection import BAH

    # Import buy and hold and universal portfolio.
    from mlfinlab.online_portfolio_selection import BAH, UP

Initialize
==========

Strategies are first initialized to create an object. Certain strategies require parameters to initialize.

.. code-block::

    # Initialize Buy and Hold.
    bah = BAH()

    # Initialize Passive Aggressive Mean Reversion.
    pamr = PAMR(optimization_method=1, epsilon=0.5, agg=10)

    # Initialize Correlation Driven Nonparametric Learning - K
    cornk = CORNK(window=2, rho=5, k=2)

Allocate
========

All strategies use ``allocate()`` to calculate the portfolio weights based on the given data. The user must supply the given data in a ``pd.DataFrame`` that is indexed by time.

Three additional options are available for the ``allocate`` method.

- ``weights`` is an option for the user to supply the initial weights.
- ``resample_by`` changes the reallocation period for the portfolio.
- ``verbose`` prints out a progress bar to allow users to follow the status.

.. code-block::

    # Initialize Buy and Hold.
    bah = BAH()

    # Allocate with no additional inputs.
    bah.allocate(price_data)

    # Allocate with monthly portfolio rebalancing.
    bah.allocate(price_data, resample_by='M')

    # Allocate with user given weights.
    bah.allocate(price_data, weights=some_weight)

    # Allocate with printed progress bar.
    bah.allocate(price_data, verbose=True)

.. automodule:: mlfinlab.online_portfolio_selection.base

    .. automethod:: OLPS.allocate

.. image:: images/allocate.png
   :width: 99 %

Result
======

Upon weights allocation the possible outputs are:

- ``self.weights`` (np.array) Final portfolio weights prediction.
- ``self.all_weights`` (pd.DataFrame) Portfolio weights for the time period.
- ``self.asset_name`` (list) Name of assets.
- ``self.number_of_assets`` (int) Number of assets.
- ``self.time`` (datetime) Time index of the given data.
- ``self.length_of_time`` (int) Number of time periods.
- ``self.relative_return`` (np.array) Relative returns of the assets.
- ``self.portfolio_return`` (pd.DataFrame) Cumulative portfolio returns over time.
- ``self.asset_prices`` (pd.DataFrame)`` Historical asset prices (daily close).

.. code-block::

    # Initialize Buy and Hold.
    bah = BAH()

    # Allocate with no additional inputs.
    bah.allocate(price_data)

    # All weights for the portfolio.
    bah.all_weights

    # Portfolio returns.
    bah.portfolio_return

.. image:: images/portfolio_return.png
   :width: 33 %

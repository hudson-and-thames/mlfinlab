.. _online_portfolio_selection-index:

==========================
Online Portfolio Selection
==========================

.. toctree::
    :maxdepth: 4
    :caption: Benchmarks
    :hidden:

    benchmarks/index

.. toctree::
    :maxdepth: 4
    :caption: Momentum
    :hidden:

    momentum/index

.. toctree::
    :maxdepth: 4
    :caption: Mean Reversion
    :hidden:

    mean_reversion/index

.. toctree::
    :maxdepth: 4
    :caption: Pattern Matching
    :hidden:

    pattern_matching/index

There are four different strategies currently implemented in the Online Portfolio Selection module.

1. Benchmarks

2. Momentum

3. Mean Reversion

4. Pattern Matching

Online Portfolio Selection
##########################

In general, most of these strategies will follow the structure of the parent class: OLPS.

The parent class exists to quickly build a new strategy. Each strategy is modularized to ensure maximum
efficiency to switch around the update algorithms.

.. automodule:: mlfinlab.online_portfolio_selection.online_portfolio_selection

    .. autoclass:: OLPS
        :members:

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

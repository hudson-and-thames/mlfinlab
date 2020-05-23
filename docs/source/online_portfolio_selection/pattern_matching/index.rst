.. _online_portfolio_selection-pattern_matching-index:

.. note::
    The portfolio optimisation module contains different algorithms that are used for asset allocation and optimising strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

================
Pattern Matching
================

.. toctree::
    :maxdepth: 3
    :caption: Correlation Driven Nonparametric Learning
    :hidden:

    correlation_driven_nonparametric_learning

.. toctree::
    :maxdepth: 3
    :caption: Symmetric Correlation Driven Nonparametric Learning
    :hidden:

    symmetric_correlation_driven_nonparametric_learning

.. toctree::
    :maxdepth: 3
    :caption: Functional Correlation Driven Nonparametric Learning
    :hidden:

    functional_correlation_driven_nonparametric_learning

Pattern matching locates similarly acting historical market windows and make future predictions based on the similarity.
Traditional quantitative strategies such as momentum and mean reversion focus on the directionality of the market trends.
The underlying assumption that the immediate past trends will continue is simple but does not always perform the best in real markets.
Pattern matching strategies combine the strengths of both by exploiting the statistical correlations of the current market window to the past.

There are seven different pattern matching strategies implemented in the Online Portfolio Selection module.

1. Correlation Driven Nonparametric Learning (CORN)

2. Correlation Driven Nonparametric Learning - Uniform (CORN-U)

3. Correlation Driven Nonparametric Learning - K (CORN-K)

4. Symmetric Correlation Driven Nonparametric Learning (SCORN)

5. Symmetric Correlation Driven Nonparametric Learning - K (SCORN-K)

6. Functional Correlation Driven Nonparametric Learning (FCORN)

7. Functional Correlation Driven Nonparametric Learning - K (FCORN-K)

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Pattern%20Matching.ipynb>`_
    provides a more detailed exploration of the strategies.
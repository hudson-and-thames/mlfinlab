.. _online_portfolio_selection-pattern_matching:

================
Pattern Matching
================

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

.. _online_portfolio_selection-pattern_matching-index:

.. note::
    The portfolio optimisation module contains different algorithms that are used for asset allocation and optimising strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

==========
Pattern Matching
==========

There are four different benchmarks strategies implemented in the Online Portfolio Selection module.

1. Correlation Driven Nonparametric Learning

2. Correlation Driven Nonparametric Learning - Uniform

3. Correlation Driven Nonparametric Learning - K

4. Symmetric Correlation Driven Nonparametric Learning

5. Symmetric Correlation Driven Nonparametric Learning - K

6. Functional Correlation Driven Nonparametric Learning

7. Functional Correlation Driven Nonparametric Learning - K

Research Notebooks
##################

The following research notebooks provides a more detailed exploration of the algorithm as outlined at the back of Ch16 in
Advances in Financial Machine Learning.

* `Chapter 16 Exercise Notebook`_

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Chapter16/Chapter16.ipynb
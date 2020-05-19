.. _online_portfolio_selection-mean_reversion-index:

.. note::
    The portfolio optimisation module contains different algorithms that are used for asset allocation and optimising strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, lets discuss about some of these implementations and the different parameters they require.

==========
Mean Reversion
==========

There are four different benchmarks strategies implemented in the Online Portfolio Selection module.

1. Passive Aggressive Mean Reversion

2. Confidence Weighted Mean Reversion

3. Online Moving Average Reversion

4. Robust Median Reversion

Research Notebooks
##################

The following research notebooks provides a more detailed exploration of the algorithm as outlined at the back of Ch16 in
Advances in Financial Machine Learning.

* `Chapter 16 Exercise Notebook`_

.. _Chapter 16 Exercise Notebook: https://github.com/hudson-and-thames/research/blob/master/Chapter16/Chapter16.ipynb
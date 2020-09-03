.. _data_generation-introduction:

============
Introduction
============

This module includes implementations of synthetic data generation algorithms.

Historical financial data has many limitations. First, they are prohibitively expensive for many users. Acquiring historical, reliable, bias-free,
stock returns data can cost thousands of dollars, without taking into account infrastructure costs.

Next, financial data contains sensitive and personally identifiable attributes of customers. Sharing this data, even between the same organization,
can be difficult and restrictive.

Furthermore, they are biased due to historical events happening in only one way. They do not explore different possibilities and different scenarios.
Historical data is only available for one of the many branches of history.

Additionally, there is a lack of important events. For example, flash crashes, world-wide economic crises, global pandemics, etc.
Without this data, it is difficult to assess if an algorithm will fare well for any event. There is no easy to test for different (and realistic)
what-if scenarios. Having an abundance of realistic financial data also can help fight against the dangers of overfitting.

Examples of financial data that can be generated include stock prices, stock returns, correlation matrices, retail banking data,
and all kinds of market microstructure data.

We are trying to close that gap and generate realistic financial data.


Financial Correlation Matrices
##############################


Financial correlation matrices are constructed by using the correlation of stock returns over a specified time frame. Usually, the Pearson’s correlation
coefficient is used to measure their linear correlation and codependence.
Correlation matrices are useful for risk management, asset allocation, hedging instrument selection, pricing models, etc.

For example, a core assumption of the Capital Asset Pricing Model (CAPM) is that investors make investment decisions based on portfolios that have the highest return
for a given risk level (in essence, investors are "mean-variance optimizers.") Risk and return are measured by the variance and mean of the portfolio returns. One way to calculate the
variance of a portfolio is by using the covariance matrix of those returns.
Usually, this covariance matrix is estimated from historical data, which makes it subject to estimation errors and bias.

Correlation and covariance matrices share some properties. The relationship between correlation and covariance is explained by the following formula

.. math::
    p_{X, Y} = \frac{cov(X, Y)}{\sigma_X \sigma_Y}

Where :math:`p` is the Pearson correlation, :math:`cov` is the covariance. :math:`\sigma_X` is the standard deviation of :math:`X`. :math:`\sigma_Y` is the standard deviation of :math:`Y`

.. warning::

    Pearson correlation only captures linear effects. If two variables have strong non-linear dependency (squared or abs for example)
    Pearson correlation won't find any pattern between them. For information about capturing non-linear dependencies, read our :ref:`codependence-introduction`
    to Codependence.


.. _implementations-labeling_excess_mean:

==========================
Labelling Excess Over Mean
==========================

Using cross-sectional data on returns of many different stocks, each observation is labeled according to whether (or how much)
its return exceeds the mean return. It is a common practice to label observations based on whether the return is positive or negative.
However, this may produce unbalanced classes, as during market booms the probability of a positive return is much higher, and
during market crashes they are lower (Coqueret and Guida, 2020). Labeling according to a benchmark such as mean return
alleviates this issue.

A dataframe containing forward total stock returns is calculated from close prices. The mean return of all companies at time  :math:`t`  in the
dataframe is used to represent the market return, and excess returns are calculated by subtracting the mean return from each stock's return
over the time period  :math:`t`. The numerical returns can then be used as-is (for regression analysis), or can be relabeled to
represent their sign (for classification analysis).

At time :math:`t`:

.. math::
    :nowrap:

    \begin{gather*}
    R_t = \{r_{t,0}, r_{t,1}, ..., r_{t,n}\} \\

    \mu_t = mean(R_t) \\

    L(R_t) = \{r_{t,0} - \mu_t, r_{t,1} - \mu_t, ..., r_{t,n} - \mu_t\}
    \end{gather*}


If categorical rather than numerical labels are desired:

.. math::
     \begin{equation}
     \begin{split}
       L(r_{t,n}) = \begin{cases}
       -1 &\ \text{if} \ \ r_{t,n} - \mu_t < 0\\
       0 &\ \text{if} \ \ r_{t,n} - \mu_t = 0\\
       1 &\ \text{if} \ \ r_{t,n} - \mu_t > 0\\
       \end{cases}
     \end{split}
     \end{equation}


The following shows the distribution of numerical excess over mean for a set of 20 stocks for the time period between Jan 2019
and May 2020.

.. image:: labeling_images/distribution_over_mean.png
   :scale: 100 %
   :align: center

.. tip::
   **Underlying Literature**

    Labeling according to excess over mean is a labeling method mentioned in Chapter 5.5.1 of the book
    Machine Learning For Factor Investing, by Coqueret, G. and Guida, T., 2020.


Implementation
##############

.. py:currentmodule:: mlfinlab.labeling.excess_over_mean

.. automodule:: mlfinlab.labeling.excess_over_mean
   :members:

Example
########
Below is an example on how to create labels of excess over mean.

.. code-block::

    import pandas as pd
    import yfinance as yf
    from mlfinlab.labeling import excess_over_mean

    # Import price data
    tickers = "AAPL MSFT AMZN GOOG"
    data = yf.download(tickers, start="2019-01-01", end="2020-05-01", group_by="ticker")
    data = data.loc[:, (slice(None), 'Adj Close')]
    data.columns = data.columns.droplevel(1)

    # Get returns over mean numerically
    excess_over_mean(data)

    # Get returns over mean as a categorical label
    excess_over_mean(data, binary=True)


Research Notebooks
##################

The following research notebooks can be used to better understand labelling excess over mean.

* `Excess Over Mean Example`_

.. _`Excess Over Mean Example`: https://github.com/hudson-and-thames/research/blob/master/Labelling/Labels%20Excess%20Over%20Mean/excess_over_mean.ipynb

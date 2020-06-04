.. _implementations-labeling_vs_benchmark:

=========================
Return Versus Benchmark
=========================

Labeling against benchmark is a simple method of labeling financial data in which time-indexed returns are labeled according to
whether they exceed a benchmark. The benchmark can be either a constant value, or a Series of values with an index which matches
that of the returns. The labels can be the numerical value of how much each observation's return exceeds the benchmark, or the sign
of the excess.

The most simple method of labeling is simply giving the sign of the return. However, sometimes it is desirable to quantify the return
compared to a benchmark to better contextualize the returns. This is commonly done by using the mean or median of multiple stocks in the market.
However, that data may not always be available, and sometimes the user might wish a specify a more custom benchmark to compare returns against.

At time :math:`t`, given that price is :math:`p_t`, benchmark is :math:`B_t` and rate of return is:

.. math::
    r_t = \frac{p_t}{p_{t-1}} - 1

The labels are give by:

.. math::
    L(r_t) = r_t - B_t

If categorical labels are desired:

 .. math::
     \begin{equation}
     \begin{split}
       L_t = \begin{cases}
       -1 &\ \text{if} \ \ r_t < B_t\\
       0 &\ \text{if} \ \ r_t = B_t\\
       1 &\ \text{if} \ \ r_t > B_t\\
       \end{cases}
     \end{split}
     \end{equation}

The following shows the returns for MSFT stock during March-April 2020, compared to the return of SPY as a benchmark during
the same time period. Green dots represent days when MSFT outperformed SPY, and red dots represent days when MSFT underperformed
SPY.

.. image:: labeling_images/MSFT_return_vs_benchmark.png
   :scale: 100 %
   :align: center

Implementation
##############

.. py:currentmodule:: mlfinlab.labeling.return_vs_benchmark
.. automodule:: mlfinlab.labeling.return_vs_benchmark
   :members:

Example
########
Below is an example on how to use the Fixed Horizon labeling technique on real data.

.. code-block::

    import pandas as pd
    from mlfinlab.labeling import return_vs_benchmark

    # Import price data
    data = pd.read_csv('../Sample-Data/stock_prices.csv', index_col='Date', parse_dates=True)

    # Get returns
    ewg_returns = data['EWG'].pct_change()
    spy_returns = data['SPY'].pct_change()

    # Create labels using SPY as a benchmark
    labels = return_vs_benchmark(ewg_returns, benchmark=spy_returns)

    # Create labels categorically
    labels = return_vs_benchmark(ewg_returns, benchmark=spy_returns, binary=True)

Research Notebook
#################

The following research notebook can be used to better understand the Fixed Horizon labeling technique.

* `Return Over Benchmark Example`_

.. _`Return Over Benchmark Example`: https://github.com/hudson-and-thames/research/blob/master/Labelling/Labeling%20vs%20Benchmark/Labeling%20vs%20Benchmark.ipynb

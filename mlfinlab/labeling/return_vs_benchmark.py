"""
Return in excess of a given benchmark.

Chapter 5, Machine Learning for Factor Investing, by Coqueret and Guida, (2020).

Work "Evaluating multiple classifiers for stock price direction prediction" by Ballings et al. (2015) uses this method
to label yearly returns over a predetermined value to compare the performance of several machine learning algorithms.
"""
import numpy as np
import pandas as pd


def return_over_benchmark(prices, benchmark=0, binary=False, resample_by=None, lag=True):
    """
    Return over benchmark labeling method. Sourced from Chapter 5.5.1 of Machine Learning for Factor Investing,
    by Coqueret, G. and Guida, T. (2020).

    Returns a Series or DataFrame of numerical or categorical returns over a given benchmark. The time index of the
    benchmark must match those of the price observations.

    :param prices: (pd.Series or pd.DataFrame) Time indexed prices to compare returns against a benchmark.
    :param benchmark: (pd.Series or float) Benchmark of returns to compare the returns from prices against for labeling.
                    Can be a constant value, or a Series matching the index of prices. If no benchmark is given, then it
                    is assumed to have a constant value of 0.
    :param binary: (bool) If False, labels are given by their numerical value of return over benchmark. If True,
                labels are given according to the sign of their excess return.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :return: (pd.Series or pd.DataFrame) Excess returns over benchmark. If binary, the labels are -1 if the
            return is below the benchmark, 1 if above, and 0 if it exactly matches the benchmark.
    """
    # Apply resample, if applicable.
    if resample_by is not None:
        prices = prices.resample(resample_by).last()

    # Check that index of benchmark matches index of prices, if benchmark is a pd.Series.
    if isinstance(benchmark, pd.Series):
        assert prices.index.equals(benchmark.index), "Index of returns and benchmark do not match. If resampling is " \
                                                      "used, index of benchmark must match index of resampled data./" \

    # Get returns from prices.
    if lag:
        returns = prices.pct_change(periods=1).shift(-1)
    else:
        returns = prices.pct_change(periods=1)

    # Subtract the benchmark from returns.
    over_benchmark = returns.sub(benchmark, axis=0)

    if binary:
        over_benchmark = over_benchmark.apply(np.sign)
    return over_benchmark

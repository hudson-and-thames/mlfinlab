"""
Return in excess of a given benchmark

Chapter 5, Machine Learning for Factor Investing, by Coqueret and Guida, (2020).
"""
import warnings
import numpy as np
import pandas as pd


def return_over_benchmark(returns, benchmark=0, binary=False):
    """
    Return over benchmark labeling method. Sourced from Chapter 5.5.1 of Machine Learning for Factor Investing,
    by Coqueret, G. and Guida, T. (2020).

    Returns a Series or DataFrame of numerical or categorical returns over a given benchmark. The time index of the
    benchmark must match those of the price observations.

    :param returns: (pd.Series or pd.DataFrame) Time indexed prices to find labels for. NaN values are fine and will
                simply result in a NaN label. Use np.pct_change() or similar to get returns from prices.
    :param benchmark: (pd.Series or float) Benchmark prices to compare the given prices against for labeling. Can be a
                constant value, or a Series matching the index of prices. If no benchmark is given, then it is assumed
                to have a constant value of 0.
    :param binary: (bool) If False, labels are given by their numerical value of return over benchmark. If True,
                labels are given according to the sign of their excess return.
    :return: (pd.Series or pd.DataFrame) of excess returns over benchmark labels. If binary, the labels are -1 if the
            return is below the benchmark, 1 if above, and 0 if it exactly matches the benchmark.
    """
    # Check that index of benchmark matches index of prices, if benchmark is a pd.Series
    if isinstance(benchmark, pd.Series):
        if not returns.index.equals(benchmark.index):
            warnings.warn("Index of returns and benchmark do not match. May result in NaN labels.", UserWarning)

    # Subtract the benchmark from prices
    over_benchmark = returns.sub(benchmark, axis=0)

    if binary:
        over_benchmark = np.sign(over_benchmark)
    return over_benchmark

.. _implementations-labeling_trend_scanning:

==============
Trend Scanning
==============

.. image:: labeling_images/trend_scanning_plot.png
   :scale: 100 %
   :align: center

Trend Scanning is both a classification and regression labeling technique introduced by Marcos Lopez de Prado in the
following lecture slides: `Advances in Financial Machine Learning, Lecture 3/10`_, and again in his text book `Machine Learning for Asset Managers`_.

.. _Advances in Financial Machine Learning, Lecture 3/10: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419
.. _Machine Learning for Asset Managers: https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545

For some trading algorithms, the researcher may not want to explicitly set a fixed profit / stop loss level, but rather detect overall
trend direction and sit in a position until the trend changes. For example, market timing strategy which holds ETFs except during volatile
periods. Trend scanning labels are designed to solve this type of problems.

This algorithm is also useful for defining market regimes between downtrend, no-trend, and uptrend.

The idea of trend-scanning labels are to fit multiple regressions from time t to t + L (L is a maximum look-forward window)
and select the one which yields maximum t-value for the slope coefficient, for a specific observation.

.. tip::
    1. Classification: By taking the sign of t-value for a given observation we can set {-1, 1} labels to define the trends as either downward or upward.
    2. Classification: By adding a minimum t-value threshold you can generate {-1, 0, 1} labels for downward, no-trend, upward.
    3. The t-values can be used as sample weights in classification problems.
    4. Regression: The t-values can be used in a regression setting to determine the magnitude of the trend.

The output of this algorithm is a DataFrame with t1 (time stamp for the farthest observation), t-value, returns for the trend, and bin.

Implementation
##############

.. py:currentmodule:: mlfinlab.labeling.trend_scanning
.. automodule:: mlfinlab.labeling.trend_scanning
   :members:

Example
########
.. code-block::

    import numpy as np
    import pandas as pd

    from mlfinlab.labeling import trend_scanning_labels

    self.eem_close = pd.read_csv('./test_data/stock_prices.csv', index_col=0, parse_dates=[0])
    # In 2008, EEM had some clear trends
    self.eem_close = self.eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.Timestamp(2008, 10, 1)]


    t_events = self.eem_close.index # Get indexes that we want to label
    # We look at a maximum of the next 20 days to define the trend, however we fit regression on samples with length >= 10
    tr_scan_labels = trend_scanning_labels(self.eem_close, t_events, look_forward_window=20, min_sample_length=10)
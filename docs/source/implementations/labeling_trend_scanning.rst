.. _implementations-labeling_trend_scanning:

==========================================
Labeling: Trend Scanning
==========================================

Trend scanning is a classification labeling technique introduced by Marcos Lopez de Prado in the following lecture slides: `Advances in Financial Machine Learning, Lecture 3/10`_

.. _Advances in Financial Machine Learning, Lecture 3/10: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419

For some trading algorithms the researcher does not want to explicitly set fix profit / stop loss levels, but rather detect overall trend direction and sit in a position until
the trend changes. For example, market timing strategy which holds ETF except volatile periods. Trend scanning labels are designed to solve this type of problems.

The idea of trend-scanning label is to fit multiple regressions from time t to t + L (L is a maximum look-forward window) and define the one which yields maximum t-value for
slope coefficient. By taking the sign of return between time with maximum t-value and time t we define the value of trend-scanning label.

.. image:: labeling_images/trend_scanning_plot.png
   :scale: 100 %
   :align: center


Implementation
~~~~~~~~~~~~~~

.. py:currentmodule:: mlfinlab.labeling.trend_scanning
.. automodule:: mlfinlab.labeling.trend_scanning
   :members:

::

    import numpy as np
    import pandas as pd

    from mlfinlab.labeling import trend_scanning_labels

    self.eem_close = pd.read_csv('./test_data/stock_prices.csv', index_col=0, parse_dates=[0])
    # In 2008, EEM had some clear trends
    self.eem_close = self.eem_close['EEM'].loc[pd.Timestamp(2008, 4, 1):pd.Timestamp(2008, 10, 1)]

    t_events = self.eem_close.index
    # We look at next 20 days to define trend, however we fit regression on samples with length >= 10
    tr_scan_labels = trend_scanning_labels(self.eem_close, t_events, look_forward_window=20, min_sample_length=10)
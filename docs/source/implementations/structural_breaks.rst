.. _implementations-structural_breaks:

=================
Structural Breaks
=================

This implementation is based on Chapter 17 of the book Advances in Financial Machine Learning. Structural breaks, like
the transition from one market regime to another, represent the shift in behaviour of market participants. They can be
classified in two general categories:

1. **CUSUM tests**: These test whether the cumulative forecasting errors significantly deviate from white noise.
2. **Explosiveness tests**: Beyond deviation from white noise, these test whether the process exhibits exponential
   growth or collapse, as this is inconsistent with a random walk or stationary process, and it is unsustainable in the long run.


.. figure:: structural_breaks_images/sadf.png
   :scale: 70 %
   :align: center
   :figclass: align-center
   :alt: structural breaks

   Image showing SADF test statistic with 5 lags and linear model.

.. py:currentmodule:: mlfinlab.structural_breaks.chow
.. automodule:: mlfinlab.structural_breaks.chow
   :members:

.. py:currentmodule:: mlfinlab.structural_breaks.cusum
.. automodule:: mlfinlab.structural_breaks.cusum
   :members:

.. py:currentmodule:: mlfinlab.structural_breaks.sadf
.. automodule:: mlfinlab.structural_breaks.sadf
   :members:

Examples
########

.. code-block::

    import pandas as pd
    import numpy as np
    from mlfinlab.structural_breaks import (get_chu_stinchcombe_white_statistics,
                                            get_chow_type_stat, get_sadf)

    bars = pd.read_csv('BARS_PATH', index_col = 0, parse_dates=[0])
    log_prices = np.log(bars.close) # see p.253, 17.4.2.1 Raw vs Log Prices

    # Chu-Stinchcombe test
    one_sided_test = get_chu_stinchcombe_white_statistics(log_prices, test_type='one_sided')
    two_sided_test = get_chu_stinchcombe_white_statistics(log_prices, test_type='two_sided')

    # Chow-type test
    chow_stats = get_chow_type_stat(log_prices, min_length=20)

    # SADF test
    linear_sadf = get_sadf(log_prices, model='linear', add_const=True, min_length=20, lags=5)

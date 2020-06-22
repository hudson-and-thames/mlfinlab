.. _statistical_arbitrage-cointegration:

=============
Cointegration
=============

Engle-Granger Test
##################

.. py:currentmodule:: mlfinlab.statistical_arbitrage.cointegration

.. autofunction:: calc_engle_granger

Johansen Test
#############

.. py:currentmodule:: mlfinlab.statistical_arbitrage.cointegration

.. autofunction:: calc_johansen

Example Code
************

.. code-block::

    import pandas as pd
    import numpy as np
    from mlfinlab.statistical_arbitrage import calc_engle_granger, calc_johansen

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Change to log prices data.
    stock_prices = np.log(stock_prices)

    # Calculate engle granger test with second and third column .
    engle_granger = calc_engle_granger(stock_prices.iloc[:, 2], stock_prices.iloc[:, 3])

    # Calculate johansen test with the first two columns, constant trend, and lag of 1.
    johansen = calc_johansen(stock_prices.iloc[:, 0:2], 0, 1)


========
Datasets
========

Mlfinlab package contains various financial datasets which can be used by a researcher as sandbox data.

Tick sample
###########

Mlfinlab provides a sample of tick data for E-Mini S&P 500 futures which can be used to test bar compression algorithms,
microstructural features, etc. Tick data sample consists of Timestamp, Price and Volume.

.. py:currentmodule:: mlfinlab.datasets.load_datasets
.. autofunction:: load_tick_sample


Dollar bars sample
##################

We also provide a sample of dollar bars for E-Mini S&P 500 futures. Data set structure:

    - Open price (open)
    - High price (high)
    - Low price (low)
    - Close price (close)
    - Volume (cum_volume)
    - Dollar volume traded (cum_dollar)
    - Number of ticks inside of bar (cum_ticks)

.. tip::
   You can find more information on dollar bars and other bar compression algorithms in
   `Data Structures <https://mlfinlab.readthedocs.io/en/latest/implementations/data_structures.html>`_

.. py:currentmodule:: mlfinlab.datasets.load_datasets
.. autofunction:: load_dollar_bar_sample




ETF prices
##########

.. py:currentmodule:: mlfinlab.datasets.load_datasets
.. autofunction:: load_dollar_bar_sample

The data set consists of close prices for: EEM, EWG, TIP, EWJ, EFA, IEF, EWQ, EWU, XLB, XLE, XLF, LQD, XLK, XLU, EPP,
FXI, VGK, VPL, SPY, TLT, BND, CSJ, DIA starting from 2008 till 2016. It can be used to test and validate portfolio
optimization techniques.


Example
#######

.. code-block::

   from mlfinlab.datasets import (load_tick_sample, load_stock_prices, load_dollar_bar_sample)

   tick_df = load_tick_sample()
   dollar_bars_df = load_dollar_bar_sample()
   stock_prices_df = load_stock_prices()

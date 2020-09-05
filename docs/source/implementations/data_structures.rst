.. _implementations-data_structures:

.. note::
    This section includes accompanying Jupyter Notebook Tutorials that are now available via the respective tier on
    `Patreon <https://www.patreon.com/HudsonThames>`_.

===============
Data Structures
===============

When analyzing financial data, unstructured data sets, in this case tick data, are commonly transformed into a structured
format referred to as bars, where a bar represents a row in a table. mlfinlab implements tick, volume, and dollar bars
using traditional standard bar methods as well as the less common information driven bars.

Standard Bars
#############

The four standard bar methods implemented share a similar underlying idea in that they take a sample of data after a
certain threshold is reached and they all result in a time series of Open, High, Low, and Close data.

1. Time bars, are sampled after a fixed interval of time has passed.
2. Tick bars, are sampled after a fixed number of ticks have taken place.
3. Volume bars, are sampled after a fixed number of contracts (volume) has been traded.
4. Dollar bars, are sampled after a fixed monetary amount has been traded.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 25) to build the more interesting features for predicting financial time series data.

.. tip::
   A fundamental paper that you need to read to have a better grasp on these concepts is:
   `Easley, David, Marcos M. López de Prado, and Maureen O’Hara. "The volume clock: Insights into the high-frequency
   paradigm." The Journal of Portfolio Management 39.1 (2012): 19-29. <https://jpm.pm-research.com/content/39/1/19.abstract>`_

.. tip::
   A threshold can be either fixed (given as ``float``) or dynamic (given as ``pd.Series``). If a dynamic threshold is used
   then there is no need to declare threshold for every observation. Values are needed only for the first observation
   (or any time before it) and later at times when the threshold is changed to a new value.
   Whenever sampling is made, the most recent threshold level is used.

   **An example for volume bars**
   We have daily observations of prices and volumes:

   +------------+------------+-----------+
   | Time       | Price      | Volume    |
   +============+============+===========+
   | 20.04.2020 | 1000       | 10        |
   +------------+------------+-----------+
   | 21.04.2020 | 990        | 10        |
   +------------+------------+-----------+
   | 22.04.2020 | 1000       | 20        |
   +------------+------------+-----------+
   | 23.04.2020 | 1100       | 10        |
   +------------+------------+-----------+
   | 24.04.2020 | 1000       | 10        |
   +------------+------------+-----------+

   And we set a dynamic threshold:

   +------------+------------+
   | Time       | Threshold  |
   +============+============+
   | 20.04.2020 | 20         |
   +------------+------------+
   | 23.04.2020 | 10         |
   +------------+------------+

   The data will be sampled as follows:

   - 20.04.2020 and 21.04.2020 into one bar, as their volume is 20.
   - 22.04.2020 as a single bar, as its volume is 20.
   - 23.04.2020 as a single bar, as it now fills the lower volume threshold of 10.
   - 24.04.2020 as a single bar again.

Time Bars
*********

These are the traditional open, high, low, close bars that traders are used to seeing. The problem with using this sampling
technique is that information doesn't arrive to market in a chronological clock, i.e. news event don't occur on the hour - every hour.

It is for this reason that Time Bars have poor statistical properties in comparison to the other sampling techniques.

.. py:currentmodule:: mlfinlab.data_structures.time_data_structures
.. autofunction:: get_time_bars


Tick Bars
*********

.. py:currentmodule:: mlfinlab.data_structures.standard_data_structures
.. autofunction:: get_tick_bars

.. code-block::

	from mlfinlab.data_structures import standard_data_structures

	# Tick Bars
	tick = standard_data_structures.get_tick_bars('FILE_PATH', threshold=5500,
	                                               batch_size=1000000, verbose=False)


Volume Bars
***********

.. py:currentmodule:: mlfinlab.data_structures.standard_data_structures
.. autofunction:: get_volume_bars


.. code-block::

	from mlfinlab.data_structures import standard_data_structures

	# Volume Bars
	volume = standard_data_structures.get_volume_bars('FILE_PATH', threshold=28000,
                                                      batch_size=1000000, verbose=False)


Dollar Bars
***********

.. py:currentmodule:: mlfinlab.data_structures.standard_data_structures
.. autofunction::  get_dollar_bars

.. code-block::

	from mlfinlab.data_structures import standard_data_structures

	# Dollar Bars
	dollar = standard_data_structures.get_dollar_bars('FILE_PATH', threshold=70000000,
	                                                   batch_size=1000000, verbose=True)

Statistical Properties
**********************

The chart below that tick, volume, and dollar bars all exhibit a distribution significantly closer to normal - versus
standard time bars:

.. image:: normality_graph.png
   :scale: 70 %
   :align: center

|

------------------------------------

|

.. note::
    This documentation and accompanying Jupyter Notebook Tutorials are now available via the respective tiers on
    `Patreon <https://www.patreon.com/HudsonThames>`_.

Information-Driven Bars 🔒
##########################

Imbalance Bars
**************

Imbalance Bars Generation Algorithm
===================================

Algorithm Logic
===============

Implementation
==============

Example
=======


-----------------------------

|

Run Bars
********


Implementation
==============


Example
=======


|

-----------------------

|

Research Notebooks 🔒
#####################

.. note::
    These and other accompanying Jupyter Notebook Tutorials are now available via the respective tier on
    `Patreon <https://www.patreon.com/HudsonThames>`_.

The following research notebooks can be used to better understand the previously discussed data structures

Standard Bars
*************

* `Getting Started`_
* `Sample Techniques`_

.. _Getting Started: https://github.com/Hudson-and-Thames-Clients/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Financial%20Data%20Structures/Getting%20Started.ipynb
.. _Sample Techniques: https://github.com/Hudson-and-Thames-Clients/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Financial%20Data%20Structures/Sample_Techniques.ipynb

Imbalance Bars
**************

* `Imbalance Bars`_

.. _Imbalance Bars: https://github.com/Hudson-and-Thames-Clients/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Financial%20Data%20Structures/Dollar-Imbalance-Bars.ipynb

|

---------------------

|

Data Preparation Tutorial
#########################

First import your tick data.

.. code-block::

   # Required Imports
   import numpy as np
   import pandas as pd

   data = pd.read_csv('data.csv')

In order to utilize the bar sampling methods presented below, our data must first be formatted properly.
Many data vendors will let you choose the format of your raw tick data files. We want to only focus on the following
3 columns: date_time, price, volume. The reason for this is to minimise the size of the csv files and the amount of time
when reading in the files.

Our data is sourced from TickData LLC which provides software called TickWrite, to aid in the formatting of saved files.
This allows us to save csv files in the format date_time, price, volume. (If you don't use TickWrite then make sure to pre-format your files)

For this tutorial we will assume that you need to first do some pre-processing and then save your data to a csv file.

.. code-block::

   # Don't convert to datetime here, it will take forever to convert
   # on account of the sheer size of tick data files.
   date_time = data['Date'] + ' ' + data['Time']
   new_data = pd.concat([date_time, data['Price'], data['Volume']], axis=1)
   new_data.columns = ['date', 'price', 'volume']


Initially, your instinct may be to pass an in-memory DataFrame object but the truth is when you're running the function
in production, your raw tick data csv files will be way too large to hold in memory. We used the subset 2011 to 2019 and
it was more than 25 gigs. It is for this reason that the mlfinlab package suggests using a file path to read the raw data
files from disk.

.. code-block::

	# Save to csv
	new_data.to_csv('FILE_PATH', index=False)

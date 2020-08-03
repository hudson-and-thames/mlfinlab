
=================
Barriers to Entry
=================

As most of you know, getting through the first 3 chapters of the book is challenging as it relies on HFT data to 
create the new financial data structures. Sourcing the HFT data is very difficult and thus we have resorted to purchasing
the full history of S&P500 Emini futures tick data from `TickData LLC`_.

We are not affiliated with TickData in any way but would like to recommend others to make use of their service. The full
history cost us about $750 and is worth every penny. They have really done a great job at cleaning the data and providing
it in a user friendly manner.

.. _TickData LLC: https://www.tickdata.com/


Sample Data
###########

TickData does offer about 20 days worth of raw tick data which can be sourced from their website `link`_.

For those of you interested in working with a two years of sample tick, volume, and dollar bars, it is provided for in the `research repo`_.

You should be able to work on a few implementations of the code with this set. 

.. _link: https://s3-us-west-2.amazonaws.com/tick-data-s3/downloads/ES_Sample.zip
.. _research repo: https://github.com/hudson-and-thames/research/tree/master/Sample-Data


Additional Sources
##################

Searching for free tick data can be a challenging task. The following three sources may help:

1. `Dukascopy`_. Offers free historical tick data for some futures, though you do have to register.
2. Most crypto exchanges offer tick data but not historical (see `Binance API`_). So you'd have to run a script for a few days.
3. `Blog Post`_: How and why I got 75Gb of free foreign exchange “Tick” data.

.. _Dukascopy: https://www.dukascopy.com/swiss/english/marketwatch/historical/
.. _Binance API: https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md
.. _Blog Post: https://towardsdatascience.com/how-and-why-i-got-75gb-of-free-foreign-exchange-tick-data-9ca78f5fa26c

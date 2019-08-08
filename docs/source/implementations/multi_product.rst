.. _implementations-multi_product:

====================
Multi-Product Series
====================

When modeling multi-product series we are faced with many unique challenges which alter the nature of the underlying time series. For example, when working with a series of instruments, weights often need to be adjusted dynamically over time. Additionally, the constituents of a multi-product series may pay out irregular coupons or dividends. mlfinlab provides a solution known as the "ETF trick"  which helps us avoid the pitfalls of many of these scenarios.


ETF Trick
=========

The ETF trick solves many of the issues that arise when dealing with multi-product series by transforming a multi-product dataset into a single dataset resembling a return ETF. This allows for the series to be traded as if it were a cashlike product regardless of the underlying complexity. An implementation of the ETF trick using mlfinlab can be seen below.

The following contains logic of vectorised ETF trick implementation. It can be used for both memory data frames (pd.DataFrame) and csv files. 
All data frames, files should be processed in a specific format, described in the examples.

.. class::  ETFTrick(self, open_df, close_df, alloc_df, costs_df, rates_df=None, index_col=0)

    :param open_df: (pd.DataFrame or string): open prices data frame or path to csv file, corresponds to o(t) from the book
    :param close_df: (pd.DataFrame or string): close prices data frame or path to csv file, corresponds to p(t)
    :param alloc_df: (pd.DataFrame or string): asset allocations data frame or path to csv file (in # of contracts), corresponds to w(t)
    :param costs_df: (pd.DataFrame or string): rebalance, carry and dividend costs of holding/rebalancing the position, corresponds to d(t)
    :param rates_df: (pd.DataFrame or string): dollar value of one point move of contract includes exchange rate, 
     futures contracts multiplies). Corresponds to phi(t)
     For example, 1$ in VIX index, equals 1000$ in VIX futures contract value.
     If None then trivial (all values equal 1.0) is generated
    :param index_col: (int): positional index of index column. Used for to determine index column in csv files

The ETFTrick object supports the following method to generate the ETF trick series

.. method::   get_etf_series(self, batch_size=1e5)

    :param batch_size: Size of the batch that you would like to make use of
    :return: pandas Series with ETF trick values starting from 1.0

An end-to-end example showing how the ETF trick can be implemented can be seen below:

SPX/EuroStoxx Hedging Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's see how mlfinlab's ETF trick is used to solve the exercise 2.3 from Chapter 2. For this exercise we use daily SPY and EUROSTOXX futures data and EUR/USD exchange rates from `link`_. Hedging weights are recalculated on a daily basis.

::

	#imports
	import pandas as pd
	import matplotlib.pyplot as plt
	import numpy as np
	import datetime as dt
	from mlfinlab.multi_product.etf_trick import ETFTrick

::

	def generate_cov_mat(row):
	    """
	    Forms covariance matrix from current data frame row using 'rolling_cov', 
	    'rolling_spx_var' and 'rolling_eur_var' column values
	    """
	    cov = row['rolling_cov']
	    spx_var = row['rolling_spx_var']
	    euro_var = row['rolling_euro_var']
	    return np.matrix([[spx_var, cov], [cov, euro_var]])

::

	# Snippet 2.1 from a book
	def pca_weights(cov, riskDist=None, risk_target = 1.):
	    """
	    Calculates hedging weights using covariance matrix(cov), risk distribution(riskDist) and risk_target
	    """
	    eVal, eVec = np.linalg.eigh(cov)
	    indices = eVal.argsort()[::-1]
	    eVal, eVec = eVal[indices], eVec[:, indices]
	    if riskDist is None:
	        riskDist = np.zeros(cov.shape[0])
	        riskDist[-1] = 1.
	    loads = risk_target * (riskDist/eVal)**.5
	    wghts = np.dot(eVec, np.reshape(loads, (-1,1)))
	    return wghts


.. _link: https://www.investing.com

Data Preprocessing
******************

Read in Data

::

	spx = pd.read_csv('FILE_PATH', usecols=['Date', 'Price', 'Open'])
	euro = pd.read_csv('FILE_PATH', usecols=['Date', 'Price', 'Open'])
	eur_usd = pd.read_csv('FILE_PATH', usecols = ['Date', 'Price'])

Rename columns to universal format

::

	spx.rename(columns = {'Date': 'date', 'Price': 'close', 'Open': 'open'}, inplace=True) 
	euro.rename(columns = {'Date': 'date', 'Price': 'close', 'Open': 'open'}, inplace=True)
	eur_usd.rename(columns = {'Date': 'date', 'Price': 'close'}, inplace=True)

Convert date column to datetime format

::

	spx['date'] = pd.to_datetime(spx.date)
	euro['date'] = pd.to_datetime(euro.date)
	eur_usd['date'] = pd.to_datetime(eur_usd.date)

Convert price data from strings to float (investing.com uses specific decimal sep format)

::

	spx.close = spx.close.apply(lambda x: x.replace(',', '')).astype(float)
	euro.close = euro.close.apply(lambda x: x.replace(',', '')).astype(float)
	spx.open = spx.open.apply(lambda x: x.replace(',', '')).astype(float)
	euro.open = euro.open.apply(lambda x: x.replace(',', '')).astype(float)

Set and sort index

::

	spx.set_index('date', inplace=True)
	euro.set_index('date', inplace=True)
	eur_usd.set_index('date', inplace=True)

	spx.sort_index(inplace=True)
	euro.sort_index(inplace=True)
	eur_usd.sort_index(inplace=True)

Exchange rate is needed only for dates when futures are traded

::

	eur_usd = eur_usd[eur_usd.index.isin(spx.index)]

Generate Covariances and Hedging Weights
****************************************

Init data frame with covariance and price data
::

	cov_df = pd.DataFrame(index=spx.index)

::

	cov_df.loc[spx.index, 'spx_close'] = spx.loc[:, 'close']
	cov_df.loc[euro.index, 'euro_close'] = euro.loc[:, 'close']
	cov_df.loc[spx.index, 'spx_open'] = spx.loc[:, 'open']
	cov_df.loc[euro.index, 'euro_open'] = euro.loc[:, 'open']
	cov_df.loc[eur_usd.index, 'eur_usd'] = eur_usd.loc[:, 'close']
	# we need to calculate EUROSTOXX returns adjusted for FX rate
	cov_df['euro_fx_adj'] = cov_df.euro_close / cov_df.eur_usd
	cov_df['spx'] = cov_df.spx_close.pct_change().fillna(0)
	cov_df['euro'] = cov_df.euro_fx_adj.pct_change().fillna(0)

Fill missing values with previous ones and sort index
::

	cov_df.update(cov_df.loc[:, ['spx', 'euro', 'spx_close', 'spx_open', 'euro_close', 'euro_open', 'eur_usd']].fillna(method='pad'))
	cov_df.sort_index(inplace=True)

Get 252 rolling covariance between SPY and EUROSTOXX, rolling variances
::

	cov_df['rolling_cov'] = cov_df['spx'].rolling(window=252).cov(cov_df['euro']) 
	cov_df['rolling_spx_var'] = cov_df['spx'].rolling(window=252).var()
	cov_df['rolling_euro_var'] = cov_df['euro'].rolling(window=252).var()

Iterate over cov_df and on each step define hedging weights using pca_weights function
::

	cov_df.dropna(inplace=True)
	for index, row in cov_df.iterrows():
	    mat = generate_cov_mat(row)
	    w = pca_weights(mat)
	    cov_df.loc[index, 'spx_w'] = w[0]
	    cov_df.loc[index, 'euro_w'] = w[1]


Prepare Data Set for ETF Trick
******************************

ETFTrick class requires open_df, close_df, alloc_df, costs_df, rates_df. Each of these data frames should be in a specific format:

1) DateTime index
2) Each column name corresponds to a ticker, number of columns in all data frames should be equal. 

In our case all data frames contain columns: 'spx' and 'euro'.

For example, we implement etf trick with 5 securities: A, B, C, D, E. 

If for the first two years only A, B and C close data is available while for the last two years only D and E data is available, close data frame format will be::

	index        A    B     C     D     E
	Year 1      22.0 7.52  7.5   NaN   NaN
	Year 1      20.7 7.96  8.2   NaN   NaN
	.....
	Year N      NaN  NaN   NaN   0.3   100.5


* **open_df**: open prices (in contract currency)
* **close_df**: close prices
* **alloc_df**: securities allocation vector
* **costs_df**: costs of holding/rebalancing the position
* **rates_df**: $ value of 1 point move of contract price. This includes exchange rates, futures multipliers

Create open_df and close_df

::

	open_df = cov_df[['spx_open', 'euro_open']]
	open_df.rename(columns = {'spx_open': 'spx', 'euro_open': 'euro'}, inplace=True)
	close_df = cov_df[['spx_close', 'euro_close']]
	close_df.rename(columns = {'spx_close': 'spx', 'euro_close': 'euro'}, inplace=True)

We need USD_EUR = 1/EUR_USD rate for EUROSTOXX price movements

::

	rates_df = 1/cov_df[['eur_usd']]
	rates_df.rename(columns = {'eur_usd': 'euro'}, inplace=True)
	rates_df['spx'] = 1.0

Allocations data frame with weights generated using pca_weights()
::

	alloc_df = cov_df[['spx_w', 'euro_w']]
	alloc_df.rename(columns={'spx_w': 'spx', 'euro_w': 'euro'}, inplace=True)

Let's assume zero rebalancing costs
::

	costs_df = alloc_df.copy()
	costs_df['spx'] = 0.0
	costs_df['euro'] = 0.0

::

	# In_memory means that all data frames are stored in memory
	# If False open_df should be a path to open data frame
	trick = ETFTrick(open_df, close_df, alloc_df,
	                costs_df, rates_df, in_memory=True)

::

	trick_series = trick.get_etf_series()


Research Notebook
==================

The following research notebook can be used to better understand the ETF trick

ETF Trick
~~~~~~~~~

* `ETF Trick Hedge`_

.. _ETF Trick Hedge: https://github.com/hudson-and-thames/research/blob/master/Chapter2/2019_04_10_ETF_trick_hedge.ipynb




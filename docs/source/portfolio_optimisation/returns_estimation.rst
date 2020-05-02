.. _portfolio_optimisation-returns_estimation:


=====================
Estimation of Returns
=====================

Accurate estimation of historical asset returns is one of the most important aspects of portfolio optimisation. At the same, it is
also one of the most difficult to calculate since most of the times, estimated returns do not correctly reflect the true underlying
returns of a portfolio/asset. Given this, there is still significant research work being published dealing with novel methods to
estimate returns and we wanted to share some of these methods with the users of mlfinlab.

This class provides functions to estimate mean asset returns. Currently, it is still in active development and we
will keep adding new methods to it.

Simple returns
##############

The `calculate_returns` function allows calculating a dataframe of returns from a dataframe of prices.
The calculation is done in the following way:

.. math::

      R_{t} = \frac{P_{t}-P_{t-1}}{P_{t-1}}

Where :math:`R_{t}` is the return for :math:`t` -th observation, and :math:`P_{t}` is the price for :math:`t` -th observation.

Annualized mean historical returns
##################################

The `calculate_mean_historical_returns` function allows calculating a mean annual return for every element in a dataframe of prices.
The calculation is done in the following way:

.. math::
      :nowrap:

      \begin{align*}
      R_{t} &= \frac{P_{t}-P_{t-1}}{P_{t-1}}
      \end{align*}

      \begin{align*}
      AnnualizedMeanReturn &= \frac{\sum_{t=0}^{T}{R_{t}}}{T} * N
      \end{align*}

Where :math:`R_{t}` is the return for :math:`t` -th observation, and :math:`P_{t}` is the price for :math:`t` -th observation,
:math:`T` is the total number of observations, :math:`N` is an average number of observations in a year.

Exponentially-weighted annualized mean of historical returns
############################################################

The `calculate_exponential_historical_returns` function allows calculating the exponentially-weighted mean annual return for every element in a dataframe of prices.
The calculation is done in the following way:

.. math::
      :nowrap:

      \begin{align*}
      R_{t} = \frac{P_{t}-P_{t-1}}{P_{t-1}}
      \end{align*}

      \begin{align*}
      Decay = \frac{2}{span+1}
      \end{align*}

      \begin{align*}
      EWMA(R)_{t} = ((R_{t} - R_{t-1}) * Decay) + R_{t-1}
      \end{align*}

      \begin{align*}
      ExponentialAnnualizedMeanReturn_{Decay} = EWMA(R)_{T} * N
      \end{align*}

Where :math:`R_{t}` is the return for :math:`t` -th observation, :math:`P_{t}` is the price for :math:`t` -th observation,
:math:`T` is the total number of observations, :math:`N` is an average number of observations in a year, :math:`EWMA(R)_{t}` is the
:math:`t` -th observation of exponentially-weighted moving average of :math:`R` .


Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.returns_estimators

    .. autoclass:: ReturnsEstimation
        :members:

        .. automethod:: __init__

Example
########
Below is an example of how to use the package functions to calculate various estimators of returns for a portfolio.

.. code-block::

    import pandas as pd
    from mlfinlab.portfolio_optimization import ReturnsEstimation

    # Import dataframe of prices for assets in a portfolio
    asset_prices = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

    # Class that contains needed functions
    ret_est = ReturnsEstimation()

    # Calculate simple returns
    assets_returns = ret_est.calculate_returns(asset_prices)

    # Calculate annualised mean historical returns for daily data
    assets_annual_returns = ret_est.calculate_mean_historical_returns(asset_prices, frequency=252)

    # Calculate exponentially-weighted annualized mean of historical returns for daily data and span of 200
    assets_exp_annual_returns = ret_est.calculate_exponential_historical_returns(asset_prices,
                                                                                 frequency=252,
                                                                                 span=200)

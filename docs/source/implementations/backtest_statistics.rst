.. _implementations-backtest_statistics:

===================
Backtest Statistics
===================

Backtest statistics module contains functions related to characteristics analysis of returns and target positions. These include filtering flips and flattenings from series of returns, calculating average holding period from series of positions, concentration of bets for positive and negative returns, drawdown and time under water calculation, Sharpe ratios - annualised, probabalistic, deflated, minimum required track record length. Each statistic has it's own function and small description of statistic in docstring and detailed list of inputs required.

Introduction
==============================
More information on functions implemented in this part, as well as use cases and deeper results descriptions can be found in the following sources:

- **Advances in Financial Machine Learning** *by* Marcos Lopez de Prado. *Page numbers in the code are referring to the pages in this book.*

- **The Sharpe Ratio Efficient Frontier** *by* David H. Bailey *and* Marcos Lopez de Prado `available here <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643>`_. *Provides a deeper understanding of Sharpe ratios implemented and Minimum track record length.*

Flattening and Flips
==============================

Snippet 14.1, page 197, Deriving Flattening and Flips from series of target positions.

Points of Flattening - When target position changes sign (For example, changing from 1.5 (long position) to -0.5 (short position) on th next timestamp)

Points of Flipping - When target position changes from nonzero to zero (For example, changing from 1.5 (long position) to 0 (no positions) on the next timestamp)

Filtering timestamps of Flattenings and Flips may be used for analysis of position changing frequency as well as checking moments when positions are closed by the strategy.

Note that function adds the last timestamp to the resulting list of timestamps (as the final position is being closed).

.. function:: timing_of_flattening_and_flips(target_positions: pd.Series) -> pd.DatetimeIndex:

    :param target_positions: (pd.Series) target position series with timestamps as indices
    :return: (pd.DatetimeIndex) timestamps of trades flattening, flipping and last bet

An example showing how Flattening and Flips function is used can be seen below::

	from mlfinlab.backtest_statistics import timing_of_flattening_and_flips

	flattening_and_flips_timestamps = timing_of_flattening_and_flips(target_positions)

Average Holding Period
==============================

Snippet 14.2, page 197, Deriving average holding period of a position.

Estimates the average holding period (in days) of a strategy, given series of target positions using average entry time pairing algorithm.
Parameters of the algorithm are calculated as follows:

1. When the size of the position in increasing


Updating EntryTime - time when a trade was opened, adjusted by increases in positions. Takes into account weight of the position increase.

        :math:`EntryTime_{0} = \frac{EntryTime_{-1}*Weight_{-1} + TimeSinceTradeStart*(Weight_{0}-Weight_{-1})}{Weight_{0}}`

2. When the size of a bet is decreasing


Capturing the :math:`HoldingTime = (EntryTime - CurrentTime)` as well as :math:`Weight` of the closed position.
If entire position is closed, setting :math:`EntryTime` to :math:`CurrentTime`.

3. Finally, calculating, using values captured in step 2


        :math:`AverageHoldingTime = \frac{\sum_{i}(HoldingTime_{i}*Weight_{i})}{\sum_{i}Weight_{i}}`

If no closed trades in the series, output is :math:`nan`

.. function:: average_holding_period(target_positions: pd.Series) -> float:

    :param target_positions: (pd.Series) target position series with timestamps as indices
    :return: (float) estimated average holding period, NaN if zero or unpredicted

An example showing how Average Holding Period function is used can be seen below::

	from mlfinlab.backtest_statistics import average_holding_period

	avg_holding_period = average_holding_period(target_positions)

Bets Concentration
==============================

Snippet 14.3, page 201, Derives the concentration of returns.

Concentration of returns measures the uniformity of returns from bets. Metric is incpired by Herfindahl-Hirschman Index and is calculated as follows:

        :math:`Weight_{i} = \frac{Return_{i}}{\sum_{i}Return_{i}}`

        :math:`SumSquares = \sum_{i}Weight_{i}^2`

        :math:`HHI = \frac{SumSquares - \frac{1}{i}}{1 - \frac{1}{i}}`

The closer concentration is to 0, the more uniform are the returns (When 0, returns are uniform). If concentration is close to 1, returns highly concentrated (When 1, only one non-zero return).

Returns :math:`nan` if less than 3 returns in series.

.. function:: bets_concentration(returns: pd.Series) -> float:

    :param returns: (pd.Series) returns from bets
    :return: (float) concentration of returns (nan if less than 3 returns)

An example showing how Bets Concentration function is used can be seen below::

	from mlfinlab.backtest_statistics import bets_concentration

	concentration = bets_concentration(returns)

All Bets Concentration
==============================

Snippet 14.3, page 201, Derives a more detailed concentration of returns.

Concentration of returns measures the uniformity of returns from bets. Metric is incpired by Herfindahl-Hirschman Index and is calculated as follows:

        :math:`Weight_{i} = \frac{Return_{i}}{\sum_{i}Return_{i}}`

        :math:`SumSquares = \sum_{i}Weight_{i}^2`

        :math:`HHI = \frac{SumSquares - \frac{1}{i}}{1 - \frac{1}{i}}`

The closer concentration is to 0, the more uniform are the returns (When 0, returns are uniform). If concentration is close to 1, returns highly concentrated (When 1, only one non-zero return).

This function calculates concentration separately for positive returns, negative returns and concentration of bets grouped by time intervals (daily, monthly etc.) separately.
If concentration of positive returns is low, there is no right fat tail in returns distribution.
If concentration of negative returns is low, there is no left fat tail in returns distribution.

If after time grouping less than 2 observations, returns third element as nan.

.. function:: all_bets_concentration(returns: pd.Series) -> float:

    :param returns: (pd.Series) returns from bets
    :param frequency: (str) desired time grouping frequency from pd.Grouper
    :return: (tuple of floats) concentration of positive, negative
                            and time grouped concentrations

An example showing how All Bets Concentration function is used with weekly group data::

	from mlfinlab.backtest_statistics import all_bets_concentration

	pos_concentr, neg_concentr, week_concentr = all_bets_concentration(returns, frequency='W')

Drawdown and Time Under Water
==============================

Snippet 14.4, page 201, Calculates drawdowns and time under water

Intuitively, a drawdown is the maximum loss suffered by an investment between two consecutive high-watermarks.

The time under water is the time elapsed between an high watermark and the moment the PnL (profit and loss) exceeds the previous maximum PnL.

Data is taken in a form of series of cumulated returns, or account balance. Can be in dollars or other currency, then returned drawdown will be in this currency units. Otherwise the drawdowns are in percentage of account at high-watermarks.

The function returns two series:

1.Drawdown series index is time of a high watermark and value of a drawdown after


2.Time under water index is time of a high watermark and how much time passed till next high watermark in years.


.. function:: drawdown_and_time_under_water(returns: pd.Series, dollars: bool = False) -> tuple:

    :param returns: (pd.Series) returns from bets
    :param dollars: (bool) flag if given dollar performance and not returns
    :return: (tuple of pd.Series) series of drawdowns and time under water
                                if dollars, then in dollars, else as a %

An example showing how Drawdown and Time Under Water function is used with account data in dollars::

	from mlfinlab.backtest_statistics import drawdown_and_time_under_water

	drawdown, tuw = drawdown_and_time_under_water(returns, dollars = True)

Annual Sharpe Ratio
==============================

Calculates Annualized Sharpe Ratio for Series of normal (not log) returns.

A usual metric of returns in relation to risk. Also takes into account number of return entries per year and risk-free rate. Calculated as:

        :math:`SharpeRatio = \frac{E[Returns] - RiskFreeRate}{\sqrt{V[Returns]}} * \sqrt{n}`

Can calculate Sharpe Ratio for both cumulative and normal returns.
Generally, the higher Sharpe Ratio is, the better.

.. function:: def sharpe_ratio(returns: pd.Series, cumulative: bool = False,
                 entries_per_year: int = 252, risk_free_rate: float = 0) -> float:

    :param returns: (pd.Series) returns
    :param cumulative: (bool) flag if returns are cumulative (no by default)
    :param entries_per_year: (int) times returns are recorded per year (days by default)
    :param risk_free_rate: (float) risk-free rate (0 by default)
    :return: (float) Annualized Sharpe Ratio

An example showing how Annualized Sharpe Ratio function is used with monthly cumulative returns data::

	from mlfinlab.backtest_statistics import sharpe_ratio

	sr = sharpe_ratio(returns, cumulative=True, entries_per_year=12)

Probabalistic Sharpe Ratio
==============================

Calculates the probabilistic Sharpe ratio (PSR) that provides an adjusted estimate of SR, by removing the inflationary effect caused by short series with skewed and/or fat-tailed returns.

Given a user-defined benchmark Sharpe ratio and an observed Sharpe ratio, PSR estimates the probability that SR ̂is greater than a hypothetical SR.

If PSR exceeds 0.95, then SR is higher than the hypothetical (benchmark) SR at the standard significance level of 5%.

Formula for calculation:

        :math:`PSR[SR^{*}] = Z[\frac{(SR - SR^{*})\sqrt{T-1}}{\sqrt{1-\gamma_3*SR+\frac{\gamma_{4}-1}{4}*SR^2}}]`

Where:

    :math:`SR^{*}` - benchmark Sharpe ratio

    :math:`SR` - estimate od Sharpe ratio

    :math:`Z[..]` - cumulative distribution function (CDF) of the standard Normal distribution

    :math:`T` - number of observed returns

    :math:`\gamma_3` - skewness of the returns

    :math:`\gamma_4` - kurtosis of the returns

.. function:: probabalistic_sharpe_ratio(observed_sr: float, benchmark_sr: float,
                                         number_of_returns: int, skewness_of_returns: float = 0,
                                         kurtosis_of_returns: float = 3) -> float:

    :param observed_sr: (float) Sharpe Ratio that is observed
    :param benchmark_sr: (float) Sharpe Ratio to which observed_SR is tested against
    :param  number_of_returns: (int) times returns are recorded for observed_SR
    :param skewness_of_returns: (float) skewness of returns (as Gaussian by default)
    :param kurtosis_of_returns: (float) kurtosis of returns (as Gaussian by default)
    :return: (float) Probabalistic Sharpe Ratio

An example showing how Probabalistic Sharpe Ratio function is used with an example of data with normal returns::

	from mlfinlab.backtest_statistics import probabalistic_sharpe_ratio

	psr = probabalistic_sharpe_ratio(1.2, 1.0, 200)

Deflated Sharpe Ratio
==============================

Calculates the deflated Sharpe ratio (DSR) - a PSR where the rejection threshold is adjusted to reflect the multiplicity of trials. DSR is estimated as PSR[SR∗], where the benchmark Sharpe ratio, SR∗, is no longer user-defined, but calculated from SR estimate trails.

DSR corrects SR for inflationary effects caused by non-Normal returns, track record length, and multiple testing/selection bias.

Given a user-defined benchmark Sharpe ratio and an observed Sharpe estimates, DSR estimates the probability that SR ̂is greater than a hypothetical SR.

If DSR exceeds 0.95, then SR is higher than the hypothetical (benchmark) SR at the standard significance level of 5%.

Hypothetical SR is calculated as:

        :math:`SR^{*} = \sqrt{V[\{SR_{n}\}]}((1-\gamma)*Z^{-1}[1-\frac{1}{N}+\gamma*Z^{-1}[1-\frac{1}{N}*e^{-1}]`

Where:

    :math:`SR^{*}` - benchmark Sharpe ratio

    :math:`\{SR_{n}\}` - trails of SR estimates

    :math:`Z[..]` - cumulative distribution function (CDF) of the standard Normal distribution

    :math:`N` - number of SR trails

    :math:`\gamma` - Euler-Mascheroni constant

    :math:`e` - Euler constant

.. function:: deflated_sharpe_ratio(observed_sr: float, sr_estimates: list,
                                    number_of_returns: int, skewness_of_returns: float = 0,
                                    kurtosis_of_returns: float = 3) -> float:

    :param observed_sr: (list) Sharpe Ratio that is being tested
    :param sr_estimates: (float) Sharpe Ratios estimates trials
    :param  number_of_returns: (int) times returns are recorded for observed_SR
    :param skewness_of_returns: (float) skewness of returns (as Gaussian by default)
    :param kurtosis_of_returns: (float) kurtosis of returns (as Gaussian by default)
    :return: (float) Deflated Sharpe Ratio

An example showing how Deflated Sharpe Ratio function is used with an example of data with normal returns::

	from mlfinlab.backtest_statistics import deflated_sharpe_ratio

	psr = deflated_sharpe_ratio(1.2, [1.0, 1.1, 1.0], 200)

Minimum Track Record Length
==============================

Calculates the Minimum Track Record Length - "How long should a track record be in order to have statistical confidence that its Sharpe ratio is above a given threshold?”

If a track record is shorter than MinTRL, we do not  have  enough  confidence that  the  observed Sharpe ratio ̂is  above  the  designated Sharpe ratio threshold.

MinTRLis expressed in terms of number of observations, not annual or calendar terms.

Minimum Track Record Length is calculated as:

        :math:`MinTRL = 1 + [1-\gamma_3*SR+\frac{\gamma_{4}-1}{4}*SR^2]*(\frac{Z_{\alpha}}{SR-SR^{*}})^2`

Where:

    :math:`SR^{*}` - benchmark Sharpe ratio

    :math:`SR` - estimate od Sharpe ratio

    :math:`Z_{\alpha}` - Z score of desired significance level

    :math:`\gamma_3` - skewness of the returns

    :math:`\gamma_4` - kurtosis of the returns

.. function:: def minimum_track_record_length(observed_sr: float, benchmark_sr: float,
                                              skewness_of_returns: float = 0, kurtosis_of_returns: float = 3,
                                              alpha: float = 0.05) -> float:

    :param observed_sr: (float) Sharpe Ratio that is being tested
    :param benchmark_sr: (float) Sharpe Ratio to which observed_SR is tested against
    :param  number_of_returns: (int) times returns are recorded for observed_SR
    :param skewness_of_returns: (float) skewness of returns (as Gaussian by default)
    :param kurtosis_of_returns: (float) kurtosis of returns (as Gaussian by default)
    :param alpha: (float) desired significance level
    :return: (float) Probabalistic Sharpe Ratio

An example showing how Minimum Track Record Length function is used with an example of data with normal returns::

	from mlfinlab.backtest_statistics import minimum_track_record_length

	min_record_length = minimum_track_record_length(1.2, 1.0)



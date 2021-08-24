"""
Implements statistics related to:
- flattening and flips
- average period of position holding
- concentration of bets
- drawdowns
- various Sharpe ratios
- minimum track record length
"""
import warnings
import pandas as pd
import scipy.stats as ss
import numpy as np


def timing_of_flattening_and_flips(target_positions: pd.Series) -> pd.DatetimeIndex:
    """
    Advances in Financial Machine Learning, Snippet 14.1, page 197

    Derives the timestamps of flattening or flipping trades from a pandas series
    of target positions. Can be used for position changes analysis, such as
    frequency and balance of position changes.

    Flattenings - times when open position is bing closed (final target position is 0).
    Flips - times when positive position is reversed to negative and vice versa.

    :param target_positions: (pd.Series) Target position series with timestamps as indices
    :return: (pd.DatetimeIndex) Timestamps of trades flattening, flipping and last bet
    """

    pass


def average_holding_period(target_positions: pd.Series) -> float:
    """
    Advances in Financial Machine Learning, Snippet 14.2, page 197

    Estimates the average holding period (in days) of a strategy, given a pandas series
    of target positions using average entry time pairing algorithm.

    Idea of an algorithm:

    * entry_time = (previous_time * weight_of_previous_position + time_since_beginning_of_trade * increase_in_position )
      / weight_of_current_position
    * holding_period ['holding_time' = time a position was held, 'weight' = weight of position closed]
    * res = weighted average time a trade was held

    :param target_positions: (pd.Series) Target position series with timestamps as indices
    :return: (float) Estimated average holding period, NaN if zero or unpredicted
    """

    pass

def bets_concentration(returns: pd.Series) -> float:
    """
    Advances in Financial Machine Learning, Snippet 14.3, page 201

    Derives the concentration of returns from given pd.Series of returns.

    Algorithm is based on Herfindahl-Hirschman Index where return weights
    are taken as an input.

    :param returns: (pd.Series) Returns from bets
    :return: (float) Concentration of returns (nan if less than 3 returns)
    """

    pass


def all_bets_concentration(returns: pd.Series, frequency: str = 'M') -> tuple:
    """
    Advances in Financial Machine Learning, Snippet 14.3, page 201

    Given a pd.Series of returns, derives concentration of positive returns, negative returns
    and concentration of bets grouped by time intervals (daily, monthly etc.).
    If after time grouping less than 3 observations, returns nan.

    Properties or results:

    * low positive_concentration ⇒ no right fat-tail of returns (desirable)
    * low negative_concentration ⇒ no left fat-tail of returns (desirable)
    * low time_concentration ⇒ bets are not concentrated in time, or are evenly concentrated (desirable)
    * positive_concentration == 0 ⇔ returns are uniform
    * positive_concentration == 1 ⇔ only one non-zero return exists

    :param returns: (pd.Series) Returns from bets
    :param frequency: (str) Desired time grouping frequency from pd.Grouper
    :return: (tuple of floats) Concentration of positive, negative and time grouped concentrations
    """

    pass

def drawdown_and_time_under_water(returns: pd.Series, dollars: bool = False) -> tuple:
    """
    Advances in Financial Machine Learning, Snippet 14.4, page 201

    Calculates drawdowns and time under water for pd.Series of either relative price of a
    portfolio or dollar price of a portfolio.

    Intuitively, a drawdown is the maximum loss suffered by an investment between two consecutive high-watermarks.
    The time under water is the time elapsed between an high watermark and the moment the PnL (profit and loss)
    exceeds the previous maximum PnL. We also append the Time under water series with period from the last
    high-watermark to the last return observed.

    Return details:

    * Drawdown series index is the time of a high watermark and the value of a
      drawdown after it.
    * Time under water index is the time of a high watermark and how much time
      passed till the next high watermark in years. Also includes time between
      the last high watermark and last observation in returns as the last element.

    :param returns: (pd.Series) Returns from bets
    :param dollars: (bool) Flag if given dollar performance and not returns.
                    If dollars, then drawdowns are in dollars, else as a %.
    :return: (tuple of pd.Series) Series of drawdowns and time under water
    """

    pass


def sharpe_ratio(returns: pd.Series, entries_per_year: int = 252, risk_free_rate: float = 0) -> float:
    """
    Calculates annualized Sharpe ratio for pd.Series of normal or log returns.

    Risk_free_rate should be given for the same period the returns are given.
    For example, if the input returns are observed in 3 months, the risk-free
    rate given should be the 3-month risk-free rate.

    :param returns: (pd.Series) Returns - normal or log
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :param risk_free_rate: (float) Risk-free rate (0 by default)
    :return: (float) Annualized Sharpe ratio
    """

    pass


def information_ratio(returns: pd.Series, benchmark: float = 0, entries_per_year: int = 252) -> float:
    """
    Calculates annualized information ratio for pd.Series of normal or log returns.

    Benchmark should be provided as a return for the same time period as that between
    input returns. For example, for the daily observations it should be the
    benchmark of daily returns.

    It is the annualized ratio between the average excess return and the tracking error.
    The excess return is measured as the portfolio’s return in excess of the benchmark’s
    return. The tracking error is estimated as the standard deviation of the excess returns.

    :param returns: (pd.Series) Returns - normal or log
    :param benchmark: (float) Benchmark for performance comparison (0 by default)
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :return: (float) Annualized information ratio
    """

    pass


def probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float, number_of_returns: int,
                               skewness_of_returns: float = 0, kurtosis_of_returns: float = 3) -> float:
    """
    Calculates the probabilistic Sharpe ratio (PSR) that provides an adjusted estimate of SR,
    by removing the inflationary effect caused by short series with skewed and/or
    fat-tailed returns.

    Given a user-defined benchmark Sharpe ratio and an observed Sharpe ratio,
    PSR estimates the probability that SR ̂is greater than a hypothetical SR.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    :param observed_sr: (float) Sharpe ratio that is observed
    :param benchmark_sr: (float) Sharpe ratio to which observed_SR is tested against
    :param number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :return: (float) Probabilistic Sharpe ratio
    """

    pass


def deflated_sharpe_ratio(observed_sr: float, sr_estimates: list, number_of_returns: int,
                          skewness_of_returns: float = 0, kurtosis_of_returns: float = 3,
                          estimates_param: bool = False, benchmark_out: bool = False) -> float:
    """
    Calculates the deflated Sharpe ratio (DSR) - a PSR where the rejection threshold is
    adjusted to reflect the multiplicity of trials. DSR is estimated as PSR[SR∗], where
    the benchmark Sharpe ratio, SR∗, is no longer user-defined, but calculated from
    SR estimate trails.

    DSR corrects SR for inflationary effects caused by non-Normal returns, track record
    length, and multiple testing/selection bias.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    Function allows the calculated SR benchmark output and usage of only
    standard deviation and number of SR trails instead of full list of trails.

    :param observed_sr: (float) Sharpe ratio that is being tested
    :param sr_estimates: (list) Sharpe ratios estimates trials list or
        properties list: [Standard deviation of estimates, Number of estimates]
        if estimates_param flag is set to True.
    :param  number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :param estimates_param: (bool) Flag to use properties of estimates instead of full list
    :param benchmark_out: (bool) Flag to output the calculated benchmark instead of DSR
    :return: (float) Deflated Sharpe ratio or Benchmark SR (if benchmark_out)
    """

    pass


def minimum_track_record_length(observed_sr: float, benchmark_sr: float,
                                skewness_of_returns: float = 0,
                                kurtosis_of_returns: float = 3,
                                alpha: float = 0.05) -> float:
    """
    Calculates the minimum track record length (MinTRL) - "How long should a track
    record be in order to have statistical confidence that its Sharpe ratio is above
    a given threshold?”

    If a track record is shorter than MinTRL, we do not  have  enough  confidence
    that  the  observed Sharpe ratio ̂is above the designated Sharpe ratio threshold.

    MinTRLis expressed in terms of number of observations, not annual or calendar terms.

    :param observed_sr: (float) Sharpe ratio that is being tested
    :param benchmark_sr: (float) Sharpe ratio to which observed_SR is tested against
    :param  number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :param alpha: (float) Desired significance level (0.05 by default)
    :return: (float) Minimum number of track records
    """

    pass

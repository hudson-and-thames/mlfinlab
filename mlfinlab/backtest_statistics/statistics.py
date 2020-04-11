"""
Implements statistics related to:
- flattening and flips
- average period of position holding
- concentration of bets
- drawdowns
- various Sharpe ratios
- minimum track record length
"""
import pandas as pd
import scipy.stats as ss
import numpy as np


def timing_of_flattening_and_flips(target_positions: pd.Series) -> pd.DatetimeIndex:
    """
    Snippet 14.1, page 197, Derives the timestamps of flattening or flipping
    trades from a pandas series of target positions. Can be used for position
    changes analysis, such as frequency and balance of position changes.

    Flattenings - times when open position is bing closed (final target position is 0).
    Flips - times when positive position is reversed to negative and vice versa.

    :param target_positions: (pd.Series) target position series with timestamps as indices
    :return: (pd.DatetimeIndex) timestamps of trades flattening, flipping and last bet
    """

    empty_positions = target_positions[(target_positions == 0)].index  # Empty positions index
    previous_positions = target_positions.shift(1)  # Timestamps pointing at previous positions

    # Index of positions where previous one wasn't empty
    previous_positions = previous_positions[(previous_positions != 0)].index

    # FLATTENING - if previous position was open, but current is empty
    flattening = empty_positions.intersection(previous_positions)

    # Multiplies current position with value of next one
    multiplied_posions = target_positions.iloc[1:] * target_positions.iloc[:-1].values

    # FLIPS - if current position has another direction compared to the next
    flips = multiplied_posions[(multiplied_posions < 0)].index
    flips_and_flattenings = flattening.union(flips).sort_values()
    if target_positions.index[-1] not in flips_and_flattenings:  # Appending with last bet
        flips_and_flattenings = flips_and_flattenings.append(target_positions.index[-1:])

    return flips_and_flattenings


def average_holding_period(target_positions: pd.Series) -> float:
    """
    Snippet 14.2, page 197, Estimates the average holding period (in days) of a strategy,
    given a pandas series of target positions using average entry time pairing algorithm.

    Idea of an algorithm:

    * entry_time = (previous_time * weight_of_previous_position + time_since_beginning_of_trade * increase_in_position )
      / weight_of_current_position
    * holding_period ['holding_time' = time a position was held, 'weight' = weight of position closed]
    * res = weighted average time a trade was held

    :param target_positions: (pd.Series) target position series with timestamps as indices
    :return: (float) estimated average holding period, NaN if zero or unpredicted
    """

    holding_period = pd.DataFrame(columns=['holding_time', 'weight'])
    entry_time = 0
    position_difference = target_positions.diff()

    # Time elapsed from the starting time for each position
    time_difference = (target_positions.index - target_positions.index[0]) / np.timedelta64(1, 'D')
    for i in range(1, target_positions.size):

        # Increased or unchanged position
        if float(position_difference.iloc[i] * target_positions.iloc[i - 1]) >= 0:
            if float(target_positions.iloc[i]) != 0:  # And not an empty position
                entry_time = (entry_time * target_positions.iloc[i - 1] +
                              time_difference[i] * position_difference.iloc[i]) / target_positions.iloc[i]

        # Decreased
        if float(position_difference.iloc[i] * target_positions.iloc[i - 1]) < 0:
            hold_time = time_difference[i] - entry_time

            # Flip of a position
            if float(target_positions.iloc[i] * target_positions.iloc[i - 1]) < 0:
                weight = abs(target_positions.iloc[i - 1])
                holding_period.loc[target_positions.index[i], ['holding_time', 'weight']] = (hold_time, weight)
                entry_time = time_difference[i]  # Reset entry time

            # Only a part of position is closed
            else:
                weight = abs(position_difference.iloc[i])
                holding_period.loc[target_positions.index[i], ['holding_time', 'weight']] = (hold_time, weight)

    if float(holding_period['weight'].sum()) > 0:  # If there were closed trades at all
        avg_holding_period = float((holding_period['holding_time'] * \
                                    holding_period['weight']).sum() / holding_period['weight'].sum())
    else:
        avg_holding_period = float('nan')

    return avg_holding_period


def bets_concentration(returns: pd.Series) -> float:
    """
    Snippet 14.3, page 201, Derives the concentration of returns from given
    pd.Series of returns.

    Algorithm is based on Herfindahl-Hirschman Index where return weights
    are taken as an input.

    :param returns: (pd.Series) returns from bets
    :return: (float) concentration of returns (nan if less than 3 returns)
    """

    if returns.size <= 2:
        return float('nan')  # If less than 3 bets
    weights = returns / returns.sum()  # Weights of each bet
    hhi = (weights ** 2).sum()  # Herfindahl-Hirschman Index for weights
    hhi = float((hhi - returns.size ** (-1)) / (1 - returns.size ** (-1)))

    return hhi


def all_bets_concentration(returns: pd.Series, frequency: str = 'M') -> tuple:
    """
    Snippet 14.3, page 201, Given a pd.Series of returns, derives concentration of positive returns, negative returns
    and concentration of bets grouped by time intervals (daily, monthly etc.). If after time grouping less than
    3 observations, returns nan.

    Properties or results:

    * low positive_concentration -> no right fat-tail of returns (desirable)
    * low negative_concentration -> no left fat-tail of returns (desirable)
    * low time_concentration -> bets are not concentrated in time, or are evenly concentrated (desirable)
    * positive_concentration == 0 ⇔ returns are uniform
    * positive_concentration == 1 ⇔ only one non-zero return exists

    :param returns: (pd.Series) returns from bets
    :param frequency: (str) desired time grouping frequency from pd.Grouper
    :return: (tuple of floats) concentration of positive, negative and time grouped concentrations
    """

    # Concentration of positive returns per bet
    positive_concentration = bets_concentration(returns[returns >= 0])

    # Concentration of negative returns per bet
    negative_concentration = bets_concentration(returns[returns < 0])

    # Concentration of bets/time period (month by default)
    time_concentration = bets_concentration(returns.groupby(pd.Grouper(freq=frequency)).count())

    return (positive_concentration, negative_concentration, time_concentration)


def drawdown_and_time_under_water(returns: pd.Series, dollars: bool = False) -> tuple:
    """
    Snippet 14.4, page 201, Calculates drawdowns and time under water for pd.Series of either relative price of a
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

    :param returns: (pd.Series) returns from bets
    :param dollars: (bool) flag if given dollar performance and not returns.
                    If dollars, then drawdowns are in dollars, else as a %.
    :return: (tuple of pd.Series) series of drawdowns and time under water
    """

    frame = returns.to_frame('pnl')
    frame['hwm'] = returns.expanding().max()  # Adding high watermarks as column

    # Grouped as min returns by high watermarks
    high_watermarks = frame.groupby('hwm').min().reset_index()
    high_watermarks.columns = ['hwm', 'min']

    # Time high watermark occurred
    high_watermarks.index = frame['hwm'].drop_duplicates(keep='first').index

    # Picking ones that had a drawdown after high watermark
    high_watermarks = high_watermarks[high_watermarks['hwm'] > high_watermarks['min']]
    if dollars:
        drawdown = high_watermarks['hwm'] - high_watermarks['min']
    else:
        drawdown = 1 - high_watermarks['min'] / high_watermarks['hwm']

    time_under_water = ((high_watermarks.index[1:] - high_watermarks.index[:-1]) / np.timedelta64(1, 'Y')).values

    # Adding also period from last High watermark to last return observed.
    time_under_water = np.append(time_under_water,
                                 (returns.index[-1] - high_watermarks.index[-1]) / np.timedelta64(1, 'Y'))

    time_under_water = pd.Series(time_under_water, index=high_watermarks.index)

    return drawdown, time_under_water


def sharpe_ratio(returns: pd.Series, entries_per_year: int = 252, risk_free_rate: float = 0) -> float:
    """
    Calculates Annualized Sharpe Ratio for pd.Series of normal or log returns.
    Risk_free_rate should be given for the same period the returns are given.
    For example, if the input returns are observed in 3 months, the risk-free
    rate given should be the 3-month risk-free rate.

    :param returns: (pd.Series) returns - normal or log
    :param entries_per_year: (int) times returns are recorded per year (daily by default)
    :param risk_free_rate: (float) risk-free rate (0 by default)
    :return: (float) Annualized Sharpe Ratio
    """

    sharpe_r = (returns.mean() - risk_free_rate) / returns.std() * (entries_per_year) ** (1 / 2)

    return sharpe_r


def information_ratio(returns: pd.Series, benchmark: float = 0, entries_per_year: int = 252) -> float:
    """
    Calculates Annualized Information Ratio for pd.Series of normal or log returns.
    Benchmark should be provided as a return for the same time period as that between
    input returns. For example, for the daily observations it should be the
    benchmark of daily returns.

    It is the annualized ratio between the average excess return and the tracking error.
    The excess return is measured as the portfolio’s return in excess of the benchmark’s
    return. The tracking error is estimated as the standard deviation of the excess returns.

    :param returns: (pd.Series) returns - normal or log
    :param benchmark: (float) benchmark for performance comparison (0 by default)
    :param entries_per_year: (int) times returns are recorded per year (daily by default)
    :return: (float) Annualized Information Ratio
    """

    excess_returns = returns - benchmark
    information_r = sharpe_ratio(excess_returns, entries_per_year)

    return information_r


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

    :param observed_sr: (float) Sharpe Ratio that is observed
    :param benchmark_sr: (float) Sharpe Ratio to which observed_SR is tested against
    :param number_of_returns: (int) times returns are recorded for observed_SR
    :param skewness_of_returns: (float) skewness of returns (as Gaussian by default)
    :param kurtosis_of_returns: (float) kurtosis of returns (as Gaussian by default)
    :return: (float) Probabilistic Sharpe Ratio
    """

    probab_sr = ss.norm.cdf(((observed_sr - benchmark_sr) * (number_of_returns - 1) ** (1 / 2)) / \
                            (1 - skewness_of_returns * observed_sr +
                             (kurtosis_of_returns - 1) / 4 * observed_sr ** 2) ** (1 / 2))

    return probab_sr


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

    :param observed_sr: (float) Sharpe Ratio that is being tested
    :param sr_estimates: (list) Sharpe Ratios estimates trials list or
        properties list: [Standard deviation of estimates, Number of estimates]
        if estimates_param flag is set to True.
    :param  number_of_returns: (int) times returns are recorded for observed_SR
    :param skewness_of_returns: (float) skewness of returns (as Gaussian by default)
    :param kurtosis_of_returns: (float) kurtosis of returns (as Gaussian by default)
    :param estimates_param: (bool) allows to use properties of estimates instead of full list
    :param benchmark_out: (bool) flag to output the calculated benchmark instead of DSR
    :return: (float) Deflated Sharpe Ratio or Benchmark SR (if benchmark_out)
    """

    # Calculating benchmark_SR from the parameters of estimates
    if estimates_param:
        benchmark_sr = sr_estimates[0] * \
                       ((1 - np.euler_gamma) * ss.norm.ppf(1 - 1 / sr_estimates[1]) +
                        np.euler_gamma * ss.norm.ppf(1 - 1 / sr_estimates[1] * np.e ** (-1)))

    # Calculating benchmark_SR from a list of estimates
    else:
        benchmark_sr = np.array(sr_estimates).std() * \
                       ((1 - np.euler_gamma) * ss.norm.ppf(1 - 1 / len(sr_estimates)) +
                        np.euler_gamma * ss.norm.ppf(1 - 1 / len(sr_estimates) * np.e ** (-1)))

    deflated_sr = probabilistic_sharpe_ratio(observed_sr, benchmark_sr, number_of_returns,
                                             skewness_of_returns, kurtosis_of_returns)

    if benchmark_out:
        return benchmark_sr

    return deflated_sr


def minimum_track_record_length(observed_sr: float, benchmark_sr: float,
                                skewness_of_returns: float = 0,
                                kurtosis_of_returns: float = 3,
                                alpha: float = 0.05) -> float:
    """
    Calculates the Minimum Track Record Length - "How long should a track record be
    in order to have statistical confidence that its Sharpe ratio is above a given
    threshold?”

    If a track record is shorter than MinTRL, we do not  have  enough  confidence
    that  the  observed Sharpe ratio ̂is above the designated Sharpe ratio threshold.

    MinTRLis expressed in terms of number of observations, not annual or calendar terms.

    :param observed_sr: (float) Sharpe Ratio that is being tested
    :param benchmark_sr: (float) Sharpe Ratio to which observed_SR is tested against
    :param  number_of_returns: (int) times returns are recorded for observed_SR
    :param skewness_of_returns: (float) skewness of returns (as Gaussian by default)
    :param kurtosis_of_returns: (float) kurtosis of returns (as Gaussian by default)
    :param alpha: (float) desired significance level (0.05 by default)
    :return: (float) Minimum number of track records
    """

    track_rec_length = 1 + (1 - skewness_of_returns * observed_sr +
                            (kurtosis_of_returns - 1) / 4 * observed_sr ** 2) * \
                       (ss.norm.ppf(1 - alpha) / (observed_sr - benchmark_sr)) ** (2)

    return track_rec_length

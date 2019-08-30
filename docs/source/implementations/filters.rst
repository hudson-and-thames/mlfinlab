.. _implementations-filters:

=======
Filters
=======

Filters are used to filter events based on some kind of trigger. For example a structural break filter can be
used to filter events where a structural break occurs. This event is then used to measure the return from the event
to some event horizon, say a day.

CUSUM Filter
============

Snippet 2.4, page 39, The Symmetric CUSUM Filter.

The CUSUM filter is a quality-control method, designed to detect a shift in the mean value of a measured quantity away from a target value.

The filter is set up to identify a sequence of upside or downside divergences from any reset level zero.

We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.

One practical aspect that makes CUSUM filters appealing is that multiple events are not triggered by raw_time_series hovering around a threshold level, which is a flaw suffered by popular market signals such as Bollinger Bands.

It will require a full run of length threshold for raw_time_series to trigger an event. Once we have obtained this subset of event-driven bars, we will let the ML algorithm determine whether the occurrence of such events constitutes actionable intelligence.

Threshold can be either fixed (float value) or dynamic (pd.Series).

Below is an implementation of the Symmetric CUSUM filter. Note: As per the book this filter is applied to closing prices but we extended it to also work on other time series such as volatility.

.. function:: cusum_filter(raw_time_series, threshold, time_stamps=True)

    :param raw_time_series: (series) of close prices (or other time series, e.g. volatility).
    :param threshold: (float or pd.Series) when the abs(change) is larger than the threshold, the function captures
      it as an event, can be dynamic if threshold is pd.Series
    :param time_stamps: (bool) default is to return a DateTimeIndex, change to false to have it return a list.
    :return: (datetime index vector) vector of datetimes when the events occurred. This is used later to sample.

An example showing how the CUSUM filter can be used to downsample a time series of close prices can be seen below::

	from mlfinlab.filters import cusum_filter

	cusum_events = cusum_filter(data['close'], threshold=0.05)

Z-score Filter
==============

Z-score filter is used to define explosive/peak points in time series. https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data

It uses rolling simple moving average, rolling simple moving standard deviation and z_score(threshold). When current time series value exceeds (rolling average + z_score * rolling std) event is sampled

.. function:: z_score_filter(raw_time_series, mean_window, std_window, z_score=3, time_stamps=True)

    :param raw_time_series: (series) of close prices (or other time series, e.g. volatility).
    :param mean_window: (int): rolling mean window
    :param std_window: (int): rolling std window
    :param z_score: (float): number of standard deviations to trigger the event
    :param time_stamps: (bool) default is to return a DateTimeIndex, change to false to have it return a list.
    :return: (datetime index vector) vector of datetimes when the events occurred. This is used later to sample.

An example of how the Z-score filter can be used to downsample a time series::

  from mlfinlb.filters import z_score_filter

  z_score_events = z_score_filter(data['close'], mean_window=100, std_window=100, z_score=3)

.. _implementations-filters:

=======
Filters
=======

Filters are used to filter events based on some kind of trigger. For example a structural break filter can be
used to filter events where a structural break occurs. In Triple-Barrier labeling, this event is then used to measure
the return from the event to some event horizon, say a day.

The core idea is that labeling every trading day is a fools errand, researchers should instead focus on forecasting specific
market anomalies or how the market moves after an event.

CUSUM Filter
############

.. py:currentmodule:: mlfinlab.filters.filters
.. autofunction::  cusum_filter


An example showing how the CUSUM filter can be used to downsample a time series of close prices can be seen below:

.. code-block::

   from mlfinlab.filters import cusum_filter

   cusum_events = cusum_filter(data['close'], threshold=0.05)


Z-Score Filter
##############

The `Z-Score <https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data>`_ filter is
used to define explosive/peak points in time series.

It uses rolling simple moving average, rolling simple moving standard deviation, and z_score(threshold). When the current
time series value exceeds (rolling average + z_score * rolling std) an event is triggered.

.. py:currentmodule:: mlfinlab.filters.filters
.. autofunction::  z_score_filter

An example of how the Z-score filter can be used to downsample a time series:

.. code-block::

   from mlfinlb.filters import z_score_filter

   z_score_events = z_score_filter(data['close'], mean_window=100, std_window=100, z_score=3)

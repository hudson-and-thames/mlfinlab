"""
Filters are used to filter events based on some kind of trigger. For example a structural break filter can be
used to filter events where a structural break occurs. This event is then used to measure the return from the event
to some event horizon, say a day.
"""

import numpy as np
import pandas as pd


# Snippet 2.4, page 39, The Symmetric CUSUM Filter.
def get_t_events(raw_price, threshold):
    """
    Snippet 2.4, page 39, The Symmetric CUSUM Filter.

    The CUSUM filter is a quality-control method, designed to detect a shift in the
    mean value of a measured quantity away from a target value. The filter is set up to
    identify a sequence of upside or downside divergences from any reset level zero.

    We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.

    One practical aspect that makes CUSUM filters appealing is that multiple events are not
    triggered by gRaw hovering around a threshold level, which is a flaw suffered by popular
    market signals such as Bollinger Bands. It will require a full run of length threshold for
    raw_price to trigger an event.

    Once we have obtained this subset of event-driven bars, we will let the ML algorithm determine
    whether the occurrence of such events constitutes actionable intelligence.

    Below is an implementation of the Symmetric CUSUM filter.

    :param raw_price: (series) of close prices.
    :param threshold: (float) when the abs(change) is larger than the threshold, the function captures
    it as an event.
    :return: (datetime index vector) vector of datetimes when the events occurred. This is used later to sample.
    """

    t_events = []
    s_pos = 0
    s_neg = 0

    # log returns
    diff = np.log(raw_price).diff()

    # Get event time stamps for the entire series
    for i in diff.index[1:]:
        pos = float(s_pos + diff.loc[i])
        neg = float(s_neg + diff.loc[i])
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)

        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)

    event_timestamps = pd.DatetimeIndex(t_events)
    return event_timestamps

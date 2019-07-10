"""
This module contains functionality for determining bet sizes for
investments based on machine learning predictions. These implementations are
based on bet sizing approaches described in Chapter 10.
"""


# imports
from mlfinlab.bet_sizing.ch10_snippets import get_signal, avg_active_signals, discrete_signal


def bet_size_probability(events, prob, num_classes, pred=None, step_size=0.0,
                         average_active=False, num_threads=1):
    """
    Calculates the bet size using the predicted probability. Note that if
    'average_active' is True, the returned pandas.Series will be twice
    the length of the orginal since the average is calculated at each
    bet's open and close.

    :param prob: (pandas.Series) The predicted probabiility.
    :param num_classes: (int) The number of predicted bet sides.
    :param pred: (pd.Series) The predicted bet side. Default value is None
        which will return a relative bet size (i.e. without multiplying
        by the side).
    :param step_size: (float) The step size at which the bet size is
        discretized, default is 0.0 which imposes no discretization.
    :param average_active: (bool) Option to average the size of active bets,
        default value is False.
    :param num_threads: (int) The number of processing threads to utilize for
        multiprocessing, default value is 1.
    :return: (pandas.Series) The bet size, with the time index.
    """
    signal_0 = get_signal(prob, num_classes, pred)
    events_0 = signal_0.to_frame('signal').join(events['t1'], how='left')

    if average_active:
        signal_1 = avg_active_signals(events_0, num_threads)
    else:
        signal_1 = events_0.signal
    if step_size > 0:
        signal_1 = discrete_signal(signal0=signal_1, step_size=step_size)

    return signal_1

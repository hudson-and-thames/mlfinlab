"""
This module contains functionality for determining bet sizes for investments based on machine learning predictions.
These implementations are based on bet sizing approaches described in Chapter 10.
"""

from mlfinlab.bet_sizing.ch10_snippets import get_signal, avg_active_signals, discrete_signal
from mlfinlab.bet_sizing.ch10_snippets import (get_w, get_t_pos, limit_price,
                                               bet_size)

def bet_size_probability(events, prob, num_classes, pred=None, step_size=0.0, average_active=False, num_threads=1):
    Calculates the bet size using the predicted probability. Note that if 'average_active' is True, the returned
    pandas.Series will be twice the length of the original since the average is calculated at each bet's open and close.
    """
    :param prob: (pandas.Series) The predicted probability.
    :param num_classes: (int) The number of predicted bet sides.
    :param pred: (pd.Series) The predicted bet side. Default value is None which will return a relative bet size
     (i.e. without multiplying by the side).
    :param step_size: (float) The step size at which the bet size is discretized, default is 0.0 which imposes no
     discretization.
    :param average_active: (bool) Option to average the size of active bets, default value is False.
    :param num_threads: (int) The number of processing threads to utilize for multiprocessing, default value is 1.
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
    

def bet_size_dynamic(current_pos, max_pos, market_price, forecast_price,
                     cal_divergence=10, cal_bet_size=0.95, func='sigmoid'):
    """
    Calculates the bet sizes and limit price as the market price and forecast price fluctuate.
    
    :param current_pos: (pandas.Series) Current position.
    :param max_pos: (pandas.Series) Maximum position
    :param market_price: (pandas.Series) Market price.
    :param forecast_price: (pandas.Series) Forecast price.
    :param cal_divergence: (float) The divergence to use in calibration.
    :param cal_bet_size: (float) The bet size to use in calibration.
    :param func: (string) Function to use for dynamic calculation. Valid
        options are: 'sigmoid'.
    :return: (pandas.DataFrame) Bet size (bet_size) and limit price (l_p).
    """
    # combine Series to DataFrame
    df = pd.concat([current_pos, max_pos, market_price, forecast_price],
                    axis=1)
    df = df.rename(columns={0: 'pos',  # current position
                            1: 'max_pos',  # maximum position
                            2: 'm_p',  # market price
                            3: 'f'})  # forecast price
    # calibrate w
    w = get_w(cal_divergence, cal_bet_size, func)
    # compute target position
    df['t_pos'] = df.apply(lambda x: get_t_pos(w, x.f, x.m_p, x.max_pos, func),
                           axis=1)
    # compute limit price
    df['l_p'] = df.apply(lambda x: limit_price(x.t_pos, x.pos, x.f, w,
                         x.max_pos, func), axis=1)
    # compute bet size
    df['bet_size'] = df.apply(lambda x: bet_size(w, x.f-x.m_p, func), axis=1)
    
    return df[['bet_size', 'l_p']]

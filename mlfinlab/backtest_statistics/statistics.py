"""
Implements the book Chapter 14 on Backtest Statistics
"""

import pandas as pd


def timing_of_flattening_and_flips(target_positions: pd.Series) -> pd.DatetimeIndex:
    """
    Snippet 14.1, page 197, Derives the timestamps of flattening or flipping
    trades from a pandas series of target positions. 
    
    :param target_positions: (pd.Series) target position series with timestamps as indices
    :return: (pd.DatetimeIndex) timestamps of trades flattening, flipping and last bet
    """
    empty_positions = target_positions[(target_positions==0)].index # empty positions index
    previous_positions = target_positions.shift(1) # timestamps pointing at previous positions
    previous_positions = previous_positions[(previous_positions!=0)].index # index of positions where previous one wasn't empty
    flattening = empty_positions.intersection(previous_positions) # FLATTENING - if previous position was open, but current is empty
    multiplied_posions = target_positions.iloc[1:] * target_positions.iloc[:-1].values # multiplies current position with value of next one
    flips = multiplied_posions[(multiplied_posions<0)].index # FLIPS - if current position has another direction compared to the next
    res = flattening.union(flips).sort_values()
    if target_positions.index[-1] not in res:
        res=res.append(target_positions.index[-1:]) # appending with last bet
    return res


def average_holding_period(target_positions: pd.Series) -> float:
    """
    Snippet 14.2, page 197, Estimates the average holding period (in days) of a strategy, 
    given a pandas series of target positions using average entry time pairing algorithm.
    
    Idea of an algorithm:
    - entry_time = (previous_time * weight_of_previous_position + 
                    time_since_beginning_of_trade * increase_in_position ) /
                    weight_of_current_position
    - holding_period ['dT' = time a position was held, 'w' = weight of position closed]
    - res = weighted average time a trade was held
    
    :param target_positions: (pd.Series) target position series with timestamps as indices
    :return: (float) estimated average holding period, NaN if zero or unpredicted
    """
    holding_period, entry_time = pd.DataFrame(columns=['dT','w']), 0
    position_difference = target_positions.diff() 
    
    # time elapsed from the starting time for each position
    time_difference = (target_positions.index-target_positions.index[0]) / np.timedelta64(1,'D')
    for i in range(1, target_positions.shape[0]):
        if float(position_difference.iloc[i] * target_positions.iloc[i-1]) >= 0: # increased or unchanged position
            if float(target_positions.iloc[i]) != 0: # and not an empty position
                entry_time = (entry_time * target_positions.iloc[i-1] + 
                              time_difference[i] * position_difference.iloc[i]) / target_positions.iloc[i]
        else: # decreased
            if float(target_positions.iloc[i] * target_positions.iloc[i-1]) < 0: # Flip of a position
                holding_period.loc[target_positions.index[i], ['dT','w']] = (time_difference[i] - entry_time, abs(target_positions.iloc[i-1]))
                entry_time = time_difference[i] # reset entry time
            else: # Only a part of position is closed
                holding_period.loc[target_positions.index[i], ['dT','w']] = (time_difference[i] - entry_time, abs(position_difference.iloc[i]))
                
    if float(holding_period['w'].sum()) > 0: # If there were closed trades at all
        res = float((holding_period['dT'] * holding_period['w']).sum() / holding_period['w'].sum())
    else:
        res = float('nan')
        
    return res


def bets_concentration(returns: pd.Series) -> float:
    """
    Snippet 14.3, page 201, Derives the concentration of bets across months
    from given pd.Series of returns.
    
    Algorithm is based on Herfindahl-Hirschman Index where return weights
    are taken as an input.
    
    :param returns: (pd.Series) returns from bets
    :return: (float) concentration of returns (nan if less than 3 returns)
    """  
    if returns.shape[0]<=2:
        return float('nan') #if less than 3 bets
    weights = returns / returns.sum() #weights of each bet
    hhi = (weights ** 2).sum() # Herfindahl-Hirschman Index for weights
    hhi = float((hhi - returns.shape[0] **(-1)) / (1 - returns.shape[0]**(-1)))
    return hhi


def positive_negative_bets_concentration(returns: pd.Series, freq: str = 'M') -> tuple:
    """
    Snippet 14.3, page 201, Derives positive, negative and time consentration of 
    bets in a given dp.Series of returns.
       
    Properties:
    - low positive_concentration -> no right fat-tail (desirable)
    - low negative_concentration -> no left fat-tail (desirable)
    - low time_concentration -> bets are not concentrated in time (desirable)
    - positive_concentration = 0 ⇔ uniform returns
    - positive_concentration = 1 ⇔ only one non-zero return
    
    :param returns: (pd.Series) returns from bets
    :param freq: (str) desired time grouping frequency from pd.Grouper
    :return: (tuple of floats) concentration of positive, negative
                            and time grouped concentrations
    """  
    # concentration of positive returns per bet
    positive_concentration = bets_concentration(returns[returns >= 0])
    # concentration of negative returns per bet
    negative_concentration = bets_concentration(returns[returns < 0]) 
    # concentration of bets/time period (month by default)
    time_concentration = bets_concentration(returns.groupby(pd.Grouper(freq='M')).count()) # concentr. bets/month
    return (positive_concentration, negative_concentration, time_concentration)


def compute_drawdown_and_time_under_water(returns: pd.Series, dollars: bool = False) -> tuple:
    """
    Snippet 14.4, page 201, Calculates drawdowns and time under water for pd.Series
    of returns or dollar performance.
    
    Intuitively, a drawdown is the maximum loss suffered by an investment between
    two consecutive high-watermarks. The time under water is the time
    elapsed between an high watermark and the moment the PnL (profit and loss) 
    exceeds the previous maximum PnL.     
    
    Return details:
    - Drawdown series index is time of a high watermark and value of a drawdown after.
    - Time under water index is time of a high watermark and how much time passed till
    next high watermark in years. 

    :param returns: (pd.Series) returns from bets
    :param dollars: (bool) flag if given dollar performance and not returns 
    :return: (tuple of pd.Series) series of drawdowns and time under water 
                                if dollars, then in dollars, else as a %
            
    """
    frame = returns.to_frame('pnl')
    frame['hwm'] = returns.expanding().max() # adding high watermarks as column
    #grouped as min returns by high watermarks
    high_watermarks = frame.groupby('hwm').min().reset_index()
    high_watermarks.columns = ['hwm','min']
    # time high watermark occured
    high_watermarks.index = frame['hwm'].drop_duplicates(keep='first').index
    #picking ones that had a drawdown after high watermark
    high_watermarks = high_watermarks[high_watermarks['hwm'] > high_watermarks['min']]
    if dollars:
        drawdown = high_watermarks['hwm'] - high_watermarks['min']
    else:
        drawdown = 1 - high_watermarks['min'] / high_watermarks['hwm']
    time_under_water = ((high_watermarks.index[1:] - 
                         high_watermarks.index[:-1]) / np.timedelta64(1,'Y')).values 
    time_under_water = pd.Series(time_under_water, index = high_watermarks.index[:-1])
    return drawdown, time_under_water


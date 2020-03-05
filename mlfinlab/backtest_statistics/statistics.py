"""
Module which implements backtest statistics as described in Chapter 14 of Advances in Financial Machine Learning.
"""

import pandas as pd

def timing_of_flattening_and_flips(target_positions):
    """
    Snippet 14.1, page 197, Derives the timestamps of flattening or flipping
    trades from a pandas series of target positions. 
    
    :param target_positions: (pd.Series) target position series with timestamps as indices
    :return: (pd.DatetimeIndex) timestamps of trades flattening, flipping and last bet
    """
    empty_positions = target_positions[(target_positions==0).all(1)].index # empty positions index
    previous_positions = target_positions.shift(1) # timestamps pointing at previous positions
    previous_positions = previous_positions[(previous_positions!=0).any(1)].index # index of positions where previous one wasn't empty
    flattening = empty_positions.intersection(previous_positions) # FLATTENING - if previous position was open, but current is empty
    multiplied_posions = target_positions.iloc[1:] * target_positions.iloc[:-1].values # multiplies current position with value of next one
    flips = multiplied_posions[(multiplied_posions<0).any(1)].index # FLIPS - if current position has another direction compared to the next
    res = flattening.union(flips).sort_values()
    if target_positions.index[-1] not in res:
        res=res.append(target_positions.index[-1:]) # appending with last bet
    return res


def average_holding_period(target_positions):
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





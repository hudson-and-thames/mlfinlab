def vwap(dollar_volume, volume):
    return sum(dollar_volume) / sum(volume)

def get_avg_tick_size(tick_size_arr):
    return np.mean(tick_size_arr)

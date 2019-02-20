import time

import pandas as pd
import numpy as np


def update_counters(cache, flag):
    # Check flag
    if flag and len(cache) > 0:
        # Update variables to latest
        cum_ticks = int(cache[-1][6])
        cum_dollar_value = np.float(cache[-1][5])
        cum_volume = cache[-1][4]
        low_price = np.float(cache[-1][2])
        high_price = np.float(cache[-1][3])
    else:
        # Reset counters
        cum_ticks, cum_dollar_value, cum_volume, cache, high_price, low_price = 0, 0, 0, [], -np.inf, np.inf

    return cum_ticks, cum_dollar_value, cum_volume, high_price, low_price


def create_bars(data, metric, threshold=50000, cache=[], flag=False):
    bars = []
    cum_ticks, cum_dollar_value, cum_volume, high_price, low_price = update_counters(cache, flag)

    # Iterate over rows
    for row in data.values:
        # Set variables
        date_time = row[0]
        price = np.float(row[1])
        volume = row[2]

        # Calculations
        cum_ticks += 1
        dollar_value = price * volume
        cum_dollar_value = cum_dollar_value + dollar_value
        cum_volume = cum_volume + volume

        # Check min max
        if price > high_price:
            high_price = price
        elif price <= low_price:
            low_price = price

        # Update cache
        cache.append([date_time, price, low_price, high_price, cum_volume, cum_dollar_value, cum_ticks])

        # If threshold reached then take a sample
        if eval(metric) >= threshold:
            # Create bars
            open_price = cache[0][1]
            low_price = min(low_price, open_price)  # If only one data point in bars then the low price isn't added, this check corrects that
            close_price = price

            # Update bars & Reset counters
            bars.append([date_time, open_price, high_price, low_price, close_price, cum_volume, cum_dollar_value, cum_ticks])
            cum_ticks, cum_dollar_value, cum_volume, cache, high_price, low_price = 0, 0, 0, [], -np.inf, np.inf

    return bars, cache



def create_dollar_bars(data, threshold=50000, cache=[], flag=False):

    bars = []
    cum_dollar_value, cum_volume, high_price, low_price = update_counters(cache, flag)

    # Iterate over rows
    for row in data.values:
        # Set variables
        date_time = row[0]
        price = np.float(row[1])
        volume = row[2]

        # Calculations
        dollar_value = price * volume
        cum_dollar_value = cum_dollar_value + dollar_value
        cum_volume = cum_volume + volume

        # Check min max
        if price > high_price:
            high_price = price
        elif price <= low_price:
            low_price = price

        # Update cache
        cache.append([date_time, price, low_price, high_price, cum_volume, cum_dollar_value])

        # If threshold reached then take a sample
        if cum_dollar_value >= threshold:
            # Create bars
            open_price = cache[0][1]
            low_price = min(low_price, open_price)  # If only one data point in bars then the low price isn't added, this check corrects that
            close_price = price

            # Update bars & Reset counters
            bars.append([date_time, open_price, high_price, low_price, close_price, cum_volume, cum_dollar_value])
            cum_dollar_value, cum_volume, cache, high_price, low_price = 0, 0, [], -np.inf, np.inf

    return bars, cache


def get_dollar_bars(file_name, threshold=50000, chunksize=20000000):

    # Variables
    count = 0
    flag = False
    final_bars = []
    cache = []

    # Read csv in batches
    for batch in pd.read_csv(file_name, chunksize=chunksize):
        print('Batch number:', count)
        bars, price_cache = create_bars(batch, metric='cum_dollar_value', threshold=threshold, cache=cache, flag=flag)

        # Append to bars list
        final_bars += bars
        count += 1

        # Set flag to True: notify function to use cache
        flag = True

    cols = ['date_time', 'open', 'high', 'low', 'close', 'cum_vol', 'cum_dollar', 'cum_ticks']
    df = pd.DataFrame(final_bars, columns=cols)
    return df


if __name__ == '__main__':

    print('Creating bars...')
    bars = get_dollar_bars(file_name="big_es_dataEStrim.csv", threshold=50000, chunksize=20000000)
    print('Writing to csv')
    bars.to_csv('dollar_bars.csv', index=False)


    # Write to csv
    # cols = ['date_time', 'open', 'high', 'low', 'close', 'cum_vol', 'cum_dollar']
    # pd.DataFrame(final_bars, columns=cols).to_csv('dollar_bars.csv', index=False)

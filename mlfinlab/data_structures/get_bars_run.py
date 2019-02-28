from mlfinlab.data_structures.data_structures import get_dollar_bars, get_tick_bars, get_volume_bars


if __name__ == '__main__':

    path = "/media/ariadne/60EC1FF4EC1FC2E8/all_es_dataEStrim.csv"

    print('Creating Dollar Bars...')
    dollar_bars = get_dollar_bars(file_path=path, threshold=70000000, batch_size=20000000)
    print('Writing to csv')
    dollar_bars.to_csv('dollar_bars.csv', index=False)

    print('Creating Volume Bars...')
    volume_bars = get_volume_bars(file_path=path, threshold=28224, batch_size=20000000)
    print('Writing to csv')
    volume_bars.to_csv('volume_bars.csv', index=False)

    print('Creating Tick Bars...')
    tick_bars = get_tick_bars(file_path=path, threshold=2800, batch_size=20000000)
    print('Writing to csv')
    tick_bars.to_csv('tick_bars.csv', index=False)
    print('Done!')

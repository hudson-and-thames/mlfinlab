def mpNumCoEvents(closeIdx, t1, molecule):
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]

    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn, tOut in t1.iteritems():
        count.loc[tIn:tOut] += 1
    return count.loc[molecule[0]:t1[molecule].max()]


def mpSampleTW(t1, numCoEvents, molecule):
    # Derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (1./mpNumCoEvents.loc[tIn:tOut].mean())
    return wght

mpNumCoEvent = mp_pandas_obj(mpNumCoEvents, ('molecule', triple_barrier_events.index), 4, closeIdx=raw_series.index, t1=triple_barrier_events['t1'])
mpNumCoEvent = mpNumCoEvent.loc[~mpNumCoEvents.index.duplicated(keep='last')]
mpNumCoEvent = mpNumCoEvent.reindex(close.index).fillna(0)
out['W'] = mpPandasObj(mpSampleTW, ('molecule', events.index), numThreads, t1=events['t1'], numCoEvents=numCoEvent)

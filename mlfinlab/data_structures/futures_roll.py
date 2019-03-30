def get_futures_roll_series(df, sec_col, current_sec_col, open_col, close_col, roll_backward=False):
    # filter out security data which is not used as current security
    df = df[df[sec_col] == df[current_sec_col]]
    df.sort_index(inplace=True)
    roll_dates = df[current_sec_col].drop_duplicates(keep='first').index
    gaps = df[close_col] * 0
    gaps.loc[roll_dates[1:]] = series[open_col].loc[roll_dates[1:]] - \
        series[close_col].loc[roll_dates[1:]
                              ]  # TODO: undertand why Marcos used iloc and list logic
    if roll_backward:
        gaps -= gaps.iloc[-1]  # roll backward
    df[close_col] -= gaps
    return df[close_col]

"""
Accumulation/Distribution Cumulative Indicator

"Classification-based Financial Markets Prediction using Deep Neural Networks" by Dixon et al. (2016) describes how
labeling data this way can be used in training deep neural networks to predict price movements.

For further details: https://www.investopedia.com/terms/a/accumulationdistribution.asp
"""


def accumulation_distribution(price):
    """
    :param price: (pd.DataFrame) Containing daily high, low, and close prices, as well as traded volume for a single
                    stock.
    :return: (pd.Series) Acculumation/Distribution (A/D) indicator for each day. The A/D indicator is cumulative, and a
                rising trend indicates a bullish sentiment while a falling indicates a bearish one. A rising price with
                falling A/D signals potential decline in price while falling price and rising A/D signals potential
                increase in price. See the documentation or
                https://www.investopedia.com/terms/a/accumulationdistribution.asp for a detailed explanation.
    """
    # Current money flow volume (CMFV)
    cmfv = price['Volume'] * ((price['Close'] - price['Low']) - (price['High'] - price['Close'])) / (
            price['High'] - price['Low'])
    cmfv_cumsum = cmfv.cumsum()

    return cmfv_cumsum

.. _implementations-labeling_matrix_flags:

=====================
Labeling Matrix Flags
=====================

Labeling price data according to a template to identify patterns in price changes is featured in the following paper:
`Cervelló-Royo, R., Guijarro, F. and Michniuk, K., 2015. Stock market trading rule based on pattern recognition and technical
analysis: Forecasting the DJIA index with intraday data. <https://www.sciencedirect.com/science/article/abs/pii/S0957417415002110>`_

This method was originally introduced in `Leigh, W., Modani, N., Purvis, R. and Roberts, T., 2002. Stock market trading rule discovery
using technical charting heuristics. <http://chart-patterns.technicalanalysis.org.uk/LMPR02.pdf>`_, which describes this method in more
detail.

A bull flag occurs when a stock's price rapidly increases, followed by a downwards trending consolidation period, followed by a breakout
increase in price confirming the original increase. As defined, "A bull flag pattern is a horizontal or downward
sloping flag of consolidation followed by a sharp rise in the positive direction, the breakout." [Leigh et al. 2002].
Being able to identify the early stages of the breakout process can lead to a profitable strategy of buying the breakout and
then selling some number of days later, when the price has theoretically stabilized again.

Cervelló-Royo et al. expand on Leigh et al.'s work by proposing a new bull flag pattern which ameliorates some weaknesses in Leigh's original template,
such as the possibility of false positives given the path the stock took. Additionally, he applies this bull flag labeling method to intraday
candlestick data, rather than just closing prices.

The bull flag labeling pattern requires the use of a template to match the price data to. Below is an example bull flag
template.

.. figure:: labeling_images/bull_flag_template.png
   :scale: 60 %
   :align: center
   :figclass: align-center
   :alt: bull flag template

   Bull flag template, from Cervelló-Royo et al. (2015), originally proposed by Leigh et al. (2002). The first 7 columns represent
   consolidation, while final 3 columns represent the early stages of the breakout.

To find the template fitting value for a given day, the relevant data window consists of the day's price and prices of a number of
preceding days. The data window is split into 10 buckets each containing a chronological tenth of
the data window. Decile cutoffs for prices in the entire data window are found. Each bucket is translated to a column with 10 elements,
such that the topmost element is the proportions of prices in the bucket that is in the top decile in the entire data window,
the second element is the proportion in the second decile, and so on until the tenth element is the proportion in the bottom decile.
Thus, a column is generated for each of 10 buckets of the data. The columns are then put together chronologically such that the first
column is on the left and last column is on the right in the resulting 10 by 10 matrix. The matrix is then multiplied element-wise
by the template matrix, and all columns are summed. The sum of all columns, finally, is the fit value for the day. If desired, the user
can specify a threshold to determine positive and negative classes. The value of the threshold depends on how strict of a classifier the user
desires, and the allowable values based on the template matrix.

The following shows the identified bull flag regions on price data in MSFT stock from 2019-2020 using the original template shown above, with
a time window of 40 days.

.. figure:: labeling_images/msft_bull_flag40.png
   :scale: 120 %
   :align: center
   :figclass: align-center
   :alt: bull flag msft

   Bull flag template from Leigh et al. (2002) applied to MSFT data. Green dots show when the template has identified the region as a
   bull flag breakout point.

The user should know what kind of timescale is desired when deciding the data window for this method. Using a small data window will catch
small, short-lived trends, while missing longer-term trends, and vice versa for being a large data window. Additionally, the choice of template
determines which kind of pattern is tracked, and should be customized with respect to the data. A template which is optimal for intraday prices, for
example, may not work nearly as well for close prices.

.. tip::
   **Simple Example**

    It's perhaps easiest to illustrate this process with a simple example. Suppose we have the following data window of 20
    prices [100, 102, ..., 118, 120, 118, ..., 104, 102]. The decile cutoffs are then [102, 104, ..., 120]. Note that these cutoffs
    are right inclusive, so a value of 102 would fit into the (100, 102) percentile. We split the data into ten chronological
    subsets such that the first subset is [100, 102]. 100% of elements in the first subset fall into the lowest decile, so the
    corresponding column would be [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]. The second subset is [104, 106], of which 1 out of 2 is in the
    2nd decile, and the other is in the 3rd decile, so the column would be [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0]. This is done
    until the entire 10 by 10 matrix is generated. This matrix is then multiplied element-wise by the template, and resulting values
    are summed to get the total template fit. The higher the fit, the better match to the template. Using the template shown above,
    the highest possible fit is 10.


Implementation
##############

.. automodule:: mlfinlab.labeling.matrix_flags

    .. autoclass:: MatrixFlagLabels
        :members:

        .. automethod:: __init__

Example
########
Below is an example on how to use the matrix flags labeling method.

.. code-block::

    import pandas as pd
    from mlfinlab.labeling.matrix_flags import MatrixFlagLabels

    # Import price data
    msft = yf.Ticker("MSFT")
    hist = msft.history(start='2020-1-1', end='2020-6-01')
    data = hist['Close']

    # Initialize with a window of 60 days.
    Flags = MatrixFlagLabels(data=data, window=60)

    # Get numerical weights based on the template (for days 60 and onwards).
    weights = Flags.apply_labeling_matrix()

    # Get categorical labels based on whether the day's weight is above 2.5.
    categorical = Flags.apply_labeling_matrix(threshold=2.5)

    # Change the template from default to user defined.
    new_template = pd.DataFrame(np.random.randint(-3, 3, size=(10, 10)))
    Flags.set_template(template=new_template)



Research Notebook
#################

The following research notebook can be used to better understand the matrix flags labeling technique.

* `Matrix Flags Example`_

.. _`Matrix Flags Example`: https://github.com/hudson-and-thames/research/blob/master/Labelling/Labels%20Matrix%20Flags/Matrix%20Flag%20Labels.ipynb





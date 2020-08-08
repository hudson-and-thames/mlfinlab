.. _implementations-labeling_fixed_time_horizon:

.. note::
    This section includes an accompanying Jupyter Notebook Tutorial that is now available via the respective tier on
    `Patreon <https://www.patreon.com/HudsonThames>`_.

====================
Fixed Horizon Method
====================

Fixed horizon labels is a classification labeling technique used in the following paper: `Dixon, M., Klabjan, D. and
Bang, J., 2016. Classification-based Financial Markets Prediction using Deep Neural Networks. <https://arxiv.org/abs/1603.08604>`_

Fixed time horizon is a common method used in labeling financial data, usually applied on time bars. The rate of return relative
to :math:`t_0` over time horizon :math:`h`, assuming that returns are lagged, is calculated as follows (M.L. de Prado, Advances in Financial Machine Learning, 2018):

.. math::
    r_{t0,t1} = \frac{p_{t1}}{p_{t0}} - 1

Where :math:`t_1` is the time bar index after a fixed horizon has passed, and :math:`p_{t0}, p_{t1}`
are prices at times :math:`t_0, t_1`. This method assigns a label based on comparison of rate of return to a threshold :math:`\tau`

 .. math::
     \begin{equation}
     \begin{split}
       L_{t0, t1} = \begin{cases}
       -1 &\ \text{if} \ \ r_{t0, t1} < -\tau\\
       0 &\ \text{if} \ \ -\tau \leq r_{t0, t1} \leq \tau\\
       1 &\ \text{if} \ \ r_{t0, t1} > \tau
       \end{cases}
     \end{split}
     \end{equation}

To avoid overlapping return windows, rather than specifying :math:`h`, the user is given the option of resampling the returns to
get the desired return period. Possible inputs for the resample period can be found `here.
<https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_.
Optionally, returns can be standardized by scaling by the mean and standard deviation of a rolling window. If threshold is a pd.Series,
**threshold.index and prices.index must match**; otherwise labels will fail to be returned. If resampling
is used, the threshold must match the index of prices after resampling. This is to avoid the user being forced to manually fill
in thresholds.

The following shows the distribution of labels for standardized returns on closing prices of SPY in the time period from Jan 2008 to July 2016
using a 20-day rolling window for the standard deviation.

.. figure:: labeling_images/fixed_horizon_labels_example.png
   :scale: 100 %
   :align: center
   :figclass: align-center
   :alt: fixed horizon example

   Distribution of labels on standardized returns on closing prices of SPY.

Though time bars are the most common format for financial data, there can be potential problems with over-reliance on time bars. Time
bars exhibit high seasonality, as trading behavior may be quite different at the open or close versus midday; thus it will not be
informative to apply the same threshold on a non-uniform distribution. Solutions include applying the fixed horizon method to tick or
volume bars instead of time bars, using data sampled at the same time every day (e.g. closing prices) or inputting a dynamic threshold
as a pd.Series corresponding to the timestamps in the dataset. However, the fixed horizon method will always fail to capture information
about the path of the prices [Lopez de Prado, 2018].

.. tip::
   **Underlying Literature**

   The following sources describe this method in more detail:

   - **Advances in Financial Machine Learning, Chapter 3.2** *by* Marcos Lopez de Prado (p. 43-44).
   - **Machine Learning for Asset Managers, Chapter 5.2** *by* Marcos Lopez de Prado (p. 65-66).


Implementation
##############

.. py:currentmodule:: mlfinlab.labeling.fixed_time_horizon
.. automodule:: mlfinlab.labeling.fixed_time_horizon
   :members:

Example
########
Below is an example on how to use the Fixed Horizon labeling technique on real data.

.. code-block::

    import pandas as pd
    import numpy as np

    from mlfinlab.labeling import fixed_time_horizon

    # Import price data.
    data = pd.read_csv('../Sample-Data/stock_prices.csv', index_col='Date', parse_dates=True)
    custom_threshold = pd.Series(np.random.random(len(data)), index = data.index)

    # Create labels.
    labels = fixed_time_horizon(prices=data, threshold=0.01, lag=True)

    # Create labels with a dynamic threshold.
    labels = fixed_time_horizon(prices=data, threshold=custom_threshold, lag=True)

    # Create labels with standardization.
    labels = fixed_time_horizon(prices=data, threshold=1, lag=True, standardized=True, window=5)

    # Create labels after resampling weekly with standardization.
    labels = fixed_time_horizon(prices=data, threshold=1, resample_by='W', lag=True,
                                standardized=True, window=4)


Research Notebook
#################

.. note::
    This and other accompanying Jupyter Notebook Tutorials are now available via the respective tier on
    `Patreon <https://www.patreon.com/HudsonThames>`_.

The following research notebook can be used to better understand the Fixed Horizon labeling technique.

* `Fixed Horizon Example`_

.. _`Fixed Horizon Example`: https://github.com/Hudson-and-Thames-Clients/research/blob/master/Labeling/Labels%20Fixed%20Horizon/Fixed%20Time%20Horizon.ipynb

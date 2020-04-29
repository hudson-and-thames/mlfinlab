.. _implementations-structural_breaks:

=================
Structural Breaks
=================

This implementation is based on Chapter 17 of the book Advances in Financial Machine Learning. Structural breaks, like
the transition from one market regime to another, represent the shift in behaviour of market participants.

The first market participant to notice the changes in the market can adapt to them before others and, consequently,
gain advantage over market participants who have not yet noticed market regime changes.

To quote Marcos Lopez de Prado, "Structural breaks offer some of the best risk/rewards".

We can classify the structural break in two general categories:

1. **CUSUM tests**: These test whether the cumulative forecasting errors significantly deviate from white noise.
2. **Explosiveness tests**: Beyond deviation from white noise, these test whether the process exhibits exponential
   growth or collapse, as this is inconsistent with a random walk or stationary process, and it is unsustainable in the long run.

.. tip::
   **Underlying Literature**

   The following sources elaborate extensively on the topic:

   - **Advances in Financial Machine Learning** *by* Marcos Lopez de Prado *Chapter 17 describes structural breaks in more detail.*
   - **Testing for Speculative Bubbles in Stock Markets: A Comparison of Alternative Methods** *by* Ulrich Homm *and* Jorg Breitung `available here <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.511.6559&rep=rep1&type=pdf>`__. *Explains the Chu-Stinchcombe-White CUSUM Test in more detail.*


.. figure:: structural_breaks_images/sadf.png
   :scale: 70 %
   :align: center
   :figclass: align-center
   :alt: structural breaks

   Image showing SADF test statistic with 5 lags and linear model.

CUSUM tests
###########

Chu-Stinchcombe-White CUSUM Test on Levels
*******************************************

We are given a set of observations :math:`t = 1, ... , T` and we assume an array of features :math:`x_{i}` to be
predictive of a value :math:`y_{t}` .

.. math::
    y_{t} = \beta_{t}x_{t} + \epsilon_{t}

Authors of the **Testing for Speculative Bubbles in Stock Markets: A Comparison of Alternative Methods** paper suggest
assuming :math:`H_{0} : \beta_{t} = 0` and therefore forecast :math:`E_{t-1}[\Delta y_{t}] = 0`. This allows to work directly
with :math:`y_{t}` instead of computing recursive least squares (RLS) estimates of :math:`\beta` .

As :math:`y_{t}` we take the log-price and calculate the standardized departure of :math:`y_{t}` relative to :math:`y_{n}`
(CUSUM statistic) with :math:`t > n` as:

.. math::
    \begin{equation}
    \begin{split}
        S_{n,t} & = (y_{t}-y_{n})(\hat\sigma_{t}\sqrt{t-n})^{-1}, \ \ t>n \\
        \hat\sigma_{t}^{2} & = (t-1)^{-1} \sum_{i=2}^{t}({\Delta y_{t_{i}}})^2 \\
    \end{split}
    \end{equation}

With the :math:`H_{0} : \beta_{t} = 0` hypothesis, :math:`S_{n, t} ~ N[0, 1]` .

We can test the null hypothesis comparing CUSUM statistic :math:`S_{n, t}` with critical value :math:`c_{\alpha}[n, t]`,
which is calculated using a one-sided test as:

.. math::

    c_{\alpha}[n, t] = \sqrt{b_{\alpha} + \log{[t-n]}}

The authors in the above paper have derived using Monte Carlo method that :math:`b_{0.05} = 4.6` .

The disadvantage is that :math:`y_{n}` is chosen arbitrarily, and results may be inconsistent due to that. This can be
fixed by estimating :math:`S_{n, t}` on backward-shifting windows :math:`n \in [1, t]` and pick:

.. math::
    \begin{equation}
    S_{t}= \sup_{n \in [1, t]} \{ S_{n, t}\}
    \end{equation}

.. py:currentmodule:: mlfinlab.structural_breaks.cusum
.. automodule:: mlfinlab.structural_breaks.cusum
   :members:

.. py:currentmodule:: mlfinlab.structural_breaks.sadf
.. automodule:: mlfinlab.structural_breaks.sadf
   :members:

Examples
########

.. code-block::

    import pandas as pd
    import numpy as np
    from mlfinlab.structural_breaks import (get_chu_stinchcombe_white_statistics,
                                            get_chow_type_stat, get_sadf)

    bars = pd.read_csv('BARS_PATH', index_col = 0, parse_dates=[0])
    log_prices = np.log(bars.close) # see p.253, 17.4.2.1 Raw vs Log Prices

    # Chu-Stinchcombe test
    one_sided_test = get_chu_stinchcombe_white_statistics(log_prices, test_type='one_sided')
    two_sided_test = get_chu_stinchcombe_white_statistics(log_prices, test_type='two_sided')

    # Chow-type test
    chow_stats = get_chow_type_stat(log_prices, min_length=20)

    # SADF test
    linear_sadf = get_sadf(log_prices, model='linear', add_const=True, min_length=20, lags=5)

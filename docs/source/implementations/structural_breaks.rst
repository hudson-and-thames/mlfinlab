.. _implementations-structural_breaks:

=================
Structural Breaks
=================

This implementation is based on Chapter 17 of the book Advances in Financial Machine Learning. Structural breaks, like
the transition from one market regime to another, represent the shift in behaviour of market participants.
If you can notice structural changes in the market, you'd better notice changes in the market.
This is because it is an opportunity to adapt to market changes before others and consequently,
it can bring revenue from market participants who have not yet noticed market regime changes.
To quote Marcos Lopez de Prado, "Structural breaks offer some of the best risk/rewards".
We can classify the structural break in two general categories and we use CUSUM tests and Explosiveness test respectively to discover those.

1. **CUSUM tests**: These test whether the cumulative forecasting errors significantly deviate from white noise.
2. **Explosiveness tests**: Beyond deviation from white noise, these test whether the process exhibits exponential
   growth or collapse, as this is inconsistent with a random walk or stationary process, and it is unsustainable in the long run.


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

.. math::
    y_{t} = \beta_{t}x_{t} + \epsilon_{t}

This method tests efficiently structural break by directly checking :math:`y_{t}`
rather than computing and checking residual of recursive least squares(RLS).
The null hypothesis is made based on  :math:`E_{t-1}[\Delta y_{t}]`.

.. math::
    \begin{equation}
    \begin{split}
        y_{t} & = \beta_{t}x_{t} + \epsilon_{t} \\
        H_{0} & : \beta_{t} = 0, \text{then}\  S_{n, t}  \text{~}\  N[0,1]
    \end{split}
    \end{equation}

When the null hypothesis is true, CUSUM statistic :math:`S_{n, t}` is calculated as follows

.. math::
    \begin{equation}
    \begin{split}
        S_{n,t} & = (y_{t}-y_{n})(\hat\sigma_{t}\sqrt{t-n})^{-1}, \ \ t>n \\
        \hat\sigma_{t}^{2} & = (t-1)^{-1} \sum_{i=2}^{t}({\Delta y_{t_{i}}})^2 \\
    \end{split}
    \end{equation}

We can test our null hypothesis comparing CUSUM statistic :math:`S_{n, t}` with critical value :math:`c_{\alpha}[n, t]`.
We can calculate the critical value at a significance level of 0.05 which is derived via Monte Carlo, :math:`b_{0.05} = 4.6`.

.. math::
    \begin{equation}
    \begin{split}
        c_{\alpha}[n, t] & = \sqrt{b_{\alpha} + \log{[t-n]}} \\
        b_{0.05} & = 4.6 \\
    \end{split}
    \end{equation}

However, :math:`y_{n}` is chosen arbitrarily, and results may be inconsistent due to that.
Here, we estimate :math:`S_{n, t}` and pick :math:`S_{t}`  as follows.

.. math::
    \begin{equation}
    S_{t}= \sup_{n \in [1, t]} \{ S_{n, t}\}
    \end{equation}

Through the below method, we can easily do chu-Stinchcombe-White CUSUM test.
The only thing you care about is deciding two-sided or one-sided test.

.. py:currentmodule:: mlfinlab.structural_breaks.cusum

.. autofunction:: get_chu_stinchcombe_white_statistics

----

Explosiveness tests
####################

Chow-Type Dickey-Fuller Test
*****************************

Chow-type Dickey-Fuller is based on an :math:`AR(1)` process.
This test is used for detecting whether the process changes from the random walk into an explosive process at time interval :math:`[1,T]`.
The drawback of this method is it is not a proper tool for checking explosive process when there is likely to be multiple regime changes
because it assumes that there is only one break and that the bubble runs up to the end of the sample.

.. math::
    \begin{equation}
    \begin{split}
      H_{0} & : y_{t} = y_{t-1} + \rho_{t} \\
      H_{1} & :
      y_{t}=\begin{cases}
      y_{t-1} + \epsilon_{t} \ \text{for} \ \ t= 1, ..., \tau^*T  \\
      \rho y_{t-1} + \epsilon_{t} \ \text{for} \ \ t= \tau^*T+1, ..., T, \text{with } \rho > 1
      \end{cases} \\
        & : \tau^* \in (0,1)
    \end{split}
    \end{equation}

.. math::
    \begin{equation}
    \begin{split}
      H_{1} & :
      y_{t}=\begin{cases}
      y_{t-1} + \epsilon_{t} \ \text{for} \ \ t= 1, ..., \tau^*T  \\
      \rho y_{t-1} + \epsilon_{t} \ \text{for} \ \ t= \tau^*T+1, ..., T, \text{with } \rho > 1
      \end{cases} \\
        & : \tau^* \in (0,1)
    \end{split}
    \end{equation}

To test this hypothesis, we fit the following specification

.. math::
    \Delta y_{t} = \delta y_{t-1} D_{t}{\tau^*} + \rho_{t}

.. math::
    \begin{equation}
    \begin{split}
      D_{t}{\tau^*} & = \begin{cases}
      0 \ \text{if} \ \ t < \tau^*T  \\
      1 \ \text{if} \ \ t \geq \tau^*T
      \end{cases} \\
        & : \tau^* \in (0,1)
    \end{split}
    \end{equation}

.. math::
    \begin{equation}
    \begin{split}
      H_{0} & : \delta = 0 \\
      H_{1} & : \delta > 1 \\
    \end{split}
    \end{equation}

Here, Dickey-Fuller-Chow(DFC) test-statistics for :math:`\tau^*`, :math:`DFC_{\tau^*} = \frac{\hat\delta}{\hat\sigma_{\delta}}`
However, we don't know the :math:`\tau`.
So, what we use Supremum Dickey-Fuller Chow(:math:`SDFC`) for the test statistic for an unknown :math:`\tau^*`

.. math::
    \begin{equation}
    SDFC= \sup_{\tau^* \in [\tau_0,1-\tau_0]} \{ DFC_{\tau^*}\}
    \end{equation}

As SDFC value is getting bigger, it is likely to lead to the rejection of the null hypothesis which means our data is random walk.

You can get a series of :math:`DFC` of your data at each time index.
To ensure that the regime is fitted with enough observations,
We need to set minimum portion of the beginning and end of the samples.


.. py:currentmodule:: mlfinlab.structural_breaks.chow

.. autofunction:: get_chow_type_stat


----

Supremum Augmented Dickey-Fuller
********************************

Through this test, we do multiple ADF test by adjusting the size of the window at various time points.
Therefore, situations with multiple regime switches and break dates can also be considered, unlike the Chow-Type Dickey-Fuller Test.


We add multiple lagged differences to the regression setting of the Chow-Type Dickey-Fuller Test and use it for specification
because it is more suitable to catch the difference between a stationary process and a periodically collapsing bubble model.

.. math::
    \begin{equation}
     \Delta y_{t} = \alpha + \beta y_{t-1} \sum_{l=1}^{L}{\gamma\Delta y_{t-l} + \epsilon_{t}}
    \end{equation}

Supremum Augmented Dickey-Fuller(SADF) test-statistics at :math:`t` is calculated as follows.


.. math::
    \begin{equation}
     SADF_{t} = \sup_{t_0 \in [1, t-\tau]}\{ADF_{t_0, t}\} = \sup_{t_0 \in [1, t-\tau]} \Bigg\{\frac{\hat\beta_{t_0,t}}{\hat\sigma_{\beta_{t_0, t}}}\Bigg\}
    \end{equation}


You can view the standard ADF test as a special case of :math:`SADF_t` where :math:`\tau=t-1`.
Bubbles can produce significantly different levels between regimes.
So, in this method, we assume using :math:`\log` prices not raw price data because :math:`\log` price is a more suitable choice
for running :math:`SADF` on long-time series data than raw price.

.. py:currentmodule:: mlfinlab.structural_breaks.sadf

.. autofunction:: get_sadf

In SADF Test, we should estimate :math:`\beta` and variance of it to fit the ADF specification.
We use this method to do that.

.. py:currentmodule:: mlfinlab.structural_breaks.sadf

.. autofunction:: get_betas


----


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

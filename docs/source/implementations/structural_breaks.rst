.. _implementations-structural_breaks:

=================
Structural Breaks
=================

This implementation is based on Chapter 17 of the book Advances in Financial Machine Learning. Structural breaks, like
the transition from one market regime to another, represent the shift in the behaviour of market participants.

The first market participant to notice the changes in the market can adapt to them before others and, consequently,
gain an advantage over market participants who have not yet noticed market regime changes.

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
   - **Tests of Equality Between Sets of Coefficients in Two Linear Regressions** *by* Gregory C. Chow `available here <http://web.sonoma.edu/users/c/cuellar/econ411/chow>`__. *A work that inspired a family of explosiveness tests.*

CUSUM tests
###########

Chu-Stinchcombe-White CUSUM Test on Levels
*******************************************

We are given a set of observations :math:`t = 1, ... , T` and we assume an array of features :math:`x_{i}` to be
predictive of a value :math:`y_{t}` .

.. math::
    y_{t} = \beta_{t}x_{t} + \epsilon_{t}

Authors of the **Testing for Speculative Bubbles in Stock Markets: A Comparison of Alternative Methods** paper suggest
assuming :math:`H_{0} : \beta_{t} = 0` and therefore forecast :math:`E_{t-1}[\Delta y_{t}] = 0`. This allows working directly
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

With the :math:`H_{0} : \beta_{t} = 0` hypothesis, :math:`S_{n, t} \sim N[0, 1]` .

We can test the null hypothesis comparing CUSUM statistic :math:`S_{n, t}` with critical value :math:`c_{\alpha}[n, t]`,
which is calculated using a one-sided test as:

.. math::

    c_{\alpha}[n, t] = \sqrt{b_{\alpha} + \log{[t-n]}}

The authors in the above paper have derived using Monte Carlo method that :math:`b_{0.05} = 4.6` .

The disadvantage is that :math:`y_{n}` is chosen arbitrarily, and results may be inconsistent due to that. This can be
fixed by estimating :math:`S_{n, t}` on backward-shifting windows :math:`n \in [1, t]` and picking:

.. math::
    \begin{equation}
    S_{t}= \sup_{n \in [1, t]} \{ S_{n, t}\}
    \end{equation}

.. py:currentmodule:: mlfinlab.structural_breaks.cusum

.. autofunction:: get_chu_stinchcombe_white_statistics

----

Explosiveness tests
####################

Chow-Type Dickey-Fuller Test
*****************************

The Chow-Type Dickey-Fuller test is based on an :math:`AR(1)` process:

.. math::

      y_{t} = \rho y_{t} + \varepsilon_{t}

where :math:`\varepsilon_{t}` is white noise.

This test is used for detecting whether the process changes from the random walk (:math:`\rho = 1`) into an explosive
process at some time :math:`\tau^{*}T`, :math:`\tau^{*} \in (0,1)`, where :math:`T` is the number of observations.

So, the hypothesis :math:`H_{0}` is tested against :math:`H_{1}`:

.. math::
    \begin{equation}
    \begin{split}
      H_{0} & : y_{t} = y_{t-1} + \varepsilon_{t} \\
      H_{1} & :
      y_{t}=\begin{cases}
      y_{t-1} + \varepsilon_{t} \ \text{for} \ \ t= 1, ..., \tau^*T  \\
      \rho y_{t-1} + \varepsilon_{t} \ \text{for} \ \ t= \tau^*T+1, ..., T, \text{ with } \rho > 1
      \end{cases}
    \end{split}
    \end{equation}

To test the hypothesis, the following specification is being fit:

.. math::
    \Delta y_{t} = \delta y_{t-1} D_{t}[\tau^*] + \varepsilon_{t}
.. math::
    \begin{equation}
    \begin{split}
      D_{t}[\tau^*] & = \begin{cases}
      0 \ \text{if} \ \ t < \tau^*T  \\
      1 \ \text{if} \ \ t \geq \tau^*T
      \end{cases}
    \end{split}
    \end{equation}

So, the hypothesis tested are now transformed to:

.. math::
    \begin{equation}
    \begin{split}
      H_{0} & : \delta = 0 \\
      H_{1} & : \delta > 1 \\
    \end{split}
    \end{equation}

And the Dickey-Fuller-Chow(DFC) test-statistics for :math:`\tau^*` is:

.. math::

    DFC_{\tau^*} = \frac{\hat\delta}{\hat\sigma_{\delta}}

As described in the **Advances in Financial Machine Learning**:

The first drawback of this method is that :math:`\tau^{*}` is unknown, and the second one is that Chow's approach
assumes that there is only one break and that the bubble runs up to the end of the sample.

To address the first issue, in the work **Tests for Parameter Instability and Structural Change With Unknown ChangePoint**
`available here <https://pdfs.semanticscholar.org/77c9/86937d205592a007df3661778a5ed4fc4e38.pdf>`__, Andrews proposed to
try all possible :math:`\tau^{*}` in an interval :math:`\tau^{*} \in [\tau_{0}, 1 - \tau_{0}]`

For the unknown :math:`\tau^{*}` the test statistic is the Supremum Dickey-Fuller Chow which is the maximum of all
:math:`T(1 - 2\tau_{0})` values of :math:`DFC_{\tau^{*}}` :

.. math::
    \begin{equation}
    SDFC = \sup_{\tau^* \in [\tau_0,1-\tau_0]} \{ DFC_{\tau^*}\}
    \end{equation}

To address the second issue, the Supremum Augmented Dickey-Fuller test was introduced.

.. py:currentmodule:: mlfinlab.structural_breaks.chow

.. autofunction:: get_chow_type_stat

----

Supremum Augmented Dickey-Fuller
********************************

This test was proposed by Phillips, Shi, and Yu in the work **Testing for Multiple Bubbles: Historical Episodes of Exuberance and Collapse in the S&P 500**
`available here <http://korora.econ.yale.edu/phillips/pubs/art/p1498.pdf>`__. The advantage
of this test is that it allows testing for multiple regimes switches (random walk to bubble and back).

The test is based on the following regression:

.. math::
     \Delta y_{t} = \alpha + \beta y_{t-1} + \sum_{l=1}^{L}{\gamma_{l} \Delta y_{t-l}} + \varepsilon_{t}

And, the hypothesis :math:`H_{0}` is tested against :math:`H_{1}`:

.. math::
    \begin{equation}
    \begin{split}
      H_{0} & : \beta \le 0 \\
      H_{1} & : \beta > 0 \\
    \end{split}
    \end{equation}

The Supremum Augmented Dickey-Fuller fits the above regression for each end point :math:`t` with backward expanding
start points and calculates the test-statistic as:

.. math::
    \begin{equation}
     SADF_{t} = \sup_{t_0 \in [1, t-\tau]}\{ADF_{t_0, t}\} = \sup_{t_0 \in [1, t-\tau]} \Bigg\{\frac{\hat\beta_{t_0,t}}{\hat\sigma_{\beta_{t_0, t}}}\Bigg\}
    \end{equation}

where :math:`\hat\beta_{t_0,t}` is estimated on the sample from :math:`t_{0}` to :math:`t`, :math:`\tau` is the minimum
sample length in the analysis, :math:`t_{0}` is the left bound of the backwards expanding window, :math:`t` iterates
through :math:`[\tau, ..., T]` .

In comparison to SDFC, which is computed only at time :math:`T`, the SADF is computed at each :math:`t \in [\tau, T]`,
recursively expanding the sample :math:`t_{0} \in [1, t - \tau]` . By doing so, the SADF does not assume a known number of
regime switches.

.. figure:: structural_breaks_images/sadf.png
   :scale: 70 %
   :align: center
   :figclass: align-center
   :alt: structural breaks

   Image showing SADF test statistic with 5 lags and linear model. The
   SADF line spikes when prices exhibit a bubble-like behavior, and returns to low levels
   when the bubble bursts.

The `model` and the `add_const` parameters of the **get_sadf** function allow for different specifications of the
regression's time trend component.

Linear model (`model='linear'`) uses a linear time trend:

.. math::

      \Delta y_{t} = \beta y_{t-1} + \sum_{l=1}^{L}{\gamma_{l} \Delta y_{t-l}} + \varepsilon_{t}

Quadratic model (`model='quadratic'`) uses a second-degree polynomial time trend:

.. math::

      \Delta y_{t} = \beta y_{t-1} + \sum_{l=1}^{L}{\gamma_{l} \Delta y_{t-l}} + \sum_{l=1}^{L}{\delta_{l}^2 \Delta y_{t-l}} + \varepsilon_{t}

Adding a constant (`add_const=True`) to those specifications results in:

.. math::

      \Delta y_{t} = \alpha + \beta y_{t-1} + \sum_{l=1}^{L}{\gamma_{l} \Delta y_{t-l}} + \varepsilon_{t}

and

.. math::

      \Delta y_{t} = \alpha + \beta y_{t-1} + \sum_{l=1}^{L}{\gamma_{l} \Delta y_{t-l}} + \sum_{l=1}^{L}{\delta_{l}^2 \Delta y_{t-l}} + \varepsilon_{t}

respectively.

.. py:currentmodule:: mlfinlab.structural_breaks.sadf

.. autofunction:: get_sadf

The function used in the SADF Test to estimate the :math:`\hat\beta_{t_0,t}` is:

.. py:currentmodule:: mlfinlab.structural_breaks.sadf

.. autofunction:: get_betas

.. tip::

   **Advances in Financial Machine Learning** book additionally describes why log prices data is more appropriate to use
   in the above tests, their computational complexity, and other details.

The SADF also allows for explosiveness testing that doesn't rely on the standard ADF specification. If the process is either
sub- or super martingale, the hypotheses :math:`H_{0}: \beta = 0, H_{1}: \beta \ne 0` can be tested under these specifications:

Polynomial trend (`model='sm_poly_1'`):

.. math::

      y_{t} = \alpha + \gamma t + \beta t^{2} + \varepsilon_{t}

Polynomial trend (`model='sm_poly_2'`):

.. math::

      log[y_{t}] = \alpha + \gamma t + \beta t^{2} + \varepsilon_{t}

Exponential trend (`model='sm_exp'`):

.. math::

      y_{t} = \alpha e^{\beta t} + \varepsilon_{t} \Rightarrow log[y_{t}] = log[\alpha] + \beta t^{2} + \xi_{t}

Power trend (`model='sm_power'`):

.. math::

      y_{t} = \alpha t^{\beta} + \varepsilon_{t} \Rightarrow log[y_{t}] = log[\alpha] + \beta log[t] + \xi_{t}

Again, the SADF fits the above regressions for each end point :math:`t` with backward expanding start points,
but the test statistic is taken as an absolute value, as we're testing both the explosive growth and collapse.
This is described in more detail in the **Advances in Financial Machine Learning** book p. 260.

The test statistic calculated (SMT for Sub/Super Martingale Tests) is:

.. math::

     SMT_{t} = \sup_{t_0 \in [1, t-\tau]} \Bigg\{\frac{ | \hat\beta_{t_0,t} | }{\hat\sigma_{\beta_{t_0, t}}}\Bigg\}

From the book:

Parameter `phi` in range (0, 1) can be used (`phi=0.5`) to penalize large sample lengths ( "this corrects for the bias that the :math:`\hat\sigma_{\beta_{t_0, t}}`
of a weak long-run bubble  may be smaller than the :math:`\hat\sigma_{\beta_{t_0, t}}` of a strong short-run bubble,
hence biasing method towards long-run bubbles" ):

.. math::

     SMT_{t} = \sup_{t_0 \in [1, t-\tau]} \Bigg\{\frac{ | \hat\beta_{t_0,t} | }{\hat\sigma_{\beta_{t_0, t}}(t-t_{0})^{\phi}}\Bigg\}

----

Examples
########

.. code-block::

    import pandas as pd
    import numpy as np
    from mlfinlab.structural_breaks import (get_chu_stinchcombe_white_statistics,
                                            get_chow_type_stat, get_sadf)

    # Importing price data
    bars = pd.read_csv('BARS_PATH', index_col = 0, parse_dates=[0])

    # Changing to log prices data
    log_prices = np.log(bars.close) # see p.253, 17.4.2.1 Raw vs Log Prices

    # Chu-Stinchcombe test (one-sided and two-sided)
    one_sided_test = get_chu_stinchcombe_white_statistics(log_prices, test_type='one_sided')
    two_sided_test = get_chu_stinchcombe_white_statistics(log_prices, test_type='two_sided')

    # Chow-type test
    chow_stats = get_chow_type_stat(log_prices, min_length=20)

    # SADF test with linear model and a constant, lag of 5 and minimum sample length of 20
    linear_sadf = get_sadf(log_prices, model='linear', add_const=True, min_length=20, lags=5)

    # Polynomial trend SMT
    sm_poly_1_sadf = get_sadf(log_prices, model='sm_poly_1', add_const=True, min_length=20, lags=5, phi=0.5)

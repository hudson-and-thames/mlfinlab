.. _online_portfolio_selection-mean_reversion-confidence_weighted_mean_reversion:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

==================================
Confidence Weighted Mean Reversion
==================================

Extending from PAMR, Confidence Weighted Mean Reversion looks at the autocovariance across all assets.
Instead of focusing on a single asset's deviation from the original price, CWMR takes in second-order
information about the portfolio vector as well to formulate a set of weights.

For CWMR, we introduce :math:`\sum`, a measure of anti-confidence in the portfolio weights, where a smaller
value represents higher confidence for the corresponding portfolio weights.

.. math::
    (\mu_{t+1}, \Sigma_{t+1}) = \underset{\mu \in \Delta_m, \Sigma}{\arg\min}D_{KL}(N(\mu,\Sigma) | N(\mu_t,\Sigma_t) )

If the returns for the period are below the threshold, :math:`\epsilon`:

.. math::
    \text{such that } Pr[b^{\top} \cdot x_t \leq \epsilon] \geq \theta \text{ and } \mu \in \Delta_m

Here the problem can be interpreted as maximizing the portfolio confidence by minimizing :math:`\Sigma` given
a confidence interval :math:`\theta` determined by the threshold, :math:`\epsilon`.

- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p_t` is the price at time :math:`t`.
- :math:`\epsilon` is the mean reversion threshold constant.
- :math:`\theta` is the confidence interval for the given distribution.
- :math:`\mu_t` is the mean of the projected weights distribution at time :math:`t`.
- :math:`\Sigma_t` is the confidence matrix in the projected weights distribution at time :math:`t`.
- :math:`N(\mu_t, \Sigma_t)` is the normal distribution for the projected weights.
- :math:`D_{KL}` is the Kullback-Leibler Divergence. More information of KL Divergence is available `here <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_.

CWMR has two variations to solve this optimization problem with CWMR-SD and CWMR-Var.

1. CWMR-SD
##########

CWMR uses the Kullback-Leibler divergence to further formulate the optimization problem as following:

.. math::
    (\mu_{t+1}, \Sigma_{t+1}) = \arg \min \frac{1}{2} \left( \log(\frac{\det \Sigma_t}{\det \Sigma}) + Tr(\Sigma_t^{-1}\Sigma) + (\mu_t-\mu)^{\top}\Sigma_t^{-1}(\mu_t - \mu) \right)

.. math::
    \text{such that } \epsilon - \mu^{\top}\cdot x_t \geq \phi x_t^{\top} \Sigma x_t\text{, } \mu^{\top} \cdot \textbf{1} = 1 \text{, and } \mu \geq 0

2. CWMR-Var
###########

The standard deviation method further assumes the PSD property of :math:`\Sigma` to refactor the equations as the following:

.. math::
    (\mu_{t+1},\Gamma_{t+1}) = \arg \min \frac{1}{2}\left(\log(\frac{\det \Gamma_t^2}{\det \Gamma^2}) +Tr(\Gamma_t^{-2}\Gamma^2) + (\mu_t - \mu)^{\top}\Gamma_t^{-2}(\mu_t -\mu) \right)

.. math::

    \text{such that }\epsilon - \mu^{\top}\cdot x_t \geq \phi || \Gamma x_t || \text{, }\Gamma \text{ is a PSD, }\mu^{\top} \cdot \textbf{1} = 1\text{, and }\mu \geq 0

- :math:`\phi` is the inverse of the cumulative distribution function for a given confidence interval.
- :math:`Tr` is the sum of the diagonal elements in a matrix.
- :math:`\Gamma` is the square root of the matrix :math:`\Sigma`.

.. warning::

    For both CWMR-Var and CWMR-SD, the calculations involve taking the inverse of a sum of another inverse matrix.
    The constant calculations of matrix inversion is extremely unstable and makes the model prone to any outliers and hyperparameters.

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Mean%20Reversion.ipynb>`_
    provides a more detailed exploration of the strategies.

Parameters
----------

CWMR in general does not have an optimal parameter. The results are extremely dependent on the
hyperparameters as seen with the case for the NYSE and TSE dataset.

For NYSE, a :math:`\epsilon` of 1 and confidence of 1 had the highest returns.

.. image:: mean_reversion_images/nyse_cwmrsd.png
   :width: 49 %

.. image:: mean_reversion_images/nyse_cwmrvar.png
   :width: 49 %

TSE has a much wider range, and it is difficult to pinpoint which parameters are actually the most useful.
At least for TSE, the SD method has higher returns with lower confidence value, whereas the VAR method
seems to indicate a congregation at 0.5.

.. image:: mean_reversion_images/tse_cwmrsd.png
   :width: 49 %

.. image:: mean_reversion_images/tse_cwmrvar.png
   :width: 49 %

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.mean_reversion.confidence_weighted_mean_reversion

    .. autoclass:: CWMR
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__

Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection import *

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute Confidence Weighted Mean Reversion - SD with no given weights, epsilon of 0.5, and confidence of 0.5.
    cwmr_sd = CWMR(confidence=0.5, epsilon=0.5, method='sd')
    cwmr_sd.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Confidence Weighted Mean Reversion - Var with given user weights, epsilon of 1, and confidence of 1.
    cwmr_var = CWMR(confidence=1, epsilon=1, method='var')
    cwmr_var.allocate(asset_prices=stock_prices, weights=some_weight)


    # Get the latest predicted weights.
    cwmr_sd.weights

    # Get all weights for the strategy.
    cwmr_var.all_weights

    # Get portfolio returns.
    cwmr_sd.portfolio_return

.. tip::

    Strategies were implemented with modifications from `Li, B., Hoi, S.C., Zhao, P. & Gopalkrishnan, V.. (2011). Confidence Weighted Mean Reversion
    Strategy for On-Line Portfolio Selection. Proceedings of the Fourteenth International
    Conference on Artificial Intelligence and Statistics, in PMLR 15:434-442.
    <https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3292&context=sis_research>`_

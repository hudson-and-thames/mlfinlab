.. _implementations-backtesting:

===============================
Backtesting by Campbell and Yan
===============================

The Backtesting module contains algorithms presented in the paper of Campbell R. Harvey and Yan Liu.
These algorithms are focused on adjusting the reported Sharpe ratios to multiple testing and calculating the required
mean return for a strategy at a given level of significance.

.. tip::
   **Underlying Literature**

   The following sources elaborate extensively on the topic:

   - **Backtesting** *by* Campbell R. Harvey *and* Yan Liu. `available here <https://papers.ssrn.com/abstract_id=2345489>`__. *The paper provides a deeper understanding of the Haircut Sharpe ratio and Profit Hurdle algorithms. The code in this module is based on the code written by the researchers.*
   - **… and the Cross-section of Expected Returns.** *by* Harvey, C.R., Y. Liu, and H. Zhu. `available here <https://faculty.fuqua.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF>`__. *Describes a structural model to capture trading strategies’ underlying distribution, referred to as the HLZ model.*
   - **The Statistics of Sharpe Ratios.** *by* Lo, A. `available here <https://alo.mit.edu/wp-content/uploads/2017/06/The-Statistics-of-Sharpe-Ratios.pdf>`__. *Gives a broader understanding of Sharpe ratio adjustments to autocorrelation and different time periods*

Haircut Sharpe Ratio
####################

Adjusts the Sharpe Ratio due to multiple testing.

This algorithm lets the user calculate the Sharpe ratio adjustments and the corresponding haircuts based on the key
parameters of the data used in the strategy backtests. For each of the adjustment methods - Bonferroni, Holm,
BHY (Benjamini, Hochberg, and Yekutieli) and the Average the algorithm calculates an adjusted p-value,
haircut Sharpe ratio, and the haircut.

The haircut is the percentage difference between the original Sharpe ratio and the new Sharpe ratio.

The inputs of the method include information about the returns that were used to calculate the observed Sharpe ratio.
In particular:

- At what frequency were the returns observed.
- The number of returns observed.
- Observed Sharpe ratio.
- Information on if an observed Sharpe ratio is annualized and if it's adjusted to the autocorrelation of returns (described in the paper by Lo, A.).
- Autocorrelation coefficient of returns.
- The number of tests in multiple testing allowed (described in the first two papers from the introduction).
- Average correlation among strategy returns.

Adjustment methods include:

- Bonferroni
- Holm
- Benjamini, Hochberg, and Yekutieli (BHY)
- Average of the methods above

The method returns np.array of adjusted p-values, adjusted Sharpe ratios, and haircuts as rows. Elements in a row are
ordered by adjustment methods in the following way [Bonferroni, Holm, BHY, Average].

Haircut Sharpe Ratio algorithm consists of the following steps:

1. We are given the observed Sharpe ratio :math:`SR` in :math:`T` periods, based on this information we can calculate the
   p-value of a single test :math:`p^S`.
2. Assuming that :math:`N` other strategies have been tried and that the average correlation of returns from the strategies
   is :math:`\rho` , we use the HLZ model to generate :math:`N` number of t-statistics from the model. We also transform the
   calculated :math:`p^S` to a t-statistic.
3. This :math:`N+1` t-statistics are transformed again to p-values, taking into account the data mining adjustment.
4. This set of :math:`N+1` p-values are fed to two models described above (Holm and BHY) to get the adjusted p-values
   with each of the methods. (Bonferroni adjustment is calculated using only the :math:`p^S` and :math:`N`)
5. The steps 2-4 are repeated multiple times (simulations).
6. For each of the two methods, we eventually have a set of :math:`p^M` values adjusted. The median of this set is the final
   adjusted p-value of the method. So, we obtained p-values for each of the three methods. We then calculate the average
   p-value as the Average of the methods.
7. The obtained p-values of each method can be then transformed back to Sharpe ratios and the haircuts can be calculated.

Implementation
**************
.. py:currentmodule:: mlfinlab.backtest_statistics.backtests

.. autoclass:: CampbellBacktesting
   :members: __init__, haircut_sharpe_ratios

Example
*******

An example showing how Haircut Sharpe Ratios method is used can be seen below:

.. code-block::

   from mlfinlab.backtests import CampbellBacktesting

   # Specify the desired number of simulations
   backtesting = CampbellBacktesting(4000)

   # In this example, annualized Sharpe ratio of 1, not adjusted to autocorrelation of returns
   # at 0.1, calculated on monthly observations of returns for two years (24 total observations),
   # with 10 multiple testing and average correlation among returns of 0.4.
   haircuts = backtesting.haircut_sharpe_ratios(sampling_frequency='M', num_obs=24,
                                                sharpe_ratio=1, annualized=True,
                                                autocorr_adjusted=False, rho_a=0.1,
                                                num_mult_test=10, rho=0.4)

   # Adjsuted Sharpe ratios by method used
   sr_adj_bonferroni = haircuts[1][0]
   sr_adj_holm = haircuts[1][1]
   sr_adj_bhy = haircuts[1][2]
   sr_adj_average = haircuts[1][3]


Profit Hurdle
#############

This algorithm calculates the Required Mean Return of a strategy at a given level of significance adjusted due to multiple testing.

The method described below works only with characteristics of monthly returns that have no autocorrelation.

The inputs of the method include information about returns data. In particular:

- The number of tests in multiple testing allowed (described in the first two papers from the introduction).
- Number of monthly returns observed.
- Significance level.
- Annual return volatility.
- Average correlation among strategy returns.

Adjustment methods include:

- Bonferroni
- Holm
- Benjamini, Hochberg, and Yekutieli (BHY)
- Average of the methods above

Profit Hurdle algorithm consists of the following steps:

1. We are given the significance level :math:`p`, strategy volatility :math:`\sigma`, the number of observations :math:`T` ,
   and the number of tests that have been concluded :math:`T` .
2. Using the HLZ model, we generate :math:`N` t-statistics assuming that the average correlation of returns is :math:`\rho` .
3. Using two methods (Holm and BHY) we calculate the threshold t-statistic that matches the :math:`p` significance level.
4. The steps 2-3 are repeated multiple times (simulations).
5. For the two methods (Holm and BHY) we have a set of t-statistics. We then take the median of t-statistics in each set
   and call it a t-statistic for the method. T-ststistic for Bonferroni is calculated based on :math:`p` and :math:`N`, as
   in the previous algorithm (Haircut Sharpe Ratios).
6. The obtained t-statistics of each method can be then transformed to mean monthly returns. We then calculate the average
   mean monthly return as the Average of the methods returns.

Implementation
**************

The method returns np.array of minimum average monthly returns by the method as elements. The order of the elements by
method is [Bonferroni, Holm, BHY, Average].

.. autoclass:: CampbellBacktesting
   :members: __init__, profit_hurdle

Example
*******

An example showing how Profit Hurdle method is used can be seen below:

.. code-block::

   from mlfinlab.backtests import CampbellBacktesting

   # Specify the desired number of simulations
   backtesting = CampbellBacktesting(4000)

   # In this example, monthly observations of returns for two years (24 total observations),
   # with 10 multiple testing, significance level of 5% and 10% annual volatility and average
   # correlation among returns of 0.4.
   monthly_ret = backtesting.profit_hurdle(num_mult_test=10, num_obs=24, alpha_sig=0.05,
                                           vol_anu=0.1, rho=0.4)

   # Minimum Average Monthly Returns by method used
   monthly_ret_bonferroni = monthly_ret[0]
   monthly_ret_holm = monthly_ret[1]
   monthly_ret_bhy = monthly_ret[2]
   monthly_ret_average = monthly_ret[3]

Research Notebooks
##################

The following research notebooks can be used to better understand how the algorithms within this module work and how
they can be used on real data.

* `Backtesting Notebook <https://github.com/hudson-and-thames/research/blob/master/Backtesting/Backtesting.ipynb>`_

.. _portfolio_optimisation-nested_clustered_optimisation:

===================================
Nested Clustered Optimization (NCO)
===================================

The NCO class includes functions related to:

- Weight allocation using the Nested Clustered Optimization (NCO) algorithm.
- Weight allocation using the Convex Optimization Solution (CVO).
- Multiple simulations for the NCO and CVO algorithms using Monte Carlo Optimization Selection (MCOS) algorithm.
- Sample data generation to use in the above functions.

.. tip::
   **Underlying Literature**

   The following sources elaborate extensively on the topic:

   - **A Robust Estimator of the Efficient Frontier** *by* Marcos Lopez de Prado `available here <https://papers.ssrn.com/sol3/abstract_id=3469961>`__. *Describes the NCO, CVO, and MCOS algorithms.*

About the Algorithm
###################

The Nested Clustered Optimization algorithm estimates optimal weight allocation to either maximize the Sharpe ratio
or minimize the variance of a portfolio.

The steps of the NCO algorithm are:

1. Get the covariance matrix of the outcomes as an input (and the vector of means if the target is to maximize the Sharpe ratio).
2. Transform the covariance matrix to the correlation matrix and calculate the distance matrix based on it.
3. Cluster the covariance matrix into subsets of highly-correlated variables.
4. Compute the optimal weights allocation (Convex Optimization Solution) for every cluster.
5. Reduce the original covariance matrix to a reduced one - where each cluster is represented by a single variable.
6. Compute the optimal weights allocation (Convex Optimization Solution) for the reduced covariance matrix.
7. Compute the final allocations as a dot-product of the allocations between the clusters and inside the clusters.

Convex Optimization Solution (CVO)
##################################

The Convex Optimization Solution is the result of convex optimization when solving a problem of calculating the optimal weight allocation
using the true covariance matrix and the true vector of means for a portfolio. The goal can be either the maximum Sharpe ratio or
minimum variance of a portfolio.

If the problem of portfolio optimization is:

.. math::

      min_{w}\frac{1}{2}w'Vw

      s.t.: w'a = 1

Where :math:`V` is the covariance matrix of elements in a portfolio, :math:`w` is the vector of weights that minimizes the
variance or maximizes the Sharpe ratio, :math:`a` is an optimal solution that defines the goal of optimization.

Then the Convex Optimization Solution to the problem is:

.. math::

      w^* = \frac{V^{-1}a}{a'V^{-1}a}

Monte Carlo Optimization Selection (MCOS)
#########################################

The Monte Carlo Optimization Selection algorithm calculates the NCO allocations and a simple optimal allocation for multiple
simulated pairs of mean vector and the covariance matrix to determine the most robust method for weight allocations for a given
pair of means vector and a covariance vector.

The steps of the MCOS algorithm are:

1. Get the covariance matrix and the means vector of the outcomes as an input (along with the simulation parameters to use).
2. Drawing the empirical covariance matrix and the empirical means vector based on the true ones.
3. If the kde_bwidth parameter is given, the empirical covariance matrix is de-noised.
4. Based on the min_var_portf parameter, either the minimum variance or the maximum Sharpe ratio is targeted in weights allocation.
5. CVO is applied to the empirical data to obtain the weights allocation.
6. NCO is applied to the empirical data to obtain the weights allocation.
7. Based on the original covariance matrix and the means vector a true optimal allocation is calculated.
8. For each weights estimation in a method, a standard deviation between the true weights and the obtained weights is calculated.
9. The error associated with each method is calculated as the mean of the standard deviation across all estimations for the method.

Sample Data Generating
######################

This method allows creating a random vector of means and a random covariance matrix that has the characteristics of securities. The elements are divided into clusters. The elements in clusters have a given level of correlation. The correlation between the clusters is set at another level. This structure is created in order to test the NCO and MCOS algorithms.

.. tip::

    These algorithms are described in more detail in the work **A Robust Estimator of the Efficient Frontier** *by* Marcos Lopez de Prado `available here <https://papers.ssrn.com/abstract_id=3469961>`_.

    Examples of using these functions are available in the `NCO Notebook <https://github.com/hudson-and-thames/research/blob/master/NCO/NCO.ipynb>`_.

Implementation
##############

.. automodule:: mlfinlab.portfolio_optimization.nco

    .. autoclass:: NCO
        :members:

        .. automethod:: __init__

Example
########
Below is an example of how to use the package functions to calculate risk metrics for a portfolio.

.. code-block::

    import pandas as pd
    from mlfinlab.portfolio_optimization import NCO

    # Import dataframe of returns for assets in a portfolio
    assets_returns = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

    # Calculate empirical covariance of assets
    assets_cov = assets_returns.cov()

    # Calculate empirical means of assets
    assets_mean = assets_returns.mean()

    # Class that contains needed functions
    nco = NCO()

    # Find optimal weights using the NCO algorithm
    w_nco = nco.allocate_nco(assets_cov, assets_mean)

    # Find optimal weights using the CVO algorithm
    w_cvo = nco.allocate_cvo(assets_cov, assets_mean)

    # Compare the NCO solutions to the CVO ones using MCOS
    # Parameters are: 10 simulations, 100 observations in a simulation
    # goal of minimum variance, no LW shrinkage
    w_cvo, w_nco = nco.allocate_mcos(assets_mean, assets_cov, 100, 10, 0.01, True, False)

    # Find the errors in estimations of NCO and CVO in simulations
    err_cvo, err_nco = nco.estim_errors_mcos(w_cvo, w_nco, assets_mean, assets_cov, True)

Research Notebooks
##################

The following research notebooks provide a more detailed exploration of the algorithm.

* `NCO Notebook`_

.. _NCO Notebook: https://github.com/hudson-and-thames/research/blob/master/NCO/NCO.ipynb

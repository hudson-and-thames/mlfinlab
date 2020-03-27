# pylint: disable=missing-module-docstring

import numpy as np
from sklearn.covariance import LedoitWolf


class NCO:
    """
    This class implements the Nested Clustered Optimization (NCO) algorithm. It is reproduced with
    modification from the following paper: `Marcos Lopez de Prado “A Robust Estimator of the Efficient Frontier”,
    (2019). <https://papers.ssrn.com/abstract_id=3469961>`_.
    """


    def __init__(self):
        """
        Initialize
        """

    @staticmethod
    def simulate_covariance(mu_vector, cov_matrix, num_obs, lw_shrinkage=False):
        """
        Derives an empirical vector of means and an empirical covariance matrix.

        Based on the set of true means vector and covariance matrix of X distributions,
        the function generates num_obs observations for every X.
        Based on these observations simulated vector of means and the simulated covariance
        matrix are obtained.

        :param mu_vector: (np.array) true means vector for X distributions
        :param cov_matrix: (np.array) true covariance matrix for X distributions
        :param num_obs: (int) number of observations to draw for every X
        :param lw_shrinkage: (bool) flag to apply Ledoit-Wolf shrinkage to X
        :return: (np.array, np.array) empirical means vector, empirical covariance matrix
        """
        # Generating a matrix of num_obs observations for X distributions
        observations = np.random.multivariate_normal(mu_vector.flatten(), cov_matrix, size=num_obs)
        # Empirical means vector calculation
        mu_simulated = observations.mean(axis=0).reshape(-1, 1)
        if lw_shrinkage:  # If applying Ledoit-Wolf shrinkage
            cov_simulated = LedoitWolf().fit(observations).covariance_
        else:  # Simple empirical covariance matrix
            cov_simulated = np.cov(observations, rowvar=False)

        return mu_simulated, cov_simulated

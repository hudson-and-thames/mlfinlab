"""
Implementations of Optimal Transport dependence measure porposed by Dr. Marti et al. : https://arxiv.org/abs/1610.09659
And implemented in the blog post by Dr. Marti: https://gmarti.gitlab.io/qfin/2020/06/25/copula-optimal-transport-dependence.html
"""
import numpy as np
import scipy.stats as ss
import ot


# pylint: disable=invalid-name

def get_ranked_observations(x: np.array, y: np.array) -> float:
    """
    Returns ranked observations - empirical copula.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :return: (tuple) Tuple with ranked observations.
    """

    Xunif = ss.rankdata(x) / len(x)
    Yunif = ss.rankdata(y) / len(y)

    return Xunif, Yunif

def get_optimal_transport_distance(x: np.array, y: np.array, normalize: bool = False) -> float:
    """
    Returns optimal transport distance between two vectors.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param normalize: (bool) Flag used to normalize the result to [0, 1]. (False by default)
    :return: (float) Optimal transport distance.
    """

    # Setting a number of observations
    nb_obs = 1000

    # Target copula with comonotonicity
    target = np.array([[i / nb_obs, i / nb_obs]
                       for i in range(nb_obs)])

    # Forget copula with independence
    forget = np.array([[u, v]
                       for u, v in zip(np.random.uniform(size=nb_obs),
                                       np.random.uniform(size=nb_obs))])

    return optimal_transport

def compute_copula_ot_dependence(empirical: np.array, target: np.array, forget: np.array, nb_obs: int) -> float:
    """
    Calculates optimalcopula transport dependence measure.

    :param empirical: (np.array) Empirical copula.
    :param target: (np.array) Target copula.
    :param forget: (np.array) Fofget copula.
    :param nb_obs: (int) Number of observations.
    :return: (float) Optimal transport dependence.
    """

    # uniform distribution on samples
    t_measure, f_measure, e_measure = (
        np.ones((nb_obs,)) / nb_obs,
        np.ones((nb_obs,)) / nb_obs,
        np.ones((nb_obs,)) / nb_obs)

    # compute the ground distance matrix between locations
    gdist_e2t = ot.dist(empirical, target)
    gdist_e2f = ot.dist(empirical, forget)

    # compute the optimal transport matrix
    e2t_ot = ot.emd(t_measure, e_measure, gdist_e2t)
    e2f_ot = ot.emd(f_measure, e_measure, gdist_e2f)

    # compute the optimal transport distance:
    # <optimal transport matrix, ground distance matrix>_F
    e2t_dist = np.trace(np.dot(np.transpose(e2t_ot), gdist_e2t))
    e2f_dist = np.trace(np.dot(np.transpose(e2f_ot), gdist_e2f))

    # compute the copula ot dependence measure
    return 1 - e2t_dist / (e2f_dist + e2t_dist)

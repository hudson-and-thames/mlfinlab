"""
Implementations of Optimal Transport dependence measure porposed by Marti et al. : https://arxiv.org/abs/1610.09659
And implemented in the blog post by Marti: https://gmarti.gitlab.io/qfin/2020/06/25/copula-optimal-transport-dependence.html
"""
import numpy as np
import scipy.stats as ss
import ot


# pylint: disable=invalid-name

def get_empirical_copula(x: np.array, y: np.array) -> np.array:
    """
    Calculate empirical copula using ranked observations.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :return: (np.array) Empirical copula.
    """

    # Calculate ranked observations
    x_unif = ss.rankdata(x) / len(x)
    y_unif = ss.rankdata(y) / len(y)

    # Empirical copula
    empirical = np.array([[x, y] for x, y in zip(x_unif, y_unif)])

    return empirical

def get_optimal_transport_distance(x: np.array, y: np.array, target_dependence: str) -> float:
    """
    Calculates optimal transport distance between two vectors.

    This implementation is based on the blog post by Marti:
    https://gmarti.gitlab.io/qfin/2020/06/25/copula-optimal-transport-dependence.html

    The target and forget copulas are being used to reference where between them does the empirical
    copula stad in the space of copulas. The forget copula used is the copula associated to
    independent random variables. The target copula is defined by the target_dependence parameter.

    Currently, these target_dependence copulas are supported:

    - ``comonotonicity`` - a comonotone copula.
    - ``countermonotonicity`` - a countermonotone copula.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param target_dependence: (str) Type of target dependence to use when measuring distance.
                                    (``comonotonicity`` by default)
    :return: (float) Optimal transport distance.
    """

    # Defining a number of observations used
    n_obs = x.shape[0]

    # Creating forget copula with independence
    forget = np.array([[u, v] for u, v in zip(np.random.uniform(size=n_obs),
                                              np.random.uniform(size=n_obs))])

    # Creating target copula with a given dependence type
    target = create_target_copula(target_dependence, n_obs)

    # Creating empirical copula from observaions
    empirical = get_empirical_copula(x, y)

    # Optimal transport distance
    copula_ot = compute_copula_ot_dependence(empirical, target, forget, n_obs)

    return copula_ot

def compute_copula_ot_dependence(empirical: np.array, target: np.array, forget: np.array, n_obs: int) -> float:
    """
    Calculates optimal copula transport dependence measure.

    :param empirical: (np.array) Empirical copula.
    :param target: (np.array) Target copula.
    :param forget: (np.array) Fofget copula.
    :param nb_obs: (int) Number of observations.
    :return: (float) Optimal transport dependence.
    """

    # Uniform distribution on samples
    t_measure, f_measure, e_measure = (np.ones((n_obs,)) / n_obs,
                                       np.ones((n_obs,)) / n_obs,
                                       np.ones((n_obs,)) / n_obs)

    # Compute the ground distance matrix between locations
    gdist_e2t = ot.dist(empirical, target)
    gdist_e2f = ot.dist(empirical, forget)

    # Compute the optimal transport matrix
    e2t_ot = ot.emd(t_measure, e_measure, gdist_e2t)
    e2f_ot = ot.emd(f_measure, e_measure, gdist_e2f)

    # Compute the optimal transport distance:
    # <optimal transport matrix, ground distance matrix>_F
    e2t_dist = np.trace(np.dot(np.transpose(e2t_ot), gdist_e2t))
    e2f_dist = np.trace(np.dot(np.transpose(e2f_ot), gdist_e2f))

    # Compute the copula ot dependence measure
    ot_measure = 1 - e2t_dist / (e2f_dist + e2t_dist)

    return ot_measure

def create_target_copula(target_dependence: str, n_obs: int) -> np.array:
    """
    Creates target copula with given dependence an number of observations.

    :param target_dependence: (str) Type of dependence to use for copula creation.[``comonotonicity``,
                                    ``countermonotonicity``]
    :param n_obs: (int) Number of observations to use for copula creation.
    :return: (np.array) Resulting copula.
    """

    if target_dependence == 'comonotonicity':
        target = np.array([[i / n_obs, i / n_obs] for i in range(n_obs)])

    elif target_dependence == 'countermonotonicity':
        target = np.array([[i / n_obs, (n_obs - i) / n_obs] for i in range(n_obs)])

    else:
        raise Exception('This type of target dependence is not supported')

    return target

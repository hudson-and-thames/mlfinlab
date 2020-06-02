"""
Implementations of mutual information (I) and variation of information (VI) codependence measures from Cornell
lecture slides: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes
"""
import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score


# pylint: disable=invalid-name

def get_optimal_number_of_bins(num_obs: int, corr_coef: float = None) -> int:
    """
    Calculates optimal number of bins for discretization based on number of observations
    and correlation coefficient (univariate case).

    Algorithms used in this function were originally proposed in the works of Hacine-Gharbi et al. (2012)
    and Hacine-Gharbi and Ravier (2018). They are described in the Cornell lecture notes:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes (p.26)

    :param num_obs: (int) Number of observations.
    :param corr_coef: (float) Correlation coefficient, used to estimate the number of bins for univariate case.
    :return: (int) Optimal number of bins.
    """

    # Univariate case
    if corr_coef is None or abs(corr_coef - 1) <= 1e-4:
        z = (8 + 324 * num_obs + 12 * (36 * num_obs + 729 * num_obs ** 2) ** .5) ** (1 / 3.)
        bins = round(z / 6. + 2. / (3 * z) + 1. / 3)

    # Bivariate case
    else:
        bins = round(2 ** -.5 * (1 + (1 + 24 * num_obs / (1. - corr_coef ** 2)) ** .5) ** .5)
    return int(bins)


def get_mutual_info(x: np.array, y: np.array, n_bins: int = None, normalize: bool = False) -> float:
    """
    Returns mutual information (I) between two vectors.

    This function uses the discretization with the optimal bins algorithm proposed in the works of
    Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).

    Read Cornell lecture notes for more information about the mutual information:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.
                         (None by default)
    :param normalize: (bool) Flag used to normalize the result to [0, 1]. (False by default)
    :return: (float) Mutual information score.
    """

    if n_bins is None:
        corr_coef = np.corrcoef(x, y)[0][1]
        n_bins = get_optimal_number_of_bins(x.shape[0], corr_coef=corr_coef)

    contingency = np.histogram2d(x, y, n_bins)[0]
    mutual_info = mutual_info_score(None, None, contingency=contingency)  # Mutual information
    if normalize is True:
        marginal_x = ss.entropy(np.histogram(x, n_bins)[0])  # Marginal for x
        marginal_y = ss.entropy(np.histogram(y, n_bins)[0])  # Marginal for y
        mutual_info /= min(marginal_x, marginal_y)
    return mutual_info


def variation_of_information_score(x: np.array, y: np.array, n_bins: int = None, normalize: bool = False) -> float:
    """
    Returns variantion of information (VI) between two vectors.

    This function uses the discretization using optimal bins algorithm proposed in the works of
    Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).

    Read Cornell lecture notes for more information about the variation of information:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.
                         (None by default)
    :param normalize: (bool) True to normalize the result to [0, 1]. (False by default)
    :return: (float) Variation of information score.
    """

    if n_bins is None:
        corr_coef = np.corrcoef(x, y)[0][1]
        n_bins = get_optimal_number_of_bins(x.shape[0], corr_coef=corr_coef)

    contingency = np.histogram2d(x, y, n_bins)[0]
    mutual_info = mutual_info_score(None, None, contingency=contingency)  # Mutual information
    marginal_x = ss.entropy(np.histogram(x, n_bins)[0])  # Marginal for x
    marginal_y = ss.entropy(np.histogram(y, n_bins)[0])  # Marginal for y
    score = marginal_x + marginal_y - 2 * mutual_info  # Variation of information

    if normalize is True:
        joint_dist = marginal_x + marginal_y - mutual_info  # Joint distribution
        score /= joint_dist

    return score

#pylint: disable=missing-docstring
import math
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_slsqp
import matplotlib.pyplot as plt


class EntropyPooling:
    """
    This class implements the Entropy Pooling algorithm proposed in the following paper: Meucci, Attilio, Fully Flexible
    Views: Theory and Practice (August 8, 2008). Fully Flexible Views: Theory and Practice. By using historical factor
    observations as a prior, EP combines it and additional investor views on the portfolio, to find a posterior
    distribution which is close to the prior and also satisfies the specified views. It also removes any assumptions on the
    distribution of the prior and produces the posterior probabilities in a non-parametric way.
    """

    def __init__(self):
        """
        Initialise.

        Class Variables:

        - ``posterior_probabilities`` - (pd.DataFrame) Final posterior probabilities calculated using Entropy Pooling algorithm.
        """

        pass

    def calculate_posterior_probabilities(self, prior_probabilities, equality_matrix=None, equality_vector=None, inequality_matrix=None,
                                          inequality_vector=None, view_confidence=1.0):
        """
        Calculate posterior probabilities from an initial set of probabilities using the Entropy Pooling algorithm.

        :param prior_probabilities: (Numpy array/Python list) List of initial probabilities of market simulations.
        :param equality_matrix: (pd.DataFrame/Numpy matrix) A (J x N1) matrix of equality constraints where N1 = number of equality views
                                                            and J = number of historical simulations. Denoted as 'H' in the "Meucci - Flexible
                                                            Views Theory & Practice" paper in formula 86 on page 22.
        :param equality_vector: (Numpy array/Python list) A vector of length J corresponding to the equality matrix. Denoted as 'h' in the "Meucci -
                                                          Flexible Views Theory & Practice" paper in formula 86 on page 22.
        :param inequality_matrix: (pd.DataFrame/Numpy matrix) A (J x N2) matrix of inequality constraints where N2 = number of inequality
                                                              views and J = number of historical simulations. Denoted as 'F' in the "Meucci -
                                                              Flexible Views Theory & Practice" paper in formula 86 on page 22.
        :param inequality_vector: (Numpy array/Python list) A vector of length J corresponding to the inequality matrix. Denoted as 'f' in the "Meucci -
                                                            Flexible Views Theory & Practice" paper in formula 86 on page 22.
        :param view_confidence: (float) An overall confidence in the specified views.
        """

        pass

    def generate_histogram(self, historical_market_vector, num_bins):
        """
        Given the final probabilities, generate the probability density histogram from the historical market data points.

        :param historical_market_vector: (pd.Series/Numpy array) Vector of historical market data.
        :param num_bins: (int) The number of bins to break the histogram into.
        :return: (plt.BarContainer) The plotted histogram figure object.
        """

        pass

    @staticmethod
    def _solve_unconstrained_optimisation(initial_guess, prior_probabilities, equality_matrix, equality_vector):
        """
        Solve the unconstrained optimisation using Lagrange multipliers. This will give us the final posterior probabilities.

        :param initial_guess: (Numpy array) An initial starting vector for the optimisation algorithm.
        :param prior_probabilities: (Numpy array) List of initial probabilities of market simulations.
        :param equality_matrix: (Numpy matrix) An (N1 x J) matrix of equality constraints where N1 = number of equality views
                                               and J = number of historical simulations.
        :param equality_vector: (Numpy array) A vector of length J corresponding to the equality matrix.
        :return: (Numpy array) Posterior probabilities.
        """

        pass

        def _cost_func(equality_lagrange_multplier):
            # pylint: disable=invalid-name
            """
            Cost function of the unconstrained optimisation problem.

            :param equality_lagrange_multplier: (Numpy matrix) The Lagrange multiplier corresponding to the equality constraints.
            :return: (float) Negative of the value of the Langrangian.
            """

            pass

        def _cost_func_jacobian(equality_lagrange_multplier):
            # pylint: disable=invalid-name
            """
            Jacobian of the cost function.

            :param equality_lagrange_multplier: (Numpy matrix) The Lagrange multiplier corresponding to the equality constraints.
            :return: (float) Negative of the value of the Langrangian gradient.
            """

            pass

        pass

    @staticmethod
    def _solve_constrained_optimisation(initial_guess, prior_probabilities, equality_matrix, equality_vector, inequality_matrix, inequality_vector,
                                        num_equality_constraints, num_inequality_constraints):
        """
        Solve the constrained optimisation using Lagrange multipliers. This will give us the final posterior probabilities.

        :param initial_guess: (Numpy array) An initial starting vector for the optimisation algorithm.
        :param prior_probabilities: (Numpy array) List of initial probabilities of market simulations.
        :param equality_matrix: (Numpy matrix) An (N1 x J) matrix of equality constraints where N1 = number of equality views
                                               and J = number of historical simulations.
        :param equality_vector: (Numpy array) A vector of length J corresponding to the equality matrix.
        :param inequality_matrix: (Numpy matrix) An (N2 x J) matrix of inequality constraints where N2 = number of inequality
                                                              views and J = number of historical simulations.
        :param inequality_vector: (Numpy array) A vector of length J corresponding to the inequality matrix.
        :param num_equality_constraints: (int) Number of equality views/constraints.
        :param num_inequality_constraints: (int) Number of inequality views/constraints.
        :return: (Numpy array) Posterior probabilities.
        """

        pass

        def _inequality_constraints_func(all_constraints_vector):
            """
            Calculate inequality cost function.

            :param all_constraints_vector: (Numpy matrix) Combined vector of all the constraints - equality and inequality.
            :return: (Numpy matrix) Vector of inequality constraints.
            """

            pass

        def _inequality_constraints_func_jacobian(all_constraints_vector):
            #pylint: disable=unused-argument
            """
            Jacobian of the inequality constraints cost function.

            :param all_constraints_vector: (Numpy matrix) Combined vector of all the constraints - equality and inequality.
            :return: (Numpy matrix) Identity matrix.
            """

            pass

        def _cost_func(lagrange_multipliers):
            # pylint: disable=invalid-name
            """
            Cost function of the constrained optimisation problem.

            :param lagrange_multipliers: (Numpy matrix) Values of the Lagrange multipliers for inequality and equality constraints.
            :return: (float) Negative of the value of the Langrangian.
            """

            pass

        def _cost_func_jacobian(lagrange_multipliers):
            #pylint: disable=invalid-unary-operand-type, invalid-name
            """
            Jacobian of the cost function.

            :param lagrange_multipliers: (Numpy matrix) Values of the Lagrange multipliers for inequality and equality constraints.
            :return: (Numpy matrix) Negative of the value of the Langrangian gradients.
            """

            pass

        pass

    @staticmethod
    def _error_checks(prior_probabilities, equality_matrix, equality_vector, inequality_matrix, inequality_vector):
        """
        Initial error checks on inputs.

        :param prior_probabilities: (Numpy array/Python list) List of initial probabilities of market simulations.
        :param equality_matrix: (pd.DataFrame/Numpy matrix) An (N1 x J) matrix of equality constraints where N1 = number of equality views
                                                            and J = number of historical simulations. Denoted as 'H' in the "Meucci - Flexible
                                                            Views Theory & Practice" paper in formula 86 on page 22.
        :param equality_vector: (Numpy array/Python list) A vector of length J corresponding to the equality matrix. Denoted as 'h' in the "Meucci -
                                                          Flexible Views Theory & Practice" paper in formula 86 on page 22.
        :param inequality_matrix: (pd.DataFrame/Numpy matrix) An (N2 x J) matrix of inequality constraints where N2 = number of inequality
                                                              views and J = number of historical simulations. Denoted as 'F' in the "Meucci -
                                                              Flexible Views Theory & Practice" paper in formula 86 on page 22.
        :param inequality_vector: (Numpy array/Python list) A vector of length J corresponding to the inequality matrix. Denoted as 'f' in the "Meucci -
                                                            Flexible Views Theory & Practice" paper in formula 86 on page 22.
        """

        pass

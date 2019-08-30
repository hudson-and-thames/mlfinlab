import numbers
from math import log, ceil

import numpy as np
import pandas as pd


def _infnone(number):
    '''
    Converts a Nonetype object to inf

    :param number: (int/float/None) a number
    :return: (float) -inf or number
    '''
    return float("-inf") if number is None else number


class CLA:
    '''
    CLA class implements the famous Critical Line Algorithm for mean-variance portfolio
    optimisation. It is reproduced with modification from the following paper:
    D.H. Bailey and M.L. Prado “An Open-Source Implementation of the Critical- Line Algorithm for
    Portfolio Optimization”,Algorithms, 6 (2013), 169-196. http://dx.doi.org/10.3390/a6010169
    '''

    def __init__(self, weight_bounds=(0, 1), calculate_returns="mean"):
        '''
        Initialise the storage arrays and some preprocessing.

        :param weight_bounds: (tuple) a tuple specifying the lower and upper bound ranges for the portfolio weights
        :param calculate_returns: (str) the method to use for calculation of expected returns.
                                        Currently supports "mean" and "exponential"
        '''

        self.weight_bounds = weight_bounds
        self.calculate_returns = calculate_returns

    def _calculate_mean_historical_returns(self, asset_prices, frequency=252):
        '''
        Calculate the annualised mean historical returns from asset price data

        :param asset_prices: (pd.DataFrame) asset price data
        :return: (np.array) returns per asset
        '''

        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.mean() * frequency
        return returns

    def _calculate_exponential_historical_returns(self, asset_prices, frequency=252, span=500):
        '''
        Calculate the exponentially-weighted mean of (daily) historical returns, giving
        higher weight to more recent data.

        :param asset_prices: (pd.DataFrame) asset price data
        :return: (np.array) returns per asset
        '''

        returns = asset_prices.pct_change().dropna(how="all")
        returns = returns.ewm(span=span).mean().iloc[-1] * frequency
        return returns

    def _init_algo(self):
        '''
        Initial setting up of the algorithm. Calculates the first free weight of the first turning point.

        :return: (list, list) asset index and the corresponding free weight value
        '''

        # Form structured array
        structured_array = np.zeros((self.expected_returns.shape[0]), dtype=[("id", int), ("mu", float)])
        expected_returns = [self.expected_returns[i][0] for i in range(self.expected_returns.shape[0])]  # dump array into list

        # Fill structured array
        structured_array[:] = list(zip(list(range(self.expected_returns.shape[0])), expected_returns))

        # Sort structured array based on increasing return value
        expected_returns = np.sort(structured_array, order="mu")

        # First free weight
        index, weights = expected_returns.shape[0], np.copy(self.lower_bounds)
        while np.sum(weights) < 1:
            index -= 1

            # Set weights one by one to the upper bounds
            weights[expected_returns[index][0]] = self.upper_bounds[expected_returns[index][0]]
        weights[expected_returns[index][0]] += 1 - np.sum(weights)
        return [expected_returns[index][0]], weights

    def _compute_bi(self, c_final, asset_bounds_i):
        '''
        Calculates which bound value to assign to a bounded asset - lower bound or upper bound.

        :param c_final: (float) a value calculated using the covariance matrices of free weights.
                          Refer to https://pdfs.semanticscholar.org/4fb1/2c1129ba5389bafe47b03e595d098d0252b9.pdf for
                          more information.
        :param asset_bounds_i: (list) a list containing the lower and upper bound values for the ith weight
        :return: bounded weight value
        '''

        bounded_asset_i = 0
        if c_final > 0:
            bounded_asset_i = asset_bounds_i[1][0]
        if c_final < 0:
            bounded_asset_i = asset_bounds_i[0][0]
        return bounded_asset_i

    def _compute_w(self, covar_f_inv, covar_fb, mean_f, w_b):
        '''
        Compute the turning point associated with the current set of free weights F

        :param covar_f_inv: (np.array) inverse of covariance matrix of free assets
        :param covar_fb: (np.array) covariance matrix between free assets and bounded assets
        :param mean_f: (np.array) expected returns of free assets
        :param w_b: (np.array) bounded asset weight values

        :return: (array, array) list of turning point weights and gamma value from the langrange equation
        '''

        # Compute gamma
        ones_f = np.ones(mean_f.shape)
        g_1 = np.dot(np.dot(ones_f.T, covar_f_inv), mean_f)
        g_2 = np.dot(np.dot(ones_f.T, covar_f_inv), ones_f)
        if w_b is None:
            g_final, w_1 = float(-self.lambdas[-1] * g_1 / g_2 + 1 / g_2), 0
        else:
            ones_b = np.ones(w_b.shape)
            g_3 = np.dot(ones_b.T, w_b)
            g_4 = np.dot(covar_f_inv, covar_fb)
            w_1 = np.dot(g_4, w_b)
            g_4 = np.dot(ones_f.T, w_1)
            g_final = float(-self.lambdas[-1] * g_1 / g_2 + (1 - g_3 + g_4) / g_2)

        # Compute weights
        w_2 = np.dot(covar_f_inv, ones_f)
        w_3 = np.dot(covar_f_inv, mean_f)
        return -w_1 + g_final * w_2 + self.lambdas[-1] * w_3, g_final

    def _compute_lambda(self, covar_f_inv, covar_fb, mean_f, w_b, asset_index, b_i):
        '''
        Calculate the lambda value in the langrange optimsation equation

        :param covar_f_inv: (np.array) inverse of covariance matrix of free assets
        :param covar_fb: (np.array) covariance matrix between free assets and bounded assets
        :param mean_f: (np.array) expected returns of free assets
        :param w_b: (np.array) bounded asset weight values
        :param asset_index: (int) index of the asset in the portfolio
        :param b_i: (list) list of upper and lower bounded weight values
        :return: (float) lambda value
        '''

        # Compute C
        ones_f = np.ones(mean_f.shape)
        c_1 = np.dot(np.dot(ones_f.T, covar_f_inv), ones_f)
        c_2 = np.dot(covar_f_inv, mean_f)
        c_3 = np.dot(np.dot(ones_f.T, covar_f_inv), mean_f)
        c_4 = np.dot(covar_f_inv, ones_f)
        c_final = -c_1 * c_2[asset_index] + c_3 * c_4[asset_index]
        if c_final == 0:
            return None, None

        # Compute bi
        if isinstance(b_i, list):
            b_i = self._compute_bi(c_final, b_i)

        # Compute Lambda
        if w_b is None:

            # All free assets
            return float((c_4[asset_index] - c_1 * b_i) / c_final), b_i

        ones_b = np.ones(w_b.shape)
        l_1 = np.dot(ones_b.T, w_b)
        l_2 = np.dot(covar_f_inv, covar_fb)
        l_3 = np.dot(l_2, w_b)
        l_2 = np.dot(ones_f.T, l_3)
        return float(((1 - l_1 + l_2) * c_4[asset_index] - c_1 * (b_i + l_3[asset_index])) / c_final), b_i

    def _get_matrices(self, free_weights):
        '''
        Calculate the required matrices between free and bounded assets

        :param free_weights: (list) list of free assets/weights
        :return: (tuple of np.array matrices) the corresponding matrices
        '''

        covar_f = self._reduce_matrix(self.cov_matrix, free_weights, free_weights)
        mean_f = self._reduce_matrix(self.expected_returns, free_weights, [0])
        bounded_weights = self._get_bounded_weights(free_weights)
        covar_fb = self._reduce_matrix(self.cov_matrix, free_weights, bounded_weights)
        w_b = self._reduce_matrix(self.weights[-1], bounded_weights, [0])
        return covar_f, covar_fb, mean_f, w_b

    def _get_bounded_weights(self, free_weights):
        '''
        Compute the list of bounded assets

        :param free_weights: (np.array) list of free weights/assets
        :return: (np.array) list of bounded assets/weights
        '''

        return self._diff_lists(list(range(self.expected_returns.shape[0])), free_weights)

    def _diff_lists(self, list_1, list_2):
        '''
        Calculate the set difference between two lists
        '''

        return list(set(list_1) - set(list_2))

    def _reduce_matrix(self, matrix, list_x, list_y):
        '''
        Reduce a matrix to the provided set of rows and columns
        '''

        return matrix[np.ix_(list_x, list_y)]

    def _purge_num_err(self, tol):
        '''
        Purge violations of inequality constraints (associated with ill-conditioned cov matrix)
        '''

        index_1 = 0
        while True:
            flag = False
            if index_1 == len(self.weights):
                break
            if abs(sum(self.weights[index_1]) - 1) > tol:
                flag = True
            else:
                for index_2 in range(self.weights[index_1].shape[0]):
                    if (
                            self.weights[index_1][index_2] - self.lower_bounds[index_2] < -tol
                            or self.weights[index_1][index_2] - self.upper_bounds[index_2] > tol
                    ):
                        flag = True
                        break
            if flag is True:
                del self.weights[index_1]
                del self.lambdas[index_1]
                del self.gammas[index_1]
                del self.free_weights[index_1]
            else:
                index_1 += 1

    def _purge_excess(self):
        '''
        Remove violations of the convex hull
        '''

        index_1, repeat = 0, False
        while True:
            if repeat is False:
                index_1 += 1
            if index_1 == len(self.weights) - 1:
                break
            weights = self.weights[index_1]
            mean = np.dot(weights.T, self.expected_returns)[0, 0]
            index_2, repeat = index_1 + 1, False
            while True:
                if index_2 == len(self.weights):
                    break
                weights = self.weights[index_2]
                mean_ = np.dot(weights.T, self.expected_returns)[0, 0]
                if mean < mean_:
                    del self.weights[index_1]
                    del self.lambdas[index_1]
                    del self.gammas[index_1]
                    del self.free_weights[index_1]
                    repeat = True
                    break
                else:
                    index_2 += 1

    def _golden_section(self, obj, left, right, **kargs):
        '''
        Golden section method. Maximum if kargs['minimum']==False is passed

        :param obj: (function) The objective function on which the extreme will be found.
        :param left: (float) The leftmost extreme of search
        :param right: (float) The rightmost extreme of search
        '''

        tol, sign, args = 1.0e-9, 1, None
        if "minimum" in kargs and kargs["minimum"] is False:
            sign = -1
        if "args" in kargs:
            args = kargs["args"]
        num_iterations = int(ceil(-2.078087 * log(tol / abs(right - left))))
        gs_ratio = 0.618033989
        complementary_gs_ratio = 1.0 - gs_ratio

        # Initialize
        x_1 = gs_ratio * left + complementary_gs_ratio * right
        x_2 = complementary_gs_ratio * left + gs_ratio * right
        f_1 = sign * obj(x_1, *args)
        f_2 = sign * obj(x_2, *args)

        # Loop
        for _ in range(num_iterations):
            if f_1 > f_2:
                left = x_1
                x_1 = x_2
                f_1 = f_2
                x_2 = complementary_gs_ratio * left + gs_ratio * right
                f_2 = sign * obj(x_2, *args)
            else:
                right = x_2
                x_2 = x_1
                f_2 = f_1
                x_1 = gs_ratio * left + complementary_gs_ratio * right
                f_1 = sign * obj(x_1, *args)

        if f_1 < f_2:
            return x_1, sign * f_1
        return x_2, sign * f_2

    def _eval_sr(self, alpha, w_0, w_1):
        '''
        Evaluate the sharpe ratio of the portfolio within the convex combination
        '''

        weights = alpha * w_0 + (1 - alpha) * w_1
        returns = np.dot(weights.T, self.expected_returns)[0, 0]
        volatility = np.dot(np.dot(weights.T, self.cov_matrix), weights)[0, 0] ** 0.5
        return returns / volatility

    def _bound_free_weight(self, free_weights):
        '''
        Add a free weight to list of bounded weights
        '''

        lambda_in = None
        i_in = None
        bi_in = None
        if len(free_weights) > 1:
            covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights)
            covar_f_inv = np.linalg.inv(covar_f)
            j = 0
            for i in free_weights:
                l, b_i = self._compute_lambda(
                    covar_f_inv, covar_fb, mean_f, w_b, j, [self.lower_bounds[i], self.upper_bounds[i]]
                )
                if _infnone(l) > _infnone(lambda_in):
                    lambda_in, i_in, bi_in = l, i, b_i
                j += 1
        return lambda_in, i_in, bi_in

    def _free_bound_weight(self, free_weights):
        '''
        Add a bounded weight to list of free weights
        '''

        lambda_out = None
        i_out = None
        if len(free_weights) < self.expected_returns.shape[0]:
            b = self._get_bounded_weights(free_weights)
            for i in b:
                covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights + [i])
                covar_f_inv = np.linalg.inv(covar_f)
                lambda_i, _ = self._compute_lambda(
                    covar_f_inv,
                    covar_fb,
                    mean_f,
                    w_b,
                    mean_f.shape[0] - 1,
                    self.weights[-1][i],
                )
                if (self.lambdas[-1] is None or lambda_i < self.lambdas[-1]) and lambda_i > _infnone(lambda_out):
                    lambda_out, i_out = lambda_i, i
        return lambda_out, i_out

    def _initialise(self, asset_prices):
        '''
        Initialise covariances, upper-counds, lower-bounds and storage buffers

        :param asset_prices: (pd.Dataframe) dataframe of asset prices
        '''

        # Initial checks
        if not isinstance(asset_prices, pd.DataFrame):
            raise ValueError("Asset prices matrix must be a dataframe")
        if not isinstance(asset_prices.index, pd.DatetimeIndex):
            raise ValueError("Asset prices dataframe must be indexed by date.")

        # Calculate the expected returns
        if self.calculate_returns == "mean":
            self.expected_returns = self._calculate_mean_historical_returns(asset_prices=asset_prices)
        else:
            self.expected_returns = self._calculate_exponential_historical_returns(asset_prices=asset_prices)
        self.expected_returns = np.array(self.expected_returns).reshape((len(self.expected_returns), 1))
        if (self.expected_returns == np.ones(self.expected_returns.shape) * self.expected_returns.mean()).all():
            self.expected_returns[-1, 0] += 1e-5

        # Calculate the covariance matrix
        self.cov_matrix = np.asarray(asset_prices.cov())

        if isinstance(self.weight_bounds[0], numbers.Real):
            self.lower_bounds = np.ones(self.expected_returns.shape) * self.weight_bounds[0]
        else:
            self.lower_bounds = np.array(self.weight_bounds[0]).reshape(self.expected_returns.shape)

        if isinstance(self.weight_bounds[0], numbers.Real):
            self.upper_bounds = np.ones(self.expected_returns.shape) * self.weight_bounds[1]
        else:
            self.upper_bounds = np.array(self.weight_bounds[1]).reshape(self.expected_returns.shape)

        # Initialise storage buffers
        self.weights = []
        self.lambdas = []
        self.gammas = []
        self.free_weights = []

    def _max_sharpe(self):
        '''
        Compute the maximum sharpe portfolio allocation

        :return: (float, np.array) tuple of max. sharpe value and the set of weight allocations
        '''

        # 1) Compute the local max SR portfolio between any two neighbor turning points
        w_sr, sharpe_ratios = [], []
        for i in range(len(self.weights) - 1):
            w_0 = np.copy(self.weights[i])
            w_1 = np.copy(self.weights[i + 1])
            kwargs = {"minimum": False, "args": (w_0, w_1)}
            alpha, sharpe_ratio = self._golden_section(self._eval_sr, 0, 1, **kwargs)
            w_sr.append(alpha * w_0 + (1 - alpha) * w_1)
            sharpe_ratios.append(sharpe_ratio)

        maximum_sharp_ratio = max(sharpe_ratios)
        weights_with_max_sharpe_ratio = w_sr[sharpe_ratios.index(maximum_sharp_ratio)]
        return maximum_sharp_ratio, weights_with_max_sharpe_ratio

    def _min_volatility(self):
        '''
        Compute minimum volatility portfolio allocation

        :return: (float, np.array) tuple of minimum variance value and the set of weight allocations
        '''

        var = []
        for w in self.weights:
            volatility = np.dot(np.dot(w.T, self.cov_matrix), w)
            var.append(volatility)
        min_var = min(var)
        return min_var ** .5, self.weights[var.index(min_var)]

    def _efficient_frontier(self, points=100):
        '''
        Compute the entire efficient frontier solution

        :param points: (int) number of efficient frontier points to be calculated
        :return: tuple of mean, variance amd weights of the frontier solutions
        '''

        mean, sigma, weights = [], [], []

        # remove the 1, to avoid duplications
        a = np.linspace(0, 1, points // len(self.weights))[:-1]
        b = list(range(len(self.weights) - 1))
        for i in b:
            w_0, w_1 = self.weights[i], self.weights[i + 1]
            if i == b[-1]:
                # include the 1 in the last iteration
                a = np.linspace(0, 1, points // len(self.weights))
            for j in a:
                w = w_1 * j + (1 - j) * w_0
                weights.append(np.copy(w))
                mean.append(np.dot(w.T, self.expected_returns)[0, 0])
                sigma.append(np.dot(np.dot(w.T, self.cov_matrix), w)[0, 0] ** 0.5)
        return mean, sigma, weights

    def allocate(self, asset_prices, solution="cla_turning_points"):
        '''
        Calculate the portfolio asset allocations using the method specified.

        :param asset_prices: (pd.Dataframe/np.array) a dataframe of historical asset prices (adj closed)
        :param solution: (str) specify the type of solution to compute. Options are: cla_turning_points, max_sharpe,
                               min_volatility, efficient_frontier
        '''

        # Some initial steps before the algorithm runs
        self._initialise(asset_prices=asset_prices)

        # Compute the turning points, free sets and weights
        free_weights, weights = self._init_algo()
        self.weights.append(np.copy(weights))  # store solution
        self.lambdas.append(None)
        self.gammas.append(None)
        self.free_weights.append(free_weights[:])
        while True:

            # 1) Bound one free weight
            lambda_in, i_in, bi_in = self._bound_free_weight(free_weights)

            # 2) Free one bounded weight
            lambda_out, i_out = self._free_bound_weight(free_weights)

            if (lambda_in is None or lambda_in < 0) and (lambda_out is None or lambda_out < 0):
                # 3) Compute minimum variance solution
                self.lambdas.append(0)
                covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights)
                covar_f_inv = np.linalg.inv(covar_f)
                mean_f = np.zeros(mean_f.shape)
            else:
                # 4) Decide whether to free a bounded weight or bound a free weight
                if _infnone(lambda_in) > _infnone(lambda_out):
                    self.lambdas.append(lambda_in)
                    free_weights.remove(i_in)
                    weights[i_in] = bi_in  # set value at the correct boundary
                else:
                    self.lambdas.append(lambda_out)
                    free_weights.append(i_out)
                covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights)
                covar_f_inv = np.linalg.inv(covar_f)

            # 5) Compute solution vector
            w_f, g = self._compute_w(covar_f_inv, covar_fb, mean_f, w_b)
            for i in range(len(free_weights)):
                weights[free_weights[i]] = w_f[i]
            self.weights.append(np.copy(weights))  # store solution
            self.gammas.append(g)
            self.free_weights.append(free_weights[:])
            if self.lambdas[-1] == 0:
                break

        # 6) Purge turning points
        self._purge_num_err(10e-10)
        self._purge_excess()

        # Compute the specified corresponding solution
        assets = asset_prices.columns
        if solution == "max_sharpe":
            self.max_sharpe, self.weights = self._max_sharpe()
            self.weights = pd.DataFrame(self.weights)
            self.weights.index = assets
            self.weights = self.weights.T
        elif solution == "min_volatility":
            self.min_var, self.weights = self._min_volatility()
            self.weights = pd.DataFrame(self.weights)
            self.weights.index = assets
            self.weights = self.weights.T
        elif solution == "efficient_frontier":
            self.means, self.sigma, self.weights = self._efficient_frontier()
            weights_copy = self.weights.copy()
            for i, turning_point in enumerate(weights_copy):
                self.weights[i] = turning_point.reshape(1, -1)[0]
            self.weights = pd.DataFrame(self.weights, columns=assets)
        else:
            # Reshape the weight matrix
            weights_copy = self.weights.copy()
            for i, turning_point in enumerate(weights_copy):
                self.weights[i] = turning_point.reshape(1, -1)[0]
            self.weights = pd.DataFrame(self.weights, columns=assets)

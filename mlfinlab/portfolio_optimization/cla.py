import numbers
import numpy as np
import pandas as pd
from math import log, ceil


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

    def _compute_lambda(self, covar_f_inv, covar_fb, mean_f, w_b, i, bi):
        '''
        Calculate the lambda value in the langrange optimsation equation

        :param covar_f_inv: (np.array) inverse of covariance matrix of free assets
        :param covar_fb: (np.array) covariance matrix between free assets and bounded assets
        :param mean_f: (np.array) expected returns of free assets
        :param w_b: (np.array) bounded asset weight values
        :param i: (int) asset index
        :param bi: (list) list of upper and lower bounded weight values
        :return: (float) lambda value
        '''

        # Compute C
        ones_f = np.ones(mean_f.shape)
        c_1 = np.dot(np.dot(ones_f.T, covar_f_inv), ones_f)
        c_2 = np.dot(covar_f_inv, mean_f)
        c_3 = np.dot(np.dot(ones_f.T, covar_f_inv), mean_f)
        c_4 = np.dot(covar_f_inv, ones_f)
        c_final = -c_1 * c_2[i] + c_3 * c_4[i]
        if c_final == 0:
            return None, None

        # Compute bi
        if type(bi) == list:
            bi = self._compute_bi(c_final, bi)

        # Compute Lambda
        if w_b is None:

            # All free assets
            return float((c_4[i] - c_1 * bi) / c_final), bi
        else:
            ones_b = np.ones(w_b.shape)
            l1 = np.dot(ones_b.T, w_b)
            l2 = np.dot(covar_f_inv, covar_fb)
            l3 = np.dot(l2, w_b)
            l2 = np.dot(ones_f.T, l3)
            return float(((1 - l1 + l2) * c_4[i] - c_1 * (bi + l3[i])) / c_final), bi

    def _get_matrices(self, f):
        '''
        Calculate the required matrices between free and bounded assets

        :param f: (list) list of free assets/weights
        :return: (tuple of np.array matrices) the corresponding matrices
        '''

        covar_f = self._reduce_matrix(self.cov_matrix, f, f)
        mean_f = self._reduce_matrix(self.expected_returns, f, [0])
        b = self._get_b(f)
        covar_fb = self._reduce_matrix(self.cov_matrix, f, b)
        w_b = self._reduce_matrix(self.weights[-1], b, [0])
        return covar_f, covar_fb, mean_f, w_b

    def _get_b(self, f):
        '''
        Compute the list of bounded assets

        :param f: (np.array) list of free weights/assets
        :return: (np.array) list of bounded assets/weights
        '''

        return self._diff_lists(list(range(self.expected_returns.shape[0])), f)

    @staticmethod
    def _diff_lists(list1, list2):
        '''
        Calculate the set difference between two lists
        '''

        return list(set(list1) - set(list2))

    @staticmethod
    def _reduce_matrix(matrix, listX, listY):
        '''
        Reduce a matrix to the provided set of rows and columns
        '''

        return matrix[np.ix_(listX, listY)]

    def _purge_num_err(self, tol):
        '''
        Purge violations of inequality constraints (associated with ill-conditioned cov matrix)
        '''

        i = 0
        while True:
            flag = False
            if i == len(self.weights):
                break
            if abs(sum(self.weights[i]) - 1) > tol:
                flag = True
            else:
                for j in range(self.weights[i].shape[0]):
                    if (
                            self.weights[i][j] - self.lower_bounds[j] < -tol
                            or self.weights[i][j] - self.upper_bounds[j] > tol
                    ):
                        flag = True
                        break
            if flag is True:
                del self.weights[i]
                del self.lambdas[i]
                del self.gammas[i]
                del self.free_weights[i]
            else:
                i += 1

    def _purge_excess(self):
        '''
        Remove violations of the convex hull
        '''

        i, repeat = 0, False
        while True:
            if repeat is False:
                i += 1
            if i == len(self.weights) - 1:
                break
            w = self.weights[i]
            mu = np.dot(w.T, self.expected_returns)[0, 0]
            j, repeat = i + 1, False
            while True:
                if j == len(self.weights):
                    break
                w = self.weights[j]
                mu_ = np.dot(w.T, self.expected_returns)[0, 0]
                if mu < mu_:
                    del self.weights[i]
                    del self.lambdas[i]
                    del self.gammas[i]
                    del self.free_weights[i]
                    repeat = True
                    break
                else:
                    j += 1

    def _golden_section(self, obj, a, b, **kargs):
        '''
        Golden section method. Maximum if kargs['minimum']==False is passed
        '''

        tol, sign, args = 1.0e-9, 1, None
        if "minimum" in kargs and kargs["minimum"] is False:
            sign = -1
        if "args" in kargs:
            args = kargs["args"]
        numIter = int(ceil(-2.078087 * log(tol / abs(b - a))))
        r = 0.618033989
        c = 1.0 - r

        # Initialize
        x1 = r * a + c * b
        x2 = c * a + r * b
        f1 = sign * obj(x1, *args)
        f2 = sign * obj(x2, *args)

        # Loop
        for i in range(numIter):
            if f1 > f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = c * a + r * b
                f2 = sign * obj(x2, *args)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = r * a + c * b
                f1 = sign * obj(x1, *args)
        if f1 < f2:
            return x1, sign * f1
        else:
            return x2, sign * f2

    def _eval_sr(self, a, w0, w1):
        '''
        Evaluate SR of the portfolio within the convex combination
        '''

        w = a * w0 + (1 - a) * w1
        b = np.dot(w.T, self.expected_returns)[0, 0]
        c = np.dot(np.dot(w.T, self.cov_matrix), w)[0, 0] ** 0.5
        return b / c

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
                l, bi = self._compute_lambda(
                    covar_f_inv, covar_fb, mean_f, w_b, j, [self.lower_bounds[i], self.upper_bounds[i]]
                )
                if _infnone(l) > _infnone(lambda_in):
                    lambda_in, i_in, bi_in = l, i, bi
                j += 1
        return lambda_in, i_in, bi_in

    def _free_bound_weight(self, free_weights):
        '''
        Add a bounded weight to list of free weights
        '''

        lambda_out = None
        i_out = None
        if len(free_weights) < self.expected_returns.shape[0]:
            b = self._get_b(free_weights)
            for i in b:
                covar_f, covar_fb, mean_f, w_b = self._get_matrices(free_weights + [i])
                covar_f_inv = np.linalg.inv(covar_f)
                l, bi = self._compute_lambda(
                    covar_f_inv,
                    covar_fb,
                    mean_f,
                    w_b,
                    mean_f.shape[0] - 1,
                    self.weights[-1][i],
                )
                if (self.lambdas[-1] is None or l < self.lambdas[-1]) and l > _infnone(lambda_out):
                    lambda_out, i_out = l, i
        return lambda_out, i_out

    def _initialise(self, asset_prices):

        # Initial checks
        if type(asset_prices) != pd.DataFrame:
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
        w_sr, sr = [], []
        for i in range(len(self.weights) - 1):
            w0 = np.copy(self.weights[i])
            w1 = np.copy(self.weights[i + 1])
            kargs = {"minimum": False, "args": (w0, w1)}
            a, b = self._golden_section(self._eval_sr, 0, 1, **kargs)
            w_sr.append(a * w0 + (1 - a) * w1)
            sr.append(b)
        return max(sr), w_sr[sr.index(max(sr))]

    def _min_volatility(self):
        '''
        Compute minimum volatility portfolio allocation

        :return: (float, np.array) tuple of minimum variance value and the set of weight allocations
        '''

        var = []
        for w in self.weights:
            a = np.dot(np.dot(w.T, self.cov_matrix), w)
            var.append(a)
        min_var = min(var)
        return min_var ** .5, self.weights[var.index(min_var)]

    def _efficient_frontier(self, points=100):
        '''
        Compute the entire efficient frontier solution

        :param points: (int) number of efficient frontier points to be calculated
        :return: tuple of mean, variance amd weights of the frontier solutions
        '''

        mu, sigma, weights = [], [], []

        # remove the 1, to avoid duplications
        a = np.linspace(0, 1, points // len(self.weights))[:-1]
        b = list(range(len(self.weights) - 1))
        for i in b:
            w0, w1 = self.weights[i], self.weights[i + 1]
            if i == b[-1]:
                # include the 1 in the last iteration
                a = np.linspace(0, 1, points // len(self.weights))
            for j in a:
                w = w1 * j + (1 - j) * w0
                weights.append(np.copy(w))
                mu.append(np.dot(w.T, self.expected_returns)[0, 0])
                sigma.append(np.dot(np.dot(w.T, self.cov_matrix), w)[0, 0] ** 0.5)
        return mu, sigma, weights

    def allocate(self, asset_prices, solution="cla_turning_points"):
        '''
        Calculate the solution of turning points satisfying the weight bounds

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
            wF, g = self._compute_w(covar_f_inv, covar_fb, mean_f, w_b)
            for i in range(len(free_weights)):
                weights[free_weights[i]] = wF[i]
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

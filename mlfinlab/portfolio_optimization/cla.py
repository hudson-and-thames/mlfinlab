import numbers
import numpy as np
import pandas as pd
from math import log, ceil


def _infnone(x):
    return float("-inf") if x is None else x

class CLA:
    def __init__(self, expected_returns, cov_matrix, weight_bounds = (0, 1)):
        '''

        :param expected_returns:
        :param cov_matrix:
        :param weight_bounds:
        '''
        
        self.expected_returns = np.array(expected_returns).reshape((len(expected_returns), 1))
        if (self.expected_returns == np.ones(self.expected_returns.shape) * self.expected_returns.mean()).all():
            self.expected_returns[-1, 0] += 1e-5
        self.cov_matrix = np.asarray(cov_matrix)
        
        if isinstance(weight_bounds[0], numbers.Real):
            self.lower_bounds = np.ones(self.expected_returns.shape) * weight_bounds[0]
        else:
            self.lower_bounds = np.array(weight_bounds[0]).reshape(self.expected_returns.shape)
        
        if isinstance(weight_bounds[0], numbers.Real):
            self.upper_bounds = np.ones(self.expected_returns.shape) * weight_bounds[1]
        else:
            self.upper_bounds = np.array(weight_bounds[1]).reshape(self.expected_returns.shape)
        
        self.weights = []  # solution
        self.lambdas = []  # lambdas
        self.gammas = []  # gammas
        self.free_weights = []  # free weights

    def _init_algo(self):
        '''

        :return:
        '''

        # Form structured array
        a = np.zeros((self.expected_returns.shape[0]), dtype=[("id", int), ("mu", float)])
        b = [self.expected_returns[i][0] for i in range(self.expected_returns.shape[0])]  # dump array into list

        # Fill structured array
        a[:] = list(zip(list(range(self.expected_returns.shape[0])), b))

        # Sort structured array
        b = np.sort(a, order = "mu")

        # First free weight
        i, w = b.shape[0], np.copy(self.lower_bounds)
        while sum(w) < 1:
            i -= 1
            w[b[i][0]] = self.upper_bounds[b[i][0]]
        w[b[i][0]] += 1 - sum(w)
        return [b[i][0]], w

    def _compute_bi(self, c, bi):
        '''

        :param c:
        :param bi:
        :return:
        '''

        if c > 0:
            bi = bi[1][0]
        if c < 0:
            bi = bi[0][0]
        return bi

    def _compute_w(self, covarF_inv, covarFB, meanF, wB):
        '''

        :param covarF_inv:
        :param covarFB:
        :param meanF:
        :param wB:
        :return:
        '''

        # 1) compute gamma
        onesF = np.ones(meanF.shape)
        g1 = np.dot(np.dot(onesF.T, covarF_inv), meanF)
        g2 = np.dot(np.dot(onesF.T, covarF_inv), onesF)
        if wB is None:
            g, w1 = float(-self.lambdas[-1] * g1 / g2 + 1 / g2), 0
        else:
            onesB = np.ones(wB.shape)
            g3 = np.dot(onesB.T, wB)
            g4 = np.dot(covarF_inv, covarFB)
            w1 = np.dot(g4, wB)
            g4 = np.dot(onesF.T, w1)
            g = float(-self.lambdas[-1] * g1 / g2 + (1 - g3 + g4) / g2)
        # 2) compute weights
        w2 = np.dot(covarF_inv, onesF)
        w3 = np.dot(covarF_inv, meanF)
        return -w1 + g * w2 + self.lambdas[-1] * w3, g

    def _compute_lambda(self, covarF_inv, covarFB, meanF, wB, i, bi):
        '''

        :param covarF_inv:
        :param covarFB:
        :param meanF:
        :param wB:
        :param i:
        :param bi:
        :return:
        '''

        # 1) C
        onesF = np.ones(meanF.shape)
        c1 = np.dot(np.dot(onesF.T, covarF_inv), onesF)
        c2 = np.dot(covarF_inv, meanF)
        c3 = np.dot(np.dot(onesF.T, covarF_inv), meanF)
        c4 = np.dot(covarF_inv, onesF)
        c = -c1 * c2[i] + c3 * c4[i]
        if c == 0:
            return None, None

        # 2) bi
        if type(bi) == list:
            bi = self._compute_bi(c, bi)

        # 3) Lambda
        if wB is None:
            # All free assets
            return float((c4[i] - c1 * bi) / c), bi
        else:
            onesB = np.ones(wB.shape)
            l1 = np.dot(onesB.T, wB)
            l2 = np.dot(covarF_inv, covarFB)
            l3 = np.dot(l2, wB)
            l2 = np.dot(onesF.T, l3)
            return float(((1 - l1 + l2) * c4[i] - c1 * (bi + l3[i])) / c), bi

    def _get_matrices(self, f):
        '''

        :param f:
        :return:
        '''
        covarF = self._reduce_matrix(self.cov_matrix, f, f)
        meanF = self._reduce_matrix(self.expected_returns, f, [0])
        b = self._get_b(f)
        covarFB = self._reduce_matrix(self.cov_matrix, f, b)
        wB = self._reduce_matrix(self.weights[-1], b, [0])
        return covarF, covarFB, meanF, wB

    def _get_b(self, f):
        return self._diff_lists(list(range(self.expected_returns.shape[0])), f)

    @staticmethod
    def _diff_lists(list1, list2):
        '''

        :param list1:
        :param list2:
        :return:
        '''
        return list(set(list1) - set(list2))

    @staticmethod
    def _reduce_matrix(matrix, listX, listY):
        '''
        Reduce a matrix to the provided list of rows and columns

        :param matrix:
        :param listX:
        :param listY:
        :return:
        '''

        return matrix[np.ix_(listX, listY)]

    def _purge_num_err(self, tol):
        # Purge violations of inequality constraints (associated with ill-conditioned cov matrix)
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
        # Remove violations of the convex hull
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
        # Golden section method. Maximum if kargs['minimum']==False is passed
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

        :param a:
        :param w0:
        :param w1:
        :return:
        '''

        w = a * w0 + (1 - a) * w1
        b = np.dot(w.T, self.expected_returns)[0, 0]
        c = np.dot(np.dot(w.T, self.cov_matrix), w)[0, 0] ** 0.5
        return b / c

    def _bound_free_weight(self, free_weights):
        '''

        :param free_weights:
        :return:
        '''

        lambda_in = None
        i_in = None
        bi_in = None
        if len(free_weights) > 1:
            covarF, covarFB, meanF, wB = self._get_matrices(free_weights)
            covarF_inv = np.linalg.inv(covarF)
            j = 0
            for i in free_weights:
                l, bi = self._compute_lambda(
                    covarF_inv, covarFB, meanF, wB, j, [self.lower_bounds[i], self.upper_bounds[i]]
                )
                if _infnone(l) > _infnone(lambda_in):
                    lambda_in, i_in, bi_in = l, i, bi
                j += 1
        return lambda_in, i_in, bi_in

    def _free_bound_weight(self, free_weights):
        '''

        :param free_weights:
        :return:
        '''

        lambda_out = None
        i_out = None
        if len(free_weights) < self.expected_returns.shape[0]:
            b = self._get_b(free_weights)
            for i in b:
                covarF, covarFB, meanF, wB = self._get_matrices(free_weights + [i])
                covarF_inv = np.linalg.inv(covarF)
                l, bi = self._compute_lambda(
                    covarF_inv,
                    covarFB,
                    meanF,
                    wB,
                    meanF.shape[0] - 1,
                    self.weights[-1][i],
                )
                if (self.lambdas[-1] is None or l < self.lambdas[-1]) and l > _infnone(lambda_out):
                    lambda_out, i_out = l, i
        return lambda_out, i_out

    def allocate(self):
        '''

        :return:
        '''

        # Compute the turning points,free sets and weights
        free_weights, w = self._init_algo()
        self.weights.append(np.copy(w))  # store solution
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
                covarF, covarFB, meanF, wB = self._get_matrices(free_weights)
                covarF_inv = np.linalg.inv(covarF)
                meanF = np.zeros(meanF.shape)
            else:
                # 4) Decide whether to free a bounded weight or bound a free weight
                if _infnone(lambda_in) > _infnone(lambda_out):
                    self.lambdas.append(lambda_in)
                    free_weights.remove(i_in)
                    w[i_in] = bi_in  # set value at the correct boundary
                else:
                    self.lambdas.append(lambda_out)
                    free_weights.append(i_out)
                covarF, covarFB, meanF, wB = self._get_matrices(free_weights)
                covarF_inv = np.linalg.inv(covarF)

            # 5) Compute solution vector
            wF, g = self._compute_w(covarF_inv, covarFB, meanF, wB)
            for i in range(len(free_weights)):
                w[free_weights[i]] = wF[i]
            self.weights.append(np.copy(w))  # store solution
            self.gammas.append(g)
            self.free_weights.append(free_weights[:])
            if self.lambdas[-1] == 0:
                break

        # 6) Purge turning points
        self._purge_num_err(10e-10)
        self._purge_excess()

    def max_sharpe(self):
        '''

        :return:
        '''

        if not self.weights:
            self.allocate()

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

    def min_volatility(self):
        '''

        :return:
        '''

        if not self.weights:
            self.allocate()
        var = []
        for w in self.weights:
            a = np.dot(np.dot(w.T, self.cov_matrix), w)
            var.append(a)
        min_var = min(var)
        return min_var**.5, self.weights[var.index(min_var)]

    def efficient_frontier(self, points = 100):
        '''

        :param points:
        :return:
        '''

        if not self.weights:
            self.allocate()

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
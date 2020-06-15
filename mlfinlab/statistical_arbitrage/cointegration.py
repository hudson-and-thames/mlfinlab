"""
Calculate Cointegration.
"""

from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def calc_engle_granger(data1, data2, trend="c", method="aeg", maxlag=None, autolag="aic",
                       return_results=None):
    """
    Wrapper function for augmented Engle-Granger two-step cointegration test, directly forked
    from `statsmodels.tsa.stattools.coint
    <https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html>`_.

    Test for no-cointegration of a univariate equation.

    The null hypothesis is no cointegration. Variables in data1 and data2 are
    assumed to be integrated of order 1, I(1).

    This uses the augmented Engle-Granger two-step cointegration test.
    Constant or trend is included in 1st stage regression, i.e. in
    cointegrating equation.

    The Null hypothesis is that there is no cointegration, the alternative
    hypothesis is that there is cointegrating relationship. If the pvalue is
    small, below a critical size, then we can reject the hypothesis that there
    is no cointegrating relationship.

    P-values and critical values are obtained through regression surface
    approximation from MacKinnon 1994 and 2010.

    If the two series are almost perfectly collinear, then computing the
    test is numerically unstable. However, the two series will be cointegrated
    under the maintained assumption that they are integrated. In this case
    the t-statistic will be set to -inf and the pvalue to zero.

    :param data1: (pd.Series) The first element in cointegrated system. Must be 1-d.

    :param data2: (pd.Series) The remaining elements in cointegrated system.

    :param trend: (str) {``c``, ``ct``, ``ctt``, ``nc``} Constant and trend order to include in
        regression.

        - ``c`` : constant only (default).
        - ``ct`` : constant and trend.
        - ``ctt`` : constant, and linear and quadratic trend.
        - ``nc`` : no constant, no trend.

    :param method: (str) {``aeg``} Only ``aeg`` (augmented Engle-Granger) is available.

    :param maxlag: None or int Argument for `adfuller`, largest or given number of lags.

    :param autolag: (str) Argument for `adfuller`, lag selection criterion.

        - If ``None``, then maxlag lags are used without lag search.
        - If ``AIC`` (default) or ``BIC``, then the number of lags is chosen to minimize the
          corresponding information criterion.
        - If ``t-stat`` based choice of maxlag.  Starts with maxlag and drops a lag until the
          t-statistic on the last lag length is significant using a 5%-sized test.

    :param return_results: (bool) For future compatibility, currently only tuple available.
        If ``True``, then a results instance is returned. Otherwise, a tuple
        with the test outcome is returned. Set `return_results=False` to
        avoid future changes in return.

    :return: (tuple) ``Coint_t``, ``Pvalue``, ``Crit_value``

        - ``coint_t``: (float) The t-statistic of unit-root test on residuals.
        - ``pvalue``: (float) MacKinnon"s approximate, asymptotic p-value based on MacKinnon (1994).
        - ``crit_value``: (dict) Critical values for the test statistic at the 1 %, 5 %, and 10 %
          levels based on regression curve. This depends on the number of observations.

    """
    return coint(data1, data2, trend=trend, method=method, maxlag=maxlag, autolag=autolag,
                 return_results=return_results)


def calc_johansen(data, det_order, k_ar_diff):
    # pylint: disable=anomalous-backslash-in-string
    """
    Wrapper function for Johansen test, directly forked
    from `statsmodels.tsa.vector_ar.vecm.coint_johansen
    <https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html>`_.

    .. warning:
        Critical values are only available for time series with 12 variables at most.

    Johansen cointegration test of the cointegration rank of a VECM.

    :param data: (pd.Series) (nobs_tot x neqs) Data to test

    :param det_order: (int)

        - ``-1``: no deterministic terms
        - ``0``: constant term
        - ``1``: linear trend

    :param k_ar_diff: (int) Nonnegative, Number of lagged differences in the model.

    :return: (tuple) (``cvm``, ``cvt``, ``eig``, ``evec``, ``ind``, ``lr1``, ``lr2``,
        ``max_eig_stat``, ``max_eig_stat_crit_vals``, ``meth``, ``r0t``, ``rkt``, ``trace_stat``,
        ``trace_stat_crit_vals``)

    - ``cvm``: (np.array) Critical values (90%, 95%, 99%) of maximum eigenvalue statistic.
    - ``cvt``: (np.array) Critical values (90%, 95%, 99%) of trace statistic
    - ``eig``: (np.array) Eigenvalues of VECM coefficient matrix.
    - ``evec``: (np.array) Eigenvectors of VECM coefficient matrix.
    - ``ind``: (np.array) Order of eigenvalues.
    - ``lr1``: (np.array) Trace statistic.
    - ``lr2``: (np.array) Maximum eigenvalue statistic.
    - ``max_eig_stat``: (np.array) Maximum eigenvalue statistic.
    - ``max_eig_stat_crit_vals``: (np.array)
    - ``meth``: (str) Test method.
    - ``r0t``: (np.array) Residuals for :math:`\delta Y`
    - ``rkt``: (np.array) Residuals for :math:`Y_{-1}`.
    - ``trace_stat``: (np.array) Trace statistic.
    - ``trace_stat_crit_vals``: (np.array) Critical values (90%, 95%, 99%) of trace statistic.

    """
    res = coint_johansen(data, det_order, k_ar_diff)
    cvm, cvt, eig, evec, ind, lr1, lr2 = res.cvm, res.cvt, res.eig, res.evec, res.ind, res.lr1, res.lr2
    max_eig_stat, max_eig_stat_crit_vals, meth = res.max_eig_stat, res.max_eig_stat_crit_vals, res.meth
    r0t, rkt, trace_stat, trace_stat_crit_vals = res.r0t, res.rkt, res.trace_stat, res.trace_stat_crit_vals

    return (cvm, cvt, eig, evec, ind, lr1, lr2, max_eig_stat, max_eig_stat_crit_vals, meth, r0t,
            rkt, trace_stat, trace_stat_crit_vals)

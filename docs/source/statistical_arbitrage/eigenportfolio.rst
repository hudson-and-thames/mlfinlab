.. _statistical_arbitrage-eigenportfolio:
.. note::

    References

    1. `Gatev, E., Goetzmann, W.N. and Rouwenhorst, K.G., 2006. Pairs trading: Performance of a
    relative-value arbitrage rule. The Review of Financial Studies, 19(3), pp.797-827.
    <https://academic.oup.com/rfs/article/19/3/797/1646694>`_


==============
Eigenportfolio
==============

.. notes::

    This section is implemented with modifications from `Avellaneda, M. and Lee, J.H., 2010. Statistical
    arbitrage in the US equities market. Quantitative Finance, 10(7), pp.761-782. <https://www.tandfonline.com/doi/pdf/10.1080/14697680903124632>`_

Principal Component Analysis (PCA) is often used as a tool for dimensional reduction. The formulation,
based on Linear Algebra, effectively identifies the directions with the largest variations with the
corresponding eigenvectors and eigenvalues.

We can calculate the principal components and the projection of the original dataset by decomposing
the covariance matrix of the original data. Because PCA is sensitive to outliers and noise, we will
normalize the data. The covariance matrix of the normalized data will then be:

.. math::
    \frac{1}{n} X^T * X

Singular Value Decomposition can also be applied to this by stating:

.. math::
    \frac{1}{n} X^T * X = U * \Sigma * V^T

:math:`U` is the left principal component, each :math:`s` in :math:`\Sigma` is the singular value, and
:math:`V^T` is the right principal componenet.

For this module, we will focus on the first method and perform an eigendecomposition to obtain the
eigenvector and eigenvalue of the data's covariance matrix. Avellaneda and Lee suggested an extremely
important concept of eigenportfolios as interpreting PCA on returns data.

They stated that the eigenvector corresponding to the largest eigenvalue of the covariance matrix
is the general market direction. The subsequent eigenvectors are considered as market-neutral to the
entire universe as these vectors are inherently orthogonal to the first eigenvector. All eigenvectors
are orthogonal to each other, and because the eigenvectors stem from the covariance matrix of the
price data, we can also interpret it as an uncorrelated variance in returns.

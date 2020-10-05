.. _codependence-codependence_marti:

.. note::
    This section includes an accompanying Jupyter Notebook Tutorial that is now available via the respective tier on
    `Patreon <https://www.patreon.com/HudsonThames>`_.

.. note::
   The following implementations and documentation closely follow the work of Gautier Marti:
   `Some contributions to the clustering of financial time series and applications to credit default swaps <https://www.researchgate.net/publication/322714557>`_.

=====================
Codependence by Marti
=====================

The work mentioned above introduces a new approach of representing the random variables that splits apart dependency and
distribution without losing any information. It also contains a distance metric between two financial time series based
on a novel approach.

According to the author's classification:

"Many statistical distances exist to measure the dissimilarity of two random variables, and therefore two i.i.d. random
processes. Such distances can be roughly classified in two families:

    1. distributional distances, [...] which focus on dissimilarity between probability distributions and quantify divergences
    in marginal behaviours,

    2. dependence distances, such as the distance correlation or copula-based kernel dependency measures [...],
    which focus on the joint behaviours of random variables, generally ignoring their distribution properties.

However, we may want to be able to discriminate random variables both on distribution and dependence. This can be
motivated, for instance, from the study of financial assets returns: are two perfectly correlated random variables
(assets returns), but one being normally distributed and the other one following a heavy-tailed distribution, similar?
From risk perspective, the answer is no [...], hence the propounded distance of this article".

.. Tip::
   Read the original work to understand the motivation behind creating the novel technique deeper and check the reference
   papers that prove the above statements.

Spearman’s Rho
##############

Following the work of Marti:

"[The Pearson correlation coefficient] suffers from several drawbacks:
- it only measures linear relationship between two variables;
- it is not robust to noise
- it may be undefined if the distribution of one of these variables have infinite second moment.

More robust correlation coefficients are copula-based dependence measures such as Spearman’s rho:

.. math::
    \rho_{S}(X, Y) &= 12 E[F_{X}(X), F_{Y}(Y)] - 3 \\
    &= \rho(F_{X}(X), F_{Y}(Y))

and its statistical estimate:

.. math::
    \hat{\rho}_{S}(X, Y) = 1 - \frac{6}{T(T^2-1)}\sum_{t=1}^{T}(X^{(t)}- Y^{(t)})^2

where :math:`X` and :math:`Y` are univariate random variables, :math:`F_{X}(X)` is the cumulative distribution
function of :math:`X` , :math:`X^{(t)}` is the :math:`t` -th sorted observation of :math:`X` , and :math:`T` is the
total number of observations".

Our method is a wrapper for the scipy spearmanr function. For more details about the function and its parameters,
please visit `scipy documentation <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html>`_.

Implementation
**************

.. py:currentmodule:: mlfinlab.codependence.gnpr_distance

.. autofunction:: spearmans_rho

Generic Parametric Representation (GPR) distance
################################################

Theoretically, Marty defines the distance :math:`d_{\Theta}` between two random variables as:

" Let :math:`\theta \in [0, 1]` . Let :math:`(X, Y) \in \nu^{2}` , where :math:`\nu` is the space of all continuous
real-valued random variables. Let :math:`G = (G_{X}, G_{Y})` , where :math:`G_{X}` and :math:`G_{Y}` are respectively
:math:`X` and :math:`Y` marginal cdfs. We define the following distance

.. math::
    d_{\Theta}^{2}(X, Y) = \Theta d_{1}^{2}(G_{X}(X), G_{Y}(Y)) + (1 - \Theta) d_{0}^{2}(G_{X}, G_{Y})

where

.. math::
    d_{1}^{2}(G_{X}(X), G_{Y}(Y)) = 3 \mathbb{E}[|G_{X}(X) - G_{Y}(Y)|^{2}]

and

.. math::
    d_{0}^{2}(G_{X}, G_{Y}) = \frac{1}{2} \int_{R} (\sqrt{\frac{d G_{X}}{d \lambda}} -
    \sqrt{\frac{d G_{Y}}{d \lambda}})^{2} d \lambda  "

For two Gaussian random variables, the distance :math:`d_{\Theta}` is therefore defined by Marti as:

" Let :math:`(X, Y)` be a bivariate Gaussian vector, with :math:`X \sim \mathcal{N}(\mu_{X}, \sigma_{X}^{2})` ,
:math:`Y \sim \mathcal{N}(\mu_{Y}, \sigma_{Y}^{2})` and :math:`\rho (X,Y)` . We obtain,

.. math::
    d_{\Theta}^{2}(X, Y) = \Theta \frac{1 - \rho_{S}}{2} + (1 - \Theta) (1 -
    \sqrt{\frac{2 \sigma_{X} \sigma_{Y}}{\sigma_{X}^{2} + \sigma_{Y}^{2}}} e^{ -
    \frac{1}{4} \frac{(\mu_{X} - \mu_{Y})^{2}}{\sigma_{X}^{2} + \sigma_{Y}^{2}}})  "

The use of this distance is referenced as the generic parametric representation (GPR) approach.

From the paper:

"GPR distance is a fast and good proxy for distance :math:`d_{\Theta}` when the first two moments :math:`\mu`
and :math:`{\sigma}` predominate. Nonetheless, for datasets which contain heavy-tailed distributions,
GPR fails to capture this information".

.. Tip::
   The process of deriving this definition as well as a proof that :math:`d_{\Theta}` is a metric is present in the work:
   `Some contributions to the clustering of financial time series and applications to credit default swaps <https://www.researchgate.net/publication/322714557>`_.


Implementation
**************

.. autofunction:: gpr_distance

Generic Non-Parametric Representation (GNPR) distance
#####################################################

The statistical estimate of the distance :math:`\tilde{d}_{\Theta}` working on realizations of the i.i.d. random variables
is defined by the author as:

" Let :math:`(X^{t})_{t=1}^{T}` and :math:`(Y^{t})_{t=1}^{T}` be :math:`T` realizations of real-valued random variables
:math:`X, Y \in \nu` respectively. An empirical distance between realizations of random variables can be defined by

.. math::
    \tilde{d}_{\Theta}^{2}((X^{t})_{t=1}^{T}, (Y^{t})_{t=1}^{T}) \stackrel{\text{a.s.}}{=}
    \Theta \tilde{d}_{1}^{2} + (1 - \Theta) \tilde{d}_{0}^{2}

where

.. math::
    \tilde{d}_{1}^{2} = \frac{3}{T(T^{2} - 1)} \sum_{t = 1}^{T} (X^{(t)} - Y^{(t)}) ^ {2}

and

.. math::
    \tilde{d}_{0}^{2} = \frac{1}{2} \sum_{k = - \infty}^{+ \infty} (\sqrt{g_{X}^{h}(hk)} - \sqrt{g_{Y}^{h}(hk)})^{2}

:math:`h` being here a suitable bandwidth, and
:math:`g_{X}^{h}(x) = \frac{1}{T} \sum_{t = 1}^{T} \mathbf{1}(\lfloor \frac{x}{h} \rfloor h \le X^{t} <
(\lfloor \frac{x}{h} \rfloor + 1)h)` being a density histogram estimating dpf :math:`g_{X}` from
:math:`(X^{t})_{t=1}^{T}` , :math:`T` realization of a random variable :math:`X \in \nu` ".

The use of this distance is referenced as the generic non-parametric representation (GNPR) approach.

As written in the paper:

" To use effectively :math:`d_{\Theta}` and its statistical estimate, it boils down to select a particular value for
:math:`\Theta` . We suggest here an exploratory approach where one can test

    (i) distribution information (θ = 0),

    (ii) dependence information (θ = 1), and

    (iii) a mix of both information (θ = 0,5).

Ideally, :math:`\Theta` should reflect the balance of dependence and distribution information in the data.
In a supervised setting, one could select an estimate :math:`\hat{\Theta}` of the right balance :math:`\Theta^{*}`
optimizing some loss function by techniques such as cross-validation. Yet, the lack of a clear loss function makes
the estimation of :math:`\Theta^{*}` difficult in an unsupervised setting".

.. note::

    The implementation of GNPR in the MlFinLab package was adjusted so that :math:`\tilde{d}_{0}^{2}`
    (dependence information distance) is being calculated using the 1D Optimal Transport Distance
    following the example in the
    `POT package documentation <https://pythonot.github.io/auto_examples/plot_OT_1D.html#sphx-glr-auto-examples-plot-ot-1d-py>`_.
    This solution was proposed by Marti.

Distributions of random variables are approximated using histograms with a given number of bins as input.

Optimal Transport Distance is then obtained from the Optimal Transportation Matrix (OTM) using
the Loss Matrix (M) as shown in `Optimal Transport blog post by Marti <https://gmarti.gitlab.io/qfin/2020/06/25/copula-optimal-transport-dependence.html>`_:

.. math::

    \tilde{d}_{0}^{2} = tr (OT^{T} * M)

where :math:`tr( \cdot )` is trace of a matrix and :math:`\cdot^{T}` is a transposed matrix.

This approach solves the issue of defining support for underlying distributions and choosing a
number of bins.


Implementation
**************

.. autofunction:: gnpr_distance

Examples
########

The following example shows how the above functions can be used:

.. code-block::

   import pandas as pd
   from mlfinlab.codependence import (spearmans_rho, gpr_distance, gnpr_distance,
                                     get_dependence_matrix)

   # Getting the dataframe with time series of returns
   data = pd.read_csv('X_FILE_PATH.csv', index_col=0, parse_dates = [0])

   element_x = 'SPY'
   element_y = 'TLT'

   # Calculating the Spearman's rho coefficient between two time series
   rho = spearmans_rho(data[element_x], data[element_y])

   # Calculating the GPR distance between two time series with both
   # distribution and dependence information
   gpr_dist = gpr_distance(data[element_x], data[element_y], theta=0.5)

   # Calculating the GNPR distance between two time series with dependence information only
   gnpr_dist = gnpr_distance(data[element_x], data[element_y], theta=1)

   # Calculating the GNPR distance between all time series with both
   # distribution and dependence information
   gnpr_matrix = get_dependence_matrix(data, dependence_method='gnpr_distance',
                                       theta=0.5)

Research Notebooks
******************

.. note::
    This and other accompanying Jupyter Notebook Tutorials are now available via the respective tier on
    `Patreon <https://www.patreon.com/HudsonThames>`_.

The following research notebook can be used to better understand the codependence metrics described above.

* `Codependence by Marti`_

.. _`Codependence by Marti`: https://github.com/Hudson-and-Thames-Clients/research/blob/master/Codependence/Codependence%20by%20Marti/codependence_by_marti.ipynb
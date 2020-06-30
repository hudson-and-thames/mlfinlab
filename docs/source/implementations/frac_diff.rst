.. _implementations-frac_diff:

====================================
Fractionally Differentiated Features
====================================

One of the challenges of quantitative analysis in finance is that time series of prices have trends or a non-constant mean.
This makes the time series is non-stationary. A non-stationary time series are hard to work with when we want to do inferential
analysis based on the variance of returns, or probability of loss.

Many supervised learning algorithms have the underlying assumption that the data is stationary. Specifically, in supervised
learning, one needs to map hitherto unseen observations to a set of labeled examples and determine the label of the new observation.

According to Marcos Lopez de Prado: “If the features are not stationary we cannot map the new observation
to a large number of known examples”. Making time series stationary often requires stationary data transformations,
such as integer differentiation. These transformations remove memory from the series. The method proposed by Marcos Lopez de Prado aims
to make data stationary while preserving as much memory as possible, as it's the memory part that has predictive power.

Fractionally differentiated features approach allows differentiating a time series to the point where the series is
stationary, but not over differencing such that we lose all predictive power.

.. tip::
   **Underlying Literature**

   The following sources elaborate extensively on the topic:

   - **Advances in Financial Machine Learning, Chapter 5** *by* Marcos Lopez de Prado. *Describes the motivation behind the Fractionally Differentiated Features and algorithms in more detail*


Fixed-width Window Fracdiff
###########################

The following description is based on **Chapter 5 of Advances in Financial Machine Learning**:

Using a positive coefficient :math:`d` the memory can be preserved:

.. math::
   \widetilde{X}_{t} = \sum_{k=0}^{\infty}\omega_{k}X_{t-k}

where :math:`X` is the original series, the :math:`\widetilde{X}` is the fractionally differentiated one, and
the weights :math:`\omega` are defined as follows:

.. math::
   \omega = \{1, -d, \frac{d(d-1)}{2!}, -\frac{d(d-1)(d-2)}{3!}, ..., (-1)^{k}\prod_{i=0}^{k-1}\frac{d-i}{k!}, ...\}

"When :math:`d` is a positive integer number, :math:`\prod_{i=0}^{k-1}\frac{d-i}{k!} = 0, \forall k > d`, and memory
beyond that point is cancelled."

Given a series of :math:`T` observations, for each window length :math:`l`, the relative weight-loss can be calculated as:

.. math::
   \lambda_{l} = \frac{\sum_{j=T-l}^{T} | \omega_{j} | }{\sum_{i=0}^{T-l} | \omega_{i} |}

The weight-loss calculation is attributed to a fact that "the initial points have a different amount of memory"
( :math:`\widetilde{X}_{T-l}` uses :math:`\{ \omega \}, k=0, .., T-l-1` ) compared to the final points
( :math:`\widetilde{X}_{T}` uses :math:`\{ \omega \}, k=0, .., T-1` ).

With a defined tolerance level :math:`\tau \in [0, 1]` a :math:`l^{*}` can be calculated so that :math:`\lambda_{l^{*}} \le \tau`
and :math:`\lambda_{l^{*}+1} > \tau`, which determines the first :math:`\{ \widetilde{X}_{t} \}_{t=1,...,l^{*}}` where "the
weight-loss is beyond the acceptable threshold :math:`\lambda_{t} > \tau` ."

Without the control of weight-loss the :math:`\widetilde{X}` series will pose a severe negative drift. This problem
is corrected by using a fixed-width window and not an expanding one.

With a fixed-width window, the weights :math:`\omega` are adjusted to :math:`\widetilde{\omega}` :

.. math::

   \widetilde{\omega}_{k} =
   \begin{cases}
   \omega_{k}, & \text{if } k \le l^{*} \\
   0, & \text{if } k > l^{*}
   \end{cases}

Therefore, the fractionally differentiated series is calculated as:

.. math::
   \widetilde{X}_{t} = \sum_{k=0}^{l^{*}}\widetilde{\omega_{k}}X_{t-k}

for :math:`t = T - l^{*} + 1, ..., T`

The following graph shows a fractionally differenced series plotted over the original closing price series:

.. figure:: frac_diff_graph.png
   :scale: 80 %
   :align: center
   :figclass: align-center
   :alt: Fractionally Differentiated Series

   Fractionally differentiated series with a fixed-width window `(Lopez de Prado 2018) <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3447398>`_

.. tip::

   A deeper analysis of the problem and the tests of the method on various futures is available in the
   **Chapter 5 of Advances in Financial Machine Learning**.


Implementation
**************

The following function implemented in mlfinlab can be used to derive fractionally differentiated features.

.. py:currentmodule:: mlfinlab.features.fracdiff
.. autofunction::  frac_diff_ffd


Stationarity With Maximum Memory Representation
###############################################

The following description is based on **Chapter 5 of Advances in Financial Machine Learning**:

Applying the fixed-width window fracdiff (FFD) method on series, the minimum coefficient :math:`d^{*}` can be computed.
With this :math:`d^{*}` the resulting fractionally differentiated series is stationary. This coefficient
:math:`d^{*}` quantifies the amount of memory that needs to be removed to achieve stationarity.

If the input series:

- is already stationary, then :math:`d^{*}=0`.
- contains a unit root, then :math:`d^{*} < 1`.
- exhibits explosive behavior (like in a bubble), then :math:`d^{*} > 1`.

A case of particular interest is :math:`0 < d^{*} \ll 1`, when the original series is "mildly non-stationary."
In this case, although differentiation is needed, a full integer differentiation removes
excessive memory (and predictive power).

The following grap shows how the output of a ``plot_min_ffd`` function looks.

.. figure:: plot_min_ffd_graph.png
   :scale: 80 %
   :align: center
   :figclass: align-center
   :alt: Minimum D value that passes the ADF test

   ADF statistic as a function of d

The right y-axis on the plot is the ADF statistic computed on the input series downsampled
to a daily frequency.

The x-axis displays the d value used to generate the series on which the ADF statistic is computed.

The left y-axis plots the correlation between the original series ( :math:`d = 0` ) and the differentiated
series at various :math:`d` values.

The horizontal dotted line is the ADF test critical value at a 95% confidence level. Based on
where the ADF statistic crosses this threshold, the minimum :math:`d` value can be defined.

The correlation coefficient at a given :math:`d` value can be used to determine the amount of memory
that was given up to achieve stationarity. (The higher the correlation - the less memory was given up)

According to Lopez de Prado:

"Virtually all finance papers attempt to recover stationarity by applying an integer
differentiation :math:`d = 1`, which means that most studies have over-differentiated
the series, that is, they have removed much more memory than was necessary to
satisfy standard econometric assumptions."

.. tip::

   An example on how the resulting figure can be analyzed is available in
   **Chapter 5 of Advances in Financial Machine Learning**.


Implementation
**************

The following function implemented in mlfinlab can be used to achieve stationarity with maximum memory representation.

.. autofunction::  plot_min_ffd


Example
#######

Given that we know the amount we want to difference our price series, fractionally differentiated features, and the
minimum d value that passes the ADF test can be derived as follows:

.. code-block::

   import numpy as np
   import pandas as pd

   from mlfinlab.features.fracdiff import frac_diff_ffd, plot_min_ffd

   # Import price data
   data = pd.read_csv('FILE_PATH')

   # Deriving the fractionally differentiated features
   frac_diff_series = frac_diff_ffd(data['close'], 0.5)

   # Plotting the graph to find the minimum d
   # Make sure the input dataframe has a 'close' column
   plot_min_ffd(data)


Research Notebook
#################

The following research notebook can be used to better understand fractionally differentiated features.

* `Fractionally Differentiated Features`_

.. _Fractionally Differentiated Features: https://github.com/hudson-and-thames/research/blob/master/Advances%20in%20Financial%20Machine%20Learning/Fractionally%20Differentiated%20Features/Chapter5_Exercises.ipynb




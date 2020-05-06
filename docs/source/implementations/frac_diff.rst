.. _implementations-frac_diff:

====================================
Fractionally Differentiated Features
====================================

One of the challenges of quantitative analysis in finance is that time series of prices have trends or a non-constant mean.
This makes the time series is non-stationary. A non-stationary time series are hard to work with when we want to do inferential
analysis based on variance of returns, or probability of loss.

Many supervised learning algorithms have the underlying assumption that the data is stationary. Specifically, in supervised
learning, one needs to map hitherto unseen observations to a set of labeled examples and determine the label of the new observation.

According to Marcos Lopez de Prado: “If the features are not stationary we cannot map the new observation
to a large number of known examples”. Making time series stationary often requires stationary data transformations,
such as integer differentiation that remove memory from the series. The method proposed by Marcos Lopez de Prado aims
to make data stationary while preserving as much memory as possible, as it's the memory part that has predictive power.

Fractionally differentiated features tackle this problem by deriving features through fractionally differentiating a time
series to the point where the series is stationary, but not over differencing such that we lose all predictive power.

.. tip::
   **Underlying Literature**

   The following sources elaborate extensively on the topic:

   - **Advances in Financial Machine Learning, Chapter 5** *by* Marcos Lopez de Prado. *Describes the motivation behind the Fractionally Differentiated Features and algorithms in more detail*

Fixed-width Window Fracdiff
###########################

The following description is based on the **Chapter 5 of Advances in Financial Machine Learning**:

Using a positive coefficient :math:`d` the memory can be preserved:

.. math::
   \widetilde{X_{t}} = \sum_{k=0}^{\infty}\omega_{k}X_{t-k}

where the :math:`\omega` weights are defined as follows:

.. math::
   \omega = \{1, -d, \frac{d(d-1)}{2!}, -\frac{d(d-1)(d-2)}{3!}, ..., (-1)^{k}\prod_{i=0}^{k-1}\frac{d-i}{k!}, ...\}

"When :math:`d` is a positive integer number, :math:`\prod_{i=0}^{k-1}\frac{d-i}{k!} = 0, \forall k > d`, and memory
beyond that point is cancelled."

The following graph shows a fractionally differenced series plotted over the original closing price series:

.. figure:: frac_diff_graph.png
   :scale: 80 %
   :align: center
   :figclass: align-center
   :alt: Fractionally Differentiated Series

   Fractionally Differentiated Series `(Lopez de Prado 2018) <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3447398>`_

Implementation
##############

The following function implemented in mlfinlab can be used to derive fractionally differentiated features.

.. py:currentmodule:: mlfinlab.features.fracdiff
.. autofunction::  frac_diff_ffd

Example
#######

Given that we know the amount we want to difference our price series, fractionally differentiated features can be derived
as follows:

.. code-block::

   import numpy as np
   import pandas as pd

   from mlfinlab.features.fracdiff import frac_diff_ffd

   data = pd.read_csv('FILE_PATH')
   frac_diff_series = frac_diff_ffd(data['close'], 0.5)


Research Notebook
#################

The following research notebook can be used to better understand fractionally differentiated features.

* `Fractionally Differentiated Features`_

.. _Fractionally Differentiated Features: https://github.com/hudson-and-thames/research/blob/master/Chapter5/Chapter5_Exercises.ipynb




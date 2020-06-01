.. _codependence-correlation_based_metrics:

.. note::
   The following implementations and documentation, closely follows the lecture notes notes from Cornell University, by Marcos Lopez de Prado:
   `Codependence (Presentation Slides) <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_.

=========================
Correlation-Based Metrics
=========================

Distance Correlation
====================

**Distance correlation** can capture not only linear association but also non-linear variable dependencies which Pearson correlation can not.
It was introduced in 2005 by Gábor J. Szekely and is described in the work
`"Measuring and testing independence by correlation of distances". <https://projecteuclid.org/download/pdfview_1/euclid.aos/1201012979>`_
It is calculated as:

.. math::
    \rho_{dist}[X, Y] = \frac{dCov[X, Y]}{\sqrt{dCov[X, X]dCov[Y,Y}}

Where :math:`dCov[X, Y]` can be interpreted as the average Hadamard product of the doubly-centered Euclidean distance matrices of
:math:`X, Y`. (`Cornell lecture slides, p.7 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_)

Values of distance correlation fall in range:

.. math::
    0 \leq \rho_{dist}[X, Y] \leq 1


Distance correlation is equal to zero if and only if the two variables are independent (in contrast to Pearson correlation
that can be zero even if the variables are dependant).

.. math::
    \rho_{dist}[X, Y] = 0 \Leftrightarrow X \perp Y

As shown in the figure below, distance correlation captures the nonlinear relationship.

.. image:: images/distance_correlation.png
   :scale: 70 %
   :align: center


The numbers in the first line are Pearson correlation values and the values in the second line are Distance correlation values.
This figure is from `"Introducing the discussion paper by Székely and Rizzo" <https://www.researchgate.net/publication/238879872_Introducing_the_discussion_paper_by_Szekely_and_Rizzo>`_
by Michale A. Newton. It provides a great overview for readers.

Implementation
==============

.. py:currentmodule:: mlfinlab.codependence.correlation

.. autofunction:: distance_correlation


Angular Distance
================

**Angular distance** is a slight modification of the correlation coefficient which satisfies all distance metric conditions.
This measure is known as the angular distance because when we use *covariance* as *inner product*, we can interpret correlation as :math:`cos\theta`.

It is a metric, because it is a linear multiple of the Euclidean distance between the vectors :math:`X, Y` (after standardization)
(`Cornell lecture slides, p.10 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_).

There are three types of angular distance: standard, absolute and squared.

Standard
========

.. figure:: images/angular_distance.png
   :scale: 70 %
   :align: center
   :figclass: align-center
   :alt: Angular Distance

   The angular distance satisfies all the conditions of a true metric, (Lopez de Prado, 2020.)


.. autofunction:: angular_distance

.. math::
    d_\rho[X, Y] = \sqrt{\frac{1}{2}(1-\rho[X,Y])}

.. math::
    d_\rho \in [0, 1]


Absolute and Squared
====================

.. figure:: images/modified_angular_distance.png
   :scale: 70 %
   :align: center
   :figclass: align-center
   :alt: Modified Angular Distance

   In some financial applications, it makes more sense to apply a modified definition of angular distance, such that the
   sign of the correlation is ignored, (Lopez de Prado, 2020)

**Absolute**

.. math::
    d_{|\rho|}[X, Y] = \sqrt{1-|\rho[X,Y]|}

.. autofunction:: absolute_angular_distance

|

**Squared**

.. math::
    d_{\rho^2}[X, Y] = \sqrt{1-{\rho[X,Y]}^2}

.. autofunction:: squared_angular_distance

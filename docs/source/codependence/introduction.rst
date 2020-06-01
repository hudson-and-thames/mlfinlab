.. _codependence-introduction:

============
Introduction
============

This module includes implementations of codependence metrics. According to Lopez de Prado:

"Two random variables are codependent when knowing the value of one helps us determine the value of the other.
This should not me confounded with the notion of causality."

Pearson correlation coefficient is the most famous and widely used measure of codependence, however, it has some drawbacks.

.. warning::

    Pearson correlation suffers from 3 major drawbacks:

    1) It captures linear effects, but if two variables have strong non-linear dependency (squared or abs for example) Pearson correlation won't find any pattern between them.
    2) Correlation is not a distance metric: it does not satisfy non-negativity and subadditivity conditions.
    3) Financial markets have non-linear patterns, which Pearson correlation fails to capture.

Pearson correlation is not the only way of measuring codependence. There are alternative and more modern measures of codependence,
which are described in the parts of this module.

.. note::
   In tis module it's discussed whether a particular metric is a true metric.
   According Arkhangel'skii, A. V. and Pontryagin, L. S. (1990), **General Topology I**:
   A metric on a set :math:`X` is a function (called a distance)

   .. math::
      d: XxX -> [0,+\infin)
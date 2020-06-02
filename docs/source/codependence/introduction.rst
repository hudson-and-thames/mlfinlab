.. _codependence-introduction:

============
Introduction
============

This module includes implementations of codependence metrics. According to Lopez de Prado:

"Two random variables are codependent when knowing the value of one helps us determine the value of the other.
This should not be confounded with the notion of causality."

Pearson correlation coefficient is the most famous and widely used measure of codependence, however, it has some drawbacks.

.. warning::

    Pearson correlation suffers from 3 major drawbacks:

    1) It captures linear effects, but if two variables have strong non-linear dependency (squared or abs for example) Pearson correlation won't find any pattern between them.
    2) Correlation is not a distance metric: it does not satisfy non-negativity and subadditivity conditions.
    3) Financial markets have non-linear patterns, which Pearson correlation fails to capture.

Pearson correlation is not the only way of measuring codependence. There are alternative and more modern measures of codependence,
which are described in the parts of this module.

.. note::
   For some methods in this module, it’s discussed whether they are true metrics.
   According to Arkhangel'skii, A. V. and Pontryagin, L. S. (1990), **General Topology I**:
   A metric on a set :math:`X` is a function (called a distance):

   .. math::
      d: X \times X \rightarrow [0,+ \infty) ;   x, y, z \in X

   for which the following three axioms are satisfied:

   1. :math:`d(x, y) = 0 \iff x = y` — identity of indiscernibles;

   2. :math:`d(x,y) = d(y,x)` — symmetry;

   3. :math:`d(x,y) \le d(x,z) + d(z,y)` — triangle inequality;

   and these imply :math:`d(x,y) \ge 0` — non-negativity.
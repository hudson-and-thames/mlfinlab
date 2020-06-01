.. _codependence-introduction:

.. note::
   The following implementations and documentation, closely follows the lecture notes notes from Cornell University, by Marcos Lopez de Prado:
   `Codependence (Presentation Slides) <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_.


============
Introduction
============

Pearson correlation coefficient is the most famous and widely used measure of codependence, however, there are some drawbacks.

.. warning::

    Pearson correlation suffers from 3 major drawbacks:

    1) It captures linear effects, but if two variables have strong non-linear dependency (squared or abs for example) Pearson correlation won't find any pattern between them.
    2) Correlation is not a distance metric: it does not satisfy non-negativity and subadditivity conditions.
    3) Financial market have non-linear patterns and correlations fails to capture them.

However, Pearson correlation is not the only way of measuring codependence. There are alternative and more modern measures of codependence, such
such as those introduced in information theory.

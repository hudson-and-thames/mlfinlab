.. _implementations-codependence:

============
Codependence
============

This module implements various dependence measures described in Dr. Marcos Lopez de Prado's slides `Codependence`_ from
Cornell University.

**Abstract**:

"Two random variables are codependent when knowing the value of one helps us determine the value of the other. This should
not be confounded with the notion of causality.

Correlation is perhaps the best known measure of codependence in econometric studies. Despite its popularity among economists,
correlation has many known limitations in the contexts of financial studies.

In these slides we will explore more modern measures of codependence, based on information theory, which overcome some of
the limitations of correlations."

.. _`Codependence`: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994

.. warning::

    Pearson correlation suffers from 2 major drawbacks:

    1) It captures linear effects, but if two variables have strong non-linear dependency (squared or abs for example) Pearson correlation won't find any pattern between them.
    2) Correlation is not a metric: it does not satisfy non-negativity and and subadditivity conditions.

Correlation-Based Metrics
#########################

**Angular Distance** is a slight modification of correlation coefficient which satisfies all metric conditions.

**Distance Correlation** was introduced in 2005 by Gábor J. Székely to capture non-linear variable dependencies.

.. py:currentmodule:: mlfinlab.codependence.correlation
.. automodule:: mlfinlab.codependence.correlation
   :members:

Information Theory Metrics
##########################

**Mutual Information** is defined as the decrease in uncertainty (or informational gain) in X that results from knowing the value of Y. Mutual information is not
a metric and needs to be normalized.

**Variation of Information** can be interpreted as the uncertainty we expect in one variable if we are told the value of another. Variation of information is a
metric because it satisfies non-negativity, symmetry and triangle inequality axioms.

.. py:currentmodule:: mlfinlab.codependence.information
.. automodule:: mlfinlab.codependence.information
   :members:



Example
#######

The following example highlights how the various metrics behave under various variable dependencies:

1. Linear
2. Squared
3.  Y = abs(X)
4. Independent variables

.. code-block::

    import numpy as np
    import matplotlib.pyplot as plt

    from mlfinlab.codependece import distance_correlation, get_mutual_info, variation_of_information_score
    from ace import model # ace package is used for max correlation estimation

    def max_correlation(x: np.array, y: np.array) -> float:
        """
        Get max correlation using ace package.
        """

        x_input = [x]
        y_input = y
        ace_model = model.Model()
        ace_model.build_model_from_xy(x_input, y_input)
        return np.corrcoef(ace_model.ace.x_transforms[0], ace_model.ace.y_transform)[0][1]

    state = np.random.RandomState(42)
    x = state.normal(size=1000)
    y_1 = 2 * x + state.normal(size=1000) / 5 # linear
    y_2 = x ** 2 + state.normal(size=1000) / 5 # squared
    y_3 = abs(x) + state.normal(size=1000) / 5 # Abs
    y_4 = np.random.RandomState(0).normal(size=1000) * np.random.RandomState(5).normal(size=1000) # independent

    for y, dependency in zip([y_1, y_2, y_3, y_4], ['linear', 'squared', 'y=|x|', 'independent']):
        text = "Pearson corr: {:0.2f} \nNorm.mutual info: {:0.2f}\nDistance correlation: {:0.2f} \nInformation variation: {:0.2f} \nMax correlation: {:0.2f}".format(
        np.corrcoef(x, y)[0, 1], get_mutual_info(x, y, normalize=True), distance_correlation(x, y), variation_of_information_score(x, y, normalize=True), max_correlation(x, y))


        # Plot relationships
        fig, ax = plt.subplots(figsize=(8,7))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        plt.title(dependency)
        ax.plot(x, y, 'ro')
        plt.savefig('{}.png'.format(dependency))


.. image:: codependence_images/linear.png
    :scale: 70 %
    :align: center

.. image:: codependence_images/squared.png
    :scale: 70 %
    :align: center

.. image:: codependence_images/abs.png
    :scale: 70 %
    :align: center

.. image:: codependence_images/independent.png
    :scale: 70 %
    :align: center

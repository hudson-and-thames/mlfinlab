.. _implementations-codependence:

============
Codependence
============

Pearson correlation numbers are the most famous and widely used to measure Codependence, but there are some drawbacks below.

.. warning::

    Pearson correlation suffers from 2 major drawbacks:

    1) It captures linear effects, but if two variables have strong non-linear dependency (squared or abs for example) Pearson correlation won't find any pattern between them.
    2) Correlation is not a metric: it does not satisfy non-negativity and and subadditivity conditions.

However, The Pearson correlation is not the only way of measuring codependence.
There are modern measures of codependence, using Euclidean distances or based on information theory,
which overcome some of the limitations of correlations.
You can find more details from `Codependence (Presentation Slides)`_ by Marcos Lopez de Prado

.. _`Codependence (Presentation Slides)`: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994


Distance Correlation
####################

**Distance Correlation** can capture not only linear assocaition but also non-linear variable dependencies which Pearson correlation can not describe.
It was introduced in 2005 by Gábor J. Székely[`wikipedia`_].

.. _`wikipedia`: https://en.wikipedia.org/wiki/Distance_correlation

- :py:func:`mlfinlab.codependence.correlation.distance_correlation`

.. math::
    \rho_{dist}[X, Y] = \frac{dCov[X, Y]}{\sqrt{dCov[X, X]dCov[Y,Y}}

.. math::
    0 \leq \rho_{dist}[X, Y] \leq 1


|  :math:`dCov[X, Y]` can be interpreted as the average Hadamard product of the doubly-centered Euclidean distance matrices of
   :math:`X, Y` [`Cornell lecture slides, p.7 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_]


|
| Unlike Pearson's correlation, when the value is zero, we can say the two variables are independent.

.. math::
    \rho_{dist}[X, Y] = 0 \Leftrightarrow X \perp Y


| As shown in the figure below, Distance Correlation captures the nonlinear relationship.

.. image:: codependence_images/distance_correlation.png
   :scale: 70 %
   :align: center


The numbers in the first line are Pearson correlation values and the values in the second line are Distance correlation values.
This figure is from '`Introducing the discussion paper by Székely and Rizzo`_' by Michale A. Newton.
You can also get great overview of the distance correlation from that paper.


.. _`Introducing the discussion paper by Székely and Rizzo`: https://www.researchgate.net/publication/238879872_Introducing_the_discussion_paper_by_Szekely_and_Rizzo

----

Correlation-Based Distance
###########################

Angular Distance
*****************

**Angular Distance** is a slight modification of correlation coefficient which satisfies all metric conditions.
This measure is known as the angular distance because when we use *covariance* as *inner product*, we can interpret correlation as :math:`cos\theta`.
It is a metric, because it is a linear multiple of the Euclidean distance between the vectors :math:`X, Y` (after standardization)
[`Cornell lecture slides, p.10 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_]. There are various modifications of angular distance.(absolute, squared).


- :py:func:`mlfinlab.codependence.correlation.angular_distance`


.. math::
    d_\rho[X, Y] = \sqrt{\frac{1}{2}(1-\rho[X,Y])}
.. math::
    d_\rho \in [0, 1]

.. image:: codependence_images/angular_distance.png
   :scale: 70 %
   :align: center


| There are alternative correlation based distance metrics. We can use those distance depend on applications.
|
|

- :py:func:`mlfinlab.codependence.correlation.absolute_angular_distance`

.. math::
    d_{|\rho|}[X, Y] = \sqrt{1-|\rho[X,Y]|}

- :py:func:`mlfinlab.codependence.correlation.squared_angular_distance`

.. math::
    d_{\rho^2}[X, Y] = \sqrt{1-{\rho[X,Y]}^2}

| Marcos Lopez de Prado's slides

    | In some financial applications, it makes more sense to apply a modified definition of angular distance, such that the sign of
    | the correlation is ignored --- Marcos Lopez de Prado, `Cornell lecture slides, p.11 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_


.. image:: codependence_images/modified_angular_distance.png
   :scale: 70 %
   :align: center

----

.. py:currentmodule:: mlfinlab.codependence.correlation
.. automodule:: mlfinlab.codependence.correlation
   :members:

----

Information Theory Metrics
##########################

we can gauge the codependence from the information theory perspective.
In information theory, (Shannon’s) entropy is a measure of information(uncertainty).

.. math::
    H[X] = -\sum\limits_{x \in S_{X}}p[x]log[p[x]]

In short, we can say that entropy is the expectation of the amount of information when we sampling from a certain probability distribution or the number of bits to transmit the target.
So, If there is correspondence between random variables, the correspondence will be reflected in entropy. For example, if two random variables are associated,
the amount of information of joint probability distribution of two random variables will be less than the sum of the information in each random variable.
This is because knowing a correspondence means knowing one random variable can reduce uncertainty about the other random variable.

.. math::
    H[X+Y]=H[X]+H[Y],  X \bot Y


Here, we have two wayw of measuring correspondence

- Mutual Information
- Variation of Information


we can check th relationships of various information measures associated with correlated variables
:math:`X` and :math:`Y` through below figure.[`Cornell lecture slides, p.24 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_]

.. image:: codependence_images/entropy_relation_diagram.png
   :scale: 70 %
   :align: center

----

Mutual Information
******************

**Mutual Information** is defined as the decrease in uncertainty (or informational gain) in X that results from knowing the value of Y.
Mutual information is not a metric and needs to be normalized. [`Cornell lecture slides, p.18 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_]

- :py:func:`mlfinlab.codependence.information.get_mutual_info`

.. math::
    \begin{align*}
    I[X, Y]=& H[X] - H[X|Y]\\
           =& H[X]+H[Y]-H[X,Y]\\
           =& \sum\limits_{x \in S_{X}} \sum\limits_{y \in S_{Y}}p[x,y]log[\frac{p[x,y]}{p[x]p[y]}]\\
    \end{align*}


----

Variation of Information
************************

**Variation of Information** can be interpreted as the uncertainty we expect in one variable if we are told the value of another. Variation of information is a
metric because it satisfies non-negativity, symmetry and triangle inequality axioms.

- :py:func:`mlfinlab.codependence.information.get_optimal_number_of_bins`

.. math::
    \begin{align*}
    VI[X,Y]=& H[X|Y] + H[Y|X]\\
           =& H[X] + H[Y]-2I[X,Y]\\
           =& 2H[X,Y]-H[X]-H[Y]\\
    \end{align*}

----

Discretization
**************

Throughout above section, we have assumed that random variables were discrete.
For continuous case, we can quantize the values and estimate :math:`H[X]`, and apply the same concepts on the binned observations.
[`Cornell lecture slides, p.26 <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994>`_]

.. math::
    \begin{align*}
    H[X] =& \int_{\infty}^{\infty}f_{X}[x_{i}]logf_{X}[x]dx\\
    \:    &\approx-\sum\limits_{i=1}^{B_{X}}f_{X}[x_{i}]logf_{X}[x_{i}]\\
    \end{align*}

.. math::
    \hat{H}[X]=-\sum\limits_{i=1}^{B_{X}}\frac{N_{i}}{N}log[\frac{N_{i}}{N}]log[\Delta_{x}]

As you can see from these equations, we need to choose the binning carefully because results may be biased.
There are optimal binning  depends on entropy case(marginal, joint).
You can optimal number of bins for discretization through below method.
This function is need for using methods for getting information based codependence.

- :py:func:`mlfinlab.codependence.information.variation_of_information_score`


----

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

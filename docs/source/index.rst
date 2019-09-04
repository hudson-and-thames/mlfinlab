.. image:: logo/ht_logo.png
   :scale: 20 %
   :align: center


|

================================================
Machine Learning Financial Laboratory (mlfinlab)
================================================


	|Build Status| |codecov| |pylint Score| |License: BSD3|

	.. |Build Status| image:: https://travis-ci.com/hudson-and-thames/mlfinlab.svg?branch=master
		:target: https://travis-ci.com/hudson-and-thames/mlfinlab

	.. |codecov| image:: https://codecov.io/gh/hudson-and-thames/mlfinlab/branch/master/graph/badge.svg
		:target: https://codecov.io/gh/hudson-and-thames/mlfinlab

	.. |pylint Score| image:: https://mperlet.github.io/pybadge/badges/10.svg

	.. |License: BSD3| image:: https://img.shields.io/github/license/hudson-and-thames/mlfinlab.svg
		:target: https://opensource.org/licenses/BSD-3-Clause

	|PyPi| |Downloads| |Python|

	.. |PyPi| image:: https://img.shields.io/pypi/v/mlfinlab.svg
		:target: https://pypi.org/project/mlfinlab/

	.. |Downloads| image:: https://img.shields.io/pypi/dm/mlfinlab.svg
		:target: https://pypi.org/project/mlfinlab/

	.. |Python| image:: https://img.shields.io/pypi/pyversions/mlfinlab.svg
		:target: https://pypi.org/project/mlfinlab/

mlfinlab is an open source package based on the research of Dr Marcos Lopez de Prado in his new book
Advances in Financial Machine Learning. This implementation started out as a spring board for a research project in the `Masters in Financial Engineering programme at WorldQuant University`_ and has grown into a mini research group called Hudson and Thames (not affiliated with the university).

.. _Masters in Financial Engineering programme at WorldQuant University: https://wqu.org/


Notes
=====

mlfinlab is a living, breathing project, and new functionalities are consistently being added.

The implementations that will be added in the future as well as the implementations that are currently supported can be seen below:

* **Part 4: Useful Financial Features**
* Working on Chapter 19: Microstructural Features (Maksim)

.. raw:: latex

    \vspace{5mm}

* **Part 3: Backtesting**
* Done Chapter 16: ML Asset Allocation
* Done Chapter 10: Bet Sizing

.. raw:: latex

    \vspace{5mm}

* **Part 2: Modelling**
* Done Chapter 8: Feature Importance
* Done Chapter 7: Cross-Validation
* Done Chapter 6: Ensemble Methods
* Done Sequential Bootstrap Ensemble

.. raw:: latex

    \vspace{5mm}

* **Part 1: Data Analysis**
* Done Chapter 5: Fractionally Differentiated Features
* Done Chapter 4: Sample Weights
* Done Chapter 3: Labeling
* Done Chapter 2: Data Structures
* Purchased high quality raw tick data.
* Email us if you would like a sample of the standard bars.

Built With
==========

* `Github`_ - Development platform and repo
* `Travis CI`_ - Continuous integration, test and deploy

.. _Github: https://github.com/hudson-and-thames/mlfinlab
.. _Travis CI: https://www.travis-ci.com

Getting Started
===============

.. toctree::
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/barriers_to_entry
   getting_started/requirements

* :doc:`getting_started/installation`

* :doc:`getting_started/barriers_to_entry`

* :doc:`getting_started/requirements`

Implementations
===============

.. toctree::
   :caption: Implementations
   :hidden:

   implementations/data_structures
   implementations/filters
   implementations/labeling
   implementations/sampling
   implementations/frac_diff
   implementations/cross_validation
   implementations/sb_bagging
   implementations/feature_importance
   implementations/bet_sizing
   implementations/portfolio_optimisation


* :doc:`implementations/data_structures`
* :doc:`implementations/filters`
* :doc:`implementations/labeling`
* :doc:`implementations/sampling`
* :doc:`implementations/frac_diff`
* :doc:`implementations/cross_validation`
* :doc:`implementations/sb_bagging`
* :doc:`implementations/feature_importance`
* :doc:`implementations/bet_sizing`
* :doc:`implementations/portfolio_optimisation`

Additional Information
======================

.. toctree::
   :caption: Additional Information
   :hidden:

   additional_information/contact
   additional_information/contributing
   additional_information/license

* :doc:`additional_information/contact`
* :doc:`additional_information/contributing`
* :doc:`additional_information/license`

:module: mlfinlab

.. image:: logo/ht_logo.png
   :scale: 15 %
   :align: center

|

================================================
Machine Learning Financial Laboratory (mlfinlab)
================================================

    |PyPi| |Python| |Build Status| |codecov| |pylint Score|

    .. |Build Status| image:: https://travis-ci.com/hudson-and-thames/mlfinlab.svg?branch=master
        :target: https://travis-ci.com/hudson-and-thames/mlfinlab

    .. |codecov| image:: https://codecov.io/gh/hudson-and-thames/mlfinlab/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/hudson-and-thames/mlfinlab

    .. |pylint Score| image:: https://mperlet.github.io/pybadge/badges/10.svg

    |Downloads|

    .. |PyPi| image:: https://img.shields.io/pypi/v/mlfinlab.svg
        :target: https://pypi.org/project/mlfinlab/

    .. |Downloads| image:: https://img.shields.io/pypi/dm/mlfinlab.svg
        :target: https://pypi.org/project/mlfinlab/

    .. |Python| image:: https://img.shields.io/pypi/pyversions/mlfinlab.svg
        :target: https://pypi.org/project/mlfinlab/


MLFinLab is a python package based on the research of Dr. Marcos Lopez de Prado (`QuantResearch.org`_) in his
new book Advances in Financial Machine Learning, Machine Learning for Asset Managers, and various additional implementations
from various authors, often from the `Journal of Financial Data Science`_.

This implementation started out as a spring board for a research project in the Masters in Financial Engineering programme
at `WorldQuant University`_ and has grown into a mini research group called`Hudson and Thames Quantitative Research`_ (not affiliated with the university).

.. _Hudson and Thames Quantitative Research: https://hudsonthames.org/
.. _WorldQuant University: https://wqu.org/
.. _Journal of Financial Data Science: https://jfds.pm-research.com/
.. _QuantResearch.org: http://www.quantresearch.org/
.. _Masters in Financial Engineering programme at WorldQuant University: https://wqu.org/

#####################
Sponsors and Donating
#####################

.. figure:: logo/support.png
   :scale: 100 %
   :align: center
   :figclass: align-center
   :alt: Platinum Sponsors

   `Become a Sponsor! <https://www.patreon.com/HudsonThames>`_

A special thank you to our sponsors! It is because of your contributions that we are able to continue the development of
academic research for open source. If you would like to become a sponsor and help support our research, please sign up
on `Patreon`_.

.. _Patreon: https://www.patreon.com/HudsonThames

*****************
Platinum Sponsor:
*****************

.. figure:: logo/machine-factor-technologies.png
   :scale: 100 %
   :align: center
   :figclass: align-center
   :alt: Platinum Sponsors

   `Machine Factor Technologies`_

**************
Gold Sponsors:
**************

.. figure:: logo/gold_sponsors.png
   :scale: 100 %
   :align: center
   :figclass: align-center
   :alt: Gold Sponsors

   `E.P. Chan & Associates`_ &  `Markov Capital`_

*******************
Supporter Sponsors:
*******************

+-----------------------+-----------------------+-----------------------+
| `John B. Keown`_      | `Roberto Spadim`_     | `Zack Gow`_           |
+-----------------------+-----------------------+-----------------------+
| `Jack Yu`_            |  Егор Тарасенок       | Joseph Matthew        |
+-----------------------+-----------------------+-----------------------+
| Justin Gerard         |  Jason Young          | `Shaun McDonogh`_     |
+-----------------------+-----------------------+-----------------------+
| `Christian Beckmann`_ |  Jeffrey Wang         | `Eugene Tartakovsky`_ |
+-----------------------+-----------------------+-----------------------+
| `Ming Wu`_            | `Richard Scheiwe`_    |  `Tianfang Wu`_       |
+-----------------------+-----------------------+-----------------------+


.. _`Machine Factor Technologies`: https://machinefactor.tech/
.. _`E.P. Chan & Associates`: https://www.epchan.com/
.. _`Markov Capital`: http://www.markovcapital.se/
.. _`John B. Keown`: https://www.linkedin.com/in/john-keown-quantitative-finance-big-data-ml/
.. _`Roberto Spadim`: https://www.linkedin.com/in/roberto-spadim/
.. _`Zack Gow`: https://www.linkedin.com/in/zackgow/
.. _`Jack Yu`: https://www.linkedin.com/in/jihao-yu/
.. _`Christian Beckmann`: https://www.linkedin.com/in/christian-beckmann/
.. _`Eugene Tartakovsky`: https://www.linkedin.com/in/etartakovsky/
.. _`Ming Wu`: https://www.linkedin.com/in/ming-yue-wu/
.. _`Richard Scheiwe`: https://www.linkedin.com/in/richardscheiwe/
.. _`Shaun McDonogh`: https://www.linkedin.com/in/shaunmcdonogh/
.. _`Tianfang Wu`: linkedin.com/in/tianfangwu


.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :hidden:

    getting_started/installation
    additional_information/contact
    getting_started/barriers_to_entry
    getting_started/researcher

.. toctree::
    :maxdepth: 2
    :caption: Feature Engineering
    :hidden:

    implementations/data_structures
    implementations/filters
    implementations/codependence
    implementations/frac_diff
    implementations/structural_breaks
    implementations/microstructural_features


.. toctree::
    :maxdepth: 2
    :caption: Labelling
    :hidden:

    implementations/tb_meta_labeling
    implementations/labeling_trend_scanning
    implementations/labeling_tail_sets

.. toctree::
    :maxdepth: 2
    :caption: Modelling
    :hidden:

    implementations/sampling
    implementations/sb_bagging
    implementations/feature_importance
    implementations/cross_validation
    implementations/EF3M
    implementations/bet_sizing

.. toctree::
    :maxdepth: 2
    :caption: Clustering
    :hidden:

    implementations/onc

.. toctree::
    :maxdepth: 2
    :caption: Backtest Overfitting
    :hidden:

    implementations/backtesting
    implementations/backtest_statistics


.. toctree::
    :maxdepth: 2
    :caption: Portfolio Optimization
    :hidden:

    portfolio_optimisation/risk_metrics
    portfolio_optimisation/returns_estimation
    portfolio_optimisation/risk_estimators
    portfolio_optimisation/mean_variance
    portfolio_optimisation/critical_line_algorithm
    portfolio_optimisation/hierarchical_risk_parity
    portfolio_optimisation/hierarchical_clustering_asset_allocation
    portfolio_optimisation/nested_clustered_optimisation

.. toctree::
    :maxdepth: 2
    :caption: Additional Information
    :hidden:

    additional_information/contributing
    additional_information/license

.. image:: logo/black_logo.png
   :scale: 15 %
   :align: center
   :target: https://hudsonthames.org/

|

================================================
Machine Learning Financial Laboratory (mlfinlab)
================================================

MlFinlab is a python package which helps portfolio managers and traders who want to leverage the power of machine learning
by providing reproducible, interpretable, and easy to use tools.

Adding MlFinLab to your companies pipeline is like adding a department of PhD researchers to your team.

.. code-block::

   pip install mlfinlab

We source all of our implementations from the most elite and peer-reviewed journals. Including publications from:

1. `The Journal of Financial Data Science <https://jfds.pm-research.com/>`_
2. `The Journal of Portfolio Management <https://jpm.pm-research.com/>`_
3. `The Journal of Algorithmic Finance <http://www.algorithmicfinance.org/>`_
4. `Cambridge University Press <https://www.cambridge.org/>`_


Documentation & Tutorials
#########################

We lower barriers to entry for all users by providing extensive `documentation <https://mlfinlab.readthedocs.io/en/latest/>`_
and `tutorial notebooks <https://github.com/hudson-and-thames/research>`_, with code examples.

Who is Hudson & Thames?
#######################

Hudson and Thames Quantitative Research is a company with a focus on implementing the most cutting edge algorithms in
quantitative finance. We productionalize all our tools in the form of libraries and provide capability to our clients.

* `Website <https://hudsonthames.org/>`_
* `Github Group <https://github.com/hudson-and-thames>`_
* `MlFinLab Documentation <https://mlfinlab.readthedocs.io/en/latest/>`_

Contact us
##########

The best place to contact the team is via the Slack channel. Alternatively you can email us at: research@hudsonthames.org.

Looking forward to hearing from you!

License
#######

This project is licensed under an all rights reserved licence and is NOT open-source.

`LICENSE.txt <https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt>`_ file for details.


.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :hidden:

    getting_started/installation
    additional_information/contact
    getting_started/barriers_to_entry
    getting_started/researcher
    getting_started/datasets
    getting_started/research_tools
    getting_started/research_notebooks
    getting_started/portfoliolab
    getting_started/arbitragelab

.. toctree::
    :maxdepth: 2
    :caption: Feature Engineering
    :hidden:

    implementations/data_structures
    implementations/filters
    implementations/frac_diff
    implementations/structural_breaks
    implementations/microstructural_features


.. toctree::
    :maxdepth: 2
    :caption: Codependence
    :hidden:

    codependence/introduction
    codependence/correlation_based_metrics
    codependence/information_theory_metrics
    codependence/codependence_marti
    codependence/codependence_matrix
    codependence/optimal_transport

.. toctree::
    :maxdepth: 2
    :caption: Data Generation
    :hidden:

    data_generation/introduction
    data_generation/corrgan
    data_generation/vine_methods
    data_generation/correlated_random_walks
    data_generation/hcbm
    data_generation/bootstrap
    data_generation/data_verification

.. toctree::
    :maxdepth: 2
    :caption: Labeling
    :hidden:

    labeling/tb_meta_labeling
    labeling/labeling_trend_scanning
    labeling/labeling_tail_sets
    labeling/labeling_fixed_time_horizon
    labeling/labeling_matrix_flags
    labeling/labeling_excess_median
    labeling/labeling_raw_return
    labeling/labeling_vs_benchmark
    labeling/labeling_excess_mean


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
    :caption: Networks
    :hidden:

    networks/introduction
    networks/mst
    networks/almst
    networks/pmfg
    networks/visualisations
    networks/dash


.. toctree::
    :maxdepth: 2
    :caption: Clustering
    :hidden:

    implementations/onc
    implementations/feature_clusters
    implementations/hierarchical_clustering

.. toctree::
    :maxdepth: 2
    :caption: Backtest Overfitting
    :hidden:

    implementations/backtesting
    implementations/backtest_statistics

.. toctree::
    :maxdepth: 2
    :caption: Portfolio Optimisation
    :hidden:

    portfolio_optimisation/portfolio_optimisation

.. toctree::
    :maxdepth: 2
    :caption: Online Portfolio Selection
    :hidden:

    online_portfolio_selection/online_portfolio_selection

.. toctree::
    :maxdepth: 2
    :caption: Optimal Mean Reversion
    :hidden:

    optimal_mean_reversion/introduction
    optimal_mean_reversion/ou_model

.. toctree::
    :maxdepth: 3
    :caption: Developer Guide
    :hidden:

    changelog

.. toctree::
    :maxdepth: 2
    :caption: Additional Information
    :hidden:

    additional_information/contributing
    additional_information/analytics
    additional_information/license

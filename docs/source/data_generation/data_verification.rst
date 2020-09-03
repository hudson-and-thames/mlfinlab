.. _data_generation-data_verification:

.. note::
   The following implementation and documentation closely follow the work of Gautier Marti:
   `CorrGAN: Sampling Realistic Financial Correlation Matrices using Generative Adversarial Networks <https://arxiv.org/pdf/1910.09504.pdf>`_.

=================
Data Verification
=================

Data verification for synthetic data is needed to confirm if it shares some properties of the original data.

Stylized Factors of Correlation Matrices
########################################

Following the work of Gautier Marti in CorrGAN, we provide function to plot and verify a synthetic matrix has the 6 stylized facts of empirical
correlation matrices.

The stylized facts are:

1. Distribution of pairwise correlations is significantly shifted to the positive.
2. Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first eigenvalue (the market).
3. Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other large eigenvalues (industries).
4. Perron-Frobenius property (first eigenvector has positive entries).
5. Hierarchical structure of correlations.
6. Scale-free property of the corresponding Minimum Spanning Tree (MST).

Implementation
##############

.. py:currentmodule:: mlfinlab.data_generation.data_verification

.. autofunction:: plot_stylized_facts

.. autofunction:: plot_pairwise_dist

.. autofunction:: plot_eigenvalues

.. autofunction:: plot_eigenvectors

.. autofunction:: plot_heirarchical_structure

.. autofunction:: plot_mst_degree_count

Example Code
############

.. code-block::

    import yfinance as yf
    from mlfinlab.data_generation.corrgan import sample_from_corrgan
    from mlfinlab.data_generation.data_verification import plot_stylized_facts

    # Download stock data from yahoo finance.
    dimensions = 3
    prices = yf.download(tickers=" ".join(["AAPL", "MSFT", "AMZN"]), period='1y')['Close']

    # Calculate correlation matrices.
    prices = prices.pct_change()
    rolling_corr = prices.rolling(252, min_periods=252//2).corr().dropna()

    # Generate same quantity of data from CorrGAN.
    corrgan_mats = sample_from_corrgan(model_loc="corrgan_models",
                                       dim=dimensions,
                                       n_samples=len(rolling_corr.index.get_level_values(0).unique()))

    # Transform from pandas to numpy array.
    empirical_mats = []
    for date, corr_mat in rolling_corr.groupby(level=0):
        empirical_mats.append(corr_mat.values)
    empirical_mats = np.array(empirical_mats)

    # Plot all stylized facts.
    plot_stylized_facts(empirical_mats, corrgan_mats)


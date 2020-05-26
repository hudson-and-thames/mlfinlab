.. _online_portfolio_selection-momentum-exponential_gradient:

.. note::
    The online portfolio selection module contains different algorithms that are used for asset allocation and optimizing strategies. Each
    algorithm is encapsulated in its own class and has a public method called ``allocate()`` which calculates the weight allocations
    on the specific user data. This way, each implementation can be called in the same way and makes it simple for users to use them.
    Next up, let's discuss some of these implementations and the different parameters they require.

====================
Exponential Gradient
====================

Exponential Gradient is a momentum strategy that focuses on the best performing asset of the last time period.
The portfolio shifts its weights to the best performing asset of the last period with an adjustment of :math:`\eta`, the learning rate.
A higher value of :math:`\eta` indicates the aggressiveness of the strategy to match the best performing assets. A lower value of :math:`\eta`
indicates the passiveness of the strategy to match the best performing assets.

.. math::
    b_{t+1} = \underset{b \in \Delta_m}{\arg\max} \: \eta \log b \cdot x_t - R(b,b_t)

- :math:`b_t` is the portfolio vector at time :math:`t`.
- :math:`x_t` is the price relative change at time :math:`t`. It is calculated by :math:`\frac{p_t}{p_{t-1}}`, where :math:`p_t` is the price at time :math:`t`.
- :math:`R(b, b_t)` is the regularization term for exponential gradient. Different update rules will use different regularization terms.
- :math:`\Delta_m` is the simplex domain. The sum of all elements is 1, and each element is in the range of [0, 1].

Exponential Gradients have an extremely efficient computational time that scales with the number of assets,
and broadly speaking, there are three update methods to iteratively update the selection of portfolio weights.

1. Multiplicative Update
########################

David Helmbold first proposed a regularization term that adopts relative entropy in his `paper <https://www.cis.upenn.edu/~mkearns/finread/portfolio.pdf>`_.

.. math::
    R(b,b_t) = \overset{m}{\underset{i=1}{\sum}}b_i \log \frac{b_i}{b_{t,i}}

Using log's first order taylor expansion of :math:`b_i`

.. math::
    \log b \cdot x_{t, i} \approx \log(b_t \cdot x_{t, i}) + \frac{x_{t, i}}{b_t \cdot x_{t, i}}(b-b_t)

Multiplicative update algorithm can be stated as the following.

.. math::
    b_{t+1} = b_t \cdot \exp \left( \eta \frac{x_t}{b_t \cdot x_t} \right) / Z

where :math:`Z` is a normalization term to sum the weights to 1.

2. Gradient Projection
######################

Instead of relative entropy, gradient projection adopts an :math:`L_2`-regularization term for the optimization equation.

.. math::
    R(b,b_t) = \frac{1}{2}\overset{m}{\underset{i=1}{\sum}}(b_i - b_{t,i})^2

Gradient projection can then be iteratively updated with the following equation.

.. math::
    b_{t+1} = b_t + \eta \cdot \left( \frac{x_t}{b_t \cdot x_t} - \frac{1}{m} \sum_{j=1}^{m} \frac{x_t}{b_t \cdot x_t} \right)


3. Expectation Maximization
###########################

Lastly, Expectation Maximization uses a :math:`\chi^2` regularization term.

.. math::
    R(b, b_t)=\frac{1}{2}\overset{m}{\underset{i=1}{\sum}}\frac{(b_i - b_{t,i})^2}{b_{t,i}}

Then the corresponding update rule becomes

.. math::
    b_{t+1} = b_t \cdot \left( \eta \cdot \left( \frac{x_t}{b_t \cdot x_t} - 1 \right) + 1 \right)

.. tip::

    The following research `notebook <https://github.com/hudson-and-thames/research/blob/master/Online%20Portfolio%20Selection/Online%20Portfolio%20Selection%20-%20Momentum.ipynb>`_
    provides a more detailed exploration of the strategies.

Parameters
----------

The optimal parameters depend on each dataset. For NYSE, a low value of :math:`\eta` was optimal, which
indicates a lack of a clear momentum strategy.

.. image:: momentum_images/nyse_eg_eta_0_1.png
   :width: 49 %

.. image:: momentum_images/nyse_eg_eta_1_100.png
   :width: 49 %

However, for the MSCI dataset, we see a high value of optimal :math:`\eta`, which indicates a presence
of a momentum strategy.

.. image:: momentum_images/msci_eg_eta_0_1.png
   :width: 49 %

.. image:: momentum_images/msci_eg_eta_1_100.png
   :width: 49 %

Implementation
--------------

.. automodule:: mlfinlab.online_portfolio_selection.momentum.exponential_gradient

    .. autoclass:: EG
        :members:
        :show-inheritance:
        :inherited-members:

        .. automethod:: __init__

Example Code
############

.. code-block::

    import pandas as pd
    from mlfinlab.online_portfolio_selection import *

    # Read in data.
    stock_prices = pd.read_csv('FILE_PATH', parse_dates=True, index_col='Date')

    # Compute Multiplicative Update with eta of 0.2 with no given weights.
    mu = EG(update_rule='MU', eta=0.2)
    mu.allocate(asset_prices=stock_prices, resample_by='W', verbose=True)

    # Compute Gradient Projection with eta of 0.5 with given weights.
    gp = EG(update_rule='GP', eta=0.5)
    gp.allocate(asset_prices=stock_prices, weights=some_weight)

    # Compute Expectation Maximization with eta of 0.8 with given weights.
    em = EG(update_rule='EM', eta=0.8)
    em.allocate(asset_prices=stock_prices, weights=some_weight)

    # Get the latest predicted weights.
    em.weights

    # Get all weights for the strategy.
    mu.all_weights

    # Get portfolio returns.
    gp.portfolio_return

.. tip::

    Strategies were implemented with modifications from `Li, B., Hoi, S. C.H., 2012. OnLine Portfolio Selection: A Survey. ACM Comput.
    Surv. V, N, Article A (December 2012), 33 pages. <https://arxiv.org/abs/1212.2129>`_

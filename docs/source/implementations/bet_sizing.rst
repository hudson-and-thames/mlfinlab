.. py:currentmodule:: mlfinlab.bet_sizing.bet_sizing, mlfinlab.bet_sizing.ef3m

==========
Bet Sizing
==========

Introduction and motivation for the bet sizing module.

Bet Sizing Methods
==================

Bet Size Probability
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: bet_size_probability

Bet Size Dynamic
~~~~~~~~~~~~~~~~
.. autofunction:: bet_size_dynamic

Bet Size Budget
~~~~~~~~~~~~~~~
.. autofunction:: bet_size_budget

Bet Size Reserve
~~~~~~~~~~~~~~~~
.. autofunction:: bet_size_reserve


See also
========

:doc:`ef3m`
~~~~~~~~~~~
The bet sizing function `bet_size_reserve` makes use of the EF3M algorithm for fitting a mixture of two Gaussian distributions to the distribution of concurrent bet sides. This functionality is encapsulated in a separate module.


.. _implementations-cross_validation:

================
Cross Validation
================

This implementation is based on Chapter 7 of the book Advances in Financial Machine Learning. The purpose of performing
cross validation is to reduce the probability of over-fitting and the book recommends it as the main tool of research.
There are two innovations compared to the classical K-Fold Cross Validation implemented in `sklearn <https://scikit-learn.org/>`_.

1. The first one is a process called **purging** which removes from the *training* set those samples that are build with
   information that overlaps samples in the *testing* set. More details on this in section 7.4.1, page 105.

.. figure:: cross_validation_images/purging.png
   :scale: 30 %
   :align: center
   :figclass: align-center
   :alt: purging image

   Image showing the process of **purging**. Figure taken from page 107 of the book.


2. The second innovation is a process called **embargo** which removes a number of observations from the *end* of the
test set. This further prevents leakage where the purging process is not enough. More details on this in section 7.4.2, page 107.


.. figure:: cross_validation_images/embargo.png
   :scale: 30 %
   :align: center
   :figclass: align-center
   :alt: embargo image

   Image showing the process of **embargo**. Figure taken from page 108 of the book.

Implementation
##############

.. py:currentmodule:: mlfinlab.cross_validation.cross_validation
.. automodule:: mlfinlab.cross_validation.cross_validation
   :members:

Research Notebooks
##################

* `Answering Chapter 7: Cross-Validation Questions <https://github.com/hudson-and-thames/research/blob/master/Chapter7_CrossValidation/Chapter7_Exercises_CrossValidation.ipynb>`_

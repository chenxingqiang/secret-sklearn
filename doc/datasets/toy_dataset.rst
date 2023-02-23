.. Places parent toc into the sidebar

:parenttoc: True

.. _toy_datasets:

Toy datasets
============

.. currentmodule:: sflearn.datasets

scikit-learn comes with a few small standard datasets that do not require to
download any file from some external website.

They can be loaded using the following functions:

.. autosummary::

   load_iris
   load_diabetes
   load_digits
   load_linnerud
   load_wine
   load_breast_cancer

These datasets are useful to quickly illustrate the behavior of the
various algorithms implemented in scikit-learn. They are however often too
small to be representative of real world machine learning tasks.

.. include:: ../../sflearn/datasets/descr/iris.rst

.. include:: ../../sflearn/datasets/descr/diabetes.rst

.. include:: ../../sflearn/datasets/descr/digits.rst

.. include:: ../../sflearn/datasets/descr/linnerud.rst

.. include:: ../../sflearn/datasets/descr/wine_data.rst

.. include:: ../../sflearn/datasets/descr/breast_cancer.rst

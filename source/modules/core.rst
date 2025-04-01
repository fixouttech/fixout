.. _core:

===========
FixOut Core
===========

.. warning ::

    FixOut is a tool centered on *sensitive features* and *their proxies*. 
    **Not providing this information correctly beforehand will compromise the obtained results**.

.. _datadist:

Data distribution 
=================

FixOut shows the data distribution centered on sensitive features.
Plots show the data frequency based on each possible value of a 
sensitive feature and the target variable :math:`y`.

**Intersectionality**. 
It also combines sensitive features to help users to verify the distribution
take into account two sensitive features at the same time.

.. figure:: ../figures/datadistr.PNG
   :target: ../figures/datadistr.PNG
   :align: center
   :scale: 40

.. note:: 
    In the current version of FixOut, this functionality is only available using the web dashboard.


.. _correlation:

Correlation analysis
====================

Correlation matrices are automatically computed using two coefficients: Pearson and Spearman.
These matrices are computed taking into account all features: sensitive or not.

FixOut also delivers a list of *non*-sensitive features that are correlated to sensitive features. 
One list is generated for one sensitive feature in decreasing order based on the absolute value of the correlation.  

.. warning ::
    Non-sensitive features that are highly correlated to sensitive features must be treated as sensitive features as well.


.. figure:: ../figures/corr.PNG
   :target: ../figures/corr.PNG
   :align: center
   :scale: 40

.. note:: 
    In the current version of FixOut, this functionality is only available using the web dashboard.

.. _fairness:

Fairness assessment
===================

FixOut automatically calculates several fairness metrics given the output of a model [EU2023]_.
The list of all fairness metrics currently available is shown below.

* Equalized Odds (EOD)
* Demographic Parity (DP)
* Equal Opportunity (EO)
* Predictive Equality (PE)
* Predictive Parity (PP)
* Conditional Accuracy Equality (CEA)

.. seealso::
    * :class:`fixout.fairness`

.. _reverse_eng:

Reverse engineering
====================

FixOut trains (and evaluates) various models to predict sensitive features using 
the target variable :math:`y`. This type of experiment is useful for analyzing biases.
The idea is to verify whether it is possible to reconstruct 
sensitive features with good performance using only :math:`y`.
See :ref:`supported_models` to check out which models can be used in this analysis.

.. seealso::
    * :class:`fixout.helper.ReverseFairness`



.. _discriminatory_model:

Discriminatory models 
======================

FixOut trains models using only sensitive features to check if its possible to achieve high performance using sensitive features only 
and then to help practionners to assess (and avoid) biases. 
See :ref:`supported_models` to check out which models can be used in this analysis.


.. seealso::
    * :class:`fixout.helper.UnfairModel`


.. _supported_models:

Models compatible with FixOut
=============================

The types of models that are compatible with FixOut are briefly shown below.
FixOut mainly supports trained models for binary classification. They must be compatible with 
the API definition of scikit-learn.

.. tab:: scikit-learn

    This is not an exhaustive list, but some examples include: Linear Regression, Decision Trees, 
    Support Vector Machines, Gaussian Naive Bayes, Random Forests, Multi-layer Perceptron, 
    and Gradient Boosting.

.. warning ::

    FixOut is still in development. **New types of models will be supported soon.**

.. :class:`sklearn.naive_bayes.GaussianNB`
.. :class:`sklearn.ensemble.RandomForestClassifier`
.. :class:`sklearn.neural_network.MLPClassifier`
.. :class:`sklearn.ensemble.GradientBoostingClassifier`

.. rubric:: References

.. [EU2023] `"Survey on fairness notions and related tensions" <https://doi.org/10.1016/j.ejdp.2023.100033>`_ .
    Alves, G., Bernier, F., Couceiro, M., Makhlouf, K., Palamidessi, C., & Zhioua, S. - 2023.
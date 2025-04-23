=====================
FixOut Web Dashboard
=====================

Intro
==================


.. warning ::

    FixOut is a tool centered on *sensitive features* and *their proxies*. 
    **Not providing this information correctly beforehand will compromise the obtained results**.

How to activate it 
==================

If you want to activate the web interface that generates the dashboard with the fairness assessment of 
your model and other functionalities, you must execute the ``run`` function **with the** 
``show`` **parameter activated** (from :class:`fixout.runner.FixOutRunner`).


    >>> fxo.run(fxa, show=True)


Once execution is complete, you will be able to see a web page like the following by accessing ``http://localhost:5000``, 
if you use the default parameters.


.. figure:: ../figures/interface_data.PNG
   :target: ../figures/interface_data.PNG
   :align: center
   :scale: 30


Main functionalities
====================

The dashboard displays FixOut's functionalities organized in tabs and centered on sensitive features.
The two first tabs are briefly described bellow: **Data** and **Metrics**.

.. note::
    Other features related to explainability and model-specific inspection are not available 
    in this version of FixOut.

Data
-----

In this tab, FixOut displays the results obtained after running pipelines maily related to data.

* **Data distribution** (see :ref:`datadist` section)
* **Correlation analysis** (see :ref:`correlation` section)
* **Reverse engineering** (see :ref:`reverse_eng` section and :class:`fixout.helper.ReverseFairness`)
* **Discriminatory models** (see :ref:`discriminatory_model` section and :class:`fixout.helper.UnfairModel`)


Metrics
--------

In this tab, the plots are obtained after calculating the fairness metrics.
See also :ref:`fairness` section and :class:`fixout.fairness`.


.. seealso::

    Check out the :ref:`core` section
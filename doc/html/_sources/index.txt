.. source documentation master file, created by
   sphinx-quickstart on Tue Jan 31 12:35:38 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Random Forests
==================================================

I started this project to better understand the way `decision trees <https://en.wikipedia.org/wiki/Decision_tree>`_ and `random forests <https://en.wikipedia.org/wiki/>`_ work.  At this point classifiers use the gini-index and the regression models use the mean square error for splitting.  All the methods are built to work with datasets that are lists of lists, where the target variable values are the right most column.  The random forests methods can also work with datasets that use `Pandas DataFrames <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_ and `Pandas Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`_.


Requirements
------------
* `Python <https://www.python.org/>`_
* `Pandas <http://pandas.pydata.org/>`_ 
* `NumPy <http://www.numpy.org/>`_
* `Sphinx <http://www.sphinx-doc.org/en/stable/>`_

Example
-------
>>> dataset = [[2.771244718, 1.784783929, 0],
	       [1.728571309, 1.169761413, 0],
	       [3.678319846, 2.81281357, 1],
	       [3.961043357, 2.61995032, 1],
	       [2.999208922, 2.209014212, 0],
	       [7.497545867, 3.162953546, 0],
	       [9.00220326, 3.339047188, 1],
	       [7.444542326, 0.476683375, 1],
	       [10.12493903, 3.234550982, 0],
	       [6.642287351, 3.319983761, 1]]
>>> data_point = pd.Series([2.0, 23.0], index=['feature_1','feature_2'])
>>> import pandas as pd
>>> df = pd.DataFrame(data=dataset,columns =['feature_1','feature_2','target'])

>>> from TreeMethods.DecisionTreeClassifier import DecisionTreeClassifier
>>> tree = DecisionTreeClassifier(max_depth=2,min_size=1)
>>> tree.fit(df,target='target')
>>> tree.predict(data_point)
0

>>> from TreeMethods.RandomForestClassifier import RandomForestClassifier
>>> forest = RandomForestClassifier(n_trees=10
                                   max_depth=5,
                                   min_size=1)
>>> forest.fit(df, target='target')
>>> forest.predict(data_point)
0


Documentation
-------------
See the submodules below for documenation:

.. toctree::
   :maxdepth: 4

   TreeMethods

Testing
-------

From the main directory run:

*py.test tests*

to run unit tests on code, more tests will be added soon.

References
----------

`An Introduction To Statistical Learning <http://www-bcf.usc.edu/~gareth/ISL/">`_

`Elements Of Statistical Learning <http://statweb.stanford.edu/~tibs/ElemStatLearn/>`_

`Scikit-learn <http://scikit-learn.org/stable/auto_examples/index.html#ensemble-methods>`_

`How to Implement Random Forest From Scratch In Python <http://machinelearningmastery.com/implement-random-forest-scratch-python/>`_

`How To Implement A Decision Tree From Scratch In Python <http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/>`_


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. source documentation master file, created by
   sphinx-quickstart on Tue Jan 31 12:35:38 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Random Forests
==================================================

This is a python pacakge to show how to build a random forest classifier and regressor in Python.

Requirements
------------
* `Python <https://www.python.org/>`_
* `Pandas <http://pandas.pydata.org/>`_ 
* `NumPy <http://www.numpy.org/>`_

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
>>>
>>> df = pd.DataFrame(data=dataset,columns =['feature_1','feature_2','target'])
>>> tree = DecisionTree(2,1)
>>> tree.fit(df,target='target')
>>> data_point = pd.Series([2.0, 23.0], index=['feature_1','feature_2'])
>>> tree.predict(data_point)
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

to run unit tests on code.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


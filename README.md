# Random Forests In Python

## Intoduction
I started this project to better understand the way <a href="https://en.wikipedia.org/wiki/Decision_tree">decision trees</a> and <a href="https://en.wikipedia.org/wiki/Random_forest">random forests</a> work.  At this point the code only the classifiers ( decision tree and random forest) based off the gini-index are working, but I am working on extending it to regression trees and random forests.  The classifiers are built to work with datasets that are lists of lists, where the target variable values are the right most column.  It can also work with datasets that use <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html">Pandas DataFrames</a> and <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html">Pandas Series</a>.

## Dependencies
The dependencies for this project are rather minimal, including,

1. <a href="https://www.python.org/">Python</a> 2.7
2. <a href="http://pandas.pydata.org/">Pandas</a>
3. <a href="http://www.numpy.org/">NumPy</a>
3. <a href="http://www.sphinx-doc.org/en/stable/">Sphinx</a> (for documentation only)

You can install all the dependencies using <a href="https://pip.pypa.io/en/stable/">pip</a> (except for python and Sphinx) by entering into the commandline,

	pip install -r requirements.txt

## Example

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
	>>> data_point = pd.Series([2.0, 23.0], index=['feature_1','feature_2'])
	>>> import pandas as pd
	>>> df = pd.DataFrame(data=dataset,columns =['feature_1','feature_2','target'])
	>>>
	>>> from TreeMethods.DecisionTreeClassifier import DecisionTreeClassifier
	>>> tree = DecisionTreeClassifier(max_depth=2,min_size=1)
	>>> tree.fit(df,target='target')
	>>>
	>>> tree.predict(data_point)
	0
	>>>
	>>> from TreeMethods.RandomForestClassifier import RandomForestClassifier
	>>> forest = RandomForestClassifier(n_trees=10
                                   max_depth=5,
                                   min_size=1)
	>>> forest.fit(df, target='target')
	>>> forest.predict(data_point)
	0

## Testing
To test the code type the following command from the terminal in the <code>RandomForest</code> directory,

<code> py.test tests</code>

More tests will be added in the near future.

## Documentation

To build the documentation on your local machine type the following commands from <code>RandomForest</code>  directory,

<code> sphinx-apidoc -F -o doc/ TreeMethods/ </code>

Then cd into the <code>doc/</code> directory and type,

<code> make html </code>

The html documentation will be in the directory <code>_build/html/</code>.  Open the file <code>index.html</code>.


## References
- <a href="http://www-bcf.usc.edu/~gareth/ISL/">An Introduction To Statistical Learning</a>

- <a href="http://statweb.stanford.edu/~tibs/ElemStatLearn/">Elements Of Statistical Learning</a>

- <a href="http://scikit-learn.org/stable/auto_examples/index.html#ensemble-methods">Scikit-learn</a>

- <a href="http://machinelearningmastery.com/implement-random-forest-scratch-python/">
How to Implement Random Forest From Scratch In Python</a>

- <a href="http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/">How To Implement A Decision Tree From Scratch In Python</a>
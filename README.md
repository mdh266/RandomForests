[![Build Status](https://travis-ci.com/mdh266/RandomForests.svg?branch=master)](https://travis-ci.com/mdh266/RandomForests)
[![codecov](https://codecov.io/gh/mdh266/RandomForests/branch/master/graph/badge.svg)](https://codecov.io/gh/mdh266/RandomForests)


# Random Forests In Python
--------------


## Intoduction
-------------
I started this project to better understand the way <a href="https://en.wikipedia.org/wiki/Decision_tree">decision trees</a> and <a href="https://en.wikipedia.org/wiki/Random_forest">random forests</a> work.  At this point the classifiers are only based off the gini-index and the regression models are based off the mean square error.  Both the classifiers and regression models are built to work with datasets that are lists of lists, where the target variable values are the right most column.  It can also work with datasets that use <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html">Pandas DataFrames</a> and <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html">Pandas Series</a>.

## Dependencies
--------------
You can install all the dependencies using <a href="https://pip.pypa.io/en/stable/">pip</a> (except for python and Sphinx) by entering into the commandline,

	pip install -r requirements.txt

## References
---------------
- <a href="http://www-bcf.usc.edu/~gareth/ISL/">An Introduction To Statistical Learning</a>

- <a href="http://statweb.stanford.edu/~tibs/ElemStatLearn/">Elements Of Statistical Learning</a>

- <a href="http://scikit-learn.org/stable/auto_examples/index.html#ensemble-methods">Scikit-learn</a>

- <a href="http://machinelearningmastery.com/implement-random-forest-scratch-python/"> How to Implement Random Forest From Scratch In Python</a>

- <a href="http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/">How To Implement A Decision Tree From Scratch In Python</a>

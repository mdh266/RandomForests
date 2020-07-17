[![Build Status](https://travis-ci.com/mdh266/RandomForests.svg?branch=master)](https://travis-ci.com/mdh266/RandomForests)


# Random Forests In Python
--------------


## Intoduction
-------------
I started this project to better understand the way <a href="https://en.wikipedia.org/wiki/Decision_tree">decision trees</a> and <a href="https://en.wikipedia.org/wiki/Random_forest">random forests</a> work.  At this point the classifiers are only based off the gini-index and the regression models are based off the mean square error.  Both the classifiers and regression models are built to work with datasets that are lists of lists, where the target variable values are the right most column.  It can also work with datasets that use <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html">Pandas DataFrames</a> and <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html">Pandas Series</a>.

## Example

	from sklearn.datasets import load_breast_cancer
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	dataset = load_breast_cancer()

	cols = [dataset.data[:,i] for i in range(4)]

	X = pd.DataFrame({k:v for k,v in zip(dataset.feature_names,cols)})
	y = pd.Series(dataset.target)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=24)

	tree   = DecisionTreeClassifier()

	model  = tree.fit(X_train,y_train)

	preds  = model.predict(X_test)

	print("Accuracy: ", accuracy_score(preds, y_test))

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
